"""Installable CLI. Score is entirely independent of API credentials and Docker."""

import argparse
import asyncio
import io
import json
import os
import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from mathlab.config import load_limits, provider_configuration
from mathlab.model import ChatCompletionsModel, ResponsesModel
from mathlab.python_tool import DockerPython, inspect_image
from mathlab.runner import run
from mathlab.runtime import AgentContext, TraceWriter
from mathlab.scoring import parse_answer, score
from mathlab.types import LabError


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run, inspect, and score maths agents")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check local configuration and Docker image")
    doctor.add_argument("--config", type=Path, default=Path("config.toml"))
    live = doctor.add_mutually_exclusive_group()
    live.add_argument("--live", action="store_true", help="Uses the API: one function-call request")
    live.add_argument(
        "--live-tools",
        action="store_true",
        help="Uses two API requests and Docker to check the complete Python tool round trip",
    )
    execute = commands.add_parser("run", help="Run an agent against questions only")
    execute.add_argument("--config", type=Path, default=Path("config.toml"))
    execute.add_argument("--agent", type=Path, required=True)
    execute.add_argument("--questions", type=Path, required=True)
    execute.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Run directory; existing names receive -2, -3, etc. automatically",
    )
    execute.add_argument(
        "--id", dest="ids", action="append", help="Select a task ID; repeat as needed"
    )
    execute.add_argument(
        "--concurrency",
        type=positive_integer,
        default=4,
        help="Maximum simultaneous questions (default: 4; use 1 for serial execution)",
    )
    progress = execute.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show progress even when stderr is redirected",
    )
    progress.add_argument("--no-progress", dest="progress", action="store_false")
    execute.set_defaults(progress=None)
    grade = commands.add_parser("score", help="Score a completed run locally (no API or Docker)")
    grade.add_argument("--run", type=Path, required=True)
    grade.add_argument("--answers", type=Path, required=True)
    return root


def create_model(settings):
    if settings.provider == "openai":
        return ResponsesModel(settings.api_key, settings.model)
    return ChatCompletionsModel(
        settings.api_key, settings.model, settings.base_url, provider=settings.provider
    )


def configured_secrets():
    return tuple(
        os.getenv(name, "") for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LOCAL_API_KEY")
    )


async def live_check(limits, settings, *, image_id=None, check_tools=False):
    model = create_model(settings)
    try:
        ctx = AgentContext(
            "doctor",
            limits,
            model,
            DockerPython(image_id) if check_tools else None,
            TraceWriter(io.StringIO(), configured_secrets()),
        )
        if check_tools:
            history = [
                {
                    "role": "user",
                    "content": "Call python with code print(1 + 1). Do not submit an answer yet.",
                }
            ]
            turn = await ctx.model(
                history,
                "For this diagnostic, call python once, then submit_answer "
                "using the observed output.",
            )
            if turn.kind != "python":
                raise LabError(
                    "Live tool check expected a Python call; this model did not request one"
                )
            observation = await ctx.python(turn.value)
            if (
                observation.exit_code != 0
                or observation.timed_out
                or observation.output_limited
                or observation.stdout.strip() != "2"
            ):
                raise LabError("Live tool check did not produce the expected Python output")
            history.extend(turn.output_items)
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": turn.call_id,
                    "output": json.dumps(observation.to_dict()),
                }
            )
            turn = await ctx.model(
                history,
                "Use submit_answer to return the integer from the tool output.",
                allow_python=False,
            )
            if parse_answer(turn.value) != 2:
                raise LabError(
                    "Live tool check returned an incorrect answer after Python execution"
                )
            return
        turn = await ctx.model(
            [{"role": "user", "content": "What is 1 + 1?"}],
            "Use submit_answer to return the integer answer, with no prose.",
            allow_python=False,
        )
        if parse_answer(turn.value) != 2:
            raise LabError("Model's function call passed validation but its smoke answer was wrong")
    finally:
        await model.close()


async def execute_run(args, limits, settings, image_id):
    model = create_model(settings)
    try:
        with tqdm(
            total=0,
            desc="Episodes",
            unit="episode",
            dynamic_ncols=True,
            disable=None if args.progress is None else not args.progress,
        ) as progress:

            def report(completed, total, failed):
                progress.total = total
                progress.set_postfix(failed=failed, refresh=False)
                progress.update(completed - progress.n)
                if completed == 0:
                    progress.refresh()

            return await run(
                agent_path=args.agent,
                questions_path=args.questions,
                out=args.out,
                limits=limits,
                model=model,
                python=DockerPython(image_id),
                model_id=settings.model,
                provider_metadata=settings.public_metadata(),
                image_id=image_id,
                selected_ids=args.ids,
                secrets=configured_secrets(),
                on_start=lambda directory: print(f"Run directory: {directory}", flush=True),
                concurrency=args.concurrency,
                on_progress=report,
            )
    finally:
        await model.close()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "score":
            summary = score(args.run, args.answers)
            print(f"{summary['correct']}/{summary['total']} correct ({summary['accuracy']:.1%})")
            print(f"Scores: {args.run / 'scores.json'}")
            return 0

        load_dotenv(Path.cwd() / ".env", override=False)
        limits = load_limits(args.config)
        if args.command == "doctor":
            issues = []
            print(f"OK configuration: {args.config}")
            try:
                settings = provider_configuration()
                print(f"OK provider: {settings.provider}; model: {settings.model}")
                print(f"OK endpoint: {settings.base_url}")
            except LabError as exc:
                issues.append(str(exc))
            try:
                image_id = inspect_image(limits.python_image)
                print(f"OK Docker image: {image_id}")
            except LabError as exc:
                issues.append(str(exc))
            if issues:
                for issue in issues:
                    print(f"FAIL {issue}", file=sys.stderr)
                return 1
            if args.live_tools:
                print(
                    f"Live tool check: two API requests plus Python execution for {settings.model}",
                    flush=True,
                )
                asyncio.run(live_check(limits, settings, image_id=image_id, check_tools=True))
                print("OK live Python tool round trip")
            elif args.live:
                print(f"Live check: using the API for one request to {settings.model}", flush=True)
                asyncio.run(live_check(limits, settings))
                print("OK live function-calling check")
            else:
                print(
                    "No model API requests made. Use --live for one API check, "
                    "or --live-tools for the Python round trip."
                )
            return 0

        settings = provider_configuration()
        image_id = inspect_image(limits.python_image)
        manifest = asyncio.run(execute_run(args, limits, settings, image_id))
        output_directory = Path(manifest["output_directory"])
        if not args.out.is_absolute():
            output_directory = Path(os.path.relpath(output_directory))
        print(
            f"Finished {len(manifest['scheduled_task_ids'])} episodes. Results: {output_directory}"
        )
        print("Score this run with the matching answer key:")
        print(
            shlex.join(
                [
                    "uv",
                    "run",
                    "mathlab",
                    "score",
                    "--run",
                    str(output_directory),
                    "--answers",
                    str(args.questions.with_name("answers.jsonl")),
                ]
            )
        )
        return 0
    except (LabError, OSError, ValueError) as exc:
        # Also redact the configured key from any local I/O diagnostic.
        diagnostic = str(exc)
        for key in configured_secrets():
            if key:
                diagnostic = diagnostic.replace(key, "[REDACTED]")
        print(f"Error: {diagnostic[:1000]}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; an active run remains incomplete and cannot be scored.", file=sys.stderr
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
