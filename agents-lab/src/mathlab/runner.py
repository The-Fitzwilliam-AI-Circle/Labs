"""Sequential episodes. Generation receives questions; only scoring reads answers."""

import asyncio
import hashlib
import importlib.util
import inspect
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mathlab.config import Limits
from mathlab.files import load_questions, sha256, write_json, write_jsonl_row
from mathlab.runtime import AgentContext, TraceWriter
from mathlab.types import EpisodeError, LabError, Result


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_run_directory(requested: Path) -> Path:
    """Atomically reserve a fresh directory, adding a suffix on collisions."""
    if not requested.name or requested.name == "..":
        raise LabError("--out must name a run directory, such as runs/experiment")
    requested.parent.mkdir(parents=True, exist_ok=True)
    candidate = requested
    suffix = 2
    while True:
        try:
            candidate.mkdir()
        except FileExistsError:
            # mkdir, rather than an exists check, also handles simultaneous runs
            # and collisions with files or dangling symlinks without overwriting.
            candidate = requested.with_name(f"{requested.name}-{suffix}")
            suffix += 1
        else:
            return candidate


def load_agent(path: Path):
    name = "mathlab_agent_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LabError(f"Cannot import agent {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        # Do not print arbitrary import-time exception bodies (may contain credentials).
        raise LabError(f"Agent import failed: {type(exc).__name__}") from exc
    if not inspect.iscoroutinefunction(getattr(module, "solve", None)):
        raise LabError(f"{path} must define async def solve(task, ctx)")
    return module


def git_state(directory: Path) -> dict:
    def git(*arguments):
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), *arguments],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    state = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if state is None else bool(state)}


async def run(
    *,
    agent_path: Path,
    questions_path: Path,
    out: Path,
    limits: Limits,
    model,
    python,
    model_id: str,
    image_id: str | None = None,
    provider_metadata: dict | None = None,
    selected_ids: list[str] | None = None,
    lock_path: Path | None = None,
    secrets: tuple[str, ...] = (),
    on_start: Callable[[Path], None] | None = None,
) -> dict:
    tasks = load_questions(questions_path)
    if selected_ids is not None:
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise LabError("Selected task IDs must be nonempty and unique")
        missing = set(selected_ids) - {task.id for task in tasks}
        if missing:
            raise LabError(f"Unknown selected task IDs: {', '.join(sorted(missing))}")
        tasks = [task for task in tasks if task.id in selected_ids]
    agent_path = agent_path.resolve()
    module = load_agent(agent_path)
    source = agent_path.read_bytes()
    out = create_run_directory(out)
    (out / "agent.py").write_bytes(source)
    lock_path = lock_path or Path.cwd() / "uv.lock"
    manifest = {
        "schema_version": 1,
        "output_directory": str(out.resolve()),
        "agent_path": str(agent_path),
        "agent_sha256": hashlib.sha256(source).hexdigest(),
        "prompt_configuration": getattr(module, "INSTRUCTIONS", None),
        "model": model_id,
        "provider_configuration": provider_metadata,
        "limits": asdict(limits),
        "scheduled_task_ids": [task.id for task in tasks],
        "questions_path": str(questions_path.resolve()),
        "questions_sha256": sha256(questions_path),
        "dependency_lock_sha256": sha256(lock_path) if lock_path.is_file() else None,
        "docker_image_id": image_id,
        "git": git_state(agent_path.parent),
        "started_at": utc_now(),
        "ended_at": None,
        "state": "incomplete",
    }
    # The API key is never recorded, even if an agent accidentally includes it in an error.
    with (
        (out / "results.jsonl").open("w", encoding="utf-8") as results_file,
        (out / "traces.jsonl").open("w", encoding="utf-8") as traces_file,
    ):
        trace = TraceWriter(traces_file, secrets + (os.getenv("OPENAI_API_KEY", ""),))
        write_json(out / "run.json", trace.redact(manifest))
        try:
            if on_start is not None:
                on_start(out)
            for task in tasks:
                ctx = AgentContext(task.id, limits, model, python, trace)
                ctx.record("episode_start", problem=task.problem)
                answer, status, error = None, "completed", None
                try:
                    async with asyncio.timeout(ctx.remaining()):
                        answer = await module.solve(task, ctx)
                        if not isinstance(answer, str):
                            raise TypeError("solve must return a string")
                        ctx.remaining()
                except (asyncio.CancelledError, KeyboardInterrupt):
                    ctx.record("episode_interrupted")
                    raise
                except TimeoutError:
                    status, error = "timeout", "Episode deadline exhausted"
                    answer = None
                except EpisodeError as exc:
                    status, error = exc.status, trace.redact(str(exc))[:1000]
                    answer = None
                except Exception as exc:
                    status = "agent_error"
                    error = trace.redact(f"{type(exc).__name__}: {exc}")[:1000]
                    answer = None
                result = Result(
                    id=task.id,
                    answer=answer,
                    status=status,
                    error=error,
                    model_requests=ctx.model_requests,
                    python_calls=ctx.python_calls,
                    elapsed_seconds=round(time.monotonic() - ctx.started, 6),
                    **asdict(ctx.usage),
                )
                row = trace.redact(asdict(result))
                write_jsonl_row(results_file, row)
                ctx.record("episode_end", result=row)
            manifest["state"] = "completed"
        finally:
            manifest["ended_at"] = utc_now()
            write_json(out / "run.json", trace.redact(manifest))
    return manifest
