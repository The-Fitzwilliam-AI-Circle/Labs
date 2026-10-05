import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mathlab.config import Limits
from mathlab.files import read_json, read_jsonl
from mathlab.runner import create_run_directory, run
from mathlab.scoring import score
from mathlab.types import LabError, ProviderError, ToolResult
from tests.conftest import ROOT, ScriptedModel, ScriptedPython, jsonl, response


def run_args(tmp_path, questions, model, **overrides):
    return {
        "agent_path": ROOT / "agents/tool.py",
        "questions_path": questions,
        "out": tmp_path / "run",
        "limits": Limits(),
        "model": model,
        "python": ScriptedPython(),
        "model_id": "scripted-test-model",
        "lock_path": ROOT / "uv.lock",
        **overrides,
    }


def test_full_loop_preserves_call_ids_and_reasoning_then_scores(tmp_path, questions):
    python_response = response("python", "print(3/4)", "python-call-42")
    model = ScriptedModel([python_response, response(value="0.75"), response(value="42")])
    tool = ScriptedPython(ToolResult("0.75\n", "", 0))
    out = tmp_path / "run"
    asyncio.run(run(**run_args(tmp_path, questions, model, python=tool)))
    continuation = model.requests[1]["history"]
    assert continuation[1:3] == python_response["output"]
    assert continuation[-1]["call_id"] == "python-call-42"
    assert json.loads(continuation[-1]["output"])["stdout"] == "0.75\n"
    assert tool.calls[0][0] == "print(3/4)"
    results = read_jsonl(out / "results.jsonl")
    assert [r["status"] for r in results] == ["completed", "completed"]
    assert results[0]["model_requests"] == 2
    manifest = read_json(out / "run.json")
    assert manifest["state"] == "completed"
    assert manifest["scheduled_task_ids"] == ["one", "two"]
    assert manifest["agent_sha256"] and manifest["dependency_lock_sha256"]
    assert (out / "agent.py").read_bytes() == (ROOT / "agents/tool.py").read_bytes()
    answers = jsonl(
        tmp_path / "answers.jsonl",
        [
            {"id": "one", "answer": "3/4"},
            {"id": "two", "answer": "42"},
        ],
    )
    assert score(out, answers)["correct"] == 2


@pytest.mark.parametrize(
    "first, status",
    [
        (ProviderError("test error"), "provider_error"),
        ({"status": "incomplete", "output": []}, "protocol_error"),
        (RuntimeError("participant bug"), "agent_error"),
    ],
)
def test_failure_records_row_and_next_task_runs(tmp_path, questions, first, status):
    model = ScriptedModel([first, response(value="42")])
    asyncio.run(run(**run_args(tmp_path, questions, model)))
    rows = read_jsonl(tmp_path / "run/results.jsonl")
    assert [row["status"] for row in rows] == [status, "completed"]
    assert rows[0]["answer"] is None


def test_scripted_five_question_participant_run(tmp_path):
    model = ScriptedModel([response(value=value) for value in ("3/4", "-3", "5050", "1/2", "7/2")])
    asyncio.run(
        run(
            **run_args(
                tmp_path,
                ROOT / "data/smoke/questions.jsonl",
                model,
                agent_path=ROOT / "agents/participant.py",
            )
        )
    )
    assert score(tmp_path / "run", ROOT / "data/smoke/answers.jsonl")["correct"] == 5


def test_generation_never_opens_keys_and_task_has_only_allowed_fields(
    tmp_path, questions, monkeypatch
):
    key = jsonl(tmp_path / "answers.jsonl", [{"id": "one", "answer": "GOLD_SENTINEL"}])
    agent = tmp_path / "agent.py"
    agent.write_text(
        "async def solve(task, ctx):\n"
        "    assert set(vars(task)) == {'id', 'problem'}\n"
        "    turn = await ctx.model([{'role':'user', 'content': task.problem}], '')\n"
        "    return turn.value\n"
    )
    original = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path != key, "Generation opened the answer key"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    model = ScriptedModel([response(), response()])
    asyncio.run(run(**run_args(tmp_path, questions, model, agent_path=agent)))
    assert "GOLD_SENTINEL" not in json.dumps(model.requests)


def test_verifier_candidate_does_not_submit_episode_or_reset_budget(tmp_path, questions):
    agent = tmp_path / "verifier.py"
    agent.write_text(
        "async def solve(task, ctx):\n"
        "    candidate = await ctx.model([], 'candidate')\n"
        "    verification = await ctx.model([], 'fresh verifier')\n"
        "    return verification.value\n"
    )
    model = ScriptedModel([response(value="999"), response(value="3/4")])
    asyncio.run(
        run(
            **run_args(
                tmp_path,
                questions,
                model,
                agent_path=agent,
                selected_ids=["one"],
            )
        )
    )
    assert read_jsonl(tmp_path / "run/results.jsonl")[0]["answer"] == "3/4"
    model = ScriptedModel([response(value="999")])
    asyncio.run(
        run(
            **run_args(
                tmp_path,
                questions,
                model,
                agent_path=agent,
                selected_ids=["one"],
                out=tmp_path / "budget",
                limits=Limits(max_model_requests=1),
            )
        )
    )
    row = read_jsonl(tmp_path / "budget/results.jsonl")[0]
    assert row["status"] == "budget_exceeded" and row["answer"] is None


def test_cancellation_marks_run_incomplete(tmp_path, questions):
    model = ScriptedModel([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run(**run_args(tmp_path, questions, model)))
    manifest = read_json(tmp_path / "run/run.json")
    assert manifest["state"] == "incomplete" and manifest["ended_at"]


def test_agent_deadline_and_non_string_return(tmp_path, questions):
    agent = tmp_path / "slow.py"
    agent.write_text(
        "import asyncio\nasync def solve(task, ctx):\n"
        "    if task.id == 'one': await asyncio.sleep(1)\n"
        "    return 42\n"
    )
    asyncio.run(
        run(
            **run_args(
                tmp_path,
                questions,
                ScriptedModel([]),
                agent_path=agent,
                limits=Limits(episode_timeout_seconds=0.02),
            )
        )
    )
    assert [r["status"] for r in read_jsonl(tmp_path / "run/results.jsonl")] == [
        "timeout",
        "agent_error",
    ]


def test_repeated_runs_preserve_every_previous_artifact(tmp_path, questions):
    first = asyncio.run(
        run(
            **run_args(
                tmp_path,
                questions,
                ScriptedModel([response(), response(value="42")]),
            )
        )
    )
    original = Path(first["output_directory"])
    snapshots = {path.name: path.read_bytes() for path in original.iterdir()}
    started = []

    def on_start(directory):
        assert read_json(directory / "run.json")["state"] == "incomplete"
        started.append(directory)

    for suffix in (2, 3):
        manifest = asyncio.run(
            run(
                **run_args(
                    tmp_path,
                    questions,
                    ScriptedModel([response(value="0"), response(value="0")]),
                    on_start=on_start,
                )
            )
        )
        directory = tmp_path / f"run-{suffix}"
        assert Path(manifest["output_directory"]) == directory
        assert read_json(directory / "run.json") == manifest
        assert read_jsonl(directory / "results.jsonl")[0]["answer"] == "0"
        assert {path.name: path.read_bytes() for path in original.iterdir()} == snapshots
    assert started == [tmp_path / "run-2", tmp_path / "run-3"]


def test_directory_reservation_skips_files_and_dangling_symlinks(tmp_path):
    base = tmp_path / "experiment"
    base.mkdir()
    (tmp_path / "experiment-2").write_text("keep")
    (tmp_path / "experiment-3").symlink_to(tmp_path / "missing")
    assert create_run_directory(base) == tmp_path / "experiment-4"
    assert (tmp_path / "experiment-2").read_text() == "keep"
    assert (tmp_path / "experiment-3").is_symlink()


def test_simultaneous_directory_reservations_are_unique(tmp_path):
    base = tmp_path / "new-parent" / "experiment"
    with ThreadPoolExecutor(max_workers=8) as pool:
        directories = list(pool.map(create_run_directory, [base] * 8))
    assert len(set(directories)) == 8
    assert all(directory.is_dir() for directory in directories)
    assert {directory.name for directory in directories} == {
        "experiment",
        *(f"experiment-{i}" for i in range(2, 9)),
    }


def test_ids_validated_before_creating_another_run(tmp_path, questions):
    model = ScriptedModel([])
    (tmp_path / "run").mkdir()
    with pytest.raises(LabError, match="Unknown selected"):
        asyncio.run(run(**run_args(tmp_path, questions, model, selected_ids=["missing"])))
    assert not model.requests
    assert not (tmp_path / "run-2").exists()
