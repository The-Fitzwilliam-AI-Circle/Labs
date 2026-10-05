import json
import shlex

import pytest

from mathlab.cli import main
from mathlab.config import Limits, load_limits
from mathlab.files import load_questions, read_json, read_jsonl, sha256
from mathlab.scoring import parse_answer
from mathlab.types import LabError
from tests.conftest import ROOT, ScriptedModel, jsonl, response
from tests.test_scoring import fixture_run


@pytest.fixture(autouse=True)
def isolated_provider_settings(monkeypatch):
    # CLI tests must not inherit the participant's current backend or credentials.
    monkeypatch.setattr("mathlab.cli.load_dotenv", lambda *args, **kwargs: None)
    for name in ("PROVIDER", "MODEL_BASE_URL", "OPENROUTER_API_KEY", "LOCAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "split, count",
    [
        ("smoke", 5),
        ("dev", 40),
        ("ladder/01-foundation", 10),
        ("ladder/02-computation", 10),
        ("ladder/03-structure", 10),
        ("ladder/04-challenge", 10),
    ],
)
def test_dataset_files_and_provenance(split, count):
    directory = ROOT / "data" / split
    tasks = load_questions(directory / "questions.jsonl")
    answers = read_jsonl(directory / "answers.jsonl")
    manifest = read_json(directory / "manifest.json")
    assert len(tasks) == len(answers) == count
    assert {task.id for task in tasks} == {row["id"] for row in answers}
    for row in answers:
        parse_answer(row["answer"])
    for name, digest in manifest["sha256"].items():
        assert sha256(directory / name) == digest
    if split == "dev":
        assert len(manifest["sources"][0]["source_revision"]) == 40
        assert len(manifest["selected_questions"]) == count


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_model_requests": -1},
        {"max_python_calls": True},
        {"python_timeout_seconds": 0},
        {"episode_timeout_seconds": float("nan")},
        {"python_output_limit_bytes": 2.5},
        {"python_image": "--privileged"},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(LabError):
        Limits(**kwargs)


def test_unknown_config_fails(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("max_model_request = 6\n")
    with pytest.raises(LabError, match="Unknown"):
        load_limits(config)


def test_score_cli_does_not_need_env_or_docker(tmp_path, monkeypatch, capsys):
    answers = fixture_run(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setattr("mathlab.cli.inspect_image", lambda *_: pytest.fail("Docker called"))
    monkeypatch.setattr("mathlab.cli.ResponsesModel", lambda *_: pytest.fail("API called"))
    assert main(["score", "--run", str(tmp_path), "--answers", str(answers)]) == 0
    assert "1/1 correct" in capsys.readouterr().out


def test_doctor_is_offline_and_reports_without_leaking_key(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-print")
    monkeypatch.setenv("MODEL", "explicit-test-model")
    monkeypatch.setattr("mathlab.cli.inspect_image", lambda *_: "sha256:test")
    monkeypatch.setattr("mathlab.cli.ResponsesModel", lambda *_: pytest.fail("API called"))
    assert main(["doctor", "--config", str(ROOT / "config.toml")]) == 0
    output = capsys.readouterr().out
    assert "No model API requests made" in output
    assert "test-key-never-print" not in output


def test_repeated_cli_runs_print_a_score_command_for_the_actual_directory(
    tmp_path,
    questions,
    monkeypatch,
    capsys,
):
    class OfflineModel(ScriptedModel):
        async def close(self):
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("MODEL", "offline-test-model")
    monkeypatch.setattr("mathlab.cli.inspect_image", lambda *_: "sha256:test")
    monkeypatch.setattr(
        "mathlab.cli.ResponsesModel",
        lambda *_: OfflineModel([response(value="3/4"), response(value="42")]),
    )
    jsonl(
        questions.with_name("answers.jsonl"),
        [
            {"id": "one", "answer": "3/4"},
            {"id": "two", "answer": "42"},
        ],
    )
    out = tmp_path / "dev tool"
    out.mkdir()
    (out / "keep.txt").write_text("previous run")
    command = [
        "run",
        "--out",
        str(out),
        "--agent",
        str(ROOT / "agents/tool.py"),
        "--questions",
        str(questions),
        "--config",
        str(ROOT / "config.toml"),
    ]
    for suffix in (2, 3):
        assert main(command) == 0
        directory = tmp_path / f"dev tool-{suffix}"
        output = capsys.readouterr().out
        assert f"Run directory: {directory}" in output
        assert f"Results: {directory}" in output
        score_line = next(line for line in output.splitlines() if line.startswith("uv run "))
        score_command = shlex.split(score_line)[3:]
        assert score_command[score_command.index("--run") + 1] == str(directory)
        assert main(score_command) == 0
        assert "2/2 correct" in capsys.readouterr().out
    assert (out / "keep.txt").read_text() == "previous run"
    assert not (out / "results.jsonl").exists()


def test_reject_duplicate_json_keys(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id":"a","id":"b","problem":"test"}\n')
    with pytest.raises(LabError, match="Duplicate JSON key"):
        load_questions(path)
    path.write_text(json.dumps({"id": "a", "problem": ""}) + "\n")
    with pytest.raises(LabError, match="empty"):
        load_questions(path)
