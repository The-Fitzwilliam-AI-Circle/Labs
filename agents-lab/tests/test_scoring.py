import json
from fractions import Fraction

import pytest

from mathlab.files import load_questions, write_json
from mathlab.scoring import parse_answer, score
from mathlab.types import DatasetError
from tests.conftest import jsonl


@pytest.mark.parametrize("value", ["3/4", "6/8", "0.75", " +0.750 ", "+3/4"])
def test_exact_equivalence(value):
    assert parse_answer(value) == Fraction(3, 4)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "The answer is 3",
        "3 kg",
        r"\frac{3}{4}",
        "1+2",
        "1e2",
        "NaN",
        "inf",
        "{1,2}",
        "3 or 4",
        "1/0",
        "1/-2",
        "1/+2",
        ".5",
        "1.",
        "1 / 2",
        "⅓",
        "٣",
        "0" * 129,
        None,
        3,
        "--3",
        "1_000",
        "1\n2",
    ],
)
def test_rejects_invalid_grammar(value):
    with pytest.raises(ValueError):
        parse_answer(value)


def result(task_id, answer="3/4", status="completed"):
    return {
        "id": task_id,
        "answer": answer,
        "status": status,
        "model_requests": 1,
        "python_calls": 0,
        "input_tokens": 20,
        "output_tokens": 10,
        "reasoning_tokens": 4,
        "unknown_usage_requests": 0,
        "elapsed_seconds": 0.5,
    }


def fixture_run(tmp_path, rows=None, scheduled=None):
    write_json(
        tmp_path / "run.json",
        {
            "schema_version": 1,
            "state": "completed",
            "scheduled_task_ids": scheduled or ["one"],
        },
    )
    jsonl(tmp_path / "results.jsonl", rows if rows is not None else [result("one")])
    answers = jsonl(tmp_path / "answers.jsonl", [{"id": "one", "answer": "6/8"}])
    return answers


def test_all_scheduled_tasks_in_denominator_and_usage(tmp_path):
    rows = [
        result("one", "0.75"),
        result("two", "2"),
        result("three", "2 apples"),
        result("four", None, "provider_error"),
    ]
    rows[-1]["unknown_usage_requests"] = 1
    answers = fixture_run(tmp_path, rows, [r["id"] for r in rows])
    jsonl(answers, [{"id": row["id"], "answer": "3/4"} for row in rows])
    summary = score(tmp_path, answers)
    assert (summary["correct"], summary["total"], summary["accuracy"]) == (1, 4, 0.25)
    assert summary["invalid_answers"] == 1
    assert summary["failure_counts"] == {"provider_error": 1}
    assert summary["resource_usage"]["output_tokens"] == 40  # Reasoning is not added twice.
    assert not summary["usage_complete"]
    assert json.loads((tmp_path / "scores.json").read_text()) == summary


@pytest.mark.parametrize("reference", ["1/0", "invalid", "1e2", 3])
def test_invalid_reference_stops_scoring(tmp_path, reference):
    answers = fixture_run(tmp_path)
    jsonl(answers, [{"id": "one", "answer": reference}])
    with pytest.raises(DatasetError, match="Invalid reference"):
        score(tmp_path, answers)
    assert not (tmp_path / "scores.json").exists()


@pytest.mark.parametrize(
    "rows, match",
    [
        ([result("one"), result("one")], "Duplicate id"),
        ([result("other")], "missing=.*one.*unexpected=.*other"),
        ([], "no records"),
    ],
)
def test_invalid_result_ids(tmp_path, rows, match):
    answers = fixture_run(tmp_path, rows)
    with pytest.raises(DatasetError, match=match):
        score(tmp_path, answers)


def test_answer_subset_and_duplicate_or_missing_references(tmp_path):
    answers = fixture_run(tmp_path)
    jsonl(answers, [{"id": "one", "answer": "3/4"}, {"id": "extra", "answer": "2"}])
    assert score(tmp_path, answers)["correct"] == 1
    jsonl(answers, [{"id": "one", "answer": "3/4"}] * 2)
    with pytest.raises(DatasetError, match="Duplicate id"):
        score(tmp_path, answers)
    jsonl(answers, [{"id": "extra", "answer": "2"}])
    with pytest.raises(DatasetError, match="Missing reference"):
        score(tmp_path, answers)


def test_reject_incomplete_run_and_corrupt_usage(tmp_path):
    answers = fixture_run(tmp_path)
    manifest = json.loads((tmp_path / "run.json").read_text())
    manifest["state"] = "incomplete"
    write_json(tmp_path / "run.json", manifest)
    with pytest.raises(DatasetError, match="incomplete"):
        score(tmp_path, answers)
    fixture_run(tmp_path, [{**result("one"), "model_requests": -1}])
    with pytest.raises(DatasetError, match="counter"):
        score(tmp_path, answers)


def test_question_files_reject_gold_fields_and_duplicate_ids(tmp_path):
    path = jsonl(
        tmp_path / "questions.jsonl",
        [{"id": "one", "problem": "Question", "answer": "GOLD", "solution": "WORKED"}],
    )
    with pytest.raises(DatasetError, match="only id and problem"):
        load_questions(path)
    jsonl(path, [{"id": "one", "problem": "Question"}] * 2)
    with pytest.raises(DatasetError, match="Duplicate id"):
        load_questions(path)
