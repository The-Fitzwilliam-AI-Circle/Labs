"""Deterministic exact rational scoring with full scheduled-ID reconciliation."""

import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from mathlab.files import read_json, read_jsonl, sha256, write_json
from mathlab.types import STATUSES, DatasetError

ANSWER = re.compile(r"[+-]?[0-9]+(?:/[0-9]+|\.[0-9]+)?\Z")
COUNTERS = (
    "model_requests",
    "python_calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "unknown_usage_requests",
)


def parse_answer(answer: str) -> Fraction:
    if not isinstance(answer, str):
        raise ValueError("Answer must be a string")
    answer = answer.strip()
    if len(answer) > 128 or not ANSWER.fullmatch(answer):
        raise ValueError("Expected an integer, unsigned-denominator fraction, or decimal")
    try:
        return Fraction(answer)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid rational answer (denominator must be nonzero)") from exc


def score(run_directory: Path, answers_path: Path) -> dict:
    manifest = read_json(run_directory / "run.json")
    if manifest.get("schema_version") != 1 or manifest.get("state") != "completed":
        raise DatasetError("Run is incomplete or uses an unsupported schema; cannot score")
    scheduled = manifest.get("scheduled_task_ids")
    if (
        not isinstance(scheduled, list)
        or not scheduled
        or any(not isinstance(task_id, str) or not task_id for task_id in scheduled)
        or len(set(scheduled)) != len(scheduled)
    ):
        raise DatasetError("Invalid or duplicate scheduled task IDs in run.json")
    references = {}
    for row in read_jsonl(answers_path):
        if set(row) != {"id", "answer"}:
            raise DatasetError(f"Reference {row['id']} must contain only id and answer")
        try:
            references[row["id"]] = parse_answer(row["answer"])
        except ValueError as exc:
            raise DatasetError(f"Invalid reference answer for {row['id']}: {exc}") from exc
    missing_answers = set(scheduled) - references.keys()
    if missing_answers:
        raise DatasetError(f"Missing reference answers: {', '.join(sorted(missing_answers))}")
    results = {row["id"]: row for row in read_jsonl(run_directory / "results.jsonl")}
    missing, unexpected = set(scheduled) - results.keys(), results.keys() - set(scheduled)
    if missing or unexpected:
        raise DatasetError(
            f"Result IDs do not match scheduled tasks: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    outcomes, failures = [], Counter()
    totals = dict.fromkeys((*COUNTERS, "elapsed_seconds"), 0)
    for task_id in scheduled:
        row = results[task_id]
        if not isinstance(row.get("status"), str) or row["status"] not in STATUSES:
            raise DatasetError(f"Unknown episode status for {task_id}")
        for key in COUNTERS:
            if type(row.get(key)) is not int or row[key] < 0:
                raise DatasetError(f"Invalid resource counter {key} for {task_id}")
            totals[key] += row[key]
        elapsed = row.get("elapsed_seconds")
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
            raise DatasetError(f"Invalid elapsed_seconds for {task_id}")
        totals["elapsed_seconds"] += elapsed
        correct = False
        if row["status"] != "completed":
            failures[row["status"]] += 1
            reason = row["status"]
        else:
            try:
                correct = parse_answer(row.get("answer")) == references[task_id]
                reason = "correct" if correct else "wrong_answer"
            except ValueError:
                reason = "invalid_answer"
        outcomes.append({"id": task_id, "correct": correct, "reason": reason})
    correct_count = sum(row["correct"] for row in outcomes)
    summary = {
        "schema_version": 1,
        "correct": correct_count,
        "total": len(scheduled),
        "accuracy": correct_count / len(scheduled),
        "invalid_answers": sum(row["reason"] == "invalid_answer" for row in outcomes),
        "failure_counts": dict(failures),
        "resource_usage": totals,
        "usage_complete": totals["unknown_usage_requests"] == 0,
        "answers_sha256": sha256(answers_path),
        "results_sha256": sha256(run_directory / "results.jsonl"),
        "outcomes": outcomes,
    }
    write_json(run_directory / "scores.json", summary)
    return summary
