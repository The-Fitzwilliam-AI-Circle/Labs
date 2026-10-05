"""Summarize completed, scored pilot runs without contacting a model.

Example:
uv run python scripts/summarize_hard_pilot.py \
    --profile original runs/hard-tool-pilot \
    --profile generous runs/hard-tool-generous runs/hard-tool-generous-b \
    --out data/hard/calibration.json

Each profile must cover every question exactly once, with identical model, agent,
image and limits within the profile. No failed cases may be silently dropped.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(label, directories, *, data_directory=None, expected_ids=None):
    data_directory = data_directory or ROOT / "data/hard"
    question_hash = digest(data_directory / "questions.jsonl")
    answer_hash = digest(data_directory / "answers.jsonl")
    dataset_ids = {row["id"] for row in read_lines(data_directory / "questions.jsonl")}
    expected = dataset_ids if expected_ids is None else set(expected_ids)
    if not expected or not expected <= dataset_ids:
        raise ValueError("Expected pilot IDs must belong to the selected dataset")
    outcomes, runs, configurations = [], [], []
    input_tokens = output_tokens = unknown_usage = 0
    for directory in directories:
        manifest = json.loads((directory / "run.json").read_text())
        scores = json.loads((directory / "scores.json").read_text())
        if manifest["state"] != "completed":
            raise ValueError(f"Incomplete run: {directory}")
        if manifest["questions_sha256"] != question_hash or scores["answers_sha256"] != answer_hash:
            raise ValueError(f"Dataset or key mismatch: {directory}")
        if scores["results_sha256"] != digest(directory / "results.jsonl"):
            raise ValueError(f"Results changed after scoring: {directory}")
        configurations.append(
            {
                **{
                    key: manifest[key]
                    for key in ("model", "agent_sha256", "limits", "docker_image_id")
                },
                "provider_configuration": manifest.get("provider_configuration"),
            }
        )
        token_limited = set()
        reasoning = []
        for row in read_lines(directory / "traces.jsonl"):
            if row["event"] == "model_response":
                response = row["raw_response"]
                if (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
                    token_limited.add(row["task_id"])
                if response.get("reasoning") and response["reasoning"] not in reasoning:
                    reasoning.append(response["reasoning"])
        grades = {row["id"]: row for row in scores["outcomes"]}
        results = read_lines(directory / "results.jsonl")
        for row in results:
            grade = grades[row["id"]]
            category = grade["reason"]
            if category == "protocol_error" and row["id"] in token_limited:
                category = "token_limit"
            elif category == "budget_exceeded":
                category = "python_budget" if "Python" in (row["error"] or "") else "request_budget"
            outcomes.append(
                {
                    "id": row["id"],
                    "correct": grade["correct"],
                    "category": category,
                    "status": row["status"],
                    "answer": row["answer"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "model_requests": row["model_requests"],
                    "python_calls": row["python_calls"],
                }
            )
        input_tokens += scores["resource_usage"]["input_tokens"]
        output_tokens += scores["resource_usage"]["output_tokens"]
        unknown_usage += scores["resource_usage"]["unknown_usage_requests"]
        runs.append(
            {
                "directory": str(directory),
                "started_at": manifest["started_at"],
                "ended_at": manifest["ended_at"],
                "results_sha256": scores["results_sha256"],
                "scores_sha256": digest(directory / "scores.json"),
                "scheduled_task_ids": manifest["scheduled_task_ids"],
                "provider_reasoning_configurations": reasoning,
            }
        )
    ids = [row["id"] for row in outcomes]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"Profile {label} must cover all expected IDs exactly once")
    if any(configuration != configurations[0] for configuration in configurations):
        raise ValueError(f"Mixed configurations in profile {label}")
    return {
        "label": label,
        **configurations[0],
        "runs": runs,
        "correct": sum(row["correct"] for row in outcomes),
        "total": len(outcomes),
        "categories": dict(sorted(Counter(row["category"] for row in outcomes).items())),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "unknown_usage_requests": unknown_usage,
        "outcomes": sorted(outcomes, key=lambda row: row["id"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", nargs="+", action="append", required=True, metavar="LABEL_OR_RUN"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profiles = []
    for profile in args.profile:
        if len(profile) < 2:
            parser.error("Each --profile needs a label and at least one run directory")
        profiles.append(summarize(profile[0], list(map(Path, profile[1:]))))
    document = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "questions_sha256": digest(ROOT / "data/hard/questions.jsonl"),
        "answers_sha256": digest(ROOT / "data/hard/answers.jsonl"),
        "method": "Twelve candidates frozen before the first live pilot; all retained. "
        "One attempt per question per budget profile. The generous profile used two "
        "disjoint batches covering all twelve questions. Public samples and related "
        "variants are not an independent held-out benchmark. No API calls were retried "
        "by this summarizer; raw run directories remain the source of evidence.",
        "profiles": profiles,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    for profile in profiles:
        print(profile["label"], f"{profile['correct']}/{profile['total']}", profile["categories"])


if __name__ == "__main__":
    main()
