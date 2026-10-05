"""Summarize the frozen development pilot; never read or run final validation.

uv run python scripts/summarize_contest_pilot.py \
    --direct runs/contest-direct-pilot --tool runs/contest-tool-pilot
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from summarize_hard_pilot import digest, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/contest"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads((DATA / "pilot.json").read_text())
    profiles = [
        summarize(label, [directory], data_directory=DATA, expected_ids=pilot["ids"])
        for label, directory in (("direct", args.direct), ("tool", args.tool))
    ]
    for key in ("model", "limits", "docker_image_id", "provider_configuration"):
        if profiles[0][key] != profiles[1][key]:
            raise ValueError(f"Baseline mismatch: {key}")
    fixture = json.loads((ROOT / "data/sources/contest-dev.json").read_text())
    rungs = {row["id"]: row["rung"] for row in fixture["records"]}
    for profile in profiles:
        profile["rungs"] = {
            rung: {
                "correct": sum(
                    row["correct"] for row in profile["outcomes"] if rungs[row["id"]] == rung
                ),
                "total": sum(rungs[row["id"]] == rung for row in profile["outcomes"]),
            }
            for rung in sorted(set(rungs.values()))
        }
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "questions_sha256": digest(DATA / "questions.jsonl"),
        "answers_sha256": digest(DATA / "answers.jsonl"),
        "pilot_sha256": digest(DATA / "pilot.json"),
        "method": "One attempt per baseline on the same twelve development questions, four per "
        "provisional rung, chosen and frozen before running. All selected cases retained. "
        "This is a small stratified diagnostic, not a measured score on all 100 development "
        "questions, a confidence claim about final validation, or a clean tool-only ablation: "
        "the direct baseline uses one request and the tool baseline may use multiple. "
        "No final-validation model runs were used for calibration. Profiles ran concurrently; "
        "elapsed times are operational observations, not a latency benchmark.",
        "profiles": profiles,
    }
    (DATA / "calibration.json").write_text(json.dumps(record, indent=2) + "\n")
    for profile in profiles:
        print(profile["label"], f"{profile['correct']}/{profile['total']}", profile["categories"])


if __name__ == "__main__":
    main()
