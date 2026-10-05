"""Reproduce a frozen contest split offline, without contacting a model.

Default: uv run python scripts/prepare_contest.py
Organizer: pass --fixture and --out for the separately stored validation split.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSE = "CC-BY-NC-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_answer(value):
    """Accept only literal rationals; never evaluate source LaTeX or expressions."""
    value = str(value).strip()
    value = re.sub(r"\\d?frac\{([0-9]+)\}\{([0-9]+)\}", r"\1/\2", value)
    if len(value) > 128 or not re.fullmatch(r"[+-]?[0-9]+(?:/[0-9]+|\.[0-9]+)?", value):
        raise ValueError("Source answer is not a supported literal rational")
    answer = str(Fraction(value))
    if len(answer) > 128:
        raise ValueError("Normalized answer exceeds scorer limit")
    return answer


def write_records(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def export(fixture, out):
    source = json.loads(fixture.read_text())
    rows = source["records"]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty split or duplicate IDs")
    problems = [row["problem"] for row in rows]
    if any(not problem.strip() for problem in problems) or len(set(problems)) != len(problems):
        raise ValueError("Empty or duplicate problems")
    questions = [{"id": row["id"], "problem": row["problem"]} for row in rows]
    answers = [
        {
            "id": row["id"],
            "answer": normalize_answer(
                row["answer_correction"]["answer"]
                if "answer_correction" in row
                else row["source_answer"]
            ),
        }
        for row in rows
    ]
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("questions.jsonl", questions), ("answers.jsonl", answers)):
        write_records(out / name, records)
    manifest = {
        "dataset": source["dataset"],
        "version": source["version"],
        "split": source["split"],
        "count": len(rows),
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "credit": "MathArena (ETH Zurich SRI Lab) and the credited competition organizers",
        "frozen_on": source["frozen_on"],
        "selection_notes": source["selection_notes"],
        "sources": source["sources"],
        "source_fixture_sha256": digest(fixture),
        "rung_counts": dict(Counter(row["rung"] for row in rows)),
        "selected_questions": [
            {key: value for key, value in row.items() if key not in ("problem", "source_answer")}
            for row in rows
        ],
        "sha256": {name: digest(out / name) for name in ("questions.jsonl", "answers.jsonl")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    # Only practice data gets runnable rungs; the final evaluation stays one frozen set.
    if source["split"] == "development":
        for rung in sorted(manifest["rung_counts"]):
            directory = out / rung
            directory.mkdir(exist_ok=True)
            selected = {row["id"] for row in rows if row["rung"] == rung}
            write_records(
                directory / "questions.jsonl", [q for q in questions if q["id"] in selected]
            )
            write_records(directory / "answers.jsonl", [a for a in answers if a["id"] in selected])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "data/sources/contest-dev.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/contest")
    args = parser.parse_args()
    manifest = export(args.fixture, args.out)
    print(f"Exported {manifest['count']} {manifest['split']} questions to {args.out}")


if __name__ == "__main__":
    main()
