"""Reproduce the hard set offline: uv run python scripts/prepare_hard.py.

Public Epoch AI samples and explicitly identified new variants. No API or network.
"""

import argparse
import hashlib
import json
from pathlib import Path

from hard_problems import ROOT, SOURCE, cases
from hard_reference import solve


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/hard")
    args = parser.parse_args()
    rows = cases()
    questions = [{"id": row["id"], "problem": row["problem"]} for row in rows]
    answers = [{"id": row["id"], "answer": solve(row["id"])} for row in rows]
    for row, answer in zip(rows, answers, strict=True):
        if "source_answer" in row:
            assert row["source_answer"] == answer["answer"], row["id"]
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, records in (("questions.jsonl", questions), ("answers.jsonl", answers)):
        path = args.out / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        hashes[name] = digest(path)
    source = json.loads(SOURCE.read_text())
    manifest = {
        "dataset": "Mathlab hard mathematics: public FrontierMath samples and new variants",
        "version": "hard-v1",
        "source_url": source["source_url"],
        "credit": source["credit"],
        "license": source["license"],
        "license_url": source["license_url"],
        "retrieved_on": source["retrieved_on"],
        "source_fixture": str(SOURCE.relative_to(ROOT)),
        "source_fixture_sha256": digest(SOURCE),
        "source_html_sha256": source["source_html_sha256"],
        "generator_sha256": digest(ROOT / "scripts/hard_problems.py"),
        "reference_sha256": digest(ROOT / "scripts/hard_reference.py"),
        "selection_notes": "Seven published integer-answer samples and five adaptations. "
        "Frozen before the first live pilot; all twelve candidates retained. Public teaching "
        "data, not the private FrontierMath benchmark or an uncontaminated held-out evaluation. "
        "Source tiers describe the originals, not independently assigned ratings for variants. "
        "References: docs/HARD_REFERENCES.md. Calibration: docs/HARD_CALIBRATION.md.",
        "selected_questions": [
            {key: value for key, value in row.items() if key not in ("problem", "source_answer")}
            for row in rows
        ],
        "sha256": hashes,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Exported {len(rows)} hard questions, exact integer keys, and source/change metadata.")


if __name__ == "__main__":
    main()
