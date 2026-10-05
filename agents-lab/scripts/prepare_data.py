"""Reproduce the reviewed local fixtures. Uses the network only when explicitly run.

Run with `uv run python scripts/prepare_data.py --offline` to regenerate from the
checked-in foundation. --source accepts a downloaded copy of the pinned upstream
test.jsonl; without either option the upstream file is downloaded.
Never called by agent execution or scoring. Reference solvers are organizer material.
"""

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from ladder import TIERS, VERSION, cases, solve

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SOURCE = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500"
DOWNLOAD = f"{SOURCE}/resolve/{REVISION}/test.jsonl"
# Reviewed statements are text-only; LaTeX notation is preserved verbatim.
SELECTION = [
    ("test/algebra/2584.json", "14/3", "Rational substitution; independently checked 2 + 5/3 + 1."),
    ("test/number_theory/572.json", "9", "Divisor count: 196 = 2^2 * 7^2."),
    (
        "test/intermediate_algebra/1197.json",
        "3/56",
        "Degree-five interpolation; exact rational check.",
    ),
    ("test/number_theory/737.json", "284", "Proper-divisor sums: 284 -> 220 -> 284."),
    ("test/intermediate_algebra/134.json", "-50", "Fifty pairs, each summing to -1."),
    ("test/counting_and_probability/525.json", "144", "Circular gaps: 3! * C(4,3) * 3!."),
    ("test/algebra/1072.json", "243/625", "Eighth geometric term: (125/9) * (3/5)^7."),
    ("test/counting_and_probability/119.json", "-125", "Constant term: C(5,2) * 10^2 * (-1/2)^3."),
    ("test/geometry/967.json", "72", "Pentagon rotation: 360/5. Statement needs no diagram."),
    ("test/precalculus/285.json", "6", "Roots include primitive sixth roots of unity."),
]
SMOKE = [
    ("What is 3 divided by 4?", "3/4"),
    ("Solve 2x + 9 = 3. What is x?", "-3"),
    ("What is the sum of the integers from 1 through 100, inclusive?", "5050"),
    ("A fair six-sided die is rolled once. What is the probability of an even result?", "1/2"),
    ("What is the value of (7/3) divided by (2/3)?", "7/2"),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(directory, questions, answers, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for filename, rows in (("questions.jsonl", questions), ("answers.jsonl", answers)):
        data = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode()
        (directory / filename).write_bytes(data)
        hashes[filename] = digest(data)
    (directory / "manifest.json").write_text(
        json.dumps({**manifest, "sha256": hashes}, indent=2) + "\n", encoding="utf-8"
    )


def foundation(source_bytes):
    source_rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    by_id = {row["unique_id"]: row for row in source_rows}
    questions, answers, selections = [], [], []
    for index, (source_id, normalized_answer, review) in enumerate(SELECTION, 1):
        row = by_id[source_id]
        original = row["answer"]
        # Each reference was reviewed; validate the explicitly supported LaTeX wrapper.
        if original.startswith(r"\frac{"):
            numerator, denominator = original[6:-1].split("}{")
            original = f"{numerator}/{denominator}"
        assert Fraction(original) == Fraction(normalized_answer), source_id
        assert "[asy]" not in row["problem"] and "\\includegraphics" not in row["problem"]
        task_id = f"dev-{index:03}"
        questions.append({"id": task_id, "problem": row["problem"]})
        answers.append({"id": task_id, "answer": str(Fraction(normalized_answer))})
        selections.append(
            {
                "id": task_id,
                "source_id": source_id,
                "subject": row["subject"],
                "level": row["level"],
                "review": review,
            }
        )
    return (
        questions,
        answers,
        {
            "dataset": "HuggingFaceH4/MATH-500",
            "source_url": SOURCE,
            "source_revision": REVISION,
            "source_file": "test.jsonl",
            "source_file_sha256": digest(source_bytes),
            "prepared_at": datetime.now(UTC).isoformat(),
            "selection_notes": (
                "Ten reviewed scalar rational answers across subjects and levels 1–5. "
                "Statements preserved verbatim; no diagrams required. "
                "References normalized by Fraction. Public development data, not held-out "
                "evaluation. Retained as the foundation rung after both lab baselines "
                "scored 10/10 with gpt-5-nano-2025-08-07."
            ),
            "selected_questions": selections,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data",
        help="Output data root; default: repository data directory",
    )
    args = parser.parse_args()
    if args.offline:
        directory = ROOT / "data/ladder" / TIERS[0]
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            if digest((directory / name).read_bytes()) != expected:
                raise ValueError(f"Foundation hash mismatch: {name}")
        questions = [
            json.loads(line) for line in (directory / "questions.jsonl").read_text().splitlines()
        ]
        answers = [
            json.loads(line) for line in (directory / "answers.jsonl").read_text().splitlines()
        ]
        manifest.pop("sha256")
    else:
        if args.source:
            source_bytes = args.source.read_bytes()
        else:
            with urllib.request.urlopen(DOWNLOAD, timeout=30) as response:
                source_bytes = response.read()
        if (
            digest(source_bytes)
            != "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
        ):
            raise ValueError("Source does not match the pinned MATH-500 file")
        questions, answers, manifest = foundation(source_bytes)
    manifest["tier"] = TIERS[0]
    selections = [{**row, "tier": TIERS[0]} for row in manifest["selected_questions"]]
    export(args.out / "ladder" / TIERS[0], questions, answers, manifest)
    new_cases = cases()
    source = {
        "dataset": "Original Mathlab difficulty ladder",
        "source_url": None,
        "source_revision": VERSION,
        "generator": "scripts/ladder.py",
        "generator_sha256": digest((ROOT / "scripts/ladder.py").read_bytes()),
        "selection_notes": "Original deterministic instances with exact reference solvers. "
        "Ten recurring families with progressively stronger computational or modelling demands. "
        "Difficulty is a design target; new questions have not been calibrated on a live model. "
        "Public development data, not held-out evaluation. Verification: tests/test_ladder.py.",
    }
    for tier in TIERS[1:]:
        group = [row for row in new_cases if row["tier"] == tier]
        tier_questions = [{"id": row["id"], "problem": row["problem"]} for row in group]
        tier_answers = [{"id": row["id"], "answer": solve(row)} for row in group]
        metadata = [{key: value for key, value in row.items() if key != "problem"} for row in group]
        export(
            args.out / "ladder" / tier,
            tier_questions,
            tier_answers,
            {**source, "tier": tier, "selected_questions": metadata},
        )
        questions.extend(tier_questions)
        answers.extend(tier_answers)
        selections.extend(metadata)
    export(
        args.out / "dev",
        questions,
        answers,
        {
            "dataset": "Mathlab development ladder",
            "source_revision": VERSION,
            "sources": [
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in ("selected_questions", "tier")
                },
                source,
            ],
            "selection_notes": "40 public development questions in four rungs of ten. "
            "Original dev-001 through dev-010 are preserved. See data/README.md for the ladder.",
            "tiers": list(TIERS),
            "selected_questions": selections,
        },
    )
    export(
        args.out / "smoke",
        [{"id": f"smoke-{i:03}", "problem": problem} for i, (problem, _) in enumerate(SMOKE, 1)],
        [{"id": f"smoke-{i:03}", "answer": answer} for i, (_, answer) in enumerate(SMOKE, 1)],
        {
            "dataset": "Mathlab hand-authored smoke fixtures",
            "source_url": None,
            "source_revision": "v0.1",
            "original_source_ids": [],
            "selection_notes": "Public plumbing checks: fraction, negative integer, sum, "
            "probability, rational arithmetic. Hand-authored for this repository.",
        },
    )
    print("Exported 5 smoke and 40 development questions, four ladder rungs, keys, and manifests.")


if __name__ == "__main__":
    main()
