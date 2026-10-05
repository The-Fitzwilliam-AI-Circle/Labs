"""Dataset integrity, exact arithmetic spot checks and safe participant packaging."""

import hashlib
import json
import uuid
from fractions import Fraction
from itertools import combinations
from math import comb, isqrt, prod
from zipfile import ZipFile

import pytest

from mathlab.files import load_questions
from mathlab.scoring import parse_answer
from scripts.export_participants import export as export_participants
from scripts.export_participants import participant_paths
from scripts.prepare_contest import export, normalize_answer
from tests.conftest import ROOT

DATA = ROOT / "data/contest"
FIXTURE = ROOT / "data/sources/contest-dev.json"


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_contest_offline_regeneration_and_rungs(tmp_path):
    manifest = export(FIXTURE, tmp_path)
    for name in ("questions.jsonl", "answers.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (DATA / name).read_bytes()
    questions = load_questions(DATA / "questions.jsonl")
    answers = read_lines(DATA / "answers.jsonl")
    assert len(questions) == len(answers) == manifest["count"] == 100
    assert {q.id for q in questions} == {a["id"] for a in answers}
    assert all(parse_answer(a["answer"]) == Fraction(a["answer"]) for a in answers)
    rung_ids = []
    for rung, count in manifest["rung_counts"].items():
        rows = read_lines(DATA / rung / "questions.jsonl")
        assert len(rows) == count
        rung_ids.extend(row["id"] for row in rows)
        for name in ("questions.jsonl", "answers.jsonl"):
            assert (tmp_path / rung / name).read_bytes() == (DATA / rung / name).read_bytes()
    assert len(rung_ids) == len(set(rung_ids)) == 100
    assert set(rung_ids) == {q.id for q in questions}
    for filename, digest in manifest["sha256"].items():
        assert hashlib.sha256((DATA / filename).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == manifest["source_fixture_sha256"]


@pytest.mark.parametrize(
    "source,expected",
    [(r"-\frac{2}{4}", "-1/2"), (r"\dfrac{24}{17}", "24/17"), ("8.5", "17/2"), (17, "17")],
)
def test_source_literal_normalization(source, expected):
    assert normalize_answer(source) == expected


@pytest.mark.parametrize(
    "source", [r"\sqrt{3}", "2^99", "1,2", "1/0", "nan", "9" * 129, "__import__('os')"]
)
def test_unsupported_source_answers_are_rejected(source):
    with pytest.raises((ValueError, ZeroDivisionError)):
        normalize_answer(source)


def test_development_keys_with_separate_exact_computations():
    answers = {a["id"]: parse_answer(a["answer"]) for a in read_lines(DATA / "answers.jsonl")}
    checks = {}
    checks["contest-hmmt-feb-2025-04"] = sum(
        Fraction(4050, 2 * j + 1).__floor__() for j in range(-1000, 1001)
    )
    checks["contest-hmmt-feb-2025-09"] = -sum(n * n for n in range(1, isqrt(2027) + 1)) % 2027

    # Unlabelled partitions: force each successive block to contain the least unused ball.
    def partitions(remaining):
        if not remaining:
            return 1
        first = min(remaining)
        total = 0
        for others in combinations(sorted(remaining - {first}), 3):
            block = {first, *others}
            if all(v - 1 in block or v + 1 in block for v in block):
                total += partitions(remaining - block)
        return total

    checks["contest-hmmt-feb-2025-13"] = partitions(set(range(1, 17)))
    checks["contest-cmimc-2025-05"] = sum(
        int(c) for n in range(77, 1000000, 77) for c in str(n) if int(c) % 2
    )
    value = valuation = 0
    for n in range(1, 811):
        value = 10 * value + n % 10
        temporary = value
        while temporary % 3 == 0:
            valuation += 1
            temporary //= 3
    checks["contest-cmimc-2025-07"] = valuation
    # Independent modular counts catch valuations above the published solution's
    # truncation: the prefix ending at position 798 has valuation eight.
    divisibility_counts = []
    for exponent in range(1, 10):
        residue = count = 0
        modulus = 3**exponent
        for n in range(1, 811):
            residue = (10 * residue + n % 10) % modulus
            count += residue == 0
        divisibility_counts.append(count)
    assert divisibility_counts == [567, 243, 81, 27, 9, 3, 1, 1, 0]
    assert sum(divisibility_counts) == valuation == 932
    checks["contest-cmimc-2025-16"] = sum(
        all(
            not (mask & (1 << (4 * r + c)) and mask & (1 << (4 * rr + cc)))
            for r in range(4)
            for c in range(4)
            for rr, cc in ((r + 1, c), (r, c + 1))
            if rr < 4 and cc < 4
        )
        and all(
            any(mask & (1 << (4 * (r + dr) + c + dc)) for dr in (0, 1) for dc in (0, 1))
            for r in range(3)
            for c in range(3)
        )
        for mask in range(1 << 16)
    )
    checks["contest-cmimc-2025-19"] = sum(
        abs(sum(Fraction(comb(18, k), 2**18) for k in range(residue, 19, 5)) - Fraction(1, 5))
        for residue in range(5)
    )
    nodes = [n * (n + 1) for n in range(1, 9)]
    checks["contest-brumo-2025-29"] = sum(
        Fraction(1, x * x) * prod(Fraction(90 - y, x - y) for y in nodes if y != x) for x in nodes
    ) - Fraction(1, 90**2)
    checks["contest-brumo-2025-26"] = ((10**726 - 1) // 727) % 1000
    checks["contest-brumo-2025-27"] = 55 * (Fraction(3, 5) - Fraction(3, 9) + Fraction(1, 13))
    for task_id, value in checks.items():
        assert answers[task_id] == value, task_id


def test_frozen_pilot_covers_all_three_rungs():
    pilot = json.loads((DATA / "pilot.json").read_text())
    records = json.loads(FIXTURE.read_text())["records"]
    assert len(pilot["ids"]) == len(set(pilot["ids"])) == 12
    for rung in ("01-contest", "02-advanced", "03-stretch"):
        assert sum(r["id"] in pilot["ids"] and r["rung"] == rung for r in records) == 4


def test_saved_contest_calibration_matches_frozen_cases_and_exact_grading():
    record = json.loads((DATA / "calibration.json").read_text())
    expected = set(json.loads((DATA / "pilot.json").read_text())["ids"])
    answers = {row["id"]: parse_answer(row["answer"]) for row in read_lines(DATA / "answers.jsonl")}
    for stem in ("questions", "answers"):
        assert (
            record[f"{stem}_sha256"]
            == hashlib.sha256((DATA / f"{stem}.jsonl").read_bytes()).hexdigest()
        )
    assert record["pilot_sha256"] == hashlib.sha256((DATA / "pilot.json").read_bytes()).hexdigest()
    assert {profile["label"] for profile in record["profiles"]} == {"direct", "tool"}
    for profile in record["profiles"]:
        assert len(profile["outcomes"]) == profile["total"] == len(expected)
        assert {row["id"] for row in profile["outcomes"]} == expected
        correct = 0
        for row in profile["outcomes"]:
            try:
                matches = (
                    row["status"] == "completed"
                    and parse_answer(row["answer"]) == answers[row["id"]]
                )
            except ValueError:
                matches = False
            assert row["correct"] == matches
            correct += matches
        assert profile["correct"] == correct
        assert sum(profile["categories"].values()) == len(expected)
        assert sum(rung["correct"] for rung in profile["rungs"].values()) == correct


def test_export_ignores_secrets_and_unlisted_data_and_refuses_overwrite(tmp_path):
    root = tmp_path / "project"
    # Populate the real allowlist, then plant files that must never be shipped.
    for relative in participant_paths(ROOT):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    secret = "private-" + uuid.uuid4().hex
    for name in (
        ".env",
        "data/heldout/answers.jsonl",
        "data/private/source.json",
        "data/contest/secret.json",
        "runs/example/traces.jsonl",
        ".git/config",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secret)
    out = tmp_path / "participants.zip"
    export_participants(root, out)
    with ZipFile(out) as archive:
        assert "agents-lab/data/contest/questions.jsonl" in archive.namelist()
        assert all(secret.encode() not in archive.read(name) for name in archive.namelist())
    with pytest.raises(FileExistsError):
        export_participants(root, out)
    (root / "agents/participant.py").unlink()
    (root / "agents/participant.py").symlink_to(ROOT / "agents/participant.py")
    with pytest.raises(ValueError, match="symlink"):
        export_participants(root, tmp_path / "unsafe.zip")
