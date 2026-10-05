"""Source-backed reductions, separate computations, exhaustive small cases and exports."""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import permutations
from math import comb, factorial, hypot, pi, prod, sin

import pytest

from mathlab.files import load_questions
from mathlab.scoring import parse_answer
from scripts.hard_problems import SOURCE, cases
from scripts.hard_reference import (
    central_perimeter,
    finite_field,
    permutation_expectation,
    polynomial,
    representations,
    small_field_count,
    solve,
)
from tests.conftest import ROOT


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["id"])
def test_hard_keys_and_published_answers(case):
    answers = {
        row["id"]: row["answer"]
        for row in map(json.loads, (ROOT / "data/hard/answers.jsonl").read_text().splitlines())
    }
    value = solve(case["id"])
    assert value == answers[case["id"]]
    assert parse_answer(value).denominator == 1
    if "source_answer" in case:
        assert value == case["source_answer"]


def test_hard_offline_exports_and_provenance(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_hard.py"), "--out", str(tmp_path)],
        check=True,
        timeout=20,
    )
    for name in ("questions.jsonl", "answers.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (ROOT / "data/hard" / name).read_bytes()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest["source_fixture_sha256"]
    tasks = load_questions(tmp_path / "questions.jsonl")
    assert len(tasks) == 12
    assert {row.id for row in tasks} == {case["id"] for case in cases()}
    assert sum(row["variant"] for row in manifest["selected_questions"]) == 5
    assert manifest["license"] == "CC-BY-4.0"


def test_saved_calibration_covers_every_question_and_agrees_with_exact_grading():
    directory = ROOT / "data/hard"
    record = json.loads((directory / "calibration.json").read_text())
    answers = {
        row["id"]: parse_answer(row["answer"])
        for row in map(json.loads, (directory / "answers.jsonl").read_text().splitlines())
    }
    for name in ("questions", "answers"):
        assert (
            record[f"{name}_sha256"]
            == hashlib.sha256((directory / f"{name}.jsonl").read_bytes()).hexdigest()
        )
    for profile in record["profiles"]:
        ids = [row["id"] for row in profile["outcomes"]]
        assert len(ids) == len(set(ids)) == profile["total"] == 12
        assert set(ids) == answers.keys()
        correct = 0
        for outcome in profile["outcomes"]:
            expected = (
                outcome["status"] == "completed"
                and parse_answer(outcome["answer"]) == answers[outcome["id"]]
            )
            assert outcome["correct"] == expected
            correct += expected
        assert profile["correct"] == correct
        assert profile["categories"] == dict(
            Counter(row["category"] for row in profile["outcomes"])
        )


@pytest.mark.parametrize("degree,value", [(19, 19), (31, 37)])
def test_polynomial_explicit_coefficients_and_normalization(degree, value):
    coefficients = {}
    for k in range((degree + 1) // 2):
        coefficient = Fraction((-1) ** k * degree * comb(degree - k, k), degree - k)
        assert coefficient.denominator == 1
        coefficients[degree - 2 * k] = int(coefficient)
    assert coefficients[degree] == 1 and coefficients[1] == -degree
    assert all(power % 2 for power in coefficients)
    assert polynomial(degree, value) == sum(c * value**power for power, c in coefficients.items())
    for z in (Fraction(2), Fraction(3, 2)):
        # Dickson/Chebyshev identity, checked independently of the evaluator recurrence.
        assert (
            sum(c * (z + 1 / z) ** power for power, c in coefficients.items())
            == z**degree + z**-degree
        )


def test_representation_count_via_partitions_and_signed_generating_function():
    def partitions(n, ceiling):
        if n == 0:
            yield ()
        for first in range(min(n, ceiling), 0, -1):
            for tail in partitions(n - first, first):
                yield (first, *tail)

    types = []
    for shape in partitions(5, 5):
        hooks = [
            shape[i] - j + sum(row > j for row in shape[i + 1 :])
            for i in range(len(shape))
            for j in range(shape[i])
        ]
        dimension = factorial(5) // prod(hooks)
        character = Fraction(
            dimension * sum(row * (row - 2 * i - 1) for i, row in enumerate(shape)), 20
        )
        negative_eigenvalues = (dimension - character) / 2
        assert negative_eigenvalues.denominator == 1
        types.append((dimension, (-1) ** int(negative_eigenvalues)))
    assert sorted(types) == sorted([(1, 1), (1, -1), (4, -1), (4, -1), (5, 1), (5, -1), (6, -1)])

    def coefficient(n, signed):
        denominator = [1]
        for size, character in types:
            result = denominator + [0] * size
            for i, c in enumerate(denominator):
                result[i + size] -= c * (character if signed else 1)
            denominator = result
        series = [1]
        for degree in range(1, n + 1):
            series.append(
                -sum(
                    denominator[k] * series[degree - k]
                    for k in range(1, min(degree + 1, len(denominator)))
                )
            )
        return series[-1]

    total, signed = coefficient(1000, False), coefficient(1000, True)
    assert representations(1000) == total == 625243878951
    assert representations(1000, True) == (total + signed) // 2 == 312621939451


def test_curve_small_fields_and_frobenius_recurrence():
    counts = [small_field_count(k) for k in (1, 2, 3)]
    assert counts == [6, 26, 126]
    # Newton identities + reciprocal symmetry of the genus-three Weil polynomial.
    traces = [0] + [5**k + 1 - counts[k - 1] for k in (1, 2, 3)]
    coefficients = [1]
    for k in (1, 2, 3):
        coefficients.append(-sum(coefficients[k - j] * traces[j] for j in range(1, k + 1)) // k)
    coefficients += [5 * coefficients[2], 25 * coefficients[1], 125]
    assert coefficients == [1, 0, 0, 0, 0, 0, 125]
    for k in range(4, 31):
        value = -sum(coefficients[j] * traces[k - j] for j in range(1, min(k, 7)))
        if k <= 6:
            value -= k * coefficients[k]
        traces.append(value)
    for degree in (18, 30):
        assert finite_field(degree) == 5**degree + 1 - traces[degree]
        assert (finite_field(degree) - 5**degree - 1) ** 2 <= 36 * 5**degree


def test_recursive_permutation_expectations_by_exhaustion():
    def transform(word):
        if not word:
            return ()
        maximum = max(word)
        split = word.index(maximum)
        return transform(word[:split]) + transform(word[split + 1 :]) + (maximum,)

    for n in range(2, 8):
        adjacent_total = all_total = 0
        for word in permutations(range(1, n + 1)):
            output = transform(word)
            positions = {value: i for i, value in enumerate(output)}
            adjacent_total += sum(positions[i + 1] < positions[i] for i in range(1, n))
            all_total += sum(
                positions[b] < positions[a] for a in range(1, n) for b in range(a + 1, n + 1)
            )
        harmonic = sum(Fraction(1, k) for k in range(1, n + 1))
        assert Fraction(adjacent_total, factorial(n)) == Fraction(n + 1, 2) - harmonic
        assert Fraction(all_total, factorial(n)) == Fraction(n * (n + 7), 4) - (n + 1) * harmonic
    n = 1000
    harmonic = sum(Fraction(1, k) for k in range(1, n + 1))
    for all_pairs in (False, True):
        exact = (
            Fraction(n * (n + 7), 4) - (n + 1) * harmonic
            if all_pairs
            else Fraction(n + 1, 2) - harmonic
        )
        assert permutation_expectation(n, all_pairs) == exact.numerator // exact.denominator


def test_geometry_against_all_pentagon_dissections():
    from math import cos

    n, p = 5, 0.3
    polygon = [(cos(2 * pi * i / n), sin(2 * pi * i / n)) for i in range(n)]
    diagonals = [(i, j) for i in range(n) for j in range(i + 1, n) if j - i not in (1, n - 1)]

    def clip(points, start, end):
        a, b = polygon[start], polygon[end]
        dx, dy = b[0] - a[0], b[1] - a[1]
        center_sign = dx * -a[1] - dy * -a[0]

        def side(point):
            return center_sign * (dx * (point[1] - a[1]) - dy * (point[0] - a[0]))

        output = []
        for left, right in zip(points, points[1:] + points[:1], strict=True):
            sl, sr = side(left), side(right)
            if sl >= 0:
                output.append(left)
            if (sl < 0) != (sr < 0):
                t = sl / (sl - sr)
                output.append(
                    (left[0] + t * (right[0] - left[0]), left[1] + t * (right[1] - left[1]))
                )
        return output

    expectation = 0
    for mask in range(1 << len(diagonals)):
        points = polygon[:]
        for bit, (i, j) in enumerate(diagonals):
            if mask & (1 << bit):
                points = clip(points, i, j)
        perimeter = sum(
            hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        )
        selected = mask.bit_count()
        expectation += perimeter * p**selected * (1 - p) ** (len(diagonals) - selected)
    assert abs(expectation - central_perimeter(n, str(p))) < 1e-12
    assert abs(central_perimeter(n, "0") - 2 * n * sin(pi / n)) < 1e-12


@pytest.mark.parametrize("n,p,expected", [(101, "0.001", 4771880153), (79, "0.002", 4471400324)])
def test_geometry_floor_at_high_precision(n, p, expected):
    exact_precision = central_perimeter(n, p, precision=60)
    lower_precision = central_perimeter(n, p, precision=35)
    assert abs(exact_precision - lower_precision) < Decimal("1e-30")
    assert int(exact_precision * 10**9) == expected
    assert int(central_perimeter(n, p) * 10**9) == expected
