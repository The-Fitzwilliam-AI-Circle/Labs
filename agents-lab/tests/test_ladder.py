"""Check full-size keys using separately implemented algorithms and tiny exhaustive cases.

These checks never call a model, Docker, or the network. Reference code stays out
of the evaluated agent. In particular, regeneration agreeing with itself is not
the sole evidence for correctness.
"""

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from functools import cache
from itertools import combinations, permutations, product
from math import gcd, prod

import pytest

from scripts import ladder
from tests.conftest import ROOT


def recurrence_check(n, modulus=None, prefix=False):
    # Polynomial reduction of x^n by the characteristic polynomial, not matrices.
    initial, coefficients = ([2, 7, 18, 65], [3, 1, 2, -5]) if prefix else ([2, 5, 11], [2, 3, 5])
    size = len(initial)

    def multiply(a, b):
        result = [0] * (2 * size - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                result[i + j] += x * y
        for degree in range(len(result) - 1, size - 1, -1):
            for shift, coefficient in enumerate(coefficients, 1):
                result[degree - shift] += result[degree] * coefficient
        return [x % modulus if modulus else x for x in result[:size]]

    result, power = [1] + [0] * (size - 1), [0, 1] + [0] * (size - 2)
    while n:
        if n % 2:
            result = multiply(result, power)
        power = multiply(power, power)
        n //= 2
    answer = sum(a * b for a, b in zip(result, initial, strict=True))
    return answer % modulus if modulus else answer


def digits_check(length, modulus, digit_sum, adjacent=False, forbid13=False):
    # Build from right to left with positional weights, unlike the reference DP.
    powers = [pow(10, i, modulus) for i in range(length)]

    @cache
    def suffix(position, remaining, residue, right):
        if not 0 <= remaining <= 9 * (length - position):
            return 0
        if position == length:
            return int(remaining == 0 and residue == 0)
        count = 0
        for digit in range(1 if position == length - 1 else 0, 10):
            if adjacent and digit == right:
                continue
            if forbid13 and digit == 1 and right == 3:
                continue
            count += suffix(
                position + 1,
                remaining - digit,
                (residue + digit * powers[position]) % modulus,
                digit if adjacent or forbid13 else -1,
            )
        return count

    return suffix(0, digit_sum, 0, -1)


def paths_check(width, height, blocked, turns=None, ceiling=None):
    blocked = set(map(tuple, blocked))
    layer = {(0, -1, 0): 1}
    for step in range(width + height):
        following = defaultdict(int)
        for (x, previous, changes), count in layer.items():
            y = step - x
            for direction in (0, 1):
                nx, ny = x + (direction == 0), y + (direction == 1)
                if nx > width or ny > height or (nx, ny) in blocked:
                    continue
                if ceiling is not None and ny > nx + ceiling:
                    continue
                new_changes = changes + int(previous != -1 and previous != direction)
                following[nx, direction, new_changes if turns is not None else 0] += count
        layer = following
    return sum(count for (_, _, used), count in layer.items() if turns is None or used == turns)


def urn_check(capacities, draws, modulus, residue, gaps):
    # Count ordered draws of distinct labelled balls; the common draws! cancels.
    layer = {(0, 0, 0): 1}
    for _ in range(draws):
        following = defaultdict(int)
        for counts, ways in layer.items():
            for colour, cap in enumerate(capacities):
                remaining = cap - counts[colour]
                if remaining:
                    updated = list(counts)
                    updated[colour] += 1
                    following[tuple(updated)] += ways * remaining
        layer = following
    numerator = denominator = 0
    for (r, g, b), ways in layer.items():
        if min(r, g, b) and (r + 2 * g + 3 * b) % modulus == residue:
            denominator += ways
            if r - g >= gaps[0] and g - b >= gaps[1]:
                numerator += ways
    return Fraction(numerator, denominator)


def permutations_check(n, offsets):
    # Ryser's permanent formula with Gray-code subset updates.
    columns = [[int(all(j != (i + d) % n for d in offsets)) for i in range(n)] for j in range(n)]
    row_sums, previous, result = [0] * n, 0, 0
    for index in range(1, 1 << n):
        gray = index ^ (index >> 1)
        changed = gray ^ previous
        column = changed.bit_length() - 1
        direction = 1 if gray & changed else -1
        row_sums = [
            total + direction * value
            for total, value in zip(row_sums, columns[column], strict=True)
        ]
        result += (-1 if (n - gray.bit_count()) % 2 else 1) * prod(row_sums)
        previous = gray
    return result


def tilings_check(height, width, holes, horizontal=None):
    # Place the domino covering the first uncovered cell, scanning cell by cell.
    holes = {(c - 1) * height + r - 1 for r, c in holes}

    @cache
    def cover(position, occupied, left):
        if left is not None and left < 0:
            return 0
        if position == height * width:
            return int(occupied == 0 and left in (None, 0))
        if position in holes:
            return 0 if occupied & 1 else cover(position + 1, occupied >> 1, left)
        if occupied & 1:
            return cover(position + 1, occupied >> 1, left)
        ways = 0
        if position % height < height - 1 and position + 1 not in holes and not occupied & 2:
            ways += cover(position + 1, (occupied >> 1) | 1, left)
        if position + height < height * width and position + height not in holes:
            ways += cover(
                position + 1,
                (occupied >> 1) | (1 << (height - 1)),
                None if left is None else left - 1,
            )
        return ways

    return cover(0, 0, horizontal)


def necklaces_check(length, ones, double_gaps=None):
    def words(n, k, h):
        if n < 3:
            return sum(
                sum(bits) == k
                and all(not (bits[i] and bits[(i + 1) % n]) for i in range(n))
                and (
                    h is None
                    or sum(
                        tuple(bits[(i + j) % n] for j in range(4)) == (1, 0, 0, 1) for i in range(n)
                    )
                    == h
                )
                for bits in product((0, 1), repeat=n)
            )
        result = 0
        for first in product((0, 1), repeat=3):
            if first[0] * first[1] or first[1] * first[2]:
                continue
            states = {(4 * first[0] + 2 * first[1] + first[2], sum(first), 0): 1}
            for _ in range(n - 3):
                following = defaultdict(int)
                for (last, used, seen), count in states.items():
                    for bit in (0, 1):
                        if (last & 1 and bit) or used + bit > k:
                            continue
                        window = (last << 1) | bit
                        new_seen = seen + int(window == 9) if h is not None else 0
                        if h is None or new_seen <= h:
                            following[window & 7, used + bit, new_seen] += count
                states = following
            for (last, used, seen), count in states.items():
                if used != k or (last & 1 and first[0]):
                    continue
                for bit in first:
                    window = (last << 1) | bit
                    seen += int(window == 9)
                    last = window & 7
                if h is None or seen == h:
                    result += count
        return result

    result = 0
    for repeats in range(1, length + 1):
        if length % repeats or ones % repeats:
            continue
        if double_gaps is not None and double_gaps % repeats:
            continue
        phi = sum(gcd(repeats, i) == 1 for i in range(1, repeats + 1))
        result += phi * words(
            length // repeats,
            ones // repeats,
            None if double_gaps is None else double_gaps // repeats,
        )
    assert result % length == 0
    return result // length


def trees_check(n, distances, weighted=False):
    # Independently construct neighbors, then use rational Gaussian elimination.
    a = [[Fraction(0)] * (n - 1) for _ in range(n - 1)]
    for i in range(n - 1):
        neighbors = {(i + sign * d) % n for sign in (-1, 1) for d in distances}
        for j in neighbors:
            weight = 1 + (i + 1) * (j + 1) % 5 if weighted else 1
            a[i][i] += weight
            if j < n - 1:
                a[i][j] -= weight
    result = Fraction(1)
    for k in range(n - 1):
        pivot = a[k][k]
        result *= pivot
        for i in range(k + 1, n - 1):
            ratio = a[i][k] / pivot
            for j in range(k + 1, n - 1):
                a[i][j] -= ratio * a[k][j]
    assert result.denominator == 1
    return result.numerator


def waiting_check(patterns, probabilities, conditional=False):
    # Track a vector of matched-prefix lengths, rather than a single suffix string.
    probabilities = list(map(Fraction, probabilities))
    states = [(0,) * len(patterns)]
    transitions = []
    for state in states:
        edges = []
        for symbol, probability in enumerate(probabilities):
            lengths = []
            for pattern, matched in zip(patterns, state, strict=True):
                text = pattern[:matched] + str(symbol)
                lengths.append(
                    max(k for k in range(len(pattern) + 1) if text.endswith(pattern[:k]))
                )
            winners = [
                i for i, (k, p) in enumerate(zip(lengths, patterns, strict=True)) if k == len(p)
            ]
            if winners:
                edges.append((None, probability, 0 in winners))
            else:
                successor = tuple(lengths)
                if successor not in states:
                    states.append(successor)
                edges.append((states.index(successor), probability, False))
        transitions.append(edges)

    def equation_solve(matrix, rhs):
        # Forward elimination and back substitution, separately from the reference solver.
        a = [list(row) + [value] for row, value in zip(matrix, rhs, strict=True)]
        for k in range(len(a)):
            for i in range(k + 1, len(a)):
                factor = a[i][k] / a[k][k]
                for j in range(k, len(a) + 1):
                    a[i][j] -= factor * a[k][j]
        result = [Fraction(0)] * len(a)
        for i in reversed(range(len(a))):
            result[i] = (a[i][-1] - sum(a[i][j] * result[j] for j in range(i + 1, len(a)))) / a[i][
                i
            ]
        return result

    matrix = [[Fraction(i == j) for j in range(len(states))] for i in range(len(states))]
    target = [Fraction(0)] * len(states)
    for i, edges in enumerate(transitions):
        for j, probability, wins in edges:
            if j is not None:
                matrix[i][j] -= probability
            elif wins:
                target[i] += probability
    if not conditional:
        return equation_solve(matrix, [Fraction(1)] * len(states))[0]
    chance = equation_solve(matrix, target)
    # Condition the chain on eventual success (Doob transform), then solve its mean time.
    conditioned = [
        [value * chance[j] / chance[i] for j, value in enumerate(row)]
        for i, row in enumerate(matrix)
    ]
    return equation_solve(conditioned, [Fraction(1)] * len(states))[0]


CHECKS = {
    "recurrence": recurrence_check,
    "digits": digits_check,
    "paths": paths_check,
    "urn": urn_check,
    "permutations": permutations_check,
    "coins": ladder.coin_check,
    "tilings": tilings_check,
    "necklaces": necklaces_check,
    "trees": trees_check,
    "waiting": waiting_check,
}


@pytest.mark.parametrize("case", ladder.cases(), ids=lambda case: case["id"])
def test_full_size_answer_with_separate_algorithm(case):
    answers = {
        row["id"]: row["answer"]
        for row in map(json.loads, (ROOT / "data/dev/answers.jsonl").read_text().splitlines())
    }
    expected = answers[case["id"]]
    assert ladder.solve(case) == expected
    assert str(CHECKS[case["family"]](**case["parameters"])) == expected
    # Avoid vacuous tasks caused by contradictory constraints.
    assert Fraction(expected) > 0
    if case["family"] == "urn":
        assert Fraction(expected) < 1


def test_foundation_preserved_byte_for_byte():
    directory = ROOT / "data/ladder/01-foundation"
    expected = {
        "questions.jsonl": "9b296aa5c474b045303460955cde150b22cd2f87e5939ea3c3781d38a78da04b",
        "answers.jsonl": "2a8a22f7d9c8703728835af4c53a37c4ff6acd9484b67a4c551f4a5949038b99",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


def test_offline_regeneration_and_disjoint_rungs(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_data.py"),
            "--offline",
            "--out",
            str(tmp_path),
        ],
        check=True,
        timeout=30,
    )
    paths = [
        path
        for split in ("smoke", "dev", "ladder")
        for path in (ROOT / "data" / split).rglob("*.json*")
    ]
    for path in paths:
        relative = path.relative_to(ROOT / "data")
        assert (tmp_path / relative).read_bytes() == path.read_bytes(), relative
    for filename in ("questions.jsonl", "answers.jsonl"):
        joined = b"".join(
            (ROOT / "data/ladder" / tier / filename).read_bytes() for tier in ladder.TIERS
        )
        assert joined == (ROOT / "data/dev" / filename).read_bytes()


def test_small_instances_against_enumeration():
    # Digit constraints, including leading zero, adjacent equal digits and 13 direction.
    for adjacent, forbid13 in product((False, True), repeat=2):
        count = sum(
            n % 7 == 0
            and sum(map(int, str(n))) == 10
            and (not adjacent or all(a != b for a, b in zip(str(n), str(n)[1:], strict=False)))
            and (not forbid13 or "13" not in str(n))
            for n in range(100, 1000)
        )
        assert ladder.digits(3, 7, 10, adjacent, forbid13) == count
    for turns in (None, 1, 2, 3, 4):
        count = 0
        for east in combinations(range(7), 4):
            directions = [int(i not in east) for i in range(7)]
            changes = sum(a != b for a, b in zip(directions, directions[1:], strict=False))
            x = y = 0
            valid = True
            for direction in directions:
                x += direction == 0
                y += direction == 1
                valid &= (x, y) != (2, 1) and y <= x + 1
            count += valid and (turns is None or changes == turns)
        assert ladder.paths(4, 3, [(2, 1)], turns, 1) == count
    for offsets in ((0, 1), (0, 1, 3)):
        count = sum(
            all(p[i] not in {(i + d) % 6 for d in offsets} for i in range(6))
            for p in permutations(range(6))
        )
        assert ladder.permutations(6, offsets) == count
    assert ladder.coins([1, 3, 4], [4, 3, 2], 12, 4, 2) == sum(
        a + 3 * b + 4 * c == 12 and a + b + c == 4 and a % 2 + b % 2 + c % 2 == 2
        for a, b, c in product(range(5), range(4), range(3))
    )


@pytest.mark.parametrize("length, ones", [(6, 2), (8, 3), (12, 4)])
def test_circular_pattern_counts_against_rotation_classes(length, ones):
    classes = defaultdict(set)
    for positions in combinations(range(length), ones):
        bits = "".join("1" if i in positions else "0" for i in range(length))
        if "11" in bits + bits[0]:
            continue
        occurrences = sum((bits + bits[:3])[i : i + 4] == "1001" for i in range(length))
        canonical = min(bits[i:] + bits[:i] for i in range(length))
        classes[occurrences].add(canonical)
    assert ladder.necklaces(length, ones) == sum(map(len, classes.values()))
    for occurrences in range(ones + 1):
        assert ladder.necklaces(length, ones, occurrences) == len(classes[occurrences])


def test_tiling_and_waiting_known_cases():
    assert ladder.tilings(2, 4, []) == 5
    assert ladder.tilings(2, 4, [], horizontal=2) == 3
    assert ladder.tilings(2, 4, [(1, 1)]) == 0
    assert ladder.waiting(["00"], ["1/2", "1/2"]) == 6
    assert ladder.waiting(["01"], ["1/2", "1/2"]) == 4
    assert ladder.waiting(["00"], ["2/5", "3/5"]) == Fraction(35, 4)
    assert ladder.waiting(["0", "1"], ["2/5", "3/5"], conditional=True) == 1
    # Against 10, 00 can only win on the first two draws (unconditional mean is 3).
    assert ladder.waiting(["00", "10"], ["1/2", "1/2"], conditional=True) == 2


def test_spanning_trees_against_edge_subset_enumeration():
    n = 6
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if min(j - i, n - j + i) in (1, 2)]
    for weighted in (False, True):
        total = 0
        for selected in combinations(edges, n - 1):
            components = list(range(n))
            weight = 1
            for i, j in selected:
                before, after = components[i], components[j]
                if before == after:
                    break
                components = [after if c == before else c for c in components]
                weight *= 1 + (i + 1) * (j + 1) % 5 if weighted else 1
            else:
                total += weight
        assert ladder.trees(n, (1, 2), weighted) == total
