"""Original, deterministic development problems and exact reference solvers.

Organizer material: never imported by the runner or supplied to an agent.
Only Python's standard library is required. Difficulty labels are design goals,
not measured model performance. See tests/test_ladder.py for separate checks.
"""

from collections import defaultdict
from fractions import Fraction
from functools import cache
from itertools import product
from math import comb, gcd

TIERS = ("01-foundation", "02-computation", "03-structure", "04-challenge")
VERSION = "ladder-v1"


def recurrence(n, modulus=None, prefix=False):
    # State at k: a_k, a_(k-1), a_(k-2), sum(a_0..a_k).
    matrix = [[2, 3, 5, 0], [1, 0, 0, 0], [0, 1, 0, 0], [2, 3, 5, 1]]
    state = [11, 5, 2, 18]

    def multiply(a, b):
        rows = [
            [sum(x * y for x, y in zip(row, col, strict=True)) for col in zip(*b, strict=True)]
            for row in a
        ]
        return [[x % modulus if modulus else x for x in row] for row in rows]

    if n < 3:
        return sum([2, 5, 11][: n + 1]) if prefix else [2, 5, 11][n]
    power = n - 2
    vector = [[x] for x in state]
    while power:
        if power & 1:
            vector = multiply(matrix, vector)
        matrix = multiply(matrix, matrix)
        power //= 2
    return vector[3 if prefix else 0][0]


def digits(length, modulus, digit_sum, adjacent=False, forbid13=False):
    # A sentinel prevents a leading zero without treating it as a real digit.
    states = {(0, 0, -1): 1}
    for position in range(length):
        next_states = defaultdict(int)
        remaining = length - position - 1
        for (remainder, total, previous), count in states.items():
            for digit in range(1 if position == 0 else 0, 10):
                if adjacent and digit == previous:
                    continue
                if forbid13 and previous == 1 and digit == 3:
                    continue
                new_sum = total + digit
                if new_sum <= digit_sum <= new_sum + 9 * remaining:
                    key = (
                        (10 * remainder + digit) % modulus,
                        new_sum,
                        digit if adjacent or forbid13 else -1,
                    )
                    next_states[key] += count
        states = next_states
    return sum(
        count for (rem, total, _), count in states.items() if rem == 0 and total == digit_sum
    )


def paths(width, height, blocked, turns=None, ceiling=None):
    blocked = set(map(tuple, blocked))

    @cache
    def visit(x, y, last, used):
        if (x, y) in blocked or (ceiling is not None and y > x + ceiling):
            return 0
        if turns is not None and used > turns:
            return 0
        if x == width and y == height:
            return int(turns is None or used == turns)
        result = 0
        for direction, dx, dy in ((0, 1, 0), (1, 0, 1)):
            if x + dx <= width and y + dy <= height:
                extra = int(last != -1 and last != direction) if turns is not None else 0
                result += visit(x + dx, y + dy, direction, used + extra)
        return result

    return visit(0, 0, -1, 0)


def urn(capacities, draws, modulus, residue, gaps):
    numerator = denominator = 0
    for red in range(capacities[0] + 1):
        for green in range(capacities[1] + 1):
            blue = draws - red - green
            if not 1 <= blue <= capacities[2] or min(red, green) < 1:
                continue
            if (red + 2 * green + 3 * blue) % modulus != residue:
                continue
            ways = comb(capacities[0], red) * comb(capacities[1], green)
            ways *= comb(capacities[2], blue)
            denominator += ways
            if red - green >= gaps[0] and green - blue >= gaps[1]:
                numerator += ways
    return Fraction(numerator, denominator)


def permutations(n, offsets):
    all_bits = (1 << n) - 1
    allowed = [all_bits ^ sum(1 << ((row + d) % n) for d in offsets) for row in range(n)]
    counts = [0] * (1 << n)
    counts[0] = 1
    for mask in range(len(counts) - 1):
        count = counts[mask]
        if not count:
            continue
        candidates = allowed[mask.bit_count()] & ~mask
        while candidates:
            bit = candidates & -candidates
            counts[mask | bit] += count
            candidates -= bit
    return counts[-1]


def coins(values, caps, target, number=None, odd=None):
    states = {(0, 0, 0): 1}
    for value, cap in zip(values, caps, strict=True):
        next_states = defaultdict(int)
        for (total, used, odds), count in states.items():
            for amount in range(min(cap, (target - total) // value) + 1):
                new_used = used + amount if number is not None else 0
                new_odds = odds + amount % 2 if odd is not None else 0
                if number is not None and new_used > number:
                    continue
                if odd is not None and new_odds > odd:
                    continue
                next_states[total + amount * value, new_used, new_odds] += count
        states = next_states
    return states.get((target, number or 0, odd or 0), 0)


def tilings(height, width, holes, horizontal=None):
    hole_masks = [0] * width
    for row, column in holes:
        hole_masks[column - 1] |= 1 << (row - 1)
    full = (1 << height) - 1

    @cache
    def fill(occupied, outgoing):
        if occupied == full:
            return ((outgoing, 1),)
        bit = next(1 << r for r in range(height) if not occupied & (1 << r))
        result = defaultdict(int)
        # A horizontal domino ends in the next column.
        for mask, count in fill(occupied | bit, outgoing | bit):
            result[mask] += count
        if bit << 1 <= full and not occupied & (bit << 1):
            for mask, count in fill(occupied | bit | (bit << 1), outgoing):
                result[mask] += count
        return tuple(result.items())

    states = {(0, 0): 1}
    for holes_here in hole_masks:
        next_states = defaultdict(int)
        for (incoming, used), count in states.items():
            if incoming & holes_here:
                continue
            for outgoing, ways in fill(incoming | holes_here, 0):
                new_used = used + outgoing.bit_count() if horizontal is not None else 0
                if horizontal is None or new_used <= horizontal:
                    next_states[outgoing, new_used] += count * ways
        states = next_states
    return states.get((0, horizontal or 0), 0)


def necklaces(length, ones, double_gaps=None):
    """Burnside, counting labelled circles via the positive gaps between ones."""

    def labelled(n, k, h):
        if k == 0:
            return int(h is None or h == 0)
        if h is None:
            return n * comb(n - k, k) // (n - k) if n >= 2 * k else 0
        # Each gap has >=1 zero; exactly h gaps have exactly two zeros.
        coefficients = [1] + [0] * (n - k)
        for _ in range(k - h):
            next_coefficients = [0] * len(coefficients)
            for total, count in enumerate(coefficients):
                for gap in range(1, len(coefficients) - total):
                    if gap != 2:
                        next_coefficients[total + gap] += count
            coefficients = next_coefficients
        remaining = n - k - 2 * h
        ways = coefficients[remaining] if 0 <= remaining < len(coefficients) else 0
        return n * comb(k, h) * ways // k

    fixed_sum = 0
    for shift in range(length):
        period = gcd(length, shift)
        repeats = length // period
        if ones % repeats or (double_gaps is not None and double_gaps % repeats):
            continue
        fixed_sum += labelled(
            period, ones // repeats, None if double_gaps is None else double_gaps // repeats
        )
    assert fixed_sum % length == 0
    return fixed_sum // length


def tree_matrix(n, distances, weighted=False):
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if min(j - i, n - (j - i)) not in distances:
                continue
            weight = 1 + ((i + 1) * (j + 1) % 5) if weighted else 1
            matrix[i][i] += weight
            matrix[j][j] += weight
            matrix[i][j] = matrix[j][i] = -weight
    return [row[:-1] for row in matrix[:-1]]


def trees(n, distances, weighted=False):
    # Fraction-free elimination of a Laplacian minor (matrix-tree theorem).
    a = tree_matrix(n, distances, weighted)
    previous = 1
    for k in range(len(a) - 1):
        pivot = a[k][k]
        assert pivot  # These connected graphs have positive definite minors.
        for i in range(k + 1, len(a)):
            for j in range(k + 1, len(a)):
                numerator = pivot * a[i][j] - a[i][k] * a[k][j]
                assert numerator % previous == 0
                a[i][j] = numerator // previous
        previous = pivot
    return a[-1][-1]


def linear_solve(matrix, values):
    a = [
        [Fraction(x) for x in row] + [Fraction(value)]
        for row, value in zip(matrix, values, strict=True)
    ]
    for k in range(len(a)):
        pivot = next(i for i in range(k, len(a)) if a[i][k])
        a[k], a[pivot] = a[pivot], a[k]
        divisor = a[k][k]
        a[k] = [x / divisor for x in a[k]]
        for i in range(len(a)):
            if i != k:
                factor = a[i][k]
                a[i] = [x - factor * y for x, y in zip(a[i], a[k], strict=True)]
    return [row[-1] for row in a]


def pattern_chain(patterns, probabilities):
    prefixes = sorted({p[:i] for p in patterns for i in range(len(p))}, key=lambda p: (len(p), p))
    matrix = [[Fraction(int(i == j)) for j in range(len(prefixes))] for i in range(len(prefixes))]
    target = [Fraction(0)] * len(prefixes)
    for i, prefix in enumerate(prefixes):
        for symbol, probability in enumerate(map(Fraction, probabilities)):
            text = prefix + str(symbol)
            winners = [p for p in patterns if text.endswith(p)]
            if winners:
                if patterns[0] in winners:
                    target[i] += probability
                continue
            suffix = max((p for p in prefixes if text.endswith(p)), key=len)
            matrix[i][prefixes.index(suffix)] -= probability
    return matrix, target


def waiting(patterns, probabilities, conditional=False):
    matrix, target = pattern_chain(patterns, probabilities)
    if not conditional:
        return linear_solve(matrix, [1] * len(matrix))[0]
    chance = linear_solve(matrix, target)
    # g_s = E_s[T * 1{first pattern wins}]; (I-Q)g = chance.
    weighted_time = linear_solve(matrix, chance)
    return weighted_time[0] / chance[0]


SOLVERS = {
    name: globals()[name]
    for name in (
        "recurrence",
        "digits",
        "paths",
        "urn",
        "permutations",
        "coins",
        "tilings",
        "necklaces",
        "trees",
        "waiting",
    )
}


def cases():
    """Stable IDs: one of each family per rung, preserving dev-001 through 010."""
    rows = []

    def add(tier, family, params, problem, technique):
        rows.append(
            {
                "tier": TIERS[tier - 1],
                "family": family,
                "parameters": params,
                "problem": problem,
                "technique": technique,
            }
        )

    for tier in (2, 3, 4):
        index = tier - 2
        n = [45, 10**12 + 39, 10**18 + 123][index]
        modulus = None if tier == 2 else 1_000_000_007
        expression = f"a_{n}" if tier != 4 else f"a_0 + a_1 + ... + a_{n}"
        request = (
            f"Find {expression}."
            if modulus is None
            else (
                f"Find the remainder when {expression} is divided by {modulus}. "
                f"Return an integer from 0 through {modulus - 1}."
            )
        )
        add(
            tier,
            "recurrence",
            dict(n=n, modulus=modulus, prefix=tier == 4),
            "Let a_0 = 2, a_1 = 5 and a_2 = 11. For every integer n >= 3, "
            "a_n = 2*a_(n-1) + 3*a_(n-2) + 5*a_(n-3). " + request,
            "Exact recurrence; fast exponentiation; augment the state for a prefix sum.",
        )

        length, divisor, total = [(10, 37, 42), (16, 23, 61), (22, 17, 83)][index]
        constraints = " Adjacent digits must be different." if tier >= 3 else ""
        if tier == 4:
            constraints += " The consecutive two-digit substring 13 must never occur."
        add(
            tier,
            "digits",
            dict(
                length=length,
                modulus=divisor,
                digit_sum=total,
                adjacent=tier >= 3,
                forbid13=tier == 4,
            ),
            f"How many positive integers have exactly {length} decimal digits, are divisible "
            f"by {divisor}, and have digit sum {total}? The first digit cannot be zero."
            + constraints,
            "Track the remainder, digit sum and relevant suffix; prune impossible sums.",
        )

        width, height, turns = [(18, 14, None), (24, 20, 17), (30, 26, 21)][index]
        blocked = [(4, 3), (9, 7), (13, 10)]
        constraints = ""
        if turns is not None:
            constraints += (
                f" The path must change direction exactly {turns} times. "
                "A change is between consecutive steps; the first step is not a change."
            )
        if tier == 4:
            constraints += " Every visited vertex (x,y) must satisfy y <= x + 2."
        add(
            tier,
            "paths",
            dict(
                width=width,
                height=height,
                blocked=blocked,
                turns=turns,
                ceiling=2 if tier == 4 else None,
            ),
            f"A lattice path goes from (0,0) to ({width},{height}), taking only steps "
            "(1,0) and (0,1). It may not visit (4,3), (9,7) or (13,10). "
            "How many such paths are there?" + constraints,
            "Count paths with position, previous direction and turns in the state.",
        )

        capacities, draws, mod, residue, gaps = [
            ((9, 11, 13), 12, 1, 0, (1, 1)),
            ((18, 21, 24), 22, 5, 0, (2, 2)),
            ((30, 35, 40), 35, 7, 3, (4, 3)),
        ][index]
        condition = "at least one ball of each colour was drawn"
        if tier >= 3:
            condition += f" and R + 2*G + 3*B has remainder {residue} modulo {mod}"
        add(
            tier,
            "urn",
            dict(capacities=capacities, draws=draws, modulus=mod, residue=residue, gaps=gaps),
            f"An urn contains {capacities[0]} red, {capacities[1]} green and "
            f"{capacities[2]} blue balls. Draw {draws} balls uniformly without replacement. "
            "Let R, G and B be the numbers of the three colours drawn. "
            f"Given that {condition}, what is the probability that "
            f"R - G >= {gaps[0]} and G - B >= {gaps[1]}? Return an exact fraction.",
            "Hypergeometric weights; normalize over the conditioning event, not all draws.",
        )

        n, offsets = [(11, (0, 1)), (15, (0, 1, 4)), (18, (0, 1, 4, 9))][index]
        add(
            tier,
            "permutations",
            dict(n=n, offsets=offsets),
            f"How many permutations p of the labelled integers 0 through {n - 1} satisfy "
            f"the following condition? For every i, p(i) must not equal (i+d) modulo {n} "
            f"for any d in {list(offsets)}. Each residue is represented from 0 to {n - 1}.",
            "A permanent: subset assignment DP or Ryser inclusion-exclusion.",
        )

        values, caps, target, number, odd = [
            ((1, 3, 7, 11, 17), (24, 18, 14, 10, 8), 180, None, None),
            ((2, 5, 9, 14, 23, 31), (18, 15, 12, 10, 8, 6), 360, 26, None),
            ((3, 7, 12, 19, 29, 43, 61), (16, 14, 12, 10, 8, 6, 4), 620, 30, 4),
        ][index]
        constraints = f" Use exactly {number} coins in total." if number is not None else ""
        if odd is not None:
            constraints += f" Exactly {odd} denominations must be used an odd number of times."
        add(
            tier,
            "coins",
            dict(values=values, caps=caps, target=target, number=number, odd=odd),
            f"Coins have denominations {list(values)}. The respective maximum available "
            f"numbers of coins are {list(caps)}. How many ways can you select coins with "
            f"total value {target}? Coins of the same denomination are indistinguishable, "
            "and order does not matter." + constraints,
            "Bounded generating functions or meet in the middle with all constraints.",
        )

        height, width = [(3, 18), (4, 24), (4, 40)][index]
        holes = (
            [(2, 5), (2, 14)]
            if tier == 2
            else [(2, c) for c in range(5, 25, 5)]
            if tier == 3
            else [(2, c) for c in range(3, 41, 7)] + [(3, c) for c in range(4, 41, 7)]
        )
        horizontal = 54 if tier == 4 else None
        constraints = (
            (
                f" Exactly {horizontal} dominoes must be horizontal (covering two cells "
                "in the same row)."
            )
            if horizontal is not None
            else ""
        )
        add(
            tier,
            "tilings",
            dict(height=height, width=width, holes=holes, horizontal=horizontal),
            f"A board has {height} rows and {width} columns, numbered starting at 1. "
            f"Remove the cells with (row,column) coordinates {holes}. "
            "How many ways can the remaining board be covered by 1-by-2 dominoes, "
            "with no gaps or overlaps? Dominoes may be horizontal or vertical. "
            "The board is fixed: rotations and reflections of a tiling are not identified."
            + constraints,
            "A frontier/profile DP; track horizontal dominoes when required.",
        )

        length, ones, double_gaps = [(36, 12, None), (60, 20, None), (84, 28, 4)][index]
        constraints = ""
        if double_gaps is not None:
            constraints = (
                f" Exactly {double_gaps} cyclic starting positions must read 1001 "
                "in four successive clockwise positions; occurrences can overlap "
                "and can cross the end of a written representation."
            )
        add(
            tier,
            "necklaces",
            dict(length=length, ones=ones, double_gaps=double_gaps),
            f"Place {ones} ones and {length - ones} zeros on a circle of {length} positions. "
            "No two ones may be adjacent, including across the end of a written representation. "
            "How many different arrangements are there if rotations are considered the same "
            "but reflections are not identified?" + constraints,
            "Burnside's lemma; count periodic fixed points, including cyclic pattern constraints.",
        )

        n, distances = [(9, (1, 3)), (12, (1, 2, 5)), (16, (1, 4, 7))][index]
        weighted = tier == 4
        rule = (
            (
                "For each eligible pair i < j, there are 1 + (((i+1)*(j+1)) modulo 5) "
                "distinct parallel edges joining i and j. There are no other edges. "
                "Choosing a different parallel edge counts as a different tree."
            )
            if weighted
            else ("There is exactly one edge for each eligible pair and no other edges.")
        )
        add(
            tier,
            "trees",
            dict(n=n, distances=distances, weighted=weighted),
            f"A graph has labelled vertices 0 through {n - 1} arranged on a circle. "
            f"A pair of distinct vertices i,j is eligible for an edge exactly when "
            f"min(abs(i-j), {n}-abs(i-j)) belongs to {list(distances)}. "
            + rule
            + f" How many spanning trees does this graph have? A spanning tree connects all "
            f"{n} vertices, has {n - 1} edges and contains no cycle.",
            "Matrix-tree theorem with an exact integer determinant; parallel-edge multiplicities.",
        )

        patterns, probabilities = [
            (("010001",), ("2/5", "3/5")),
            (("012010", "201202"), ("1/3", "1/3", "1/3")),
            (("010201", "201020", "112011"), ("1/2", "1/3", "1/6")),
        ][index]
        distribution = ", ".join(f"P({s}) = {p}" for s, p in enumerate(probabilities))
        condition = (
            (
                f" Conditional on {patterns[0]} being the first stopping pattern, "
                "what is the expected value of T?"
            )
            if tier == 4
            else (" What is the expected value of T?")
        )
        add(
            tier,
            "waiting",
            dict(patterns=patterns, probabilities=probabilities, conditional=tier == 4),
            "Independent symbols are drawn with " + distribution + ". Starting with an empty "
            f"sequence, stop as soon as any of {list(patterns)} appears as a consecutive "
            "substring. Let T be the total number of symbols drawn, including those in "
            "the stopping pattern. Patterns can overlap with earlier partial matches."
            + condition
            + " Return an exact fraction.",
            "Prefix automaton and exact absorbing-chain equations; condition on the winner.",
        )

    for i, row in enumerate(rows, 11):
        row["id"] = f"dev-{i:03}"
    return rows


def solve(case):
    return str(SOLVERS[case["family"]](**case["parameters"]))


def coin_check(values, caps, target, number=None, odd=None):
    """Independent full-size meet-in-the-middle check for bounded coin counts."""
    midpoint = len(values) // 2

    def half(start, stop):
        counts = defaultdict(int)
        for amounts in product(*(range(cap + 1) for cap in caps[start:stop])):
            total = sum(v * a for v, a in zip(values[start:stop], amounts, strict=True))
            counts[
                total,
                sum(amounts) if number is not None else 0,
                sum(a % 2 for a in amounts) if odd is not None else 0,
            ] += 1
        return counts

    left, right = half(0, midpoint), half(midpoint, len(values))
    return sum(
        count * right.get((target - total, (number or 0) - used, (odd or 0) - odds), 0)
        for (total, used, odds), count in left.items()
    )
