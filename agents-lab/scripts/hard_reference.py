"""Reference reductions for the hard set; organizer material, never agent input.

Mathematical justifications and source attribution are in docs/HARD_REFERENCES.md.
Computations use the standard library; no Sage, internet, or model is required.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from math import cos, floor, pi, sin, sqrt


def polynomial(degree, value):
    previous, current = 2, value
    for _ in range(2, degree + 1):
        previous, current = current, value * current - previous
    return current


def representations(dimension, determinant_one=False):
    # Irreducible dimensions and determinant characters of S_5.
    types = [(1, 0), (1, 1), (4, 1), (4, 1), (5, 0), (5, 1), (6, 1)]
    ways = [[0, 0] for _ in range(dimension + 1)]
    ways[0][0] = 1
    for size, sign in types:
        for total in range(size, dimension + 1):
            for parity in (0, 1):
                ways[total][parity] += ways[total - size][parity ^ sign]
    return ways[dimension][0] if determinant_one else sum(ways[dimension])


def small_field_count(degree):
    """Count the projective quartic over F_5, F_25 or F_125 by exhaustion."""
    modulus = {1: [0, 1], 2: [2, 0, 1], 3: [1, 1, 0, 1]}[degree]
    size = 5**degree
    elements = [[(x // 5**i) % 5 for i in range(degree)] for x in range(size)]
    addition, multiplication = [], []
    for x in elements:
        sums, products = [], []
        for y in elements:
            sums.append(
                sum(((a + b) % 5) * 5**i for i, (a, b) in enumerate(zip(x, y, strict=True)))
            )
            coefficients = [0] * (2 * degree - 1)
            for i, a in enumerate(x):
                for j, b in enumerate(y):
                    coefficients[i + j] += a * b
            for power in range(2 * degree - 2, degree - 1, -1):
                factor = coefficients[power]
                for j in range(degree):
                    coefficients[power - degree + j] -= factor * modulus[j]
            products.append(sum((coefficients[i] % 5) * 5**i for i in range(degree)))
        addition.append(sums)
        multiplication.append(products)
    cubes = [multiplication[x][multiplication[x][x]] for x in range(size)]
    affine = sum(
        addition[multiplication[cubes[x]][y]][addition[cubes[y]][x]] == 0
        for x in range(size)
        for y in range(size)
    )
    return affine + 2  # The two projective points with z=0.


def finite_field(degree):
    # Smooth plane quartic, genus 3. Counts for degrees 1,2,3 determine L(t)=1+125t^6.
    if degree % 6:
        trace = 0
    else:
        trace = 6 * (-125) ** (degree // 6)
    return 5**degree + 1 - trace


def permutation_expectation(n, all_pairs=False):
    # Use Euler–Maclaurin around an exactly summable anchor. The first omitted term
    # bounds the harmonic-number error; propagate it through the expectation.
    with localcontext() as context:
        context.prec = 70
        anchor = 1000
        anchor_harmonic = sum(Decimal(1) / i for i in range(1, anchor + 1))
        coefficients = [
            Fraction(-1, 12),
            Fraction(1, 120),
            Fraction(-1, 252),
            Fraction(1, 240),
            Fraction(-1, 132),
            Fraction(691, 32760),
        ]

        def expansion(value):
            x = Decimal(value)
            return (
                x.ln()
                + 1 / (2 * x)
                + sum(
                    Decimal(c.numerator) / c.denominator / x ** (2 * i)
                    for i, c in enumerate(coefficients, 1)
                )
            )

        harmonic = anchor_harmonic - expansion(anchor) + expansion(n)
        error = 1 / (12 * Decimal(anchor) ** 14) + 1 / (12 * Decimal(n) ** 14)
        error += Decimal("1e-55")  # Far exceeds accumulated 70-digit rounding error here.
        if all_pairs:
            expectation = Decimal(n) * (n + 7) / 4 - (n + 1) * harmonic
            error *= n + 1
        else:
            expectation = Decimal(n + 1) / 2 - harmonic
        lower, upper = floor(expectation - error), floor(expectation + error)
        assert lower == upper, "Precision is insufficient to certify this floor"
        return lower


def prime_sieve(limit):
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for number in range(2, int(limit**0.5) + 1):
        if sieve[number]:
            start = number * number
            sieve[start::number] = b"\x00" * ((limit - start) // number + 1)
    return [number for number in range(2, limit + 1) if sieve[number]]


def asymptotics():
    # Published additive-divisor asymptotic: C = (1/2) prod_p (1 - 2/p^2 + 1/p^3).
    # For omitted p > B, sum 2/p^2 < 2/B, hence product >= 1 - 2/B.
    with localcontext() as context:
        context.prec = 50
        limit = 100_000
        upper = Decimal(1) / 2
        for p in prime_sieve(limit):
            x = Decimal(p)
            upper *= 1 - 2 / x**2 + 1 / x**3
        lower = upper * (1 - Decimal(2) / limit)
        # A conservative allowance for Decimal rounding is immaterial to the floor.
        assert floor(1000 * (lower - Decimal("1e-40"))) == floor(1000 * (upper + Decimal("1e-40")))
        return floor(1000 * upper)


def decimal_vertices(n, precision):
    pi_text = "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
    with localcontext() as context:
        context.prec = precision
        decimal_pi = Decimal(pi_text)
        points = []
        for k in range(n):
            angle = 2 * decimal_pi * k / n
            if angle > decimal_pi:
                angle -= 2 * decimal_pi
            sine = term_s = angle
            cosine = term_c = Decimal(1)
            for j in range(1, precision + 5):
                term_s *= -angle * angle / ((2 * j) * (2 * j + 1))
                term_c *= -angle * angle / ((2 * j - 1) * (2 * j))
                sine += term_s
                cosine += term_c
            points.append((cosine, sine))
        return points


def central_perimeter(n, probability, precision=None):
    """Integrate the chance each chord segment borders the central cell.

    Float is the fast reference; Decimal recomputation checks floors at much higher
    precision. Odd n ensures no selected diagonal passes through the center.
    """
    assert n % 2 == 1
    if precision is not None:
        with localcontext() as context:
            context.prec = precision
            return _central_perimeter(n, Decimal(probability), decimal_vertices(n, precision))
    vertices = [(cos(2 * pi * k / n), sin(2 * pi * k / n)) for k in range(n)]
    return _central_perimeter(n, float(probability), vertices)


def _central_perimeter(n, probability, vertices):
    # Orient each diagonal's affine half-plane so the origin is on its positive side.
    boundaries = []
    for i in range(n):
        for j in range(i + 1, n):
            if j - i in (1, n - 1):
                continue
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            a, b, c = yi - yj, xj - xi, xi * yj - xj * yi
            if c < 0:
                a, b, c = -a, -b, -c
            boundaries.append((i, j, a, b, c))
    q = 1 - probability
    powers = [q**k for k in range(len(boundaries) + 1)]
    expectation = 0
    x0, y0 = vertices[0]
    for distance in range(1, (n + 1) // 2):
        xd, yd = vertices[distance]
        blockers = 0
        events = []
        for i, j, a, b, c in boundaries:
            if i == 0 and j == distance:
                continue
            left = 0 if 0 in (i, j) else a * x0 + b * y0 + c
            right = 0 if distance in (i, j) else a * xd + b * yd + c
            if left < 0 or (left == 0 and right < 0):
                blockers += 1
            if left * right < 0:
                events.append((left / (left - right), 1 if right < 0 else -1))
        last = integral = 0
        for position, change in sorted(events):
            integral += (position - last) * powers[blockers]
            blockers += change
            last = position
        integral += (1 - last) * powers[blockers]
        squared_length = (xd - x0) ** 2 + (yd - y0) ** 2
        length = (
            squared_length.sqrt() if isinstance(squared_length, Decimal) else sqrt(squared_length)
        )
        inclusion = 1 if distance == 1 else probability
        expectation += n * inclusion * length * integral
    return expectation


def banach():
    # Published extremal construction proves N=f^3(3), f(m)=m*2^m.
    # v_2(f(m))=v_2(m)+m, so do not construct N itself.
    first = 3 * 2**3
    second = first * 2**first
    return 3 + first + second


def solve(task_id):
    computations = {
        "hard-001": lambda: polynomial(19, 19),
        "hard-002": lambda: representations(1000),
        "hard-003": lambda: finite_field(18),
        "hard-004": lambda: permutation_expectation(10**12),
        "hard-005": lambda: floor(10**9 * central_perimeter(101, "0.001")),
        "hard-006": asymptotics,
        "hard-007": banach,
        "hard-008": lambda: representations(1000, determinant_one=True),
        "hard-009": lambda: finite_field(30),
        "hard-010": lambda: permutation_expectation(10**12, all_pairs=True),
        "hard-011": lambda: polynomial(31, 37),
        "hard-012": lambda: floor(10**9 * central_perimeter(79, "0.002")),
    }
    return str(computations[task_id]())
