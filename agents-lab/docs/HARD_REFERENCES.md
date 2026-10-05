# Hard-set references — organizer spoilers

The source problems and their mathematical reductions come from **Epoch AI,
[FrontierMath Sample Problems](https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems)**,
accessed 5 October 2026, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The local source fixture retains the original statements and published answers.
The manifest records every adaptation. These are public samples and new variants,
not the private FrontierMath evaluation set, and Epoch AI does not endorse this lab.

The difficult part of these questions is finding and justifying the reduction.
`scripts/hard_reference.py` implements the resulting computations. Its speed does
not imply that the mathematical discovery is easy. Some reductions depend on deep
theorems and the published proofs; we have not independently reproved all of those
theorems. Tests separately check arithmetic, small instances, boundary cases and
numerical precision. A published answer alone is not our only numerical check.

## hard-001 and hard-011: polynomial classification

The source uses polynomial monodromy and the prime degree to identify the normalized
Chebyshev polynomial `p(x) = 2 T_d(x/2)`. For degrees 19 and 31, this is monic, odd,
real, and has linear coefficient `-d`. Its equal-value locus has the diagonal plus
`(d-1)/2` non-linear components, so it meets the required factorization condition.
The source gives the classification argument ruling out other normalized possibilities.
The same argument applies to the variant's prime degree 31.

Evaluate using `P_0(x)=2`, `P_1(x)=x`, `P_d(x)=x P_(d-1)(x)-P_(d-2)(x)`.
Tests independently use the explicit binomial coefficient formula and verify
`P_d(z+1/z)=z^d+z^(-d)` with exact fractions.

| ID | Evaluation | Answer |
| --- | --- | --- |
| hard-001 | P_19(19) | 1876572071974094803391179 |
| hard-011 | P_31(37) | 4021687983592856545155979653421541070749098038637 |

## hard-002 and hard-008: matrix orbits and determinant characters

The non-commuting pairs in the relations form the chain `1–2–4–3`. With involutions,
the displayed commutator equation is equivalent to the braid relation. Thus these
are representations of `S_5`, and simultaneous conjugacy is representation isomorphism.
Over the complex numbers, a representation is a direct sum of irreducibles, whose
dimensions are `1,1,4,4,5,5,6`. Count multiplicities with total dimension 1000:

`[t^1000] 1 / ((1-t)^2 (1-t^4)^2 (1-t^5)^2 (1-t^6)) = 625243878951`.

For the new determinant-one variant, the signs of a transposition on these seven
irreducibles are respectively `+,-,-,-,+,-,-`. All four generators are conjugate
transpositions, so the parity of the combined determinant character must be even.
Tracking this parity gives **312621939451**. Tests independently derive dimensions
and signs from partitions of 5, hook lengths and transposition characters, then use
a signed rational generating function rather than the reference's two-state DP.

## hard-003 and hard-009: points on a curve over a huge finite field

For `x^3 y + y^3 z + z^3 x = 0`, the projective plane quartic is smooth in characteristic
5, hence has genus 3. If a singular point had a zero coordinate, the derivatives
would force all coordinates to be zero. If all were nonzero, multiplying the three
derivative equations would force `27 = -1` in characteristic 5, a contradiction.

Exhaustive arithmetic over `F_5`, `F_25`, and `F_125` gives 6, 26, and 126 projective
points. Newton identities and reciprocal symmetry determine the Weil polynomial
`L(t)=1+125t^6`. Consequently the six Frobenius eigenvalues satisfy `alpha^6=-125`.
For a multiple `d` of 6, the point count is `5^d + 1 - 6*(-125)^(d/6)`.

- Degree 18: **3814708984376**, agreeing with the corrected published solution.
- Degree 30 (new variant): **931322574798583984376**.

The test implements the small extension fields explicitly and independently reconstructs
the power sums. No enumeration over the enormous target field is needed.

## hard-004 and hard-010: the recursive permutation map

For values `a<b`, the source proves that `b` precedes `a` after the map exactly when
some `c>b` occurs between `b` and `a` in the original order `b,...,c,...,a`.
Considering the relative order of `a,b` and the `n-b` larger values gives probability
`1/2 - 1/(n-b+2)`. Sum these probabilities by linearity of expectation.

- Consecutive values: `E = (n+1)/2 - H_n`.
- All pairs (new variant): `E = n(n+7)/4 - (n+1) H_n`.

At `n=10^12`, the floors are **499999999972** and **249999999973541763219141**.
The latter is not the inversion count of a uniformly random output permutation:
the recursively transformed output is not uniform.

Tests exhaust all input permutations up to size 7 and check both formulas. For the
large floors, the reference uses Euler–Maclaurin with an explicit remainder bound
and 70-digit Decimal arithmetic; both ends of the resulting interval give the same
integer. Exact rational harmonic numbers provide an additional smaller-size check.

## hard-005 and hard-012: perimeter of the central random cell

For a point on a potential boundary segment, let `B` be the number of other diagonals
separating that point from the center. The probability that none is selected is
`(1-p)^B`. Integrate this along each polygon edge and diagonal, additionally multiply
by `p` for a diagonal's own selection, and add by linearity of expectation.

`B` is constant between chord intersections, so this is a finite sum. Rotational
symmetry reduces the calculation to one chord of each cyclic length. The polygons
are odd-sided, so the center never lies on a diagonal. Crucially, this counts the
boundary of the central cell, not the total length of every selected diagonal.

| ID | n | p | Expected perimeter, approximately | Requested floor |
| --- | --- | --- | --- | --- |
| hard-005 | 101 | 0.001 | 4.7718801532200280090358766683 | 4771880153 |
| hard-012 | 79 | 0.002 | 4.4714003245955479354532082539 | 4471400324 |

The reference reproduces the published first answer. Tests also check every subset
of the five diagonals of a pentagon by explicit polygon clipping, an independent
geometric computation. The full instances agree at 35 and 60 decimal digits and
with the fast floating-point implementation. The requested floors are well separated
from the integer boundaries at these precisions.

## hard-006: an additive-divisor asymptotic

The ordered count equals `sum_(m<=x) d_2(m) d_3(m+1)`. The published solution invokes
the known additive-divisor asymptotic to obtain `alpha=1`, `beta=3`, and

`C = (1/2) product_(primes p) (1 - 2/p^2 + 1/p^3)`.

The reference computes a finite product to 100,000 in Decimal arithmetic. The omitted
product lies between `1-2/100000` and 1, using the sum over all larger integers as an
upper bound on the prime tail. Both resulting bounds for `1000*C` have floor **214**.
This checks the numerical answer; the asymptotic reduction relies on the mathematical
result cited and explained in Epoch AI's solution, not on fitting finite samples.

## hard-007: the extremal sequence in the Tsirelson construction

The source's lower construction and upper-bound argument show that the largest
index is `N=f(f(f(3)))` where `f(m)=m*2^m`. Its published proof treats the support
constraints and pointwise closure; these are essential and are not established by
merely running the short arithmetic routine.

Since `v_2(f(m)) = v_2(m)+m`, the requested exponent is
`3 + 24 + 24*2^24 = 402653211`. Constructing N itself is unnecessary. This is the
most specialized background question in the set; it is an optional stretch exercise.

## Progressive hints for teaching

Give one hint at a time and label such runs as **hint-assisted**, keeping their scores
separate from the unassisted pilot. Related original/variant pairs are not independent
test items and must not be split across a purported held-out evaluation.

| Family | First hint | Stronger hint |
| --- | --- | --- |
| Polynomial | Investigate polynomials with unusually reducible p(x)-p(y). | Try the normalized Chebyshev family and check the linear coefficient. |
| Matrix orbits | Interpret the four matrices as generators of a finite group. | The Coxeter diagram is a four-vertex path; count irreducible multiplicities. |
| Finite field | Exhaustion over the target field is unnecessary. | A smooth genus-three curve's first three extension counts determine its Weil polynomial. |
| Permutations | Work with one pair of values and linearity of expectation. | Characterize the larger separator that makes the pair inverted after recursion. |
| Geometry | A diagonal contributes only the part exposed to the central cell. | Integrate the probability that no selected diagonal separates a boundary point from the center. |
| Asymptotics | Rewrite the number of tuples with divisor functions. | This is a shifted additive-divisor correlation, whose constant is an Euler product. |
| Tsirelson space | Bound how far a decreasing positive tail can extend at each amplitude scale. | The relevant growth function is f(m)=m*2^m. |
