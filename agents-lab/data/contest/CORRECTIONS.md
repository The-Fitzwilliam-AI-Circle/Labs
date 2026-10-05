# Reference corrections

## contest-v1.1: CMIMC 2025 problem 7

`contest-cmimc-2025-07` asks for the sum of 3-adic valuations of the first 810
prefixes of the digit string `12345678901234567890...`.

Both the MathArena key and the [original contest solution, problem 7](https://cmimc.com/archive/cmimc2025/algnt_sol.pdf)
state **930**. Exact arithmetic gives **932** for the published statement.

Two computations verify the correction: building each integer and repeatedly
dividing by 3; and separately counting divisible prefixes modulo each power of 3.
The valuation frequencies are:

| Valuation | Number of prefixes |
| ---: | ---: |
| 0 | 243 |
| 1 | 324 |
| 2 | 162 |
| 3 | 54 |
| 4 | 18 |
| 5 | 6 |
| 6 | 2 |
| 8 | 1 |

The counts sum to 810 and their weighted sum is 932. In particular, the prefix
ending at position **798** is divisible by 3⁸ but not 3⁹. The published derivation
counts the relevant branch only through valuation 6, omitting two contributions.
The corrected branch sum is 204, giving `324 + 2 × 202 + 204 = 932`.

The question and all selected IDs remain unchanged. The fixture preserves the
original source answer, adds an explicit `answer_correction`, and regenerates a
key of 932. Tests cover both exact computations. This item was not in the twelve
pilot questions, so the correction does not change either pilot score. It was
found by offline verification, not by selecting around a model's performance.
The organizer retains the original key and freeze record in `history/contest-v1/`.
All validation questions, keys and frozen hashes remain unchanged.
