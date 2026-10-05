# Hard mathematics set

**Optional capstone after the [100-question contest bank](../contest/README.md).** It contains
twelve problems spanning algebra, representation theory, finite fields, combinatorial
probability, geometry, analytic number theory and a Banach-space construction.
The original forty questions remain available as warm-up exercises.

Seven problems come from **Epoch AI's public
[FrontierMath samples](https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems)**;
five are explicitly identified new variants. The source is credited under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This is a teaching adaptation,
not the private FrontierMath benchmark or an official FrontierMath score.

The initial live pilot used `gpt-5-nano-2025-08-07` with the unchanged tool agent and
your existing settings. It scored **0/12**: four wrong submissions and eight episodes
that exhausted token or tool allowances. With the larger budget below, the same agent
scored **2/12 (16.7%)**: eight wrong submissions and two budget/truncation failures.
See [the calibration report](../../docs/HARD_CALIBRATION.md) for the full result,
limits, per-question outcomes and the distinction between wrong answers and execution
failures. All twelve candidates were fixed before the first pilot and retained.

## Run the set

From the repository root, the recommended hard-set configuration gives more room
for reasoning: 16,384 output tokens per request, eight model requests, eight Python
calls and 240 seconds per episode. Python still has the same five-second timeout,
256 MiB memory limit, packages and network isolation. The original `config.toml`
is unchanged.

```bash
uv run mathlab run --agent agents/tool.py \
  --questions data/hard/questions.jsonl --config config.hard.toml --out runs/hard-tool
uv run mathlab score --run runs/hard-tool --answers data/hard/answers.jsonl
```

To compare the direct agent, use `--agent agents/direct.py` with the same questions
and configuration and a separate output name. To reproduce the original-budget
pilot, omit `--config config.hard.toml`. Repeated runs get numbered directories;
use the score command printed at the end of each run.

For a lab session, start with three distinct families:

```bash
uv run mathlab run --agent agents/participant.py \
  --questions data/hard/questions.jsonl --config config.hard.toml \
  --id hard-002 --id hard-004 --id hard-005 --out runs/hard-starter
```

The full key still scores this subset. All twelve represent up to 48 minutes of
episode time at the recommended deadline: roughly twelve minutes plus overhead
at the default `--concurrency 4`, or 48 minutes plus overhead with `--concurrency 1`.
Use subsets while iterating and rehearse the chosen concurrency on your backend.

## Problems and progression

| ID | Problem | Origin |
| --- | --- | --- |
| hard-001 | Recover a degree-19 polynomial from a factorization condition | Published sample |
| hard-002 | Count simultaneous-conjugacy orbits of matrix tuples | Published sample |
| hard-003 | Count projective curve points over a huge finite field | Published sample; scaling notation clarified |
| hard-004 | Expected consecutive-value inversions after a recursive permutation map | Published sample |
| hard-005 | Expected perimeter of a central random polygonal cell | Published sample; independence explicit |
| hard-006 | Leading constant of a five-variable Diophantine asymptotic | Published sample; positive ordered entries explicit |
| hard-007 | Extremal sequence in a Tsirelson-space construction | Published sample; specialized stretch problem |
| hard-008 | Matrix orbits with determinant-one constraints | New variant of hard-002 |
| hard-009 | Curve points over an extension of degree 30 | New variant of hard-003 |
| hard-010 | All inversions after the recursive map | New variant of hard-004 |
| hard-011 | Degree-31 polynomial with a different evaluation point | New variant of hard-001 |
| hard-012 | A different random-polygon perimeter distribution | New variant of hard-005 |

For a climb within a family, solve the original and then its variant. The determinant
and all-inversions variants change the mathematical reduction; the other variants
check transfer to new parameters. Public samples may be familiar to a model, which
is one reason to include variants. Related pairs do not constitute independent
held-out test items.

These questions require recognizing mathematical structure before computation.
A successful Python execution alone is a weak check. Ask the agent to establish
the reduction, test small cases, check normalization and justify any asymptotic or
numerical approximation before returning an answer.

The organizer's [reference and hint guide](../../docs/HARD_REFERENCES.md) contains
spoilers and progressive hints. Hint-assisted runs are useful teaching experiments,
but must be reported separately from the unassisted baseline.

## Verification and reproduction

All seven published answers are reproduced by the reference computations. New variant
answers have explicit derivations and numerical checks. Tests use separate polynomial
formulas, character calculations, small finite fields, exhaustive permutations, complete
small-polygon enumeration and high-precision arithmetic. Deep classification and
asymptotic results rely on the published mathematical reductions, as explained in
the reference guide; the tests do not replace those proofs.

All twelve reference computations also completed in the actual five-second Python
sandbox; the slowest observed call, including startup and cleanup, was 0.20 seconds.
The target-field enumeration and naive matrix searches are intentionally impractical;
the known mathematical reductions make the final computations small.

```bash
uv run python scripts/prepare_hard.py                  # Offline data regeneration
uv run pytest tests/test_hard.py                       # Offline mathematical/export checks
uv run python scripts/check_hard_runtime.py            # Docker; no model API calls
```

Use `--out /path/to/data-copy` with `prepare_hard.py` to reproduce the dataset elsewhere.
Question files contain only IDs and statements. The runner never imports the reference
solver or reads the answer key. Source statements, licensing, hashes and adaptation
notes are recorded in the source fixture and manifest.

The pilot is a small teaching calibration, not a guarantee that a particular question
will fail on every rerun. Keep model, agent, selected IDs and limits fixed when comparing
improvements, and separate wrong mathematics from budgets, truncations and timeouts.
