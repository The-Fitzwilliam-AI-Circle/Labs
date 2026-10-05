> **Main lab:** use the [100-question contest bank](contest/README.md), with three
> practice rungs and a separate 60-question final-validation set held by the organizer.
> The four rungs below remain available as warm-ups; [FrontierMath](hard/README.md)
> is an optional capstone.

# The question ladder

The tool baseline scored 36/40 on the ladder below, so these forty questions now
serve as warm-up and agent-debugging exercises. Use the contest bank linked above
for the main session.

There are **40 development questions in four rungs of ten**, plus five smoke checks.
Start at rung 2 if you have already solved the original set. The original statements,
answers and IDs `dev-001` through `dev-010` are unchanged.

| Rung | IDs | What it exercises | Question file |
| --- | --- | --- | --- |
| 1 — Foundation | dev-001–010 | Short mathematical reasoning; deciding when a tool is useful | `ladder/01-foundation/questions.jsonl` |
| 2 — Computation | dev-011–020 | Exact arithmetic, finite counting, probability and reliable code | `ladder/02-computation/questions.jsonl` |
| 3 — Structure | dev-021–030 | Efficient algorithms, state representation, large recurrences and overlapping patterns | `ladder/03-structure/questions.jsonl` |
| 4 — Challenge | dev-031–040 | Combine constraints, cyclic symmetry, weighted graphs and conditional stopping times | `ladder/04-challenge/questions.jsonl` |

Paths in this table are relative to `data/`. Every rung has a matching `answers.jsonl`
and provenance manifest. `data/dev/questions.jsonl` is the ordered concatenation of
all four rungs, so the existing development commands now run all forty questions.
The smoke set remains unchanged.

These are intended difficulty levels, not measured model scores or a guarantee that
every question is harder than every question in the previous rung. The original ten
reached 10/10 with both lab baselines on `gpt-5-nano-2025-08-07`. A subsequent tool-agent
run scored 36/40 on the full ladder (three wrong submissions and one timeout).
Keep the model and budgets fixed while
climbing, and measure both accuracy and resource use.

## Run one rung

These commands still work for the original warm-up ladder. For a substantially harder
challenge, use the commands in [data/hard/README.md](hard/README.md) instead.

From the repository root:

```bash
uv run mathlab run --agent agents/direct.py \
  --questions data/ladder/02-computation/questions.jsonl --out runs/rung2-direct
uv run mathlab run --agent agents/tool.py \
  --questions data/ladder/02-computation/questions.jsonl --out runs/rung2-tool

uv run mathlab score --run runs/rung2-direct \
  --answers data/ladder/02-computation/answers.jsonl
uv run mathlab score --run runs/rung2-tool \
  --answers data/ladder/02-computation/answers.jsonl
```

For subsequent runs, use the score command printed by the runner: the new output
directory receives a numerical suffix. To climb, change the question path to
`03-structure` or `04-challenge` and give the run a corresponding name.

For a shorter first comparison, append these same filters to both run commands:

```bash
--id dev-011 --id dev-014 --id dev-015 --id dev-017 --id dev-020
```

This samples five different families. The full rung answer key still works for
scoring a subset. Record the actual denominator and selected IDs when comparing.
Forty questions with multiple agent versions can take much longer than a lab session;
use subsets while iterating and full rungs for the final comparison.

## Follow a family up the ladder

The ten families recur in the same order, allowing a successful approach on one
rung to be extended on the next. Later variants add constraints or change what
must be computed; they are not just arithmetic with longer numbers.

| Family | Computation | Structure | Challenge |
| --- | --- | --- | --- |
| Recurrences | dev-011: exact term | dev-021: enormous index modulo a prime | dev-031: enormous prefix sum |
| Digit constraints | dev-012: divisibility and digit sum | dev-022: unequal adjacent digits | dev-032: also forbid a substring |
| Lattice paths | dev-013: blocked vertices | dev-023: exact number of turns | dev-033: also stay below a boundary |
| Urn probability | dev-014: conditioned colour counts | dev-024: condition on a residue | dev-034: larger population and stronger inequalities |
| Permutations | dev-015: two forbidden offsets | dev-025: three forbidden offsets | dev-035: four forbidden offsets on 18 labels |
| Coin selections | dev-016: bounded supplies | dev-026: exact total number of coins | dev-036: also constrain odd multiplicities |
| Domino tilings | dev-017: board with holes | dev-027: wider board with holes | dev-037: also fix horizontal domino count |
| Circular arrangements | dev-018: rotations and no adjacent ones | dev-028: larger periodicity calculation | dev-038: also count a cyclic pattern exactly |
| Spanning trees | dev-019: cyclic graph | dev-029: denser graph | dev-039: distinct parallel edges |
| Waiting times | dev-020: one overlapping pattern | dev-030: race between two patterns | dev-040: biased symbols and condition on the winner |

If you get stuck, inspect whether the agent correctly represented the constraints
before changing its prompt. Useful experiments include exact fractions, checking a
small enumerated example, replacing a large brute-force search with a smaller state,
or asking a verifier to inspect boundary and conditioning cases. Fix budget or
protocol failures before treating them as mathematical errors.

## Provenance, answer checks and reproduction

Rung 1 retains the pinned MATH-500 selection and original source metadata. Rungs
2–4 contain original, deterministic instances authored for this repository. All
answers are exact integers or fractions and fit the existing 128-character format.
Questions contain only `id` and `problem`; reference answers remain in separate files.

Organizer material includes:

- `scripts/ladder.py`: problem parameters and exact reference solvers using the standard library.
- `tests/test_ladder.py`: a separate full-size calculation for every new answer, plus
  small exhaustive and known-answer checks for boundary conditions.
- Per-rung manifests: provenance, parameters, intended techniques and file hashes.

Examples of cross-checks are matrix powers versus polynomial reduction, assignment
DP versus Ryser's formula, bounded coin DP versus meet in the middle, and gap counting
versus cyclic binary-word counting. The probability and tiling solvers also have
separate implementations. Reference solutions and manifests contain spoilers.

Regenerate the committed data without network access or model calls:

```bash
uv run python scripts/prepare_data.py --offline
uv run pytest tests/test_ladder.py tests/test_data_cli.py
```

Use `--out /path/to/data-copy` to regenerate somewhere else. To reproduce the
foundation from upstream instead, pass `--source /path/to/test.jsonl`; the script
requires the pinned source hash. With neither `--offline` nor `--source`, it downloads
that pinned file. Nothing is downloaded by normal agent runs or scoring.

To check the thirty reference implementations against the configured Python sandbox:

```bash
uv run python scripts/check_ladder_runtime.py
```

This requires Docker and the built lab image, and makes no API calls. It checks that
an efficient known solution fits the Python budget; a model may still choose a slow
algorithm, make a modelling error, or exhaust its episode budget.

These are public development exercises. Related families are useful for teaching,
but their scores are not independent evidence of broad generalization. Do not copy
keys, reference code or question-ID lookups into an agent. The contest bank now has
a separate 60-question final set; the [organizer workflow](../docs/ORGANIZER.md)
keeps its key absent from the generation worker.
