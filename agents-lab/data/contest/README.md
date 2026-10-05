# Contest practice bank

**100 development questions**, intended to bridge the forty-question warm-up and
the twelve FrontierMath capstones. A separate organizer-held bank contains
**60 final-validation questions**. Practice questions and keys are public to the
participants; validation questions are released only after final agents are frozen.

## Choose a working set

| Rung | Questions | Question file |
| --- | ---: | --- |
| Contest | 36 | `data/contest/01-contest/questions.jsonl` |
| Advanced | 35 | `data/contest/02-advanced/questions.jsonl` |
| Stretch | 29 | `data/contest/03-stretch/questions.jsonl` |
| Entire development bank | 100 | `data/contest/questions.jsonl` |

Rungs use relative position within source contests/rounds as a rough ordering.
They are **not calibrated item difficulty ratings**: an individual “Contest” task
can be harder than a “Stretch” task. Questions span algebra, number theory,
combinatorics, probability, geometry and some calculus. The rational-answer filter
underrepresents geometry with radical answers.

For a short first comparison, use these four development problems:

```bash
uv run mathlab run --agent agents/tool.py --config config.hard.toml \
  --questions data/contest/questions.jsonl --out runs/contest-tool \
  --id contest-hmmt-feb-2025-13 --id contest-hmmt-feb-2025-15 \
  --id contest-smt-2025-25 --id contest-hmmt-feb-2025-09
```

Repeat with `agents/direct.py` and `--out runs/contest-direct`. Copy the score
command printed at completion; the key is `data/contest/answers.jsonl`.
Remove the `--id` arguments to run all 100. To run one rung, change `--questions`
and use that rung's answer file. Both baselines must use the same model, IDs and
configuration. A four-question run has a configured maximum of 16 minutes;
a full 100-question run has a maximum of 6 hours 40 minutes per agent, excluding
small setup/cleanup overheads. Iterate on subsets during the session.

Some examples of the actual tasks:

- Partition 16 numbered balls into four indistinguishable boxes of four, requiring
  each ball to have a numerically adjacent neighbour in its box (`hmmt-feb-2025-13`).
- Count coin placements in a 4×4 grid with no adjacent coins and no empty 2×2 block
  (`cmimc-2025-16`).
- Find the expected number of draws from a 54-card deck until an ace, king and queen
  have all appeared (`brumo-2025-27`).
- Extract a coefficient modulo a prime from a degree-2026 interpolation polynomial
  (`hmmt-feb-2025-09`).

Prefix these abbreviated IDs with `contest-` when using `--id`.

## Sources and reproducibility

These are selected **existing competition problems**, not generated parameter
variants. The development sources are MathArena's verified transcriptions of
[HMMT February 2024](https://huggingface.co/datasets/MathArena/hmmt_feb_2024),
[HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025),
[CMIMC 2025](https://huggingface.co/datasets/MathArena/cmimc_2025),
[SMT 2025](https://huggingface.co/datasets/MathArena/smt_2025) and
[BRUMO 2025](https://huggingface.co/datasets/MathArena/brumo_2025).

The source snapshots, selection and pilot IDs were frozen before the live lab
pilot. No model-specific item results were used to select either split.
`manifest.json` records each source revision, downloaded file hash, source index,
topic, ordering proxy and any transcription change. Unsupported symbolic/list
answers, a tolerance-based decimal answer, incomplete diagram cases and an
ambiguous statement were excluded. Source LaTeX fractions are normalized with
exact rational arithmetic. The mathematics is unchanged apart from documented
typo/markup repairs and removal of one redundant drawing.

Keys use published MathArena references, with one documented correction from
930 to 932 found by exact arithmetic; see [CORRECTIONS.md](CORRECTIONS.md).
Ten development keys also have separate
exact computational checks in `tests/test_contest.py`; this is not an independent
proof of every reference answer. If a discrepancy appears, inspect the statement,
source and saved trace before changing a key. Version any substantive correction.

Regenerate the frozen development files offline:

```bash
uv run python scripts/prepare_contest.py
```

See [LICENSE.md](LICENSE.md) for attribution and **CC BY-NC-SA 4.0** terms. The
participant archive includes this notice and development provenance.

## Calibration and final evaluation

A first pilot with `gpt-5-nano-2025-08-07` and `config.hard.toml` gave:

| Baseline | Correct | Wrong answers | Token truncations |
| --- | ---: | ---: | ---: |
| Direct | 9/12 (75%) | 1 | 2 |
| Tool | 8/12 (66.7%) | 3 | 1 |

The same twelve practice questions were used once per agent. The tool's mistakes
included generating one extra decimal digit and modelling the wrong card-drawing
stopping condition. Tool access alone does not ensure a better answer.

`pilot.json` fixes those twelve questions, four per rung. `calibration.json`
records exact input hashes, resource limits and all failure categories. This small diagnostic does not establish the accuracy of all 100
practice questions, and the rung labels should not be interpreted as measured
success probabilities. See [the validation log](../../docs/VALIDATION.md).

The **60-question final set** uses different competition editions. It is kept in
the organizer bundle outside the participant project, together with its key,
source snapshots, exclusions and frozen hashes. It has not been used for model
calibration. See [the organizer workflow](../../docs/ORGANIZER.md).

“Held out” means held out from lab participants' development and feedback. These
are public contest questions, so model pretraining exposure is unknown. Methods
can recur across contests; exact and numeral-normalized duplicates were checked,
and a closely related Hamiltonian-grid family was excluded from validation.

Use the existing `data/dev` for warm-ups and `data/hard` for optional capstones.
Their questions, IDs and answer keys are unchanged, so previous runs still score.
