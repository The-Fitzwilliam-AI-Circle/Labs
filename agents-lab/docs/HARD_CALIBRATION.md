# Hard-set live calibration

Recorded on 5 October 2026 with **`gpt-5-nano-2025-08-07`**, the unchanged `agents/tool.py`,
and the existing Docker image. The provider reported **medium** reasoning effort in
both profiles. Model selection, tool implementation and agent instructions were not
weakened to make the questions fail.

| Set / profile | Correct | Wrong submitted answers | Token truncations | Tool/request budget failures | Wall-clock timeouts | Provider errors |
| --- | --- | --- | --- | --- | --- | --- |
| Earlier 40-question development ladder | 36/40 (90.0%) | 3 | 0 | 0 | 1 | 0 |
| Hard set, original limits | **0/12 (0.0%)** | 4 | 6 | 2 | 0 | 0 |
| Hard set, generous limits | **2/12 (16.7%)** | 8 | 1 | 1 | 0 | 0 |

The generous profile's two successes were `hard-002` and `hard-010`. Among its ten
episodes that returned an answer, two were correct. Thus the low score is not
explained solely by the resource failures.

## Limits and execution

| Limit | Original profile | Generous profile |
| --- | --- | --- |
| Configuration | `config.toml` | `config.hard.toml` |
| Model requests per episode | 6 | 8 |
| Output tokens per request, including reasoning | 8,192 | 16,384 |
| Python calls per episode | 4 | 8 |
| Episode deadline | 90 seconds | 240 seconds |
| Python execution deadline | 5 seconds | 5 seconds |
| Python memory | 256 MiB | 256 MiB |

The original run's two budget failures exhausted Python calls. The generous run's
one budget failure exhausted model requests. Token truncations were verified in
the provider's `incomplete_details.reason = max_output_tokens`; the harness records
these as `protocol_error`. They are not counted as wrong mathematical submissions.
All reported usage was known; no provider-error episode was included as evidence
of mathematical difficulty.

Every reference computation completed in the real five-second sandbox. The slowest
observed reference call, including startup and cleanup, was 0.20 seconds. Efficient
known solutions fit the tool limits; discovering the reduction is the main challenge.

## Per-question results

| ID | Original limits | Generous limits |
| --- | --- | --- |
| hard-001 | Python-call budget | Wrong answer |
| hard-002 | Token truncation | Correct |
| hard-003 | Token truncation | Wrong answer |
| hard-004 | Python-call budget | Wrong answer |
| hard-005 | Token truncation | Wrong answer |
| hard-006 | Wrong answer | Wrong answer |
| hard-007 | Wrong answer | Wrong answer |
| hard-008 | Token truncation | Wrong answer |
| hard-009 | Token truncation | Wrong answer |
| hard-010 | Wrong answer | Correct |
| hard-011 | Wrong answer | Model-request budget |
| hard-012 | Token truncation | Token truncation |

The [reference guide](HARD_REFERENCES.md) gives the keys, mathematical reductions,
checks and hints. For example, the generous run returned `5^18+1` on `hard-003`,
omitting the nonzero Frobenius trace; it returned `500` on `hard-006`, instead of
the verified Euler-product floor `214`. These are numerical disagreements with
the verified references, not equivalent answer formats.

## Evidence and reproducibility

All twelve questions were frozen before the first pilot. Seven are public samples
from **Epoch AI's [FrontierMath sample page](https://epoch.ai/frontiermath/tiers-1-4/benchmark-problems)**
and five are new documented variants. All twelve candidates were retained; the score
was not calculated by selecting only failed questions after testing.

Each profile has one attempt per question. The generous profile ran in two disjoint
batches covering all twelve IDs exactly once. Every episode started with a fresh
model conversation. The original profile and parts of the generous profile overlapped
in wall-clock execution; the recorded times are not a controlled latency benchmark.

The full source records are:

- Original: `runs/hard-tool-pilot/` — all twelve questions.
- Generous A: `runs/hard-tool-generous/` — IDs 001, 002, 003, 007, 008, 009.
- Generous B: `runs/hard-tool-generous-b/` — IDs 004, 005, 006, 010, 011, 012.
- Earlier baseline: `runs/dev-tool-3/` — the original forty-question ladder.

The portable [calibration record](../data/hard/calibration.json) contains all answers,
outcome categories, limits, per-question resource use and run hashes, without
credentials or raw model reasoning. Raw runs are retained locally under `runs/`
and are excluded from version control.

Questions SHA-256: `405392b2a5f862bd82272b0bbc88e98967a2137edc201a89ce707d9edafcd0e7`.
Answers SHA-256: `0d65f8d993b234576d3d622658243f3c4b0d029ce2e1d647c4e9df6ccf4eb2ae`.
The question hash matched all three run manifests, confirming no statement changed
during the pilot.

Recreate the summary from the completed, scored local runs:

```bash
uv run python scripts/summarize_hard_pilot.py \
  --profile original runs/hard-tool-pilot \
  --profile generous runs/hard-tool-generous runs/hard-tool-generous-b \
  --out data/hard/calibration.json
```

The summarizer rejects incomplete coverage, duplicate IDs, mixed configurations,
changed results, or mismatched question/answer hashes. It makes no API calls.

## Interpretation for the lab

This is a small calibration of one model and one agent, not a general benchmark
ranking or a guarantee that every question fails on every rerun. Public problems
may be familiar to models, and related original/variant pairs are not independent.
No claim is made about performance on private FrontierMath problems.

The earlier forty questions remain useful warm-ups. Use the hard set as a capstone,
starting with three questions per session and the more generous configuration.
When introducing a mathematical hint, label the new run as hint-assisted. Compare
accuracy and error causes as well as resource use; reducing a token failure to a
completed wrong answer is progress in execution, but it is not a mathematical solve.
