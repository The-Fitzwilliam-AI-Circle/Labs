# Validation and pilot status

Updated on 5 October 2026. The main bank now contains **100 contest development
questions** and **60 organizer-held final-validation questions**. The older
forty-question ladder remains a warm-up; twelve FrontierMath questions remain
optional capstones. Their IDs and keys have not changed.

## Provider support and event setup

- `uv run pytest -q`: **202 passed**, four Docker integration tests deselected.
- Ruff lint and formatting checks pass.
- OpenAI Responses remains the default. OpenRouter and local/self-hosted
  OpenAI-compatible Chat Completions providers are selectable through `.env`.
- Twenty new tests cover provider selection, credential separation, SDK wire
  requests, native reasoning/tool history replay, exact action validation,
  token accounting, missing usage, truncations, no retries and error redaction.
  They also check that the live-tool diagnostic rejects incorrect Python output.
- `uv run mathlab doctor --live` passed against the configured
  `gpt-5-nano-2025-08-07` on OpenAI.
- `uv run mathlab doctor --live-tools --config config.hard.toml` passed against
  the same model, using two real API requests and actual Docker execution. This
  exercises a Python function call, its observation, and the final submission.
- OpenRouter and local adapters passed offline SDK transport tests. Particular
  remote models and local servers still require a live rehearsal and capacity check.
- Run manifests now include public provider/endpoint metadata. All three provider
  credentials are redacted from traces/diagnostics. Hosted keys never fall back
  into another provider's configuration. Existing runs remain scoreable.
- Participant ZIPs are generated locally and excluded from Git.
  [PROVIDERS.md](PROVIDERS.md) covers configuration and checks;
  [EVENT_RUNBOOK.md](EVENT_RUNBOOK.md) covers advance setup, a two-hour schedule,
  separate model tracks and final evaluation after frozen submissions.

## Contest-bank verification

- `uv run pytest -q`: **182 passed**, four Docker integration tests deselected.
- Ruff lint and formatting checks pass. Existing harness behavior is unchanged;
  both live pilots exercised the existing model and Docker integrations.

- The development set has 36 contest, 35 advanced and 29 stretch questions. Rungs
  are rough source-position orderings, not measured item difficulty ratings.
- Development uses five competition editions; validation uses three disjoint
  editions. Both selections were fixed before lab model runs, without selecting
  failures from published model outputs. Exact and numeral-normalized duplicates
  are absent across the final set and all public lab questions. One closely
  related Hamiltonian-grid family was explicitly excluded from final validation.
- Source revisions, downloaded file hashes, licenses, source indices, statement
  edits and excluded candidates are preserved. All keys meet the exact rational
  scorer contract. Missing diagrams, an ambiguous statement, proof/list outputs
  and unsupported symbolic/tolerance answers were excluded.
- Ten development keys have independent exact checks. They found a published key
  error: CMIMC 2025 problem 7 is 932 rather than 930. Two separate computations
  verify this; see [the correction record](../data/contest/CORRECTIONS.md). The
  original source answer and original frozen key are preserved. No pilot question
  or validation file changed, and the correction was independent of model outputs.
- Offline regeneration reproduces both split exports. The organizer's private
  audit checks frozen hashes and the mapping between opaque worker IDs and keys.
  Answer verification is source-backed plus these spot checks, not an independent
  mathematical proof of every problem in the bank.
- The participant ZIP uses an explicit allowlist. Tests plant secrets in `.env`,
  private/heldout folders, unlisted data and runs and verify they are not exported.
  Export rejects symlinks and never overwrites a previous archive. The actual
  package is also audited against every held-out question and source ID.
- Final validation has **not been run against a model**. It must remain unused
  for tuning until final agents and evaluation policy are frozen. The clean-worker
  workflow is documented in [ORGANIZER.md](ORGANIZER.md).

The development pilot uses the same twelve cases for direct and tool baselines,
four per rung, fixed in `data/contest/pilot.json` before execution. The model is
`gpt-5-nano-2025-08-07`, with `config.hard.toml` limits. Every selected case is kept
in the score. Results and hashes are recorded in `data/contest/calibration.json`.

The direct baseline scored **9/12 (75%)**: one wrong answer and two output-token
truncations. The tool baseline scored **8/12 (66.7%)**: three wrong answers and one output-token
truncation. The wrong answers were substantive: a graph model admitted invalid
friend/enemy configurations, a repetend calculation generated 727 digits instead
of 726, and a card-drawing program waited for all twelve special cards instead of
one card from each required rank. There were no tool execution failures or provider
errors in this pilot. This is more useful headroom than the warm-up's 36/40, with
more successes than the FrontierMath capstone's 2/12 under the same generous limits.
It does not establish full-bank accuracy or final-validation difficulty.

## Earlier hard-set verification

- `uv run pytest -q`: **166 passed**, four Docker integration tests deselected.
- Ruff lint and format checks pass.
- Seven public Epoch AI sample answers are reproduced; five documented variants
  have explicit mathematical reductions and recomputed keys.
- Separate checks cover polynomial coefficients and identities, representation
  characters and generating functions, explicit small finite fields, exhaustive
  permutations, complete small-polygon dissections, and higher-precision geometry.
- Harmonic-number and Euler-product computations use error bounds to verify the
  requested integer floors. The reference guide distinguishes numerical verification
  from the deep mathematical results supplied by the published proofs.
- The hard-set generator reproduces the question, answer and manifest files offline.
- `uv run python scripts/check_hard_runtime.py`: **12/12 references passed** in
  the real five-second Docker sandbox. The slowest observed call, including startup
  and cleanup, was **0.20 seconds**.
- The original-budget live pilot scored **0/12**, comprising four wrong submissions,
  six output-token truncations and two exhausted Python-call budgets. A second
  profile with increased reasoning and tool allowances scored **2/12 (16.7%)**:
  eight wrong answers, one output-token truncation and one model-request budget
  failure. Neither profile had wall-clock timeouts or provider errors. Complete
  results and source run hashes are in [HARD_CALIBRATION.md](HARD_CALIBRATION.md).

References and numerical verification details: [HARD_REFERENCES.md](HARD_REFERENCES.md).
The source is Epoch AI's public sample page, credited under CC BY 4.0; this lab is
not an official FrontierMath evaluation. All twelve candidates were frozen before
the first live run and retained. Calibration uses the existing API credentials;
offline tests and reference checks do not contact a model.

## Earlier forty-question ladder verification

- `uv run pytest -q`: **144 passed**, four Docker integration tests deselected.
- `uv run pytest -m integration -q`: **4 passed**, covering actual container
  execution, isolation, timeout/output-limit handling and cancellation cleanup.
- Ruff lint and format checks pass.
- All thirty new reference answers agree with a separate full-size calculation.
  Cross-checks include matrix powers versus polynomial reduction, assignment DP
  versus Ryser's formula, bounded coin DP versus meet in the middle, circular gap
  counting versus cyclic word counting, and separate probability/path/tiling solvers.
- Small exhaustive cases check digit restrictions, path turns, restricted permutations,
  coin constraints, cyclic pattern counts and weighted spanning trees. Known-answer
  cases check tilings, overlapping stopping patterns and conditional waiting times.
- All answer strings pass the existing exact-answer parser. IDs and keys match,
  each rung has ten questions, and the four rungs concatenate to the full set.
- The original foundation question and answer files retain their previous SHA-256
  hashes. Existing task IDs and reference answers have not changed.
- `scripts/prepare_data.py --offline --out <temporary directory>` reproduces every
  committed JSON/JSONL dataset file byte for byte, including provenance manifests.
- `uv run python scripts/check_ladder_runtime.py`: **30/30** exact references
  completed in the real configured Docker sandbox with the five-second timeout,
  256 MiB memory limit and no network. The slowest observed call, including cleanup,
  was **0.47 seconds**. This establishes feasibility for the supplied algorithms;
  it does not establish model accuracy or guarantee a model will choose those algorithms.

Image checked: `sha256:d9d7cf6ad46d32989eb61359ee1393f074ef332d5542039fe6fda84f68389489`.
The reference runtime checker and all answer checks use no model API calls.

The saved `runs/dev-direct-2` and `runs/dev-tool-2` records show **10/10** for both
baselines on the original questions with `gpt-5-nano-2025-08-07`. They are evidence
of the foundation's ceiling, not scores on the expanded set. The configured output
allowance is 8,192 tokens per request in `config.toml`. The later 36/40 tool run showed
that this expanded ladder was also too easy for the intended challenge.
The [data guide](../data/README.md) contains runnable commands and the family map.
