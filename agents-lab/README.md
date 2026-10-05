# Maths Agent Lab

A small Python lab for the Fitzwilliam AI Circle. Edit one agent function,
run maths questions, inspect model and Python interactions, and score final answers
with exact arithmetic. Model weights stay fixed.

## Setup and first run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and start Docker.
Run commands from the `agents-lab/` directory in the Circle's Labs repository
(or the root of an extracted participant package). `uv` installs Python 3.12 when needed.

```bash
uv sync --locked
cp .env.example .env
# Set PROVIDER, MODEL and the provider-specific key/endpoint in .env.
docker build -f Dockerfile.python -t mathlab-python:v0 .

uv run mathlab doctor
uv run mathlab doctor --live-tools

uv run mathlab run --agent agents/participant.py \
  --questions data/smoke/questions.jsonl --out runs/first-attempt
uv run mathlab score --run runs/first-attempt --answers data/smoke/answers.jsonl
```

`doctor` checks configuration and the local image without contacting the model.
`doctor --live-tools` uses two API requests and Docker to verify the complete
Python tool loop. `doctor --live` is a smaller one-request submission check.
An API key is separate from a ChatGPT subscription. The organizer must choose a
model ID; there is no automatic model selection. The development sets use
`gpt-5-nano-2025-08-07` in calibration. See the [contest guide](data/contest/README.md)
and [hard-set calibration](docs/HARD_CALIBRATION.md) for results and limits.

Repeating a run command automatically creates a new directory. For example,
`--out runs/dev-tool` uses `runs/dev-tool` first, then `runs/dev-tool-2`,
`runs/dev-tool-3`, and so on, skipping names that already exist. Previous runs are
preserved. The CLI prints the actual directory when the run starts and a score
command for that specific run when it finishes.

Runs process **four questions concurrently** by default. Set `--concurrency N`
to change the number of active episodes, or `--concurrency 1` for serial execution:

```bash
uv run mathlab run --agent agents/tool.py --config config.hard.toml \
  --questions data/contest/questions.jsonl --out runs/contest-tool --concurrency 4
```

A tqdm progress bar shows completed/total episodes, elapsed time, estimated time
remaining and execution failures. Incorrect maths answers are identified later by
`score`, so the failure count is not an accuracy score. Progress appears on stderr
in a terminal; use `--progress` to show it in redirected logs or `--no-progress`
to hide it. Failed episodes also advance the bar.

Each episode keeps its own request/tool budget and starts its deadline when a worker
picks it up. Results are flushed as episodes finish, so row order may differ from
question order; scoring matches task IDs. Trace events from different questions can
interleave: filter by `task_id` and follow its `event_order`. Ctrl-C cancels active
work and cleans up before leaving the run incomplete.

Use the same concurrency for comparisons; it is recorded in `run.json`. Higher
concurrency increases simultaneous API requests and Docker containers, so reduce it
if the provider rate-limits requests or the laptop/server is overloaded. Five pairs
using the default can have up to 20 episodes active across the room. Keep mutable
episode state inside `solve`, rather than in shared module globals. Agent code must
yield through async calls; CPU-heavy host code blocks the shared event loop.

Choose `PROVIDER=openai`, `openrouter`, or `local`. Existing configurations default
to OpenAI. The [provider guide](docs/PROVIDERS.md) has copyable settings for hosted
APIs and OpenAI-compatible local servers. No agent changes are needed when switching.
For organizers, the [event runbook](docs/EVENT_RUNBOOK.md) covers advance setup,
a two-hour session, model tracks and final evaluation.

Generation reads only questions; scoring reads the answer key afterward.
Scoring needs neither credentials nor Docker.
The five smoke questions check plumbing; they do not measure difficult reasoning.

## Edit your agent

Start in [agents/participant.py](agents/participant.py), an editable copy of
[agents/tool.py](agents/tool.py). Its contract is:

```python
async def solve(task: Task, ctx: AgentContext) -> str:
    # task has only id and problem; return a string such as "7/2".
    ...
```

Edit `INSTRUCTIONS` or the loop. Make every model request through `ctx.model(...)`
and every Python execution through `ctx.python(...)`. The context owns the request
allowance, Python allowance, usage totals, deadline, and trace. Starting a new
history for another candidate or verifier uses the **same context and budget**.

The model can request `python` or `submit_answer`. A `submit_answer` action is a
candidate returned to your code; only `solve` returning submits the episode. If you
continue that conversation, first append a matching `function_call_output` for
the candidate's call ID. Fresh verifier conversations do not need that output.
Preserve `turn.output_items` in order, including returned reasoning items.

Python has the standard library, SymPy 1.14.0, and mpmath 1.3.0. It runs in a fresh
container on every call: imports, variables, and files never persist. Print the
values you need. Tool errors are observations that the agent can repair.

Try changing one thing at a time: require an arithmetic check, improve the prompt,
or compare a candidate with a fresh verifier. The lab walkthrough in
[docs/LAB.md](docs/LAB.md) gives a short sequence of exercises.

## Compare the baselines

Use the **100-question contest bank** for the main lab. It bridges the earlier
forty-question warm-up and the twelve FrontierMath capstones. The bank has three
practice rungs (36 contest, 35 advanced, 29 stretch), plus **60 separately held-out
final-validation questions** for the organizer. See the [contest guide](data/contest/README.md)
for examples, sources and the practice pilot.

Start with four questions and use the same configuration for both agents:

```bash
uv run mathlab run --agent agents/direct.py --config config.hard.toml \
  --questions data/contest/questions.jsonl --out runs/contest-direct \
  --id contest-hmmt-feb-2025-13 --id contest-hmmt-feb-2025-15 \
  --id contest-smt-2025-25 --id contest-hmmt-feb-2025-09
uv run mathlab run --agent agents/tool.py --config config.hard.toml \
  --questions data/contest/questions.jsonl --out runs/contest-tool \
  --id contest-hmmt-feb-2025-13 --id contest-hmmt-feb-2025-15 \
  --id contest-smt-2025-25 --id contest-hmmt-feb-2025-09
uv run mathlab score --run runs/contest-direct --answers data/contest/answers.jsonl
uv run mathlab score --run runs/contest-tool --answers data/contest/answers.jsonl
```

Remove `--id` arguments for all 100 practice questions, or select a rung from the
guide. The optional `config.hard.toml` allows up to four minutes per question;
full-bank runs belong outside the interactive session. The original `config.toml`
is unchanged. For an extra challenge, use the [twelve FrontierMath capstones](data/hard/README.md).

Use the same `MODEL` and configuration for both. The direct agent makes one request
with no Python access. The tool agent can use the configured episode budget.
On repeat runs, copy the score command printed by `mathlab run` to grade the new
directory. An explicit `mathlab score --run runs/dev-tool` always grades that
exact directory. The suggested command assumes the matching key is named
`answers.jsonl` beside the question file; adjust `--answers` if yours is elsewhere.
`--id smoke-001 --id smoke-003` selects a subset; scoring still accepts the full key.
Treat pilot scores as measurements of this model, agent and budget. Record mathematical
errors separately from token truncations, tool-budget exhaustion and timeouts.

## Read a run

| File | What to inspect |
| --- | --- |
| `run.json` | Model, resolved limits, concurrency, scheduled IDs, hashes, source state, image ID, completion |
| `agent.py` | Snapshot of the evaluated agent, including its editable prompt |
| `results.jsonl` | One terminal result per task, answer, status, usage, duration, bounded error |
| `traces.jsonl` | Ordered requests, returned provider items, Python code, observations, and usage |
| `scores.json` | Accuracy, per-task grading reasons, failures, aggregate usage, input hashes |

For a readable trace without extra dependencies:

```bash
uv run python -m json.tool --json-lines runs/first-attempt/traces.jsonl
```

Find a task ID and follow `event_order`: `model_request`, `model_response`,
`python_request`, `python_result`, and `episode_end`. Tool observations include
stdout, stderr, exit code, timeout, and output-limit flags. Only provider-returned
reasoning data is preserved; no hidden reasoning is inferred or reconstructed.

An execution can be `completed`, `budget_exceeded`, `timeout`, `provider_error`,
`protocol_error`, or `agent_error`. A completed answer can still be wrong. Every
scheduled task counts in the score. Missing, extra, or duplicate results are errors;
interrupted runs remain incomplete and cannot be scored. Individual task failures
produce a result row and execution continues with the next question.

## Answer format and limits

Answers are at most 128 characters after trimming whitespace: optional leading
sign, then an integer, a fraction with unsigned numerator and denominator, or a
decimal with digits on both sides of the point. `3/4`, `6/8`, and `0.75` are equal.
Prose, LaTeX, units, expressions, scientific notation, sets, and zero denominators
are invalid. An invalid participant answer scores zero; an invalid reference stops
scoring. Scores are reproducible for saved outputs; fresh model outputs may vary.

Defaults in `config.toml`: six requests, 8,192 output tokens per request, four Python
calls, 90 seconds per episode, five seconds per Python invocation including startup,
and 8 KiB of combined tool output. All actual request attempts count, including
failures; SDK retries are disabled. No extra request forces an answer after exhaustion.
Unknown usage is flagged. Reasoning tokens are a subset of output tokens and are
not added twice. These are resource limits, not a hard monetary spending cap.

Each Python container has no network or host mounts, runs as a non-root user with a
read-only root, drops capabilities, and has one CPU, 256 MiB RAM, no additional swap,
64 processes, and 16 MiB temporary storage. Output is bounded while reading.
Explicit container removal runs after success, error, timeout, and cancellation;
cleanup has a short grace period outside the execution deadline. The image tag is
resolved to an immutable local image ID before a run. The Docker base tag can change
on rebuild; the recorded image ID identifies the version actually evaluated.

Participant agent code runs on the host and is trusted. The container isolates model
Python, not arbitrary participant submissions. Development answers are public and
may be inspected. Final-validation material is held in the separate organizer
bundle. Use a clean generation worker with questions only, then score on the
organizer machine; a different readable directory is not a security boundary.
The [organizer guide](docs/ORGANIZER.md) explains packaging, frozen submissions
and final evaluation.

## Development and troubleshooting

```bash
uv run pytest                         # Offline; scripted model and tool responses
uv run ruff check .
uv run ruff format --check .
uv run pytest -m integration          # Explicit Docker checks; build the image first
```

The integration tests require Docker and fail clearly when it is unavailable.
Routine tests never contact the API. Use `doctor --live-tools` and the smoke runs above
for explicit live checks. See [docs/VALIDATION.md](docs/VALIDATION.md) for the current
verification status and remaining pilot work.

| Symptom | Action |
| --- | --- |
| Missing `MODEL` or key | Fill `.env` for the selected `PROVIDER`; existing environment variables take precedence |
| Docker image unavailable | Start Docker and run the documented build command |
| Repeating a run | Reuse the same `--out`; the CLI creates a numbered directory and prints its score command |
| `protocol_error` | Inspect `raw_response`; incomplete responses may need a larger token limit |
| `provider_error` | Check model access, key, quota, and API availability; retries consume budget |
| HTTP 429 with `credit_balance_exhausted` | Add API credits in the billing settings for the organization associated with your key |
| HTTP 429 with `insufficient_quota` or a spend/usage-limit code | Check API billing and the relevant organization/project limits before retrying |
| HTTP 429 with `rate_limit_exceeded` or `slow_down` | Wait before retrying and reduce request frequency; check the model's API limits |
| `budget_exceeded` / `timeout` | Inspect the trace; simplify the loop or adjust pilot limits |
| `invalid_answer` | Return only an integer, fraction, or decimal string |
| Incomplete run | Repeat the run command to create a new directory; do not edit completion metadata to force scoring |

## Data and design

The main [contest bank](data/contest/README.md) has **100 practice questions** from
HMMT, CMIMC, SMT and BRUMO, with pinned MathArena sources and exact scalar keys.
The organizer holds **60 final-validation questions from separate editions** outside
this project. No validation model run was used to select or calibrate the bank.
Contest data carries **CC BY-NC-SA 4.0**; see its license notice. These are public
questions, so held out from the lab does not mean unseen during model training.

Create a participant package that excludes credentials, runs and final-validation
material:

```bash
uv run python scripts/export_participants.py --out dist/mathlab-participants-v2.zip
```

The [organizer guide](docs/ORGANIZER.md) covers final submissions and grading.
Five smoke questions, the earlier forty-question [warm-up ladder](data/README.md),
and twelve [FrontierMath capstones](data/hard/README.md) remain available. Their
IDs and answer files are unchanged, preserving saved runs. Each set has separate
question/answer files and provenance. Exported questions contain no answer fields.

The original brief is preserved in [docs/STARTER_SPEC.md](docs/STARTER_SPEC.md).
Implementation references: [uv projects](https://docs.astral.sh/uv/guides/projects/),
[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling),
[OpenAI Python SDK](https://developers.openai.com/api/reference/python),
[Docker execution](https://docs.docker.com/engine/containers/run/), and
[Python Fraction](https://docs.python.org/3.12/library/fractions.html).
