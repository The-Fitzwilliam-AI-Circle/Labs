# Maths Agent Lab Starter Repository

Version 0.1 — 5 October 2026

Suggested repository location: `docs/STARTER_SPEC.md`

## 1 Purpose and first milestone

Build a small Python repository for the Fitzwilliam AI reading-group lab. Participants should be able to edit one agent function, run a few maths questions, inspect its tool interactions, and score its final answers.

The first usable version should contain a direct baseline, a Python-enabled baseline, and a participant copy of that baseline. Use one configured model, local files, and a command-line interface. This is inference-time agent development: model weights stay fixed.

Start with five hand-authored smoke questions and ten curated development questions from MATH-500. Expand to the previously proposed 30 development and roughly 40 held-out questions after the runner works and the model has been piloted. MATH-500 supplies separate problem, solution, answer, subject, level, and source ID fields; export only the fields needed for each local file. [1]

**First milestone:** from a clean checkout, a developer can configure model access, run the supplied tool agent on five questions, view the resulting trace, and obtain a reproducible score without editing infrastructure.

## 2 Implementation choices

| Component | Choice for version 0.1 |
| --- | --- |
| Language and environment | Python 3.12 and `uv`; commit `pyproject.toml`, `.python-version`, and `uv.lock` |
| Interface | An installable `mathlab` command using standard-library `argparse` |
| Model access | Official OpenAI Python SDK using the Responses API; one configured model ID |
| Runtime dependencies | `openai` and `python-dotenv`; use the standard library for the rest of the host application |
| Python tool | A small Docker image containing Python 3.12 and a pinned SymPy version |
| Data and records | UTF-8 JSONL and JSON files |
| Grading | Exact integer and rational comparison with `fractions.Fraction` |
| Development checks | `pytest` and `ruff`, with mocked model calls for routine tests |

The model provider is an implementation default, not a requirement of the workshop. Keep its SDK-specific code in `model.py` so it can be replaced later. Require the organizer to set the model ID explicitly after a pilot; do not guess a model or automatically select one.

Resolve compatible dependency versions during implementation and commit the resulting lockfile. The `uv` project workflow supports a shared lockfile and running commands inside the project environment. [2]

Keep the first version to a CLI and readable files. A web interface, database, agent framework, distributed workers, RL training, and a multi-provider abstraction are outside this initial build.

## 3 Repository contents

| Path | Responsibility |
| --- | --- |
| `README.md` | Setup, the first run, how to edit an agent, how to read results, and troubleshooting |
| `pyproject.toml`, `uv.lock`, `.python-version` | Package, console entry point, dependencies, and Python version |
| `.env.example` | `OPENAI_API_KEY` and `MODEL` placeholders; load `.env` in the CLI |
| `.gitignore` | Exclude `.env`, virtual environments, run output, and private evaluation data |
| `config.toml` | Default request, tool, and time limits |
| `Dockerfile.python` | Build the Python tool image before running agents |
| `agents/direct.py` | One model request with no Python access |
| `agents/tool.py` | Reference model–Python interaction loop |
| `agents/participant.py` | Editable copy of the reference tool agent |
| `src/mathlab/types.py` | Small dataclasses for tasks, model actions, tool observations, and results |
| `src/mathlab/model.py` | Responses API adapter, schemas, usage extraction, and protocol validation |
| `src/mathlab/runtime.py` | Per-problem context, shared counters, deadline, and trace recording |
| `src/mathlab/python_tool.py` | Isolated execution, bounded output, and container cleanup |
| `src/mathlab/runner.py` | Load questions, invoke agents, and write one result per problem |
| `src/mathlab/scoring.py` | Validate files, compare answers, and write the score summary |
| `src/mathlab/cli.py` | `doctor`, `run`, and `score` commands |
| `data/smoke/`, `data/dev/` | Separate question and answer JSONL files plus provenance manifests |
| `tests/` | Scoring, protocol, budget, runner, and Docker checks |

Include normal package initialization files. Load the selected local agent file with `importlib`; resolve relative paths from the invocation directory. Agent code is trusted participant code, so a plugin discovery system is unnecessary.

## 4 Developer workflow

The README should support this sequence:

```bash
uv sync --locked
cp .env.example .env
# Set OPENAI_API_KEY and MODEL in .env.
docker build -f Dockerfile.python -t mathlab-python:v0 .

uv run mathlab doctor
uv run mathlab doctor --live

uv run mathlab run \
  --agent agents/participant.py \
  --questions data/smoke/questions.jsonl \
  --out runs/first-attempt

uv run mathlab score \
  --run runs/first-attempt \
  --answers data/smoke/answers.jsonl
```

`doctor` checks configuration and the local Docker image without contacting the model. `doctor --live` additionally makes one small model request to verify the selected model’s function-calling support; say explicitly that it uses the API.

Require a fresh output directory for each run; refuse to overwrite an existing run. Start with sequential execution. A bounded concurrency option can follow once the basic loop works.

`run` receives questions only. `score` loads the answer key after generation has finished. Scoring should work without API credentials or Docker.

## 5 Agent interface and model loop

The participant-facing contract is:

```python
@dataclass(frozen=True)
class Task:
    id: str
    problem: str


async def solve(task: Task, ctx: AgentContext) -> str:
    """Return one final answer, for example '-3' or '7/2'."""
```

`Task` must never contain the reference answer or worked solution. Construct it explicitly from allowed question fields rather than passing through a raw dataset row.

`AgentContext` exposes two asynchronous operations:

- `ctx.model(history, instructions, allow_python=True) -> ModelTurn`: make a budgeted request and record its usage and trace.
- `ctx.python(code) -> ToolResult`: run a budgeted Python invocation and record its result.

The context owns all counters and the episode deadline. Reusing it for another conversation or verifier must reuse the same budget. Participants must make their model and tool calls through this context.

### Model protocol

Expose two native function tools:

| Tool | Arguments | Meaning |
| --- | --- | --- |
| `python` | `{"code": "..."}` | Execute a self-contained Python program |
| `submit_answer` | `{"answer": "7/2"}` | Return a candidate answer from this model conversation |

Use strict object schemas, require the named string argument, and disallow extra properties. Set `tool_choice="required"` and `parallel_tool_calls=False`. The direct baseline exposes only `submit_answer`; the tool baseline exposes both. [3]

`ModelTurn` should contain `kind` (`python` or `answer`), `value` (code or answer), `call_id`, and the complete provider output items needed to continue the conversation. Preserve those items, including required reasoning items, and associate each function output with its original call ID. Do not reconstruct conversation state from concatenated response text. [3]

The tool baseline follows this sequence:

1. Start a fresh history containing the question and instructions.
2. Request one action through the context.
3. Append the provider output items to the history.
4. For `python`, execute the code and append a structured function-call output with the matching call ID.
5. For `submit_answer`, return its answer string from `solve`.
6. Repeat while the context permits another operation.

For step 4, serialize the observation as a JSON string inside the provider input item:

```python
history.append(
    {
        "type": "function_call_output",
        "call_id": turn.call_id,
        "output": json.dumps(tool_result_dict),
    }
)
```

**Only `solve` returning submits the episode’s final answer.** A model’s `submit_answer` action returns a candidate to the participant’s code. This lets a later participant implementation run a verifier or compare candidates before returning its final choice.

Start a fresh history for each candidate or verifier and reuse the same `AgentContext`. Continuing a history after `submit_answer` requires first appending its matching function-call output.

Return a clear `protocol_error` for malformed arguments, unknown tools, missing function calls, unexpected multiple calls, or incomplete model responses. Never silently execute only the first of several actions. Include a bounded diagnostic and preserve the raw response in the trace where available.

The system instructions should explain the answer format, available libraries, stateless Python execution, and configured limits. Keep them as editable constants in each agent file. Returning execution errors as observations permits the agent to repair its code.

## 6 Python execution

Use a fresh container for every call. Variables, imports, and files do not persist; every code snippet must be self-contained. Build the image once before the lab and install all permitted packages at build time.

Send code over stdin with an argument-list subprocess call. Never interpolate model-generated code into a shell command or execute it in the host Python process.

Configure the container with no network, no host mounts, no credentials, no Docker socket, a non-root user, a read-only root filesystem, a small writable temporary filesystem, dropped capabilities, and resource limits. Docker documents these execution controls. [4]

Use the following initial limits: one CPU, 256 MiB memory, no additional swap allowance, 64 processes, a 16 MiB temporary filesystem, and a five-second invocation deadline including startup. These are pilot defaults, not measured requirements.

Return a structured observation:

```json
{"stdout":"42\n","stderr":"","exit_code":0,
 "timed_out":false,"output_limited":false}
```

Retain at most 8 KiB of combined stdout and stderr. Use bounded readers while the program runs; if it exceeds the output allowance, stop it and mark `output_limited=true`. Do not collect unlimited output and truncate afterward.

Name each container uniquely and explicitly remove it in cleanup after success, error, timeout, or cancellation. Killing the local Docker client alone is insufficient cleanup. A tool failure is an observation the agent may recover from while it has remaining budget.

Version 0.1 assumes cooperative participant code running on the host. The isolated Python tool does not turn the complete runner into a secure service for untrusted submissions.

## 7 Data and exact scoring

Question file:

```json
{"id":"smoke-001","problem":"What is 3 divided by 4?"}
```

Answer file:

```json
{"id":"smoke-001","answer":"3/4"}
```

Keep provenance in a separate `manifest.json`: dataset name, source URL, source revision or download date, original source IDs, selection notes, and hashes of the exported files. Smoke questions are hand-authored public fixtures. Development questions are a reviewed MATH-500 selection with text-only statements and scalar rational answers. Normalize the reference answers during preparation and omit worked solutions from the exported question file. [1]

Development answers may be committed and inspected by participants. Final evaluation questions and answers are withheld from the participant repository. When that evaluation is added, generation receives the questions, and the organizer grades completed results separately; the answer key must not be readable by the generation process. A different directory on the same readable filesystem is not a security boundary.

### Answer grammar

After stripping surrounding whitespace, accept an optional leading sign followed by an integer, a fraction of unsigned integers, or a decimal with digits on both sides of the point. Examples: `-3`, `7/2`, `0.75`. Require a nonzero fraction denominator and limit the answer string to 128 characters.

Convert accepted strings to `Fraction` directly and compare exact values. Thus `3/4`, `6/8`, and `0.75` are equivalent. `Fraction` supports exact conversion of rational strings and decimal strings. [5]

Reject prose, units, LaTeX, arithmetic expressions, scientific notation, sets, multiple answers, and non-finite values. Do not use `eval`, an LLM judge, floating-point tolerances, or last-number extraction. Keep symbolic-equivalence grading for a later extension.

An invalid participant answer scores zero with an `invalid_answer` reason. An invalid reference answer is a dataset error and stops scoring.

Before scoring, validate unique IDs and reconcile the answer file and result rows against the run’s scheduled task IDs. Duplicate, missing, or unexpected result IDs fail clearly. An answer key may contain extra entries when scoring a selected subset, but every scheduled ID must have exactly one valid reference answer.

Score every scheduled problem in the denominator. A correctly formatted wrong answer or a failed episode scores zero. A missing row indicates an incomplete or corrupt run; it must not silently disappear from the denominator.

## 8 Limits and run records

Put initial settings in `config.toml`:

```toml
max_model_requests = 6
max_output_tokens_per_request = 2048
max_python_calls = 4
episode_timeout_seconds = 90
python_timeout_seconds = 5
python_output_limit_bytes = 8192
```

These values are starting defaults for a pilot. Model compatibility and useful task difficulty still need to be checked.

Count each actual model request attempt before sending it, including failed requests. Disable SDK automatic retries with `max_retries=0`; retry behaviour is configurable in the official SDK. Any later retry policy must consume the same counters and remaining time. [6]

All candidate, verifier, and recovery calls count toward the episode limits. Do not make an extra uncounted request to force a final answer. Reject a further operation before it exceeds its request or tool allowance. Apply the remaining episode deadline to in-flight model calls and tool execution.

These limits cap requests, per-request generation, tool executions, and elapsed time. Log actual input and output usage; do not claim a hard monetary spending cap. Reasoning usage is already included in Responses API output-token totals, so record its breakdown without adding it again. If a failed request has unknown usage, mark that explicitly. [7]

The runner should produce:

| File | Contents |
| --- | --- |
| `run.json` | Schema version, agent path, exact model ID, resolved limits, scheduled task IDs, question-file hash, start/end time, run completion state, dependency-lock hash, and Docker image ID |
| `results.jsonl` | One terminal record per scheduled task, including failures |
| `traces.jsonl` | Task ID, event order, model inputs and returned outputs, tool code and observations, durations, and reported usage |
| `scores.json` | Written by `score`: correct/total, accuracy, invalid answers, failure counts, and aggregate resource usage |

Record the Git commit and dirty state when available, plus a copy or hash of the agent file and its prompt configuration. This identifies what was actually evaluated without requiring a source-control service.

Example result:

```json
{"id":"smoke-001","answer":"3/4","status":"completed",
 "model_requests":2,"python_calls":1,
 "input_tokens":780,"output_tokens":155,"elapsed_seconds":4.2}
```

Use `completed`, `budget_exceeded`, `timeout`, `provider_error`, `protocol_error`, and `agent_error` as episode statuses. A completed episode can still have a wrong answer. Keep grading outcomes separate from execution status.

Flush results after each episode. Record a bounded error and continue after an individual task failure. Mark interrupted runs incomplete so scoring can reject them. A brief terminal summary is sufficient; never put credentials or environment-variable dumps in traces. Preserve only model reasoning content actually returned by the provider.

## 9 Build order and completion criteria

Implement in four small steps:

1. **Local evaluation path.** Add package setup, question loading, result writing, and exact scoring. Use scripted fake model responses in tests. Make the CLI work without API access.
2. **Direct baseline.** Add the model adapter, strict answer action, configuration, request accounting, and one real-model smoke run.
3. **Python loop.** Build the image, add bounded execution and cleanup, then implement the tool baseline and participant copy.
4. **First development run.** Add ten reviewed development questions, document the workflow, and compare the two baselines using the same configuration.

Treat the following as the initial acceptance checks:

- A clean checkout installs using the lockfile and the documented setup.
- A scripted model can request Python, receive its output, and return an answer through the full runner.
- Equivalent fractions and decimals score identically; invalid answers and wrong answers score zero.
- Invalid reference answers, duplicate IDs, and missing result rows produce clear errors.
- A new conversation or verifier cannot reset the shared budget; exhausted limits prevent another operation.
- A Python timeout and excessive output produce bounded observations and leave no running container.
- The Python tool cannot read host files or credentials or use the network.
- A task failure produces a result row and does not prevent the next task from running.
- Gold answers and worked solutions never enter model requests or Python tool inputs.
- Both supplied agents complete a small live smoke run with the selected model, and their traces and scores can be inspected.

Keep routine tests offline. Mark Docker checks as integration tests and make live API checks explicit. The implementation is ready for initial use when a participant can edit `agents/participant.py`, run five questions, and explain the outcome from the saved records.

## 10 Reference documentation

These references support the selected data source and library behaviour. The repository design, file formats, defaults, and build sequence above are proposed implementation decisions.

1. [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — source fields and development-problem selection.
2. [uv project workflow](https://docs.astral.sh/uv/guides/projects/) — project setup, lockfiles, and command execution.
3. [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling) — schemas, tool selection, call IDs, and conversation continuation.
4. [Docker container execution](https://docs.docker.com/engine/containers/run/) — filesystem, user, network, and resource controls.
5. [Python fractions](https://docs.python.org/3.12/library/fractions.html) — exact rational conversion and comparison.
6. [Official OpenAI Python SDK](https://github.com/openai/openai-python) — async client, error handling, and retry configuration.
7. [Responses API create reference](https://developers.openai.com/api/reference/resources/responses/methods/create) — output limits and token-usage fields.

Reference documentation checked on 5 October 2026. Confirm the chosen model’s supported settings during implementation and record the tested model ID in the README.
