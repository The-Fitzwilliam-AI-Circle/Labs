# Running the agent lab

Plan for **around 10 people bringing their own laptops**, working in roughly five
pairs. The schedule below assumes a **two-hour session; duration is still to be
confirmed**. The aim is to improve an inference-time agent while keeping its model
fixed within each scored track.

Use one primary laptop per pair for scored experiments; the other can be used to
read traces, discuss changes and provide a fallback. Rotate the person writing code
halfway through. Aim to have every laptop ready, with at least one verified working
machine per pair before the day. If the organizer is included in the ten people,
four pairs plus an organizer and helper is another practical arrangement.

Use a common hosted API for the main competition so participants do not need a GPU
or model weights on their laptops. If using self-hosted inference, provide one
rehearsed shared server and test it with five simultaneous agent runs. Participants
still need local Docker for Python execution. Keep experiments with models running
on individual laptops in the exploration track.

## Competition format

Give participants all **100 practice questions and answers**. Reading solutions,
examining failed traces and testing ideas is part of learning. Require agent code
to solve the supplied problem; embedding answers, question-ID lookups or retrieving
public answer pages is outside the lab rules. Let participants use the 40 older
questions for warm-ups and the 12 FrontierMath questions as optional capstones.

Keep **both questions and answers** for all 60 final-validation cases private until
submissions close. Giving out only the questions still permits special-case code,
manual solving and searching public sources. That can be a valid open-question
competition, but it measures a different objective from performance on unseen lab
problems. Here the final set serves as a test set. Do not offer a feedback endpoint
or repeated submissions against it during development.

Use one organizer-selected, rehearsed model/backend for the main competition.
Permit other providers in an exploration track or maintain separate leaderboards
per pinned model/backend. Otherwise improvements from stronger models, larger
hardware or different routing will be mixed up with improvements to agent design.
Each team should compare its revision with the baseline on the same questions,
same model and same limits. Do not claim an identical inference budget merely
because two different providers accept the same token-limit number.

## One week before

1. Choose the primary backend, exact model, `config.hard.toml` or another frozen
   budget profile, and a fallback backend. Rehearse the fallback too. For the
   simplest first event, use the already piloted OpenAI model as the primary;
   keep OpenRouter/local as optional until their deployment passes rehearsal.
2. Run a representative **development** sample on each intended model. Inspect
   tool-use traces as well as scores. Aim for useful headroom, for example roughly
   40–75% on the session's working questions; this is a planning target, not a
   property guaranteed by the rung names. Current evidence is only a 12-question
   practice pilot: direct 9/12, tool 8/12 on GPT-5 nano. Do not calibrate on the
   final set. Broaden the development rehearsal if the new backend changes difficulty.
3. Prepare an inference access plan: individual/team credentials with a spending
   allowance, or an organizer-managed endpoint with team access. Plan for five
   teams. Check funding and
   concurrent capacity. Avoid circulating one unrestricted personal key. Never
   include credentials in the ZIP or submitted code.
4. Estimate event usage from the saved rehearsal token totals, intended attempts
   per pair and number of pairs; leave headroom. Test five simultaneous agent runs,
   particularly on a shared GPU. Avoid an on-the-day model download or
   weight conversion as a required setup step.
5. Export the participant package. Send it, setup instructions, selected model
   details and a success checklist in advance. Ask participants to complete setup
   **before arrival**. Prepare one preconfigured spare workstation or a working
   partner station for installation failures.
6. Collect laptop operating systems and any installation restrictions in advance.
   Rehearse the platforms people will actually bring, and check the room's Wi-Fi,
   power sockets and access to the chosen inference endpoint.

## Participant preflight: 1–2 days before

Participants need a terminal, editor, uv, working Docker and the supplied package.
On Windows, use a consistent WSL2-based environment with Docker integration if
that is the organizer's tested platform; rehearse it rather than assuming the
macOS setup covers it. Docker's licensing and installation are handled through
the organization's normal setup process.

From the extracted project:

```bash
uv sync --locked
cp .env.example .env
# Set PROVIDER, MODEL and the selected provider's credentials/endpoint in .env.
docker build -f Dockerfile.python -t mathlab-python:v0 .
uv run mathlab doctor --config config.hard.toml
uv run mathlab doctor --live-tools --config config.hard.toml
uv run mathlab run --agent agents/tool.py --config config.hard.toml \
  --questions data/smoke/questions.jsonl --out runs/preflight
```

Score using the printed command. Send the organizer only confirmation that the
checks passed and the smoke score—not `.env` or an API key. The five smoke questions
check setup, not competitive ability. The `--live-tools` diagnostic verifies an actual Python call and the model
consuming its result. The organizer should also rehearse harder practice cases
with the chosen model. See [PROVIDERS.md](PROVIDERS.md).

## On the day: two hours

| Time | Activity | Outcome |
| --- | --- | --- |
| 0–15 min | Pair up; preflight checks; use a spare/partner station for unresolved setup | Everyone can run and score an episode |
| 15–30 min | Run direct/tool baselines on the same four practice questions from the README | A saved baseline and an understood trace |
| 30–65 min | Change one agent behavior; iterate on a small practice subset | A concrete hypothesis, code change and comparison |
| 65–90 min | Try fresh practice cases and a harder rung; compare against the baseline | Evidence beyond the initially tuned examples |
| 90–105 min | Freeze and submit agent plus allowed support files and a short explanation | Immutable final submissions |
| 105–120 min | Show mechanisms, regressions and costs; discuss the practice leaderboard | Learning from both successful and failed changes |

Use a common working subset for the initial exercise so pairs can compare findings.
Later allow exploration of the remaining practice bank. One organizer can circulate
between the pairs; if a second helper is available, have them handle setup/access
issues while the organizer helps interpret traces. Change one thing at a
time: better mathematical checks, state handling, a verifier, or a code-debugging
loop. If there are only 90 minutes, shorten experimentation; keep setup preflight
and the submission freeze.

The full final evaluation is **after the interactive session**. Sixty episodes at
the current four-minute deadline allow up to four hours per team with this
sequential runner. Actual runtime may be shorter; measure it during the development
rehearsal. Do not promise a full 60-question leaderboard in the last ten minutes.
Use practice results for the live discussion and publish final results afterward.
Five team submissions require **300 final episodes**, plus any organizer baseline
runs. Include these in the inference allowance and evaluation schedule.

## Final evaluation and fairness

Use the [organizer workflow](ORGANIZER.md): run each frozen submission with opaque-ID
questions in a clean worker with no key or private source files, then score on the
organizer machine. Hash all support files as well as the entrypoint; the harness
automatically snapshots only the entrypoint. Keep the model, server settings,
Docker image and episode limits fixed within a track. Record local weight revision,
quantization, context window and tool parser externally alongside the run metadata.

Announce attempt and outage rules before submissions. A simple policy is one run
per team on all 60 questions, with all ordinary wrong answers, truncations and
agent budget failures counted. A verified infrastructure outage may justify a
documented rerun under the same rule for every affected team; keep the original
run rather than selecting whichever result scores better. A shared server must
have consistent capacity during scoring. Evaluate sequentially or with a fixed,
load-tested worker count to avoid queue pressure deciding the ranking.

Report correct/60 plus invalid answers, failure categories, tokens and time. Cost
or latency awards should be separate and only compare measurements made under
comparable infrastructure. Release per-question final feedback after all teams
have finished. Retire the final set for future cohorts once it has become familiar.

## Setup triage

| Failure | First check |
| --- | --- |
| Docker unavailable | Start Docker; confirm the prebuilt image; use the spare machine if installation is blocked |
| 401/403 | Provider-specific key, access permissions and exact model ID |
| 429 | Funding/quota versus request-rate limit; pause and resolve the shared cause |
| Connection refused | Local model server running, correct port, and reachable hostname |
| Tool/schema rejection | Model supports functions; local server has the correct template/parser |
| Answer arrives as plain text | Choose/configure a function-calling model; text alone does not satisfy this agent contract |
| Token limit or timeout | Inspect the failure; use the predeclared profile rather than changing competition limits per team |

The important remaining rehearsal is operational: the chosen accounts, hardware,
network, participant operating systems and concurrent load. Automated integration
tests cannot establish that those event-specific resources are ready.
