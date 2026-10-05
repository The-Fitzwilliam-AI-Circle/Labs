# Organizer workflow: practice and final validation

The new bank has **100 practice questions and 60 final-validation questions**.
Use “final validation” as the lab's final test: participants tune on development
questions and receive final-validation results only after their agent is frozen.
Repeatedly exposing validation scores turns it into another development set.

For advance setup and the session schedule, see [EVENT_RUNBOOK.md](EVENT_RUNBOOK.md).
Backend settings and compatibility checks are in [PROVIDERS.md](PROVIDERS.md).

## Distribute practice materials

The organizer bundle lives alongside this project in `../agents-lab-organizer/`.
It contains validation questions, the answer key, provenance, source snapshots,
the exclusion log and `freeze.json`. **Do not distribute that folder.** It is
outside this Git repository and outside the package export allowlist.

The public repository does not include this private bundle or its utilities.
Keep it outside the entire Labs checkout. When following the commands below from
`Labs/agents-lab/`, substitute the actual private bundle path for
`../agents-lab-organizer/`; that example describes the standalone workspace layout.

Build a shareable participant ZIP:

```bash
uv run python scripts/export_participants.py --out dist/mathlab-participants.zip
```

The exporter includes the lab harness, three starter agents, configuration,
practice data, practice keys, tests and guides. It excludes `.env`, Git history,
runs, unlisted data files and all organizer material. Existing ZIPs are not
overwritten; choose a versioned filename for another release. `PACKAGE.json`
contains hashes of the included files. Share this ZIP, rather than zipping the
whole working directory or its parent.

## Freeze the evaluation before submissions

The delivered validation split was selected before the practice pilot and has
not been run against the baseline model. Its 60 questions come from editions
absent from development. Original source selection is retained privately; public
development fixtures contain only development records. Dataset hashes are in
the organizer's `freeze.json` and split manifest. Back up the organizer folder.

Use the same 60 questions, provider/backend, model snapshot, Docker image and limits for all final
submissions. `config.hard.toml` is the initial contest profile used in calibration:
eight model requests, eight Python calls, 16,384 output tokens per request and
240 seconds per question. Set the final policy before seeing validation results.
Keep all 60 cases in the denominator, including timeouts and tool/token failures.
An entire final run has a configured ceiling of four hours per agent; schedule
it after the interactive session and account for API usage.

Have each pair submit its complete agent code and any allowed support files.
Record a hash of the whole submission; this harness automatically snapshots and
hashes the selected `agent.py`, but does **not** freeze imported helper modules.
Do not permit code changes between exposing final questions and running the agent.

## Generate with questions, score separately with the key

Run final generation in a **clean worker environment** with only the participant
package, frozen submission, chosen configuration, credentials and a copy of the
validation **questions**. Never copy the organizer folder, answer key, raw source
snapshots or private manifests into that worker. Source IDs are removed from the
worker questions by the organizer's `prepare_worker_questions.py` utility.

On the organizer machine, prepare a fresh questions-only file:

```bash
uv run python ../agents-lab-organizer/prepare_worker_questions.py \
  --out ../agents-lab-organizer/worker-input/questions.jsonl
```

Copy **only that JSONL file** to the worker as `evaluation/questions.jsonl`.
Then, on the worker:

```bash
uv run mathlab run --agent agents/participant.py --config config.hard.toml \
  --questions evaluation/questions.jsonl --out runs/final
```

Ignore the suggested local score command: there is deliberately no worker key.
Return the full resulting run directory to the organizer. Grade it on the
organizer machine, using the paired opaque-ID key produced by the utility:

```bash
uv run mathlab score --run received/final \
  --answers ../agents-lab-organizer/worker-key/answers.jsonl
```

Use the **same prepared questions and key for every team**, and preserve the
returned run's actual directory name if it was numbered. The organizer's
`worker-key/manifest.json` records the question hash, key hash and mapping back
to the frozen validation set. Verify the returned `run.json` question hash,
scheduled IDs, agent snapshot, `provider_configuration`, model and limits before
accepting a result. Separate model/backend tracks if participants explore different
providers; a single mixed-model leaderboard does not isolate agent improvements.

The current harness trusts host-side participant Python. A sibling folder or a
different readable directory on the same host does **not** hide answers from it.
The clean worker workflow keeps the private key absent from generation; it is
not a sandbox for hostile submissions. Participants must still follow the lab
contract: model calls through `ctx.model`, code through `ctx.python`, no external
answer lookup or embedded question/answer tables. Public contest text can be
searched online, so opaque IDs do not make it secret or prove no training exposure.

## Report the final result

Report correct/60, failed episodes, invalid answers, tokens, Python calls and time.
Compare agents on the **same cases** and inspect which items changed. Sixty cases
are substantially more informative than twelve but do not resolve tiny ranking
differences: one question is 1.67 percentage points. A single stochastic run is
also not a measurement of repeatability. For formal comparisons, predeclare equal
repeated attempts and aggregate them without choosing the best attempt.

Release final per-question feedback after all teams finish. If a source defect is
confirmed, document it and apply the same correction/exclusion to every team;
keep the original frozen files and results. A reused and discussed final set
should be replaced for a later cohort if participants can access the feedback.
