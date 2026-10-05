# Participant walkthrough

Allow roughly 60–90 minutes after setup. Work in pairs if useful. Keep the model
and backend fixed throughout the comparison: the thing you are changing is the inference-time
agent program.

## 1. Establish a baseline (15 minutes)

Complete the [advance setup](EVENT_RUNBOOK.md) and choose the organizer-approved
[provider configuration](PROVIDERS.md). Follow the README setup. Run the direct agent and the participant agent against the
five smoke questions, using separate output directories, then score both runs.

Open a trace. Identify the question, a model action, a Python observation if one was
requested, and the final answer. A tool-capable model may answer without using Python;
tool availability alone does not require execution.

Discuss: what does the model decide, and what does the host program enforce? Can a
completed episode score zero? Where is the reference answer read?

## 2. Choose a rung (15–25 minutes)

Use the **[100-question contest practice bank](../data/contest/README.md)**.
Start with the four-question comparison in the main README, using
`--config config.hard.toml`. Choose a small working set from these rungs:

| Rung | Questions | Directory under `data/contest/` |
| --- | ---: | --- |
| Contest | 36 | `01-contest` |
| Advanced | 35 | `02-advanced` |
| Stretch | 29 | `03-stretch` |

These orderings are provisional. Compare actual traces before deciding whether a
particular problem is a useful next step. For extra difficulty, choose a FrontierMath
capstone from `data/hard`; for a warm-up, choose a rung from `data/ladder`.
Full 100-question runs are best scheduled outside the session.

The organizer reserves **60 different questions for final evaluation**. Those
questions and their scores are unavailable while you develop. Keep your agent
problem-independent; source-ID lookup tables do not count as solving mathematics.

Keep the same model, limits and selected IDs for each comparison. Record the rung,
IDs and actual denominator:

| Agent / change | Correct / attempted | Invalid answers | Failed episodes | Requests | Python calls | Output tokens | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Direct | | | | | | | |
| Tool baseline | | | | | | | |
| Participant revision | | | | | | | |

Use the aggregate counters in `scores.json`. Mark usage as partial if
`usage_complete` is false. Inspect at least one failure or unnecessarily expensive
success. Development answers are available for inspection, but do not insert them
or question-ID lookups into an agent: that would measure access to the key.

## 3. Make one change and climb (20–30 minutes)

Edit only `agents/participant.py`. Choose one experiment:

- Improve the instructions so the model uses exact arithmetic and prints useful checks.
- Require a calculation or consistency check before returning a candidate.
- Ask a fresh verifier to assess a candidate, then return your chosen final answer.
- Have the agent check a tiny enumerated instance before scaling its algorithm.
- Improve how it tracks constraints or handles cyclic boundaries and conditioning.

For a verifier, start a fresh history and reuse `ctx`; do not instantiate a new context
or call the provider directly. A candidate is returned from `ctx.model(...)`; the
episode ends when your function returns. All conversations share the six-request
and four-Python-call defaults in `config.toml`. The optional `config.hard.toml` raises
these to eight requests and eight Python calls; reserve capacity before adding a check.

Repeat the run command: a numbered directory is created automatically, keeping
your previous attempt. Use the printed score command and compare traces for the
same tasks. If a failure comes from the protocol or budget rather than maths, fix
that first. Once a change helps on your working set, test it on fresh practice questions
from the same rung, then try the next rung. Development rungs contain different
contest problems; they are not parameter variants of a shared template. If both baselines saturate a rung, climb; if both fail, distinguish
incorrect mathematics from runtime limits before changing the model.

## 4. Explain the result (10–15 minutes)

Give a short explanation grounded in saved records:

- What changed in your agent, and why should that help?
- Which answers improved or regressed?
- How much extra work did the agent use?
- Did a trace show the mechanism you expected?
- Which result might be noise or overfitting to this rung or a recurring family?

The [hard-set calibration](HARD_CALIBRATION.md) distinguishes wrong answers from
exhausted budgets and token truncations. The [organizer reference guide](HARD_REFERENCES.md)
contains progressive mathematical hints; if you use them, record the run as
hint-assisted and compare it with the unassisted baseline separately.

These public exercises remain a small teaching set. Repeat promising experiments
before claiming an improvement, and report results per set and selected IDs. Related
original/variant pairs must not be treated as independent held-out data. Tool access
does not guarantee higher accuracy, and the direct and tool agents also have different
request allowances.

Submit your frozen agent and any support files when the organizer closes development.
The organizer then runs every submission on the same sixty final-validation questions
with the same model and limits, and releases results after all submissions are evaluated.
The [organizer workflow](ORGANIZER.md) keeps the final answer key off the generation worker.
