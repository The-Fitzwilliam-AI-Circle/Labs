# Scaling Lab — Attendee Pack

*Fitzwilliam AI Circle, 'Scaling' month. Read this before the session and pick your lane.*

---

## What we're doing

We're going to reproduce the central finding of the scaling-laws literature — that model loss falls as a predictable power law in compute, and that you must grow model size and training data together — using Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat). Then we'll run a prediction contest where you forecast a model run nobody has shown you.

You do **not** need a GPU, a cloud account, or any ML background to fully participate and to win the contest. The lab is built so that the core exercise works from a laptop and a CSV. GPU work is an optional bonus that enriches the shared dataset.

## The reading

Four papers. You'll get more out of the session if you've at least skimmed the first three and read Sutton properly (it's short).

| # | Paper | Skim for | Link |
|---|-------|----------|------|
| 1 | Kaplan et al. 2020 | The idea that loss vs compute is a straight line on log-log axes | <https://arxiv.org/abs/2001.08361> |
| 2 | Brown et al. 2020 (GPT-3) | "Few-shot in-context learning" — what scale unlocked | <https://arxiv.org/abs/2005.14165> |
| 3 | Hoffmann et al. 2022 (Chinchilla) | Why Kaplan's allocation advice was wrong | <https://arxiv.org/abs/2203.15556> |
| Bonus | Sutton, The Bitter Lesson | Read this one fully — it's two pages | <https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf> |

If you only have 20 minutes: read Sutton fully, read the Chinchilla abstract and look at its Figure 1, and skim the GPT-3 abstract for what "few-shot" means.

## Pick your lane

Lanes are about *compute access*, not skill. Choose honestly based on what you can actually run by lab day. The lab succeeds for you completely in Lane 0.

### Lane 0 — No GPU (the default; most people)

You work entirely from the shared dataset Neil generated. You'll plot the scaling curves, fit the power law, and enter the prediction contest. **This is the whole intellectual payload of the lab.** A laptop with Python (or just Google Colab in the browser) is all you need. If you do nothing else, do this — and you can still win the contest.

Requirement: a working Python environment OR a Google account for Colab. The fitting notebook (next section) runs in either.

### Lane 1 — Modest GPU (one A100/H100 on RunPod, or a Colab GPU session)

Everything in Lane 0, plus: run **one or two small depths yourself** and drop your own data point onto the shared plot. The thrill is watching your dot land on the line everyone else's data drew.

**Recommended path: RunPod (same team account as Lab 1).** You already have an invite to the **Fitzwilliam AI Circle** RunPod team from the GPT-2 lab — log in, switch to the team account in the top-left dropdown, deploy a **single H100 80GB SXM** pod from Secure Cloud, pick a PyTorch-CUDA template, and open Jupyter Lab. A single-GPU run is ~minutes for d8 and costs a few dollars of team credit. (Free Colab GPU also works; expect smaller batch and slower wall-clock.)

**Use this exact command** (Neil validated it in prep — do not improvise the batch size, you'll hit out-of-memory):

```bash
git clone https://github.com/karpathy/nanochat.git && cd nanochat
uv sync --extra gpu && source .venv/bin/activate

# Single small model, single GPU. --device_batch_size is the OOM dial.
# Starting values (Neil will confirm the validated number on the day):
#   80 GB card (H100/A100 on RunPod):  --device_batch_size=32
#   24 GB card (RTX 4090/3090):        --device_batch_size=8
# If you OOM, halve the value (32 → 16 → 8 → 4 → 2 → 1) and retry.
OMP_NUM_THREADS=1 python -m scripts.base_train -- \
    --depth=8 \
    --run="myname_d8" \
    --model-tag="myname_d8" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
    --device_batch_size=32
```

A single-GPU run is ~8× slower than the 8-GPU node but produces near-identical results (nanochat auto-switches to gradient accumulation). A d8 is still minutes-scale.

After the run finishes, grab the row written to `~/.cache/nanochat/scaling_laws_results/results.csv` and bring it on a USB stick or paste it into the shared sheet — we'll splice it into the room's combined dataset live.

### Lane 2 — Full 8×GPU node (only if you genuinely have one)

You contribute the *expensive* points back to the shared dataset — a high depth (d16/d20) or a replication seed. Coordinate with Neil beforehand; the lab budget can reimburse you because your run benefits the whole room. You'll run `bash runs/scaling_laws_lab.sh` against the agreed `(flops, depth)` cell.

## The fitting notebook

You'll be given `scaling_fit.ipynb` (also runnable as the plain script `scaling_fit.py`). It loads the shared CSV and walks you through:

1. Plotting `val_bpb` vs `flops_budget`, one line per depth (log-log).
2. Finding the compute-optimal depth at each budget — the minimum of each U-curve.
3. Fitting `val_bpb ≈ a · C^(−b)` to those minima and reading off the exponent `b`.
4. Looking at `core_score` and seeing how much noisier the *capability* metric is than the *loss* metric.
5. Producing your single prediction for the contest.

It runs in Colab (no install — File → Upload notebook) or locally (`pip install pandas numpy matplotlib scipy jupyter`, then `jupyter lab scaling_fit.ipynb`). It has TODO cells — the point is you do the fit, not that you watch it happen.

## The prediction contest

**The setup:** Neil trained one model — a specific depth at a specific compute budget — that is **not in your dataset**. You don't know which.

**Your job:** From the visible data, fit the scaling law and predict that held-out run's `val_bpb`.

**Submission format:** one number, plus one sentence on your reasoning. Hand it in (or call it out) *before* the reveal. Example: `predicted val_bpb = 0.842 — fitted power law on the per-budget minima, extrapolated to the stated held-out FLOPs.`

**Winner:** closest absolute error on `val_bpb`. Tie-break: best one-sentence reasoning.

**Why this is fair to everyone:** the winning move is reasoning correctly about a power law and being honest about extrapolation uncertainty. Owning a GPU buys you nothing here. Someone who has never trained a model and reasoned carefully from the CSV will beat someone who ran their own model but extrapolated sloppily. That's the point.

A genuine hint, not a trick: the loss metric (`val_bpb`) behaves; the capability metric (`core_score`) is noisy at this scale. Predict the one that sits on a clean line.

## What to bring

- A laptop (or a Google account for Colab).
- The reading at least skimmed.
- Your chosen lane decided in advance — tell Neil if you're Lane 2 so he can coordinate a cell for you.
- Curiosity about why the entire industry bets billions on the assumption that the straight line you'll fit by hand keeps going.

## Appendix — The one piece of math, for those who want it

Skip this if equations aren't your thing; you can win the contest without it.

The scaling law is just:

```
L(C) ≈ a · C^(−b)
```

`L` is loss (here, validation bits-per-byte), `C` is compute in FLOPs, `a` and `b` are fitted constants. Take logs of both sides:

```
log L = log a − b · log C
```

That's a straight line: plot `log L` against `log C` and the slope is `−b`. Fitting the law is just fitting a line in log-log space. That is the entire mathematical content of Kaplan et al.

The Chinchilla refinement. Model the compute-optimal number of parameters as `N* ≈ k₁ · C^p` and the compute-optimal number of training tokens as `D* ≈ k₂ · C^q`. The ratio is then:

```
D*/N* = (k₂/k₁) · C^(q−p)
```

Chinchilla's empirical finding is `p ≈ q ≈ 0.5`. Therefore the exponent on `C` is `q − p ≈ 0`, so `C^0 = 1`, and:

```
D*/N* ≈ k₂/k₁ = a constant, independent of compute
```

This is the whole game. Because the optimal tokens-per-parameter ratio doesn't depend on scale, you can fix one constant and generate the entire family of compute-optimal models by turning a single dial — which is exactly what nanochat's `--depth` does. Chinchilla measured that constant at ≈ 20. nanochat (different optimiser, smaller scale) measures ≈ 8. The *form* of the law is robust; the *constant* is regime-dependent. Holding those two facts apart is the most useful habit this lab can give you.
