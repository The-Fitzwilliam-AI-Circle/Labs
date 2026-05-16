# Scaling Lab — Fitzwilliam AI Circle

The 'Scaling' month of the Circle. We reproduce the central finding of the scaling-laws literature — that model loss falls as a predictable power law in compute, and that model size and training data must grow together — on real nanochat runs Neil generates beforehand. Then we run a prediction contest where attendees forecast a held-out run nobody has seen.

The lab is designed so that **the full intellectual payload works from a laptop and a CSV.** GPU work is optional and additive.

## Files in this folder

| File | For whom | When |
|---|---|---|
| [01_prep_runbook.md](01_prep_runbook.md) | Facilitator (Neil) only | Run on an 8×H100 node before the session. Produces the dataset. |
| [02_facilitator_guide.md](02_facilitator_guide.md) | Facilitator (Neil) only | The in-room script. Read on the day. |
| [03_attendee_pack.md](03_attendee_pack.md) | Every attendee | Distribute before the session. Pre-reading + lane selection. |
| [scaling_fit.ipynb](scaling_fit.ipynb) | Every attendee | The fitting notebook used in Block C. Runs in Colab or locally. |
| [scaling_fit.py](scaling_fit.py) | Attendees who prefer a script | Plain-script mirror of the notebook (VS Code / PyCharm cells). |

## Distributed at the session

Two CSVs Neil generates from the prep runbook are distributed separately:

- `ai_circle_scaling_dataset.csv` — the shared dataset (given to all attendees in Block C).
- `HELDOUT_DO_NOT_DISTRIBUTE.csv` — kept private, revealed at the end of Block D.

## The four papers

| # | Paper | The one move it makes |
|---|-------|----------------------|
| 1 | [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) | Loss is a predictable power law in compute |
| 2 | [Brown et al. 2020 (GPT-3)](https://arxiv.org/abs/2005.14165) | Scale produces emergent few-shot ability — the payoff |
| 3 | [Hoffmann et al. 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556) | Grow params and data together; the ratio is a constant |
| Bonus | [Sutton, The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) | General + compute beats clever + handcrafted, every time |

## Reference

- nanochat: <https://github.com/karpathy/nanochat>
- nanochat Discussion #420 (the worked solution this lab reproduces): <https://github.com/karpathy/nanochat/discussions/420>
