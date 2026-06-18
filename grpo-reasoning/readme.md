# Advanced GRPO — train a reasoning model

You're going to take a **base** model — `Qwen3-4B-Base`, with no instruct or reasoning tuning — and turn it into a model that shows its working, using **GRPO** (Group Relative Policy Optimization), the reinforcement-learning algorithm behind DeepSeek-R1. By the end of the session you'll have a LoRA adapter that makes the model reason inside explicit tags and answer maths problems, and you'll have watched the reward climb live in the training log.

The whole lab is a single notebook — [advanced_grpo.ipynb](advanced_grpo.ipynb) — adapted from Unsloth's *Advanced GRPO* recipe for a RunPod Jupyter pod. The "advanced" part is two tricks that make reward move *visibly* within one session instead of after hundreds of steps:

1. A short **SFT warm-up** so the model reliably emits our custom reasoning format before RL even starts.
2. **Four shaped reward functions** (exact format, partial format, exact answer, numeric closeness) instead of one, so there's almost always *some* gradient signal to climb.

The core of the lab — and where you'll want to spend the discussion time — is **section 6, the reward functions**. That's where the actual learning signal is designed, and it's the part you'd change first for a different task.

## Before the session — read this on your laptop

Clone the repo so you can read this README and skim the notebook ahead of time:

```bash
git clone https://github.com/The-Fitzwilliam-AI-Circle/Labs.git
cd Labs/grpo-reasoning
```

You don't need to install anything locally. All the work happens on a RunPod GPU — your laptop just needs a browser. Open `advanced_grpo.ipynb` (GitHub renders it, or open it in VS Code / Jupyter) to get a feel for the flow: install → load model → define the reasoning format → SFT warm-up → GRPO data prep → **reward functions** → GRPO training → inference → optional export.

You don't have to understand every cell in advance. Do come having thought about one question: **what would you reward, and how much, to teach a model to reason?** The notebook's answer is four stacked functions — come ready to argue with them.

## Pod requirements

This lab needs more VRAM than the GPT-2 lab — vLLM holds the model for fast rollouts *and* we train a LoRA on top.

- **GPU: ≥ 24 GB VRAM.** RTX 4090 / 3090 / L4 / A10 / A40 / A100 are all fine. The 4090 (24 GB) is the default and matches the cards we usually run. If only 16 GB cards are free, set `load_in_4bit = True` in the model-load cell (section 2) and it'll still run.
- **Template: the stock `RunPod PyTorch 2.x` template.** Unlike the GPT-2 lab, this lab does **not** use a custom pod template — the notebook's first cell `pip install`s everything it needs (Unsloth, vLLM, a pinned `transformers`/`trl`). Don't reuse the `Fitzwilliam-GPT-lab` template; it's built for tinygrad and ships a smaller GPU by default.
- **Container disk: ~40 GB+.** vLLM + Unsloth + Torch are large; the template's 20 GB default can fill up mid-install.
- **Volume: ~30 GB at `/workspace`** (the template default mount). The notebook sets `HF_HOME=/workspace/hf_cache` so the ~8 GB of model weights and the datasets persist on the volume and survive a pod stop/start instead of re-downloading.

## Setup — RunPod (the venue path)

About 10 minutes start to finish, then you're training on a GPU. Steps 1–2 are the same team-account flow as every Fitzwilliam lab; if you've done a lab before, you already have the invite.

### 1. Accept the team invite

At the start of the session you'll get an email from RunPod inviting you to the **Fitzwilliam AI Circle** team account. Click the link, sign in or create a RunPod account, and accept. The team account is how we pay for everyone's compute centrally — you don't need a credit card or your own credit. We'll get everyone set up at the start of the lab.

### 2. Switch to the team account

In the RunPod dashboard, use the **top-left dropdown** to switch from your personal account to **Fitzwilliam AI Circle**. If you skip this you'll be on your personal (probably unfunded) account.

### 3. Deploy a pod

- Go to **Pods** → **Deploy**.
- **GPU:** pick **RTX 4090** in **Secure Cloud** (RTX 3090 also fine). If 4090s are out, an **A40** or **A100** gives extra headroom — and unlike the GPT-2 lab, the bigger cards *are* worth it here if you want to raise `num_generations` or use longer reasoning traces.
- **Template:** choose **`RunPod PyTorch 2.x`** (the official one). No custom template needed.
- **Disk:** bump **Container Disk to ~40 GB** and **Volume to ~30 GB** (mounted at `/workspace`). See pod requirements above for why.
- Click **Deploy** and wait ~60 seconds for it to boot.

### 4. Connect — Jupyter Lab (recommended)

Once the pod shows **Running**, click **Connect** and open the **Jupyter Lab** link (port 8888) in a new tab. This is the natural home for a notebook lab: file browser, terminal, and notebook editor in one UI, nothing to install locally.

Open a terminal inside Jupyter Lab (**File → New → Terminal**) and pull the repo into the persistent volume so your work survives a pod stop:

```bash
cd /workspace
git clone https://github.com/The-Fitzwilliam-AI-Circle/Labs.git
```

Then in the file browser, navigate to `Labs/grpo-reasoning/` and double-click **`advanced_grpo.ipynb`** to open it.

> **Prefer your own editor?** You can also connect over SSH and open the notebook in VS Code / Cursor (Remote-SSH + the Jupyter extension) or JetBrains. The SSH-key setup is identical to the [gpt2-tinygrad lab](../gpt2-tinygrad/readme.md) (see its "Option B — Full SSH" section) — set it up once and reuse it. Jupyter Lab is the path of least resistance and what the rest of this README assumes.

### 5. Run it

Click **Run → Run All Cells**. The first install cell takes a few minutes.

> **If `import unsloth` fails the first time, restart the kernel once and re-run** (**Kernel → Restart Kernel**, then Run All again). That's a known vLLM/Unsloth first-install ordering quirk on a fresh pod, not a real error — it almost always works on the second pass.

What you're watching for, in order:
- **Section 4 (SFT warm-up)** finishes in a couple of minutes — afterwards the model already emits the `<start_working_out>…</start_working_out><SOLUTION>…</SOLUTION>` format.
- **Section 7 (GRPO training)** is the long pole: roughly **20–45 min for 100 steps** depending on GPU. Watch the **`reward`** column in the log table climb. Section 6's `check_numbers` reward also prints a live question/answer/response sample every 5 steps, so you can literally read the model getting better.
- **Section 8 (inference)** compares the model *without* the GRPO LoRA against *with* it on "what is the sqrt of 101?" — the before/after is the payoff.

For a tighter live slot, drop `max_steps` to ~50 in section 7. For a real run, comment out `max_steps` and set `num_train_epochs = 1`.

### 6. When you're done — STOP THE POD

The team's RunPod credit is shared, so a forgotten running pod burns credit other people would have used. On the **Pods** page, click **Stop**.

- **Stop** pauses compute billing and keeps your `/workspace` volume (trivial storage cost). Resume later and pick up where you left off — your weights cache and saved LoRA are still there.
- **Terminate** wipes the pod *and* the volume. Only do this if you've pushed anything you want to keep (e.g. the LoRA) off the pod first.

For a single session, **Stop** when you finish.

## Saving / exporting your model (optional)

Section 9 is all gated behind `if False:` — flip a flag to `True` to act on it. Set your Hugging Face token first if you're pushing to the Hub.

- Merge the LoRA into 16-bit weights, or push merged weights to the Hub.
- Save just the LoRA adapter (small — easy to download off the pod).
- Export GGUF for llama.cpp / Ollama.

Section 8 already calls `model.save_lora("grpo_saved_lora")`, so even without section 9 your trained adapter is on the pod. Download it from the Jupyter file browser before you terminate.

## What's in this folder

| File | What it is |
|---|---|
| [advanced_grpo.ipynb](advanced_grpo.ipynb) | **The lab.** The full pipeline end-to-end — `Run All` on a RunPod pod. Self-contained; the first cell installs all dependencies, so there's no `pyproject.toml` / `uv sync` for this lab. |
| readme.md | This file. |

## Stuck?

- **`import unsloth` fails on first run.** Restart the kernel once and Run All again (see step 5). It's a first-install ordering quirk, not a real failure.
- **CUDA out of memory** during GRPO. Lower `num_generations` (try 2) in section 7 first — it's the group size and the biggest memory lever. Then lower `gpu_memory_utilization` (e.g. 0.85) in section 2, and/or set `load_in_4bit = True` in section 2 if you're on a 16 GB card.
- **Disk full / `No space left on device`** mid-install. The container disk filled up. Terminate, redeploy with a larger **Container Disk** (~40 GB+), and re-run.
- **Model weights re-download every time** you restart the pod. `HF_HOME` isn't pointing at the volume. Confirm the install cell ran (`os.environ["HF_HOME"] = "/workspace/hf_cache"`) and that your volume is mounted at `/workspace`.
- **Pod template / GPU doesn't appear.** You're in your personal account, not the team account. Top-left dropdown → **Fitzwilliam AI Circle**.
- **Reward stays flat / negative.** The model isn't emitting valid format, so the answer rewards never fire. Check the live sample print from `check_numbers` (section 6) — if there are no `<SOLUTION>` tags, the SFT warm-up (section 4) didn't take; re-run it and confirm the format-check cell after it shows the tags.
- **Laptop went to sleep mid-run.** Training runs on the pod, not your laptop — the run keeps going. Reopen Jupyter Lab and reconnect to the kernel from the **Kernel** menu.

If none of those, ask Neil.

## Reference

- Unsloth notebooks (the recipe this is adapted from lives here, alongside 250+ others): <https://github.com/unslothai/notebooks>
- Unsloth's Reinforcement Learning guide — background on GRPO and reward design: <https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide>
- GRPO is the RL algorithm behind DeepSeek-R1: [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
- SFT warm-up data: [`unsloth/OpenMathReasoning-mini`](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)
- GRPO training data: [`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)
