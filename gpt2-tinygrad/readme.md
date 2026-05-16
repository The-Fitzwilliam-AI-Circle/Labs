# GPT-2 from scratch (in tinygrad)

You're going to implement GPT-2 from scratch in [tinygrad](https://github.com/tinygrad/tinygrad). The skeleton you edit has the scaffolding — config dataclass, weight initialisation, training loop, optimiser, dataset loader, causal mask, scheduler — and the model itself is missing: token + positional embeddings, attention, FFN, the decoder block, and the forward pass. By the end of the session you'll have a working transformer that you can train on TinyShakespeare and use to generate Shakespeare-flavoured text.

## Before the session — read this on your laptop

Clone this repo on your laptop first so you can read the README and skim the skeleton ahead of time:

```bash
git clone https://github.com/The-Fitzwilliam-AI-Circle/Labs.git
cd Labs/gpt2-tinygrad
```

You don't need to install anything locally. The actual work happens on a RunPod GPU; your laptop just needs a browser (and optionally an SSH client if you want to use your own editor — see below). You can open `skeleton_easy.py` in your editor of choice to get a feel for what you'll be implementing.

## Pick your difficulty

There are three skeleton files. Pick one and edit it. They all expose the same public API to `train.py`, so the runnable command is the same regardless of difficulty.

| File | What's filled in for you | What you implement |
|---|---|---|
| [skeleton_easy.py](skeleton_easy.py) | Config defaults, init, causal mask, optimiser, scheduler, training loop, dataset, every `__init__` | Just the algorithm bodies: embedding `__call__`s, attention, FFN, decoder block `__call__`, forward pass, loss, generate |
| [skeleton_medium.py](skeleton_medium.py) | `_init_weights`, dataset, `train` outer loop | Everything in easy, **plus** the `GPT2Config` dataclass, `make_causal_mask`, every `__init__` (structural assembly), `CosineScheduleWithWarmup`, `make_optimizer`, and `train_step` |
| [skeleton_hard.py](skeleton_hard.py) | Just `from tinygrad import Tensor`. Have fun. | Everything. |

Default is easy. Run with `--difficulty medium` or `--difficulty hard` (see Setup below) to switch.

## Repo layout

| File | What it is |
|---|---|
| `skeleton_{easy,medium,hard}.py` | **What you edit.** See the table above. Each TODO maps to a step in the roadmap below. |
| [train.py](train.py) | **What you run.** Loads TinyShakespeare, tokenises it with the GPT-2 BPE, trains the model from your chosen skeleton, then samples some text. Picks the skeleton via `--difficulty`. |
| [tinyshakespeare.txt](tinyshakespeare.txt) | The training corpus. ~1.1MB of Shakespeare's plays as a single text file. Already in the repo, don't worry about it. |
| [pyproject.toml](pyproject.toml) | Project + dependency declaration for `uv`. |

`main.py` (the full reference implementation) is **not in the repo by design.** If you go looking for it, you won't find it — that's intentional. Neil may share it after the session.

## Setup — RunPod (the venue path)

This is what you do at the venue. About 10 minutes start to finish, then you're training on a 4090.

### 1. Accept the team invite

At the start of the session you'll receive an email from RunPod inviting you to the **Fitzwilliam AI Circle** team account. Click the link, follow the prompts to create or sign in to a RunPod account, and accept the invite.

The team account is how we pay for everyone's compute centrally — you don't need to add a credit card or buy credit yourself.

We will get everyone set up on this at the start of the lab. 

### 2. Switch to the team account

After logging in to RunPod, look at the **top-left dropdown** in the dashboard. By default you're in your personal account. Click the dropdown and switch to **Fitzwilliam AI Circle**.

If you don't switch, you'll be looking at your personal (possibly unfunded) account and won't see the lab template.

### 3. Deploy a pod from the lab template

- Go to **Pods** → **Deploy**.
- Under **Pod Template**, pick **`Fitzwilliam-GPT-lab`**. This template has CUDA, Python 3.11, the right environment variables, and a 20 GB persistent volume mounted at `/workspace` already configured.
- **GPU:** pick **RTX 4090** in **Secure Cloud**. RTX 3090 also works if 4090s are out. Don't pick A100/H100 — they're overkill for this lab and burn through team credit faster.
- Leave everything else at the template's defaults.
- Click **Deploy**. Wait ~60 seconds for the pod to boot.

### 4. Connect to the pod

Once the pod's status shows **Running**, click **Connect** on the pod's row. You have three connection options. Pick whichever fits your workflow — they're not mutually exclusive, you can use more than one against the same pod.

#### Option A — Jupyter Lab (recommended for first-timers)

Click the **Jupyter Lab** link (port 8888) to open it in a new browser tab. You get a file browser, a terminal, and notebook editor in one UI, no setup on your laptop required.

Open a terminal inside Jupyter Lab via **File → New → Terminal**. That's where you run the commands in step 5.

To edit code, double-click `skeleton_easy.py` in the file browser sidebar — it opens in a tab. Save with `Cmd/Ctrl-S`. You can have the editor open in one tab and the terminal in another and flip between them.

This is the path of least resistance and what the rest of the README assumes by default.

#### Option B — Full SSH (if you use VS Code, Cursor, JetBrains, emacs etc )

Best if you'd rather use your own editor against the remote pod — VS Code Remote-SSH, Cursor, JetBrains Gateway, or just `ssh` + your usual terminal setup. Setup is a bit more involved but you do it once and reuse it for every pod afterwards.

**One-time setup (on your laptop):**

If you don't already have an SSH key, generate one:

```bash
ssh-keygen -t ed25519 -C "your@email.com"
# Press enter to accept the default location (~/.ssh/id_ed25519)
# Optionally set a passphrase
```

Then copy the public key to your clipboard:

```bash
# macOS
pbcopy < ~/.ssh/id_ed25519.pub

# Linux
xclip -sel clip < ~/.ssh/id_ed25519.pub

# Or just print it and copy manually
cat ~/.ssh/id_ed25519.pub
```

In RunPod, go to **Settings → SSH Public Keys**, paste the public key, save. This is per-user — you only need to do it once.

**Per-pod (every time you deploy):**

After the pod is running, click **Connect** on the pod and find the **SSH over exposed TCP** section (not "Basic SSH Terminal" — that's the limited version, see Option C). It'll show a command like:

```bash
ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519
```

Copy and run that in your terminal. You're now in the pod's shell. Continue with step 5.

**Using VS Code or Cursor:** install the **Remote - SSH** extension, then **Cmd/Ctrl-Shift-P → Remote-SSH: Connect to Host → Add New SSH Host**, paste the same command. After it connects, **File → Open Folder → /workspace/Labs/gpt2-tinygrad**. You're now editing files on the pod with full editor features.

**Using JetBrains Gateway:** new SSH connection, point at the same host/port/key, pick the project directory after connecting.

#### Option C — Web Terminal (basic shell, no extras)

The simplest option. Click **Start Web Terminal** and then **Connect to Web Terminal** to open a bare shell in your browser. No file browser, no editor — you'd edit files via `nano` or `vim` from the shell. Mostly useful for quick poking around or as a fallback if Jupyter Lab is misbehaving.

This is sometimes labelled just "SSH" in the RunPod UI, but it's not a real SSH connection — it doesn't support SCP, SFTP, or external editor integrations. For those, use Option B.

### 5. Set up the lab on the pod

Whichever connection option you picked, you should now have a shell on the pod. Run:

```bash
# uv isn't preinstalled, so install it first.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone into the persistent volume so your work survives pod stops.
cd /workspace
git clone https://github.com/The-Fitzwilliam-AI-Circle/Labs.git
cd Labs
uv sync
```

Verify the GPU is visible to tinygrad before you start implementing:

```bash
uv run python -c "from tinygrad import Tensor; print(Tensor.ones(10).realize().device)"
# Should print: CUDA
```

If it prints `CLANG` or `CPU`, the template's `CUDA=1` env var isn't being respected. Run `unset CLANG` and try again, then ping Neil if it still doesn't work.

### 6. Edit and run

Edit `skeleton_easy.py` using whichever method matches your connection choice — Jupyter Lab's file browser, your local VS Code/Cursor over SSH, or terminal-based editors like vim/nano. The file lives at `/workspace/Labs/gpt2-tinygrad/skeleton_easy.py`.

To run training (in any pod terminal):

```bash
uv run python train.py                      # default: easy
uv run python train.py --difficulty medium
uv run python train.py --difficulty hard
```

That command will fail until you've implemented enough of your chosen skeleton to make a forward pass go through. That's expected — see the roadmap below.

### 7. When you're done — STOP THE POD

This is the only step that costs you (well, Neil) money to forget. The team's RunPod credit is shared, so a forgotten running pod burns credit that other people would have used.

On the **Pods** page, click **Stop** on your pod. **Stop ≠ Terminate:**

- **Stop** pauses billing for compute and keeps your `/workspace` volume around (~$0.10/GB/hr storage rate, trivial). You can resume later and pick up where you left off.
- **Terminate** wipes the pod *and* the volume. Only do this if you've pushed your work to GitHub.

For a single-session lab, **stop** when you finish. If you're not coming back, **terminate** after pushing.

**Note on SSH and stopped pods:** when you stop and resume a pod, RunPod gives it a new SSH host/port. Your public key persists in your RunPod account, but you'll need to copy the new SSH command from the **Connect** panel after each restart. VS Code Remote-SSH users will need to update the host entry in `~/.ssh/config` (or just delete and re-add the host).

## Setup — your own laptop (the fallback path)

If you want to keep tinkering after the session, or you're prepping ahead of time and don't have a RunPod pod yet. You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# You should already have cloned the repo from the "Before the session" step.
cd Labs
uv sync
uv run python train.py
```

**Apple Silicon caveat (handled for you).** On macOS, tinygrad defaults to the Metal backend, which can hit a shader-buffer limit (31 buffers per kernel) on the kernels this model fuses. The skeleton has a platform-guarded fallback at the top of the file:

```python
if platform.system() == "Darwin":
    os.environ["DEV"] = "CLANG"   # CPU backend on Mac
```

So macOS uses CPU and Linux/CUDA (RunPod) uses GPU automatically. CPU is plenty fast for the toy smoke tests, but full training on TinyShakespeare will be slow on a laptop — RunPod is the path that produces results in 10 minutes.

If you'd like to try Metal anyway (it may be fine on recent tinygrad), comment that block out and run; if you see "shader buffers" or "fuser" errors, put it back.

**First-run kernel compile.** Tinygrad JIT-compiles a CUDA/Metal/Clang kernel for each unique tensor op pattern the first time it sees one. Expect the first forward pass to hang for 10–60 seconds while it does this. **It's not stuck.** Subsequent forward passes reuse the compiled kernels and are fast. The same applies on the first backward pass and the first call to `greedy_generate`. You'll see roughly three "long pauses" during a training+sampling run, then everything cooks.

## Implementation roadmap

Implement in this order. Each step has a checkpoint — a way to verify it works before moving on. The smoke test at the bottom of your skeleton runs a tiny config end-to-end; you can run `uv run python skeleton_easy.py` (or `_medium.py`, etc.) after each step to see what still breaks.

The roadmap below is described in terms of the easy skeleton. Medium and hard expand the same six steps with extra implementation work — the docstring at the top of each file lists exactly what's stubbed for that difficulty.

### Step 1 — Token + positional embeddings

Implement `TokenEmbedding.__call__` and `LearnedPositionalEmbedding.__call__`.

**You'll know it works when:** with a fresh `GPT2(cfg)` and tokens of shape `(B, T)`, `model.token_embed(tokens).shape == (B, T, d_model)` and the positional embedding adds something of the same shape.

### Step 2 — Scaled dot-product attention + multi-head attention

Implement `scaled_dot_product_attention` and `MultiHeadAttention.__call__`.

**You'll know it works when:** the causal mask actually masks. Build two batches that share a prefix but differ in the suffix; pass them through the attention with the causal mask; the activations over the shared prefix should match. If they don't, your mask is wrong or applied in the wrong place.

### Step 3 — Position-wise feed-forward + decoder block

Implement `PositionwiseFeedForward.__call__` and `DecoderBlock.__call__`. Remember pre-norm, not post-norm.

**You'll know it works when:** a single-layer model trains on TinyShakespeare and the loss drops over the first ~50 steps. If it sits at ~10.8 forever, the residual or the norm is in the wrong place.

### Step 4 — Stack blocks + final layer norm + tied unembedding

Implement `GPT2.forward` (and `GPT2.__call__`). The output projection is tied to the input embedding — use `token_embed.weight.transpose()`, don't introduce a new `Linear`.

**You'll know it works when:** `model(tokens, causal_mask).shape == (B, T, vocab_size)`. The smoke test at the bottom of your skeleton checks this.

### Step 5 — Loss + train

Implement `lm_loss`. Then run `uv run python train.py`.

**You'll know it works when:** the loss falls from ~10.8 (uniform random over 50,257 tokens is `ln(50257) ≈ 10.83`) to ~4.x within a few hundred steps. If it stays flat, the loss masking is probably wrong or your gradients aren't flowing.

### Step 6 — Generate

Implement `greedy_generate`. `train.py` calls it after training and prints the result.

**You'll know it works when:** the output is recognisably English-shaped — you'll see real words, the rhythm of Shakespeare's character names ("KING", "GLOUCESTER:"), and roughly sensible punctuation. It won't be coherent. That's fine.

## Hyperparameters

Use these. They train in ~10 minutes on a 4090 and produce recognisable Shakespeare:

```
d_model    = 384
n_heads    = 6
n_layers   = 6
d_ff       = 1536       # 4 * d_model
max_seq_len = 256
batch_size = 32
n_steps    = 2000
```

These are the defaults in `GPT2Config` and `train.py`. **Don't change them until you have a working baseline.** The full GPT-2 124M config (d_model=768, n_layers=12) will not converge to anything useful in 10 minutes and will mostly just confuse you about whether your implementation is correct.

## Dataset

`tinyshakespeare.txt` is the concatenation of Shakespeare's plays — about 1.1MB of plain text, ~300k tokens after BPE. It's already in the repo. The tokeniser is OpenAI's GPT-2 BPE via [tiktoken](https://github.com/openai/tiktoken):

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
```

`train.py` handles loading and tokenisation; you don't need to touch it.

## Stretch goals

If you finish early, in rough order of usefulness:

- **KV cache for generation.** Right now `greedy_generate` re-runs the whole forward pass each step — O(T²) total. Cache the K and V tensors per layer so each new token costs O(T).
- **Gradient clipping.** Add `||grad||_2 < 1.0` clipping before `optimizer.step()`. Stabilises long training runs.
- **Decay / no-decay parameter groups.** AdamW is currently applied uniformly — the standard refinement is to exclude biases and LayerNorm gains from weight decay. Splits cleanly with `get_parameters`.
- **Try TinyStories.** Swap the dataset for [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and see how outputs change with a different distribution.
- **Scale up to GPT-2 124M.** `d_model=768, n_heads=12, n_layers=12, max_seq_len=1024`. Will need a longer training run.

## Stuck?

Things that catch people most often:

- **Pod template doesn't appear in the dropdown.** You're in your personal account, not the team account. Top-left dropdown → switch to Fitzwilliam AI Circle.
- **GPU isn't being used (on RunPod).** Check `Tensor.ones(10).realize().device` — should print `CUDA`. If it says `CLANG` or `CPU`, run `unset CLANG` in the same shell session and retry. The template sets `CUDA=1`, so this only happens if you've shadowed it somehow.
- **`tinygrad.codegen` errors mentioning shader buffers on macOS** (laptop path only). You're on Metal. The platform guard at the top of the skeleton should be activating the CPU backend — make sure you haven't deleted or moved the `if platform.system() == "Darwin": os.environ["DEV"] = "CLANG"` block, and that it sits *before* the tinygrad import.
- **First forward pass hangs for ~30s.** That's the JIT kernel compile, not a bug. See the warning above the roadmap. Wait it out.
- **Loss is NaN.** Lower the learning rate (try 1e-4 instead of 2.5e-4) and/or add gradient clipping. Often happens if the softmax sees `-inf` everywhere because the mask was built wrong.
- **Loss starts at ~10.8 and stays there.** Your model is producing uniform-random logits. Most common cause: forward pass not actually wired through (e.g. embedding output not flowing into blocks, or `ln_f` skipped, or unembedding weight not tied correctly).
- **Pod won't deploy / "no GPUs available".** RTX 4090s in Secure Cloud occasionally fill up. Either pick a different region from the dropdown, switch to Community Cloud (slightly less reliable but cheaper), or fall back to RTX 3090 / RTX A5000.
- **My laptop went to sleep mid-training run.** Training is happening on the pod, not your laptop, so the run keeps going. Reopen Jupyter Lab — you may need to reconnect to the kernel from the Kernel menu — and the run is still there.
- **`ssh: Permission denied (publickey)`.** Your public key isn't on your RunPod account, or you're trying to use the wrong private key. Check **Settings → SSH Public Keys** in RunPod, then run with the `-i` flag matching the key you uploaded: `ssh ... -i ~/.ssh/id_ed25519`.
- **VS Code Remote-SSH connection fails after a pod restart.** RunPod gives the pod a new host/port on each restart. Either delete the old host entry from `~/.ssh/config` and re-add it from the new Connect panel, or update it in place.

If none of those, ask Neil.