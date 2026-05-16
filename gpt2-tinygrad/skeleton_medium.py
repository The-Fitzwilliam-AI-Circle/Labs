"""
GPT-2 from scratch -- ARIA Circle lab skeleton (medium difficulty).

Implement the bodies of the methods marked `raise NotImplementedError(...)`,
in roughly this order:

    Step 1:  GPT2Config (dataclass fields + d_k property)
             make_causal_mask
             TokenEmbedding.__call__
             LearnedPositionalEmbedding (__init__ + __call__)
    Step 2:  scaled_dot_product_attention
             MultiHeadAttention (__init__ + __call__)
    Step 3:  PositionwiseFeedForward (__init__ + __call__)
             DecoderBlock (__init__ + __call__)
    Step 4:  GPT2 (__init__ + forward + __call__)
    Step 5:  lm_loss
             CosineScheduleWithWarmup (__init__ + lr)
             make_optimizer
             train_step
    Step 6:  greedy_generate

Already implemented for you (don't touch unless you want to):
    - _init_weights (GPT-2's depth-aware init scheme)
    - ToyLMDataset (used only by the smoke test in this file)
    - train (the outer training loop -- relies on the things above)
    - the smoke test at the bottom

The smoke test runs a tiny config end-to-end. Run this file
(`uv run python skeleton_medium.py`) at any point to see how far you've
got -- it'll fail loudly until everything is in place.

The real training run on TinyShakespeare lives in train.py.
"""

import os
import platform
# On macOS, Metal caps shader buffers at 31 per kernel; tinygrad's fuser
# can produce kernels above that limit for this model, so we fall back
# to the CPU (Clang) backend. On Linux/CUDA (e.g. RunPod) we want the
# GPU, so this only fires on Darwin. If Metal works fine on your machine
# you can comment this out -- it's defensive.
if platform.system() == "Darwin":
    os.environ["DEV"] = "CLANG"

from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
import numpy as np
from dataclasses import dataclass
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Config -- lab-sized defaults (trains in ~10 min on a 4090)
# ---------------------------------------------------------------------------
# For reference, the real GPT-2 sizes are scalings of n_layers / n_heads / d_model:
#   small  124M:  n_layers=12, n_heads=12, d_model=768   (the released "GPT-2")
#   medium 355M:  n_layers=24, n_heads=16, d_model=1024
#   large  774M:  n_layers=36, n_heads=20, d_model=1280
#   xl    1558M:  n_layers=48, n_heads=25, d_model=1600
# Don't switch to one of these until you have the lab-sized model converging.
# ---------------------------------------------------------------------------

@dataclass
class GPT2Config:
    """Hyperparameters for the model."""
    # TODO Step 1: declare the dataclass fields above and add a d_k property.
    pass


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TokenEmbedding:
    """Plain embedding matrix, no sqrt(d_model) scaling.

    Weight shape: (vocab_size, d_model).

    __call__:
        x:      (B, T)            int token ids in [0, vocab_size)
        return: (B, T, d_model)   float
    """
    def __init__(self, vocab_size: int, d_model: int):
        self.embedding = nn.Embedding(vocab_size, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        # x: (B, T) int -> (B, T, d_model)
        raise NotImplementedError("Step 1: implement token embedding lookup")
    
class LearnedPositionalEmbedding:
    """Learned absolute position embeddings (the GPT-2 design).

    Weight shape: (max_len, d_model).

    __call__:
        x:      (B, T, d_model)   float token embeddings
        return: (B, T, d_model)   float, x + pos_emb after dropout
        Requires T <= max_len.
    """
    def __init__(self, max_len: int, d_model: int, dropout: float):
        # Store self.embedding (an nn.Embedding(max_len, d_model)) and
        # self.dropout (the float dropout rate). _init_weights expects
        # self.embedding.weight to exist.
        raise NotImplementedError("Step 1: build positional embedding state")

    def __call__(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model) -> same shape
        # Build a (B, T) tensor of positions [0..T-1], embed it, add to x,
        # then apply dropout.
        raise NotImplementedError("Step 1: apply positional embedding")


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None,
    dropout: float = 0.0,
) -> Tensor:
    """Scaled dot-product attention.

    Inputs:
        q:    (B, h, T_q, d_k)
        k:    (B, h, T_k, d_k)
        v:    (B, h, T_k, d_v)            usually d_v == d_k
        mask: broadcastable to (B, h, T_q, T_k), additive (0 keep, -inf block).
              Often passed as (1, 1, T, T).

    Returns:
        (B, h, T_q, d_v)
    """
    # scores = q @ k^T / sqrt(d_k); add mask if given; softmax over the last
    # axis; dropout the attention weights; then weight v by them.
    raise NotImplementedError("Step 2: implement scaled dot-product attention")


class MultiHeadAttention:
    """Self-attention only, always causal.

    Linear weights w_q / w_k / w_v / w_o each have shape (d_model, d_model).
    w_o is a residual-stream write -- gets a smaller init in _init_weights.

    __call__:
        x:    (B, T, d_model)
        mask: broadcastable to (B, n_heads, T, T), additive (0 keep, -inf block);
              typically (1, 1, T, T) from make_causal_mask.
        return: (B, T, d_model)

    Internal heads run on (B, n_heads, T, d_k) where d_k = d_model // n_heads.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        # Assert d_model % n_heads == 0. Store d_model, n_heads, d_k, dropout.
        # Build four nn.Linear(d_model, d_model): w_q, w_k, w_v, w_o.
        # _init_weights expects these names exactly.
        raise NotImplementedError("Step 2: build multi-head attention state")

    def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Project to Q, K, V; reshape to (B, n_heads, T, d_k); call
        # scaled_dot_product_attention; recombine heads back to (B, T, d_model);
        # project out with w_o.
        raise NotImplementedError("Step 2: implement multi-head attention")


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class PositionwiseFeedForward:
    """Two linear transforms with GELU in between.

    tinygrad's .gelu() is the tanh approximation, matching GPT-2.

    Weights:
        w_1: (d_model, d_ff)   bias (d_ff,)
        w_2: (d_ff, d_model)   bias (d_model,)
    w_2 is a residual-stream write -- gets scaled init.

    __call__:
        x:      (B, T, d_model)
        return: (B, T, d_model)
        Internal hidden activation has shape (B, T, d_ff).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        # Build two nn.Linear with bias=True: w_1 (d_model -> d_ff) and
        # w_2 (d_ff -> d_model). Store self.dropout. _init_weights expects
        # these names.
        raise NotImplementedError("Step 3: build FFN state")

    def __call__(self, x: Tensor) -> Tensor:
        # w_1 -> gelu -> dropout -> w_2.
        raise NotImplementedError("Step 3: apply position-wise FFN")


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock:
    """One GPT-2 block: pre-norm self-attention + pre-norm FFN, residuals on top.

    Pre-norm:
        x = x + sublayer(LayerNorm(x))
    (vs GPT-1's post-norm: x = LayerNorm(x + sublayer(x)).)

    The residual stream is now the unnormalised highway running end-to-end.
    Each sublayer reads a normalised view of it and writes back into the raw
    stream.

    __call__:
        x:           (B, T, d_model)
        causal_mask: broadcastable to (B, n_heads, T, T); typically (1, 1, T, T).
        return:      (B, T, d_model)
    """
    def __init__(self, cfg: GPT2Config):
        # Build self.norm_1 (LayerNorm over d_model), self.self_attention
        # (MultiHeadAttention), self.norm_2 (LayerNorm), self.ffn
        # (PositionwiseFeedForward), self.dropout (the float).
        # _init_weights expects these exact attribute names.
        raise NotImplementedError("Step 3: build the decoder block state")

    def __call__(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        # Pre-norm attention residual, then pre-norm FFN residual.
        # Apply self.dropout to each sublayer output before adding.
        raise NotImplementedError("Step 3: apply the pre-norm decoder block")


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GPT2:
    """Decoder-only Transformer language model.

    Forward pass:
        tokens -> embed -> + pos -> N pre-norm blocks -> final LayerNorm
               -> tie-weighted projection to vocab logits.

    The output projection is tied to the input embedding -- there is no
    separate out_proj parameter. Use token_embed.weight.transpose() to
    project final hidden states back to vocab.

    Forward I/O:
        tokens:      (B, T)            int32 token ids in [0, vocab_size).
                                       Requires T <= cfg.max_seq_len.
        causal_mask: broadcastable to (B, n_heads, T, T), additive
                     (0 keep, -inf block); typically (1, 1, T, T).
        return:      (B, T, vocab_size)  float logits (pre-softmax).
    """
    def __init__(self, cfg: GPT2Config):
        # Build self.cfg, self.token_embed, self.pos_embed, self.blocks
        # (a list of DecoderBlock), and self.ln_f (a final LayerNorm).
        # Then call _init_weights(self) to apply the GPT-2 init scheme.
        raise NotImplementedError("Step 4: assemble the GPT2 model")

    def forward(self, tokens: Tensor, causal_mask: Tensor) -> Tensor:
        """Run a forward pass.

        tokens:      (B, T) int32 in [0, vocab_size).
        causal_mask: broadcastable to (B, n_heads, T, T); typically (1, 1, T, T).
        return:      (B, T, vocab_size) float logits.
        """
        # Embed tokens, add positions, run through self.blocks, apply ln_f,
        # project to vocab using the tied embedding weight.
        raise NotImplementedError("Step 4: implement the forward pass")

    def __call__(self, tokens: Tensor, causal_mask: Tensor) -> Tensor:
        raise NotImplementedError("Step 4: delegate to self.forward")


# ---------------------------------------------------------------------------
# Initialisation -- already implemented
# ---------------------------------------------------------------------------

def _init_weights(model: "GPT2") -> None:
    """GPT-2 init scheme.

    All Linear / Embedding weights ~ N(0, 0.02), biases zero, LayerNorm
    gamma=1 / beta=0 (which is tinygrad's default, so we leave it alone).

    The residual-write projections -- w_o in each attention block and w_2
    in each FFN -- are re-initialised with std = 0.02 / sqrt(2 * n_layers).
    Each layer makes two writes to the residual stream; without this
    down-scaling, activation variance grows linearly with depth.
    """
    std = 0.02
    n_layers = model.cfg.n_layers

    # Token + position embeddings
    model.token_embed.embedding.weight.assign(
        Tensor.normal(*model.token_embed.embedding.weight.shape, mean=0.0, std=std)
    )
    model.pos_embed.embedding.weight.assign(
        Tensor.normal(*model.pos_embed.embedding.weight.shape, mean=0.0, std=std)
    )

    # Per-block linears
    for block in model.blocks:
        for lin in (block.self_attention.w_q,
                    block.self_attention.w_k,
                    block.self_attention.w_v,
                    block.ffn.w_1):
            lin.weight.assign(Tensor.normal(*lin.weight.shape, mean=0.0, std=std))
            if lin.bias is not None:
                lin.bias.assign(Tensor.zeros(*lin.bias.shape))

        # Residual-write projections: scaled std
        scaled_std = std / math.sqrt(2 * n_layers)
        for lin in (block.self_attention.w_o, block.ffn.w_2):
            lin.weight.assign(Tensor.normal(*lin.weight.shape, mean=0.0, std=scaled_std))
            if lin.bias is not None:
                lin.bias.assign(Tensor.zeros(*lin.bias.shape))


# ---------------------------------------------------------------------------
# Masks -- already implemented
# ---------------------------------------------------------------------------

def make_causal_mask(size: int) -> Tensor:
    """Build an additive causal mask of shape (1, 1, T, T):
    -inf above the diagonal, 0 on and below. Broadcasts over (B, n_heads, T, T)
    when added to attention scores.
    """
    # One way: build (size, 1) and (1, size) index grids, compare to find
    # "future" positions, use Tensor.where to put -inf there and 0 elsewhere,
    # then reshape to (1, 1, T, T).
    raise NotImplementedError("Step 1: build the causal mask")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def lm_loss(logits: Tensor, target: Tensor, pad_idx: int) -> Tensor:
    """Cross-entropy for next-token prediction. No label smoothing.

    Args:
        logits:  (B, T, V) float, pre-softmax.
        target:  (B, T)    int  token ids in [0, V).
        pad_idx: int, positions where target == pad_idx are excluded from the mean.

    Returns:
        Scalar Tensor (shape ()) -- mean NLL over non-pad positions.
    """
    # log_softmax along the vocab axis, gather the target log-probs (one_hot
    # works fine here), mask out positions where target == pad_idx, return
    # the mean over the remaining positions.
    raise NotImplementedError("Step 5: implement language-modelling cross-entropy")


# ---------------------------------------------------------------------------
# Optimiser + schedule -- already implemented
# ---------------------------------------------------------------------------

class CosineScheduleWithWarmup:
    """Linear warmup -> cosine anneal to 0.

    lr(step):
        step:   int   current training step (0-indexed).
        return: float learning rate for that step.
    """
    def __init__(self, max_lr: float, warmup_steps: int, total_steps: int):
        # Stash the three numbers; .lr(step) does the actual computation.
        raise NotImplementedError("Step 5: store schedule parameters")

    def lr(self, step: int) -> float:
        # If step < warmup_steps:   linear ramp from 0 -> max_lr.
        # Otherwise:                cosine anneal from max_lr -> 0 over the
        #                           remaining (total_steps - warmup_steps).
        raise NotImplementedError("Step 5: compute the per-step learning rate")


def make_optimizer(model: GPT2, lr: float = 2.5e-4, weight_decay: float = 0.01):
    """AdamW over all parameters. Weight decay is applied uniformly here;
    the standard refinement is to split into decay / no-decay groups,
    excluding biases and LayerNorm gains.
    """
    # Use get_parameters(model) to grab every trainable Tensor, and return
    # an AdamW with b1=0.9, b2=0.999, eps=1e-8.
    raise NotImplementedError("Step 5: build the AdamW optimiser")


# ---------------------------------------------------------------------------
# Toy data -- used by the smoke test below. The real TinyShakespeare loader
# lives in train.py.
# ---------------------------------------------------------------------------

class ToyLMDataset:
    """Random next-token task for the smoke test.

    Yields (tokens_in, tokens_out, causal_mask) tuples. tokens_out is
    tokens_in shifted by one; ids start at 1 so pad_idx=0 is never produced.
    """
    def __init__(self, vocab_size: int, seq_len: int, n_batches: int, batch_size: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_batches = n_batches
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            data = np.random.randint(1, self.vocab_size, (self.batch_size, self.seq_len + 1))
            tokens_in  = Tensor(data[:, :-1], dtype=dtypes.int32)
            tokens_out = Tensor(data[:, 1:],  dtype=dtypes.int32)
            causal_mask = make_causal_mask(self.seq_len)
            yield tokens_in, tokens_out, causal_mask


# ---------------------------------------------------------------------------
# Train / inference loops -- already implemented
# ---------------------------------------------------------------------------

def train_step(model: GPT2, optimizer: AdamW, batch) -> float:
    """One fwd/bwd/step pass on a single batch.

    batch: (tokens_in, tokens_out, causal_mask).
    Returns the scalar loss value as a Python float.
    """
    # Forward through the model, compute lm_loss, zero grads, backward,
    # optimizer.step(), return loss.numpy().item().
    raise NotImplementedError("Step 5: implement one training step")


def train(cfg: GPT2Config, dataset, n_steps: int, max_lr: float = 2.5e-4) -> "GPT2":
    """Build a fresh GPT2 from cfg and train for up to n_steps batches.

    dataset: any iterable yielding (tokens_in, tokens_out, causal_mask) tuples.
    Returns the trained model.
    """
    model = GPT2(cfg)
    schedule = CosineScheduleWithWarmup(
        max_lr=max_lr,
        warmup_steps=min(200, max(1, n_steps // 10)),
        total_steps=n_steps,
    )
    optimizer = make_optimizer(model, lr=0.0)

    step = 0
    with Tensor.train():
        for batch in dataset:
            if step >= n_steps:
                break

            optimizer.lr = schedule.lr(step)
            loss = train_step(model, optimizer, batch)

            if step % 50 == 0:
                print(f"  step {step:4d}  loss {loss:.4f}  lr {optimizer.lr:.2e}")

            step += 1

    return model


def greedy_generate(
    model: GPT2, prompt: Tensor, max_new_tokens: int,
) -> Tensor:
    """Naive autoregressive generation -- no KV cache.

    Args:
        prompt:         (B, T_prompt) int32 token ids.
        max_new_tokens: number of tokens to append (one per forward pass).

    Returns:
        (B, T_prompt + max_new_tokens) int32. Requires
        T_prompt + max_new_tokens <= cfg.max_seq_len.

    Stretch goal: cache K, V from earlier steps so the per-step forward is
    O(T) rather than O(T^2).
    """
    # Loop max_new_tokens times. Each step:
    #   - rebuild causal_mask for the current length
    #   - forward pass to get logits (B, T, V)
    #   - take logits at the final position, argmax over vocab
    #   - concatenate the new token onto `tokens` along dim=1
    raise NotImplementedError("Step 6: implement greedy autoregressive generation")


# ---------------------------------------------------------------------------
# Smoke test -- tiny config, runs end-to-end. Run this file to sanity-check
# your implementation. The real TinyShakespeare run is in train.py.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = GPT2Config(
        vocab_size=100,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        max_seq_len=32, dropout=0.0,
    )

    # ------------------------------------------------------------------
    # 1. Forward pass -- check output shape
    # ------------------------------------------------------------------
    print("=== 1. Forward pass shape check ===")
    model = GPT2(cfg)
    B, T = 2, 6

    tokens = Tensor(np.random.randint(1, cfg.vocab_size, (B, T)), dtype=dtypes.int32)
    causal_mask = make_causal_mask(T)

    logits = model(tokens, causal_mask)
    print(f"  logits.shape = {logits.shape}  (expected ({B}, {T}, {cfg.vocab_size}))")
    assert logits.shape == (B, T, cfg.vocab_size), f"shape mismatch: {logits.shape}"

    n_params_total = len(get_parameters(model))
    print(f"  total parameter tensors: {n_params_total}")
    print("  (no separate out_proj.weight -- output projection is token_embed.weight^T)")
    print("  PASSED\n")

    # ------------------------------------------------------------------
    # 2. Overfit on toy LM task -- loss should fall
    # ------------------------------------------------------------------
    print("=== 2. LM overfit (300 steps) ===")
    dataset = ToyLMDataset(vocab_size=20, seq_len=8, n_batches=400, batch_size=32)
    trained_model = train(cfg, dataset, n_steps=300)
    print("  Training done\n")

    # ------------------------------------------------------------------
    # 3. Greedy generate from a short prompt
    # ------------------------------------------------------------------
    print("=== 3. Greedy generation ===")
    prompt_arr = np.array([[3, 5, 7]])
    prompt = Tensor(prompt_arr, dtype=dtypes.int32)
    generated = greedy_generate(trained_model, prompt, max_new_tokens=7)
    print(f"  prompt:    {prompt_arr[0].tolist()}")
    print(f"  generated: {generated.numpy()[0].tolist()}")