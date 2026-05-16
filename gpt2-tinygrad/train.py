"""
Train the GPT-2 you wrote in skeleton_<difficulty>.py on TinyShakespeare.

Usage:
    uv run python train.py                        # default: easy
    uv run python train.py --difficulty medium
    uv run python train.py --difficulty hard

Loads tinyshakespeare.txt, BPE-tokenises it with tiktoken (the GPT-2
encoding, vocab=50257), trains for n_steps with the lab-sized config from
GPT2Config, then samples ~200 tokens from a prompt.

Loss should fall from ~10.8 (uniform random over vocab) to ~4.x within a
few hundred steps. If it stays flat, something in the model or loss is
wrong -- go back to the skeleton.
"""

import os
import platform
# Apple Silicon Metal workaround -- see the same guard in skeleton_*.py.
# On Linux/CUDA (RunPod) we want the GPU, so this only fires on Darwin.
if platform.system() == "Darwin":
    os.environ.setdefault("DEV", "CLANG")

import argparse
import importlib

import numpy as np
import tiktoken
from tinygrad import Tensor, dtypes


def _load_skeleton(difficulty: str):
    """Import the chosen skeleton module and pull out the symbols we need.

    Imports lazily so a half-empty skeleton (e.g. the hard one) only fails
    when train.py actually tries to use it, not at module-load time.
    """
    module_name = f"skeleton_{difficulty}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Couldn't import {module_name}.py. "
            f"Make sure the file exists in this directory."
        ) from e

    required = ["GPT2Config", "GPT2", "train", "greedy_generate", "make_causal_mask"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise SystemExit(
            f"{module_name}.py is missing the following symbols required by "
            f"train.py: {missing}. Implement them in your skeleton, then re-run."
        )

    return mod


def load_tokens(path: str) -> np.ndarray:
    """Read the corpus and BPE-tokenise it once into a flat int32 array."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(text)
    return np.array(ids, dtype=np.int32)


class ShakespeareDataset:
    """Streams random (B, T) windows out of a flat token array.

    Each step yields:
        tokens_in:   (B, T)   int32
        tokens_out:  (B, T)   int32, the next-token target (shifted by 1)
        causal_mask: (1, 1, T, T) additive mask
    """
    def __init__(self, tokens: np.ndarray, batch_size: int, seq_len: int,
                 n_batches: int, make_causal_mask):
        self.tokens = tokens
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_batches = n_batches
        self.mask = make_causal_mask(seq_len)

    def __iter__(self):
        N = self.tokens.shape[0]
        for _ in range(self.n_batches):
            # Random start index per sequence, leaving room for the +1 shift.
            starts = np.random.randint(0, N - self.seq_len - 1, size=self.batch_size)
            chunks = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts])
            tokens_in  = Tensor(chunks[:, :-1], dtype=dtypes.int32)
            tokens_out = Tensor(chunks[:, 1:],  dtype=dtypes.int32)
            yield tokens_in, tokens_out, self.mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--difficulty", default="easy", choices=["easy", "medium", "hard"],
        help="Which skeleton_<difficulty>.py to import the model from "
             "(default: easy).",
    )
    args = parser.parse_args()

    print(f"Loading skeleton: skeleton_{args.difficulty}.py")
    mod = _load_skeleton(args.difficulty)
    GPT2Config       = mod.GPT2Config
    train            = mod.train
    greedy_generate  = mod.greedy_generate
    make_causal_mask = mod.make_causal_mask

    cfg = GPT2Config(
        vocab_size=50257,
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
        max_seq_len=256,
        dropout=0.1,
    )

    batch_size = 32
    n_steps = 2000

    print(f"Loading and tokenising tinyshakespeare.txt...")
    tokens = load_tokens("tinyshakespeare.txt")
    print(f"  {tokens.shape[0]} tokens (BPE, gpt2 encoding)")

    dataset = ShakespeareDataset(
        tokens=tokens,
        batch_size=batch_size,
        seq_len=cfg.max_seq_len,
        n_batches=n_steps,
        make_causal_mask=make_causal_mask,
    )

    print(f"Training: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, batch={batch_size}, seq_len={cfg.max_seq_len}, "
          f"steps={n_steps}")
    model = train(cfg, dataset, n_steps=n_steps)

    print("\n=== Sampling ===")
    enc = tiktoken.get_encoding("gpt2")
    prompt_text = "ROMEO:"
    prompt_ids = enc.encode(prompt_text)
    prompt = Tensor(np.array([prompt_ids], dtype=np.int32), dtype=dtypes.int32)

    generated = greedy_generate(model, prompt, max_new_tokens=200)
    out_ids = generated.numpy()[0].tolist()
    print(enc.decode(out_ids))


if __name__ == "__main__":
    main()
