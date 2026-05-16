# %% [markdown]
# # Scaling Lab — Fitting Notebook (plain .py mirror of scaling_fit.ipynb)
#
# Fitzwilliam AI Circle, 'Scaling' month.
#
# This is the plain-script mirror of `scaling_fit.ipynb` for anyone who prefers
# a .py file (VS Code interactive, PyCharm cells, plain python). The cells are
# delimited with `# %%` so VS Code / PyCharm render them as a notebook.
#
# **Cells marked `TODO` are yours to complete.** The point is that *you* do the fit.
#
# Install: `pip install pandas numpy matplotlib scipy`

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %% [markdown]
# ## 1. Load the shared dataset
#
# Put `ai_circle_scaling_dataset.csv` next to this script.
#
# CSV columns (written by nanochat's `runs/scaling_laws.sh`):
#
# ```
# flops_budget, depth, model_dim, num_params, num_scaling_params,
# num_iterations, tokens_trained, param_data_ratio, val_bpb,
# core_score, train_time_sec, seed, commit
# ```

# %%
CSV_PATH = "ai_circle_scaling_dataset.csv"
df = pd.read_csv(CSV_PATH)
print(f"{len(df)} rows | commit {df['commit'].iloc[0] if 'commit' in df else 'n/a'}")
print("Budgets:", sorted(df["flops_budget"].unique()))
print("Depths :", sorted(df["depth"].unique()))
df.head()

# %% [markdown]
# ## 2. Sanity check: the C ≈ 6ND identity
#
# Total training compute ≈ 6 × (params) × (tokens). If this doesn't hold to ~1%,
# something is wrong with the dataset and every downstream conclusion is suspect.

# %%
df["flops_check"] = 6 * df["num_scaling_params"] * df["tokens_trained"]
df["flops_ratio"] = df["flops_check"] / df["flops_budget"]
print(df[["flops_budget", "depth", "flops_ratio"]].describe().loc[["mean", "min", "max"]])
# Expect flops_ratio clustered tightly around 1.0 (within a few %).

# %% [markdown]
# ## 3. The U-curves: val_bpb vs depth, one curve per FLOPs budget
#
# Fix a compute budget. Sweep model depth. Small models train for many tokens,
# big models for few — all at the *same* total compute. One depth strikes the
# balance and reaches the lowest loss. As the budget grows, the optimal depth
# shifts right. **That shift is the Chinchilla result.**

# %%
agg = (
    df.groupby(["flops_budget", "depth"])
    .agg(val_bpb_mean=("val_bpb", "mean"),
         val_bpb_std=("val_bpb", "std"),
         core_mean=("core_score", "mean"),
         core_std=("core_score", "std"))
    .reset_index()
)

plt.figure(figsize=(8, 5))
for budget in sorted(agg["flops_budget"].unique()):
    sub = agg[agg["flops_budget"] == budget].sort_values("depth")
    plt.errorbar(sub["depth"], sub["val_bpb_mean"], yerr=sub["val_bpb_std"],
                 marker="o", capsize=3, label=f"{budget:.0e} FLOPs")
plt.xlabel("depth")
plt.ylabel("val_bpb (validation loss, bits/byte)")
plt.title("U-curves: each budget has one compute-optimal depth")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 4. TODO — Extract the compute-optimal frontier

# %%
# TODO: build a DataFrame `frontier` with columns ['flops_budget', 'best_val_bpb'].
# Hint: groupby flops_budget on `agg`, take the row with min val_bpb_mean.
#
# frontier = (
#     agg.loc[agg.groupby("flops_budget")["val_bpb_mean"].idxmin()]
#        .rename(columns={"val_bpb_mean": "best_val_bpb"})
#        [["flops_budget", "best_val_bpb"]]
#        .reset_index(drop=True)
# )
# print(frontier)
frontier = None  # <-- replace with your answer

# %% [markdown]
# ## 5. TODO — Fit the power law  L ≈ a · C^(−b)

# %%
def power_law(C, a, b):
    return a * np.power(C, -b)

# TODO: fit `power_law` to frontier['flops_budget'], frontier['best_val_bpb'].
# popt, _ = curve_fit(power_law, frontier["flops_budget"],
#                     frontier["best_val_bpb"], p0=[1.0, 0.05])
# a, b = popt
# print(f"a = {a:.4f},  b = {b:.5f}")
a, b = None, None  # <-- replace

# %% [markdown]
# ## 6. Plot your fit against the data

# %%
if frontier is not None and a is not None:
    C = np.array(frontier["flops_budget"], dtype=float)
    plt.figure(figsize=(8, 5))
    plt.loglog(C, frontier["best_val_bpb"], "o", ms=10, label="compute-optimal points")
    Cs = np.logspace(np.log10(C.min()) - 0.2, np.log10(C.max()) + 0.5, 100)
    plt.loglog(Cs, power_law(Cs, a, b), "--",
               label=f"fit: {a:.3f}·C^(-{b:.4f})")
    plt.xlabel("compute (FLOPs)")
    plt.ylabel("compute-optimal val_bpb")
    plt.title("The scaling law you just reproduced")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()
else:
    print("Complete sections 4 and 5 first.")

# %% [markdown]
# ## 7. The noisy twin: core_score

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for budget in sorted(agg["flops_budget"].unique()):
    sub = agg[agg["flops_budget"] == budget].sort_values("depth")
    ax[0].plot(sub["depth"], sub["val_bpb_mean"], "o-", label=f"{budget:.0e}")
    ax[1].errorbar(sub["depth"], sub["core_mean"], yerr=sub["core_std"],
                    marker="s", capsize=3, label=f"{budget:.0e}")
ax[0].set_title("val_bpb — clean"); ax[0].set_xlabel("depth"); ax[0].set_ylabel("val_bpb")
ax[1].set_title("core_score — noisy"); ax[1].set_xlabel("depth"); ax[1].set_ylabel("CORE")
for a_ in ax: a_.legend(); a_.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 8. TODO — Your contest prediction

# %%
HELD_OUT_FLOPS = None  # <-- Neil gives you this on the day, e.g. 4.5e18

# TODO:
# prediction = power_law(HELD_OUT_FLOPS, a, b)
# print(f"My predicted val_bpb = {prediction:.4f}")
# print("Reasoning: fitted L = a·C^-b on the per-budget compute-optimal minima, "
#       "extrapolated to the held-out FLOPs. val_bpb chosen over core_score "
#       "because it sits on a clean power law; core is too noisy to extrapolate.")
