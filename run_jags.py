"""
JAGS logistic-mix2 sampler for genetic information estimation.

Implements logistic_mix2_mvnorm in JAGS via model_mix2.jags.
Uses reduce_to_relatives() to work on the small relatives-only submatrix,
then runs JAGS via the CLI (jags_utils.py — no pyjags required).

Install requirements:
    sudo apt install jags   (or conda install -c conda-forge jags)
"""

import numpy as np
from geneticinfo_functions import (
    simulate_casecontrol_related,
    print_casecontrol_summary,
    reduce_to_relatives,
)
from jags_utils import run_jags

# ------------------------------------------------------------------ #
# 1.  Simulate case-control data
# ------------------------------------------------------------------ #
print("Simulating case-control data ...")
y_np, A_np, L_np, info = simulate_casecontrol_related(
    n_fullsib_pairs=400_000,
    n_fullsib_trips=200_000,
    n_halfsib_pairs=40_000,
    n_unrelated=4_000,
    mu=1.0,
    seed=42,
)
print_casecontrol_summary(info)

# ------------------------------------------------------------------ #
# 2.  Reduce to relatives-only subsample
# ------------------------------------------------------------------ #
y_red, A_red, L_red, kept_idx, block_sizes = reduce_to_relatives(A_np, y_np)
M_red = len(y_red)
p_obs_full = float(y_np.mean())     # prevalence in the full case-control sample
p_obs_red  = float(y_red.mean())    # prevalence in the relatives-only subsample

print(f"\n=== Relatives-only subsample ===")
print(f"  M_red        = {M_red}")
print(f"  Block sizes  = {sorted(set(block_sizes), reverse=True)}")
print(f"  n_blocks     = {len(block_sizes)}")
print(f"  p_obs (full) = {p_obs_full:.4f}")
print(f"  p_obs (red)  = {p_obs_red:.4f}")

logit_pobs = float(np.log(p_obs_full / (1.0 - p_obs_full)))

# ------------------------------------------------------------------ #
# 3.  Assemble JAGS data
# ------------------------------------------------------------------ #
jags_data = dict(
    M         = M_red,
    L         = L_red,                       # M_red x M_red Cholesky factor
    y         = y_red.astype(float),         # binary outcomes
    p_obs     = p_obs_full,                  # mixing proportion prior
    logit_pobs= logit_pobs,
    tau_beta0 = 1.0 / 25.0,                 # beta0 prior precision (sd=5)
    tau_s     = 1.0 / 25.0,                 # s prior precision (sd=5, T>0)
    zero_M    = np.zeros(M_red),             # reserved for dmnorm variant
)

# Initial values — start W near the class-conditional means
inits_fn = lambda: dict(
    s     = float(np.abs(np.random.normal(1.0, 0.3))),
    beta0 = logit_pobs + float(np.random.normal(0, 0.1)),
    z     = y_red.astype(float),            # initialise class = observed status
    W     = np.where(y_red == 1, 0.5, -0.5).astype(float),
)

# ------------------------------------------------------------------ #
# 4.  Run JAGS
# ------------------------------------------------------------------ #
n_chains  = 4
n_warmup  = 2_000
n_samples = 2_000

print(f"\nRunning JAGS ({n_chains} chains, {n_warmup} warmup + {n_samples} samples) ...")

samples = run_jags(
    model_file = "model_mix2.jags",
    data       = jags_data,
    inits      = [inits_fn() for _ in range(n_chains)],
    monitors   = ["mu", "s", "beta0"],
    n_adapt    = 1000,
    n_burnin   = n_warmup,
    n_samples  = n_samples,
    n_chains   = n_chains,
    verbose    = True,
)

# ------------------------------------------------------------------ #
# 5.  Posterior summaries
# ------------------------------------------------------------------ #
mu_all  = samples["mu"].flatten()       # shape: (n_samples * n_chains,)  [after flatten]
s_all   = samples["s"].flatten()
b0_all  = samples["beta0"].flatten()

print(f"\n=== Posterior summaries (true mu = {info['mu']:.4f}) ===")
for name, arr in [("mu", mu_all), ("s", s_all), ("beta0", b0_all)]:
    lo, hi = np.percentile(arr, [5, 95])
    print(f"  {name:6s}: mean={arr.mean():.4f}  sd={arr.std():.4f}  "
          f"90% CI=[{lo:.4f}, {hi:.4f}]")

# Per-chain R-hat (simple version)
mu_chains = samples["mu"]              # shape: (n_samples, n_chains)
B = mu_chains.mean(axis=0).var(ddof=1) * n_samples     # between-chain var
W_var = mu_chains.var(axis=0, ddof=1).mean()            # within-chain var
V_hat = W_var * (n_samples - 1) / n_samples + B / n_samples
r_hat = np.sqrt(V_hat / W_var) if W_var > 0 else float("nan")
print(f"\n  R-hat (mu): {r_hat:.4f}  (target < 1.01)")

# ------------------------------------------------------------------ #
# 6.  Trace + density plot
# ------------------------------------------------------------------ #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = plt.cm.tab10.colors

for c in range(n_chains):
    axes[0].plot(mu_chains[:, c], linewidth=0.4, alpha=0.8,
                 color=colors[c], label=f"chain {c+1}")
axes[0].axhline(info["mu"], color="red", linestyle="--", lw=1.5,
                label=f"true mu={info['mu']:.2f}")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("mu")
axes[0].set_title("Trace of mu (JAGS)")
axes[0].legend(fontsize=7)

axes[1].hist(mu_all, bins=60, density=True, alpha=0.6,
             edgecolor="black", linewidth=0.3)
axes[1].axvline(info["mu"], color="red", linestyle="--", lw=1.5,
                label=f"true mu={info['mu']:.2f}")
axes[1].set_xlabel("mu")
axes[1].set_ylabel("Density")
axes[1].set_title("Posterior of mu (JAGS, all chains)")
axes[1].legend()

fig.tight_layout()
fig.savefig("mu_jags.png", dpi=150)
print("\nPlot saved to mu_jags.png")
