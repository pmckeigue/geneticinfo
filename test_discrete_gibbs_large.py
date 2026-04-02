"""Test lr_discrete_blockdiag with DiscreteHMCGibbs on reduced relatives-only data."""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import numpy as np

num_chains = min(4, jax.device_count())
numpyro.set_host_device_count(num_chains)

from geneticinfo_functions import (
    simulate_casecontrol_related, print_casecontrol_summary,
    lr_discrete_blockdiag, reduce_to_relatives,
)

# --- Simulate ---
y_np, A_np, L_np, info = simulate_casecontrol_related(
    n_fullsib_pairs=400_000, n_halfsib_pairs=40_000, n_unrelated=4000,
)
print_casecontrol_summary(info)

M_orig = info["M"]
p_obs = float(y_np.mean())

# --- Reduce to relatives only ---
y_red, A_red, L_red, kept_idx, block_sizes = reduce_to_relatives(A_np, y_np)
M_red = len(kept_idx)
n_blocks = len(block_sizes)
block_size = block_sizes[0] if block_sizes else 0

print(f"\n=== Block decomposition ===")
print(f"  Original M:       {M_orig}")
print(f"  Reduced M:        {M_red}")
print(f"  Number of blocks: {n_blocks}")
print(f"  Block size:       {block_size}")
print(f"  Reduction ratio:  {M_red/M_orig:.3f}")

# Extract per-block Cholesky factors and group by size.
# block_sizes lists the size of each block in the order they appear in L_red.
from collections import defaultdict
L_by_size = defaultdict(list)   # size -> list of (s,s) arrays
y_by_size = defaultdict(list)   # size -> list of scalars
ind_offset = 0
y_offset   = 0
for s in block_sizes:
    L_by_size[s].append(L_red[ind_offset:ind_offset + s, ind_offset:ind_offset + s].copy())
    y_by_size[s].extend(y_red[y_offset:y_offset + s].tolist())
    ind_offset += s
    y_offset   += s

unique_sizes = sorted(L_by_size, reverse=True)   # largest first
L_list = [jnp.array(np.stack(L_by_size[s]), dtype=jnp.float64) for s in unique_sizes]
y_list = [jnp.array(y_by_size[s], dtype=jnp.float64) for s in unique_sizes]

print(f"\n{'='*60}")
print(f"lr_discrete_blockdiag with DiscreteHMCGibbs  (M_red={M_red})")
print(f"{'='*60}")

inner_kernel = NUTS(lr_discrete_blockdiag, max_tree_depth=8)
kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
print(f"  Chains:           {num_chains}  (devices: {jax.device_count()})")
mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000,
            num_chains=num_chains, chain_method="parallel")
mcmc.run(random.PRNGKey(0),
         L_list=L_list, y_list=y_list, sizes=unique_sizes, p_obs=p_obs)

samples = mcmc.get_samples()
mu_samples = samples["mu"]
s_samples = samples["s"]
p_samples = samples["p"]

print(f"\n  True: mu={info['mu']:.4f}, s={info['s']:.4f}")
print(f"  mu:  mean={float(mu_samples.mean()):.4f}, "
      f"median={float(jnp.median(mu_samples)):.4f}, "
      f"sd={float(mu_samples.std()):.4f}")
print(f"  s:   mean={float(s_samples.mean()):.4f}, "
      f"median={float(jnp.median(s_samples)):.4f}, "
      f"sd={float(s_samples.std()):.4f}")
print(f"  p:   mean={float(p_samples.mean()):.4f}, "
      f"median={float(jnp.median(p_samples)):.4f}")

# Print summary for global parameters
import io, sys
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
mcmc.print_summary(prob=0.9, exclude_deterministic=False)
sys.stdout = old
for line in buf.getvalue().splitlines():
    if any(k in line for k in ["mu ", " s ", " p ", "n_eff", "r_hat", "diverge"]):
        print("  " + line)

# --- Trace and posterior density plots for mu ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# mu_samples shape: (num_samples * num_chains,) — reshape to per-chain
mu_np = np.array(mu_samples)
n_samples_per_chain = len(mu_np) // num_chains
mu_chains = mu_np.reshape(num_chains, n_samples_per_chain)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Trace plot — one colour per chain
colors = plt.cm.tab10.colors
for c in range(num_chains):
    axes[0].plot(mu_chains[c], linewidth=0.3, alpha=0.7, color=colors[c],
                 label=f"chain {c+1}")
axes[0].axhline(info["mu"], color="red", linestyle="--", linewidth=1.5,
                label=f'true mu={info["mu"]:.2f}')
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("mu")
axes[0].set_title(f"Trace of mu ({num_chains} chains)")
axes[0].legend(fontsize=7)

# Posterior density — pooled across chains
axes[1].hist(mu_np, bins=60, density=True, alpha=0.6, edgecolor="black", linewidth=0.3)
axes[1].axvline(info["mu"], color="red", linestyle="--", linewidth=1.5,
                label=f'true mu={info["mu"]:.2f}')
axes[1].set_xlabel("mu")
axes[1].set_ylabel("Density")
axes[1].set_title(f"Posterior density of mu ({num_chains} chains pooled)")
axes[1].legend()

fig.tight_layout()
fig.savefig("mu_trace_density.png", dpi=150)
print(f"\nPlot saved to mu_trace_density.png")
