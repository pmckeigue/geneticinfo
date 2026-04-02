"""
Compare PG-Gibbs vs DiscreteHMCGibbs on the synthetic dataset.
Run with: /home/pmckeigue/venv/bin/python3 run_discrete_gibbs_synthetic.py
"""
import os, time
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"  # per GPU; 4 chains × 4 GPUs

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from jax import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

N_CHAINS  = 4
N_WARMUP  = 500
N_SAMPLES = 2000

numpyro.set_host_device_count(N_CHAINS)
print(f"JAX devices: {jax.devices()}")
print(f"Using {N_CHAINS} chains × {N_WARMUP} warmup + {N_SAMPLES} samples\n")

# ── 1. Synthetic data ─────────────────────────────────────────────────────────
import compress_dataset, geninfo

L_syn, y_syn = compress_dataset.generate_synthetic("compressed_model.npz", seed=0)
M = L_syn.shape[0]
print(f"Synthetic data: M={M}  prevalence={y_syn.mean():.4f}\n")

# ── 2. PG-Gibbs ───────────────────────────────────────────────────────────────
print("=" * 60)
print("PG-Gibbs (CPU, block-diagonal)")
print("=" * 60)
t0 = time.perf_counter()
result_pg = geninfo.sample_posterior(
    L_syn, y_syn,
    corr_threshold=0.025,
    n_warmup=N_WARMUP,
    n_samples=N_SAMPLES,
    n_chains=N_CHAINS,
    n_blas_threads=1,
    progress_bar=False,
)
t_pg = time.perf_counter() - t0
print(f"  Wall time: {t_pg:.1f} s\n")

pg_mu    = np.stack([d["mu"]    for d in result_pg["chain_dicts"]])  # (4, N_SAMPLES)
pg_beta0 = np.stack([d["beta0"] for d in result_pg["chain_dicts"]])
pg_p     = np.stack([d["p"]     for d in result_pg["chain_dicts"]])
pg_ll    = np.stack([d["ll"]    for d in result_pg["chain_dicts"]])

# ── 3. DiscreteHMCGibbs ───────────────────────────────────────────────────────
print("=" * 60)
print("DiscreteHMCGibbs + NUTS (GPU, full L matrix)")
print("=" * 60)

from geneticinfo_functions import lr_discrete

L_jax = jnp.array(L_syn, dtype=jnp.float64)
y_jax = jnp.array(y_syn, dtype=jnp.float64)

inner_kernel = NUTS(lr_discrete, max_tree_depth=8)
kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
mcmc = MCMC(kernel,
            num_warmup=N_WARMUP,
            num_samples=N_SAMPLES,
            num_chains=N_CHAINS,
            chain_method="parallel",
            progress_bar=True)

t0 = time.perf_counter()
mcmc.run(random.PRNGKey(0),
         M=M, L=L_jax, y=y_jax,
         muprior=dist.HalfCauchy(1.0))  # match PG-Gibbs prior
t_dhg = time.perf_counter() - t0
print(f"\n  Wall time: {t_dhg:.1f} s\n")

samples = mcmc.get_samples(group_by_chain=True)
dhg_mu   = np.array(samples["mu"])    # (4, N_SAMPLES)
dhg_p    = np.array(samples["p"])
dhg_s    = np.array(samples["s"])     # sqrt(2*mu), deterministic
dhg_beta0 = np.log(dhg_p / (1.0 - dhg_p))  # logit(p)

# ── 4. Summary statistics ────────────────────────────────────────────────────
import arviz as az

def ess_rhat(chains):
    """chains: (n_chains, n_samples)"""
    idata = az.convert_to_inference_data({"mu": chains})
    summ  = az.summary(idata, var_names=["mu"])
    return float(summ["ess_bulk"].iloc[0]), float(summ["r_hat"].iloc[0])

pg_ess,  pg_rhat  = ess_rhat(pg_mu)
dhg_ess, dhg_rhat = ess_rhat(dhg_mu)

print(f"{'':30s}  {'PG-Gibbs':>12}  {'DiscHMCGibbs':>12}")
print(f"{'Wall time (s)':30s}  {t_pg:>12.1f}  {t_dhg:>12.1f}")
print(f"{'mu mean':30s}  {pg_mu.mean():>12.4f}  {dhg_mu.mean():>12.4f}")
print(f"{'mu median':30s}  {np.median(pg_mu):>12.4f}  {np.median(dhg_mu):>12.4f}")
print(f"{'mu sd':30s}  {pg_mu.std():>12.4f}  {dhg_mu.std():>12.4f}")
print(f"{'ESS_bulk (mu)':30s}  {pg_ess:>12.0f}  {dhg_ess:>12.0f}")
print(f"{'R-hat (mu)':30s}  {pg_rhat:>12.4f}  {dhg_rhat:>12.4f}")
print(f"{'ESS / wall-time':30s}  {pg_ess/t_pg:>12.1f}  {dhg_ess/t_dhg:>12.1f}")

# ── 5. Trace + KDE figure ─────────────────────────────────────────────────────
colors = ["steelblue", "firebrick", "seagreen", "darkorange"]
iters  = np.arange(N_SAMPLES)

params_pg  = {"mu (nats)": pg_mu,   "beta0": pg_beta0, "p": pg_p,   "log-lik": pg_ll}
params_dhg = {"mu (nats)": dhg_mu, "beta0": dhg_beta0, "p": dhg_p}

fig, axes = plt.subplots(4, 4, figsize=(18, 12))
fig.suptitle(
    f"PG-Gibbs (left pair) vs DiscreteHMCGibbs (right pair)\n"
    f"Synthetic data M={M}  |  {N_CHAINS} chains × {N_SAMPLES} samples  "
    f"|  PG: {t_pg:.0f}s  DHG: {t_dhg:.0f}s",
    fontsize=11,
)

param_names = ["mu (nats)", "beta0", "p", "log-lik"]

for row, name in enumerate(param_names):
    pg_chains  = params_pg.get(name)
    dhg_chains = params_dhg.get(name)

    # PG-Gibbs trace
    ax = axes[row, 0]
    if pg_chains is not None:
        for cid in range(N_CHAINS):
            ax.plot(iters, pg_chains[cid], color=colors[cid], lw=0.5, alpha=0.8,
                    label=f"c{cid}" if row == 0 else None)
    ax.set_ylabel(name, fontsize=9)
    if row == 0:
        ax.set_title("PG-Gibbs  trace", fontsize=9)
        ax.legend(fontsize=7, ncol=4)
    if row == 3:
        ax.set_xlabel("Sample")

    # PG-Gibbs KDE
    ax = axes[row, 1]
    if pg_chains is not None:
        for cid in range(N_CHAINS):
            v = pg_chains[cid]
            g = np.linspace(v.min(), v.max(), 300)
            ax.plot(gaussian_kde(v)(g), g, color=colors[cid], lw=1.0)
    if row == 0:
        ax.set_title("PG-Gibbs  density", fontsize=9)
    ax.set_yticklabels([]); ax.tick_params(left=False)
    if row == 3:
        ax.set_xlabel("Density")

    # DHG trace
    ax = axes[row, 2]
    if dhg_chains is not None:
        for cid in range(N_CHAINS):
            ax.plot(iters, dhg_chains[cid], color=colors[cid], lw=0.5, alpha=0.8)
    if row == 0:
        ax.set_title("DiscreteHMCGibbs  trace", fontsize=9)
    if row == 3:
        ax.set_xlabel("Sample")

    # DHG KDE
    ax = axes[row, 3]
    if dhg_chains is not None:
        for cid in range(N_CHAINS):
            v = dhg_chains[cid]
            g = np.linspace(v.min(), v.max(), 300)
            ax.plot(gaussian_kde(v)(g), g, color=colors[cid], lw=1.0)
    if row == 0:
        ax.set_title("DiscreteHMCGibbs  density", fontsize=9)
    ax.set_yticklabels([]); ax.tick_params(left=False)
    if row == 3:
        ax.set_xlabel("Density")

plt.tight_layout()
fig.savefig("compare_samplers.png", dpi=150, bbox_inches="tight")
print("\nSaved compare_samplers.png")
