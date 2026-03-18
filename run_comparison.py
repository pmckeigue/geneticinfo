"""
Run both DiscreteHMCGibbs and PG-Gibbs on the same case-control dataset
simulated from a population with full-sib pairs, full-sib triplets, and
half-sib pairs.  True mu = 1.0, K = 0.01.

DiscreteHMCGibbs (JAX/NumPyro, GPU):
  - Operates on M_rel individuals in relative-only blocks (all sizes >= 2)
  - Fits lr_discrete_blockdiag; 4 chains, 2000 warmup + 2000 samples

PG-Gibbs (NumPy, CPU):
  - Operates on full case-control sample; excludes singletons from mu update
  - 4 chains via fork, 1000 warmup + 5000 samples
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"

import time
import multiprocessing as mp
import numpy as np
import arviz as az
from tqdm import tqdm

# ── simulation parameters ────────────────────────────────────────────────────
MU_TRUE          = 1.0
K                = 0.01
SEED_DATA        = 42
N_FULLSIB_PAIRS  = 400_000
N_FULLSIB_TRIPS  = 200_000
N_HALFSIB_PAIRS  = 40_000
N_UNRELATED      = 4_000

# ── 1. Simulate once ─────────────────────────────────────────────────────────
print("=" * 60)
print("Simulating case-control dataset ...")
print("=" * 60)
from geneticinfo import simulate_casecontrol_related, print_casecontrol_summary
y_sample, A_sample, L_sample, info, g_sample = simulate_casecontrol_related(
    n_fullsib_pairs=N_FULLSIB_PAIRS,
    n_fullsib_trips=N_FULLSIB_TRIPS,
    n_halfsib_pairs=N_HALFSIB_PAIRS,
    n_unrelated=N_UNRELATED,
    K=K, mu=MU_TRUE, seed=SEED_DATA,
    return_genotypic_values=True,
)
print_casecontrol_summary(info)
M = info["M"]

# ── 2. Build shared block structure ──────────────────────────────────────────
from polyagamma_gibbs import infer_blocks_from_L, BlockStructure, ChainConfig
from pg_gibbs_vectorized import GroupedBlocks, _worker_preloaded

L_np = np.asarray(L_sample, dtype=np.float64)
y_np = np.asarray(y_sample, dtype=np.float64)

perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
L_perm = L_np[perm_rows, :][:, perm_cols]
y_perm = y_np[perm_rows]

bs = BlockStructure(L_perm, col_slices, row_slices)
gb = GroupedBlocks.from_block_structure(bs)
sizes_str = "  ".join(f"size-{s}:{gb.L_by_size[s].shape[0]}" for s in gb.sizes)
print(f"\nBlock structure:  M={gb.M}  blocks={bs.n_blocks}  {sizes_str}")

n2 = gb.L_by_size[2].shape[0] if 2 in gb.L_by_size else 0
n3 = gb.L_by_size[3].shape[0] if 3 in gb.L_by_size else 0
M_rel = sum(gb.L_by_size[s].shape[0] * s for s in gb.sizes if s >= 2)
print(f"  Related (size>=2): {M_rel}  ({n2} pairs, {n3} triplets)")

# Build a relatives-only GroupedBlocks (singletons discarded entirely).
# Blocks are re-indexed 0..M_rel-1 in size-descending order so that the
# PG-Gibbs sampler sees the same M_rel individuals as DiscreteHMCGibbs.
rel_sizes = sorted([s for s in gb.sizes if s >= 2], reverse=True)
rel_idx_ordered = np.concatenate([gb.idx_by_size[s] for s in rel_sizes])
y_rel = y_perm[rel_idx_ordered]

new_L_by_size   = {}
new_idx_by_size = {}
offset = 0
for s in rel_sizes:
    n_s = gb.L_by_size[s].shape[0]
    new_L_by_size[s]   = gb.L_by_size[s]
    new_idx_by_size[s] = np.arange(offset, offset + n_s * s)
    offset += n_s * s

gb_rel = GroupedBlocks(
    M=M_rel,
    sizes=rel_sizes,
    L_by_size=new_L_by_size,
    idx_by_size=new_idx_by_size,
)
print(f"  Relatives-only GroupedBlocks: M={gb_rel.M}  "
      + "  ".join(f"size-{s}:{gb_rel.L_by_size[s].shape[0]}" for s in gb_rel.sizes))


# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: PG-Gibbs  (full case-control sample; singletons excluded from μ)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PG-Gibbs  (4 chains × 5000 samples, 1000 warmup, full sample M)")
print("=" * 60)

NUM_CHAINS_PG = 4
cfg_pg = ChainConfig(
    n_warmup=1000, n_samples=5000,
    prior_loc=0.0, prior_scale=1.0,
    beta0_sd=5.0, slice_w=1.5, slice_m=20, slice_max_steps=250,
    inner_latent_cycles=1, learn_p=True, p_prior_conc=200.0,
    project_affects_beta0=True, diag_every=200, data_flag=True,
)
# Use the full permuted dataset (M=29,496).  Singletons constrain beta0 via
# their 50% case rate, while being excluded from the mu likelihood through
# min_block_size=2 inside build_theta_eig_cache_fast.
jobs_pg = [(cid, gb, y_perm, 42 + cid, cfg_pg, MU_TRUE)
           for cid in range(NUM_CHAINS_PG)]

t0_pg = time.perf_counter()
ctx  = mp.get_context("fork")
lock = ctx.RLock()
with ctx.Pool(processes=NUM_CHAINS_PG,
              initializer=tqdm.set_lock, initargs=(lock,)) as pool:
    chains_pg = pool.map(_worker_preloaded, jobs_pg)
t_pg = time.perf_counter() - t0_pg

mu_pg = np.concatenate([o["mu"] for o in chains_pg])
az_pg = az.convert_to_inference_data(
    {"mu": np.stack([o["mu"] for o in chains_pg])})
summ_pg   = az.summary(az_pg, var_names=["mu"])
ess_pg    = float(summ_pg["ess_bulk"].iloc[0])
rhat_pg   = float(summ_pg["r_hat"].iloc[0])
ci90_pg   = (float(np.percentile(mu_pg, 5)), float(np.percentile(mu_pg, 95)))

print(f"\nPG-Gibbs results:")
print(f"  Wall time:    {t_pg:.1f} s")
print(f"  M = {M} total; mu likelihood uses M_rel = {M_rel} "
      f"({n2} pairs + {n3} triplets; singletons excluded from mu update only)")
print(f"  mu: mean={mu_pg.mean():.3f}  median={np.median(mu_pg):.3f}  "
      f"sd={mu_pg.std():.3f}  90%CI=[{ci90_pg[0]:.3f},{ci90_pg[1]:.3f}]")
print(f"  ESS_bulk={ess_pg:.0f}  r_hat={rhat_pg:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: DiscreteHMCGibbs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DiscreteHMCGibbs  (4 chains × 2000 samples, 2000 warmup)")
print("=" * 60)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from geneticinfo import lr_discrete_blockdiag

NUM_CHAINS_HMC = 4
numpyro.set_host_device_count(NUM_CHAINS_HMC)

# Build per-size block lists for lr_discrete_blockdiag.
# Order: largest blocks first (size-3 before size-2) so individuals are
# laid out consistently with gb_rel.
assert n2 > 0, "No size-2 blocks found"
p_obs = float(np.mean(y_sample))

hmc_sizes  = []
hmc_L_list = []
hmc_y_list = []
for s in sorted(gb.sizes, reverse=True):
    if s < 2:
        continue
    hmc_sizes.append(s)
    hmc_L_list.append(jnp.array(gb.L_by_size[s], dtype=jnp.float64))
    hmc_y_list.append(jnp.array(y_perm[gb.idx_by_size[s]], dtype=jnp.float64))

M_hmc = sum(L.shape[0] * s for L, s in zip(hmc_L_list, hmc_sizes))
size_str = "  ".join(f"{L.shape[0]} size-{s}" for L, s in zip(hmc_L_list, hmc_sizes))
print(f"  Using M_rel={M_hmc}: {size_str}")

inner_kernel = NUTS(lr_discrete_blockdiag, max_tree_depth=8)
kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000,
            num_chains=NUM_CHAINS_HMC, chain_method="parallel",
            progress_bar=True)

t0_hmc = time.perf_counter()
mcmc.run(random.PRNGKey(0),
         L_list=hmc_L_list, y_list=hmc_y_list, sizes=hmc_sizes, p_obs=p_obs)
t_hmc = time.perf_counter() - t0_hmc

samples_hmc = mcmc.get_samples()
mu_hmc = np.array(samples_hmc["mu"])
az_hmc = az.convert_to_inference_data(
    {"mu": mu_hmc.reshape(NUM_CHAINS_HMC, -1)})
summ_hmc  = az.summary(az_hmc, var_names=["mu"])
ess_hmc   = float(summ_hmc["ess_bulk"].iloc[0])
rhat_hmc  = float(summ_hmc["r_hat"].iloc[0])
ci90_hmc  = (float(np.percentile(mu_hmc, 5)), float(np.percentile(mu_hmc, 95)))

print(f"\nDiscreteHMCGibbs results:")
print(f"  Wall time:    {t_hmc:.1f} s")
print(f"  M_rel = {M_hmc} ({size_str})")
print(f"  mu: mean={mu_hmc.mean():.3f}  median={float(np.median(mu_hmc)):.3f}  "
      f"sd={mu_hmc.std():.3f}  90%CI=[{ci90_hmc[0]:.3f},{ci90_hmc[1]:.3f}]")
print(f"  ESS_bulk={ess_hmc:.0f}  r_hat={rhat_hmc:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print(f"  True mu = {MU_TRUE}   K = {K}   M = {M:,}   seed = {SEED_DATA}")
print("=" * 60)
print(f"  {'Algorithm':<22}  {'median':>7}  {'sd':>6}  {'90% CI':>18}  "
      f"{'ESS':>6}  {'r_hat':>6}  {'wall':>8}")
print(f"  {'PG-Gibbs':<22}  {np.median(mu_pg):>7.3f}  {mu_pg.std():>6.3f}  "
      f"  [{ci90_pg[0]:.3f},{ci90_pg[1]:.3f}]  "
      f"{ess_pg:>6.0f}  {rhat_pg:>6.3f}  {t_pg:>7.1f}s")
print(f"  {'DiscreteHMCGibbs':<22}  {float(np.median(mu_hmc)):>7.3f}  {mu_hmc.std():>6.3f}  "
      f"  [{ci90_hmc[0]:.3f},{ci90_hmc[1]:.3f}]  "
      f"{ess_hmc:>6.0f}  {rhat_hmc:>6.3f}  {t_hmc:>7.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS: prior / posterior / likelihood for each algorithm
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PRIOR_SCALE = 1.0   # half-Cauchy scale used by both samplers

def halfcauchy_pdf(x, scale=PRIOR_SCALE):
    return 2.0 / (np.pi * scale * (1.0 + (x / scale) ** 2))

def make_likelihood(samples):
    """Estimate likelihood as posterior / prior (importance-weight KDE)."""
    prior_vals = np.maximum(halfcauchy_pdf(samples), 1e-300)
    w = 1.0 / prior_vals
    w /= w.sum()
    kde = gaussian_kde(samples, weights=w)
    return kde

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, samples, label, wall, m_rel_used, n_samples_str, color_post in [
    (axes[0], mu_pg,  "PG-Gibbs",
     t_pg,  f"M={M} (singletons excluded from μ update only)",
     f"4 chains × 5,000 samples", "steelblue"),
    (axes[1], mu_hmc, "DiscreteHMCGibbs",
     t_hmc, f"{2*n2} ({n2} pairs; {n3} triplet(s) excluded)",
     f"4 chains × 2,000 samples", "darkorange"),
]:
    mu_max  = max(4.0, float(np.percentile(samples, 99)) * 1.6)
    mu_grid = np.linspace(1e-4, mu_max, 600)

    prior_d = halfcauchy_pdf(mu_grid)
    post_d  = gaussian_kde(samples)(mu_grid)
    lik_d   = make_likelihood(samples)(mu_grid)
    lik_d  /= np.trapezoid(lik_d, mu_grid)

    ax.plot(mu_grid, prior_d, color="gray",  lw=1.8, ls="--",
            label=f"Prior  [half-Cauchy(scale={PRIOR_SCALE:.0f})]")
    ax.plot(mu_grid, post_d,  color=color_post, lw=2.2,
            label="Posterior")
    ax.plot(mu_grid, lik_d,   color="firebrick", lw=2.0, ls="-.",
            label="Likelihood  (posterior / prior)")
    ax.axvline(MU_TRUE, color="black", ls=":", lw=1.5,
               label=f"True μ = {MU_TRUE:.1f}")

    med = float(np.median(samples))
    ci5, ci95 = float(np.percentile(samples, 5)), float(np.percentile(samples, 95))
    ess = ess_pg if "PG" in label else ess_hmc
    rhat = rhat_pg if "PG" in label else rhat_hmc

    ax.set_xlabel("μ  (genetic information, nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{label}\n"
        f"{n_samples_str}   M_rel = {m_rel_used}\n"
        f"median={med:.3f}   90% CI=[{ci5:.3f}, {ci95:.3f}]",
        fontsize=10,
    )
    ax.text(0.97, 0.95,
            f"ESS = {ess:.0f}\n$\\hat{{R}}$ = {rhat:.3f}\nwall = {wall:.0f} s",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.legend(fontsize=9)
    ax.set_xlim(0, mu_max)
    ax.set_ylim(bottom=0)

fig.suptitle(
    f"Prior, posterior, and likelihood for μ\n"
    f"Same dataset: M={M:,}   K={K}   true μ={MU_TRUE}   seed={SEED_DATA}",
    fontsize=11, y=1.01,
)
fig.tight_layout()
fig.savefig("comparison_prior_posterior_likelihood.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: comparison_prior_posterior_likelihood.png")
