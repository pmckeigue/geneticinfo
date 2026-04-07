"""
Head-to-head comparison of PG-Gibbs (collapsed_phi) and DiscreteHMCGibbs
on the same simulated dataset, with identical half-Cauchy(scale=1) prior on μ.

Requires 4 GPUs (Tesla V100 or similar) for the DiscreteHMCGibbs run.
PG-Gibbs runs on CPU (4 cores).

Outputs (6 plots, one posterior/trace/pairs per sampler):
  comparison_pg_posterior.png / comparison_pg_trace.png / comparison_pg_pairs.png
  comparison_hmc_posterior.png / comparison_hmc_trace.png / comparison_hmc_pairs.png
"""
from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"

import dataclasses
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import arviz as az

MU_TRUE         = 2.0
K               = 0.01
SEED_DATA       = 42
N_FULLSIB_PAIRS = 400_000
N_FULLSIB_TRIPS = 200_000
N_HALFSIB_PAIRS =  40_000
N_UNRELATED     =   4_000

# ── 1. Simulate ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Simulating case-control dataset ...")
print("=" * 60)
from geneticinfo_functions import simulate_casecontrol_related, print_casecontrol_summary

y_sample, A_sample, L_sample, info, g_sample = simulate_casecontrol_related(
    n_fullsib_pairs=N_FULLSIB_PAIRS, n_fullsib_trips=N_FULLSIB_TRIPS,
    n_halfsib_pairs=N_HALFSIB_PAIRS, n_unrelated=N_UNRELATED,
    K=K, mu=MU_TRUE, seed=SEED_DATA, return_genotypic_values=True,
)
print_casecontrol_summary(info)
M = info["M"]
p_obs = float(np.mean(y_sample))

# ── 2. Build shared block structure ──────────────────────────────────────────
from polyagamma_gibbs import infer_blocks_from_L, BlockStructure
from pg_gibbs_vectorized import GroupedBlocks

L_np = np.asarray(L_sample, dtype=np.float64)
y_np = np.asarray(y_sample, dtype=np.float64)
perm_rows, perm_cols, row_slices, col_slices = infer_blocks_from_L(L_np, tol_rel=0.0)
L_perm = L_np[perm_rows, :][:, perm_cols]
y_perm = y_np[perm_rows]
bs = BlockStructure(L_perm, col_slices, row_slices)
gb = GroupedBlocks.from_block_structure(bs)
print(f"\nBlock structure:  M={gb.M}  blocks={bs.n_blocks}  "
      + "  ".join(f"size-{s}:{gb.L_by_size[s].shape[0]}" for s in gb.sizes))

M_rel = sum(gb.L_by_size[s].shape[0] * s for s in gb.sizes if s >= 2)
print(f"  Relatives only: M_rel={M_rel}")

# ── 3. PG-Gibbs (collapsed_phi, half-Cauchy prior) ───────────────────────────
print(f"\n{'=' * 60}")
print(f"PG-Gibbs (collapsed_phi, half-Cauchy prior)  4 chains × 5000 samples")
print("=" * 60)

# Build dense relatives-only L for sample_posterior
rel_sizes = sorted([s for s in gb.sizes if s >= 2], reverse=True)
rel_idx_ordered = np.concatenate([gb.idx_by_size[s] for s in rel_sizes])
y_rel = y_perm[rel_idx_ordered]

new_L_by_size, new_idx_by_size = {}, {}
offset = 0
for s in rel_sizes:
    n_s = gb.L_by_size[s].shape[0]
    new_L_by_size[s]   = gb.L_by_size[s]
    new_idx_by_size[s] = np.arange(offset, offset + n_s * s)
    offset += n_s * s

gb_rel = GroupedBlocks(M=M_rel, sizes=rel_sizes,
                       L_by_size=new_L_by_size, idx_by_size=new_idx_by_size)

def dense_from_groupedblocks(gb_):
    L = np.zeros((gb_.M, gb_.M), dtype=np.float64)
    for s in gb_.sizes:
        Ls  = gb_.L_by_size[s]
        idx = gb_.idx_by_size[s].reshape(-1, s)
        for b in range(Ls.shape[0]):
            sl = idx[b]
            L[np.ix_(sl, sl)] = Ls[b]
    return L

L_rel = dense_from_groupedblocks(gb_rel)

from geneticinfo import sample_posterior, summarize_and_plot, plot_trace, plot_pairs

t0 = time.perf_counter()
result_pg = sample_posterior(
    L_rel, y_rel,
    n_warmup=2000, n_samples=5000, n_chains=4, n_blas_threads=1,
    mu_prior_scale=1.0,
    mu_prior_df=1.0,          # half-Cauchy — same prior as DiscreteHMCGibbs
    use_collapsed_phi=True,
    jags_adapt=True,          # adapt slice widths during warmup (needed for heavy-tailed prior)
    n_warmup_phase1=300,      # 300 fixed-width iterations before adaptation starts
)
t_pg = time.perf_counter() - t0
result_pg["_wall"] = t_pg

mu_pg = result_pg["mu_all"]
mu_mat_pg = np.vstack([d["mu"] for d in result_pg["chain_dicts"]])
summ_pg = az.summary(az.convert_to_inference_data({"mu": mu_mat_pg}), var_names=["mu"])
ess_pg  = float(summ_pg["ess_bulk"].iloc[0])
rhat_pg = float(summ_pg["r_hat"].iloc[0])
ci90_pg = (float(np.percentile(mu_pg, 5)), float(np.percentile(mu_pg, 95)))
print(f"  median={np.median(mu_pg):.3f}  sd={mu_pg.std():.3f}  "
      f"90%CI=[{ci90_pg[0]:.3f},{ci90_pg[1]:.3f}]  "
      f"ESS={ess_pg:.0f}  r_hat={rhat_pg:.3f}  wall={t_pg:.0f}s")

summarize_and_plot(result_pg, outfile="comparison_pg_posterior.png", mu_true=MU_TRUE,
                   title=f"PG-Gibbs collapsed_phi  (half-Cauchy prior, true μ={MU_TRUE})")
plot_trace(result_pg, outfile="comparison_pg_trace.png",
           title=f"PG-Gibbs collapsed_phi  (half-Cauchy prior)")
plot_pairs(result_pg, outfile="comparison_pg_pairs.png",
           title=f"PG-Gibbs collapsed_phi  (half-Cauchy prior, true μ={MU_TRUE})")

# ── 4. DiscreteHMCGibbs (half-Cauchy prior, default) ─────────────────────────
print(f"\n{'=' * 60}")
print(f"DiscreteHMCGibbs (half-Cauchy prior)  4 chains × 2000 samples")
print("=" * 60)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from geneticinfo_functions import lr_discrete_blockdiag

NUM_CHAINS = 4
N_WARMUP   = 1000
N_SAMPLES  = 2000
numpyro.set_host_device_count(NUM_CHAINS)

hmc_sizes, hmc_L_list, hmc_y_list = [], [], []
for s in sorted(gb.sizes, reverse=True):
    if s < 2:
        continue
    hmc_sizes.append(s)
    hmc_L_list.append(jnp.array(gb.L_by_size[s], dtype=jnp.float64))
    hmc_y_list.append(jnp.array(y_perm[gb.idx_by_size[s]], dtype=jnp.float64))

inner_kernel = NUTS(lr_discrete_blockdiag, max_tree_depth=8)
kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
            num_chains=NUM_CHAINS, chain_method="parallel", progress_bar=True)

t0 = time.perf_counter()
mcmc.run(random.PRNGKey(0), L_list=hmc_L_list, y_list=hmc_y_list,
         sizes=hmc_sizes, p_obs=p_obs)
         # muprior defaults to HalfCauchy(scale=1) in lr_discrete_blockdiag
t_hmc = time.perf_counter() - t0

mcmc.print_summary(exclude_deterministic=True)

# ── 5. Build HMC result dict ──────────────────────────────────────────────────
samps = mcmc.get_samples(group_by_chain=True)
mu_chains_hmc   = np.array(samps["mu"])    # (n_chains, n_samples)
p_chains_hmc    = np.array(samps["p"])
n_ch, n_sa      = mu_chains_hmc.shape

beta0_hmc = (np.log(np.clip(p_chains_hmc, 1e-9, 1 - 1e-9))
             - np.log(np.clip(1 - p_chains_hmc, 1e-9, 1 - 1e-9)))

# Bernoulli log-likelihood using sampled G
G_hmc = np.array(samps["G"])          # (n_chains, n_samples, M_rel)
eta   = beta0_hmc[:, :, None] + G_hmc
ll_pos = -np.log1p(np.exp(-np.clip(eta, -500, 500)))
ll_neg = -np.log1p(np.exp( np.clip(eta, -500, 500)))
ll_hmc = (y_rel[None, None, :] * ll_pos
          + (1 - y_rel)[None, None, :] * ll_neg).sum(axis=-1)

mu_all_hmc = mu_chains_hmc.ravel()

@dataclasses.dataclass
class _Cfg:
    n_warmup: int
    n_samples: int

result_hmc = {
    "mu_chains": mu_chains_hmc,
    "mu_all":    mu_all_hmc,
    "chain_dicts": [
        {"mu": mu_chains_hmc[c], "beta0": beta0_hmc[c], "p": p_chains_hmc[c],
         "ll": ll_hmc[c], "mu_warmup": np.empty(0), "ll_warmup": np.empty(0),
         "chain_id": c}
        for c in range(n_ch)
    ],
    "cfg":            _Cfg(n_warmup=0, n_samples=n_sa),
    "mu_prior_scale": 1.0,
    "mu_prior_df":    1.0,   # half-Cauchy (df=1)
    "block_info":     {},
}

# ── 6. HMC plots ──────────────────────────────────────────────────────────────
summarize_and_plot(result_hmc, outfile="comparison_hmc_posterior.png", mu_true=MU_TRUE,
                   title=f"DiscreteHMCGibbs  (half-Cauchy prior, true μ={MU_TRUE})")
plot_trace(result_hmc, outfile="comparison_hmc_trace.png",
           title=f"DiscreteHMCGibbs  (half-Cauchy prior)")
plot_pairs(result_hmc, outfile="comparison_hmc_pairs.png",
           title=f"DiscreteHMCGibbs  (half-Cauchy prior, true μ={MU_TRUE})")

summ_hmc = az.summary(az.convert_to_inference_data({"mu": mu_chains_hmc}),
                       var_names=["mu"])
ess_hmc  = float(summ_hmc["ess_bulk"].iloc[0])
rhat_hmc = float(summ_hmc["r_hat"].iloc[0])
ci90_hmc = (float(np.percentile(mu_all_hmc, 5)), float(np.percentile(mu_all_hmc, 95)))
print(f"  median={np.median(mu_all_hmc):.3f}  sd={mu_all_hmc.std():.3f}  "
      f"90%CI=[{ci90_hmc[0]:.3f},{ci90_hmc[1]:.3f}]  "
      f"ESS={ess_hmc:.0f}  r_hat={rhat_hmc:.3f}  wall={t_hmc:.0f}s")

# ── 7. Comparison table ───────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("COMPARISON SUMMARY  (both samplers: half-Cauchy(scale=1) prior on μ)")
print(f"  True μ = {MU_TRUE}   K = {K}   M_rel = {M_rel}   seed = {SEED_DATA}")
print("=" * 60)
print(f"  {'Algorithm':<32}  {'median':>7}  {'sd':>6}  {'90% CI':>18}  "
      f"{'ESS':>5}  {'ESS/s':>6}  {'r_hat':>6}  {'wall':>8}")
for name, mu_all_, ess_, rhat_, ci90_, t_, n_samp in [
    ("PG-Gibbs (collapsed_phi)",  mu_pg,      ess_pg,  rhat_pg,  ci90_pg,  t_pg,
     4 * 5000),
    ("DiscreteHMCGibbs",          mu_all_hmc, ess_hmc, rhat_hmc, ci90_hmc, t_hmc,
     NUM_CHAINS * N_SAMPLES),
]:
    print(f"  {name:<32}  {np.median(mu_all_):>7.3f}  {mu_all_.std():>6.3f}  "
          f"  [{ci90_[0]:.3f},{ci90_[1]:.3f}]  "
          f"{ess_:>5.0f}  {ess_/t_:>6.2f}  {rhat_:>6.3f}  {t_:>7.1f}s")
