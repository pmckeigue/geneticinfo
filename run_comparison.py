"""
Compare PG-Gibbs (baseline vs use_phi_correction) and optionally DiscreteHMCGibbs
on the same simulated case-control dataset.

True mu = 2.0, K = 0.01.  Simulated pedigree: full-sib pairs, full-sib triplets,
half-sib pairs, and unrelated individuals.

Set RUN_HMC = True to also run DiscreteHMCGibbs (requires 4 GPUs).
"""
from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")

# ── flags ────────────────────────────────────────────────────────────────────
RUN_HMC = False   # set True to also run DiscreteHMCGibbs (requires 4 GPUs)

# ── simulation parameters ────────────────────────────────────────────────────
MU_TRUE         = 2.0
K               = 0.01
SEED_DATA       = 42
N_FULLSIB_PAIRS = 400_000
N_FULLSIB_TRIPS = 200_000
N_HALFSIB_PAIRS =  40_000
N_UNRELATED     =   4_000

# ── 1. Simulate once ─────────────────────────────────────────────────────────
print("=" * 60)
print("Simulating case-control dataset ...")
print("=" * 60)
from geneticinfo_functions import simulate_casecontrol_related, print_casecontrol_summary
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

# ── 2. Build block structure ──────────────────────────────────────────────────
from polyagamma_gibbs import infer_blocks_from_L, BlockStructure
from pg_gibbs_vectorized import GroupedBlocks

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

# Relatives-only GroupedBlocks (singletons discarded)
rel_sizes = sorted([s for s in gb.sizes if s >= 2], reverse=True)
rel_idx_ordered = np.concatenate([gb.idx_by_size[s] for s in rel_sizes])
y_rel = y_perm[rel_idx_ordered]

new_L_by_size  = {}
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

# Dense block-diagonal L for relatives only (input to sample_posterior)
def dense_from_groupedblocks(gb) -> np.ndarray:
    L = np.zeros((gb.M, gb.M), dtype=np.float64)
    for s in gb.sizes:
        Ls  = gb.L_by_size[s]
        idx = gb.idx_by_size[s].reshape(-1, s)
        for b in range(Ls.shape[0]):
            sl = idx[b]
            L[np.ix_(sl, sl)] = Ls[b]
    return L

L_rel = dense_from_groupedblocks(gb_rel)

# ── 3. PG-Gibbs: baseline and use_phi_correction ─────────────────────────────
from geneticinfo import (
    build_blocks, sample_posterior, summarize_and_plot, plot_trace, plot_pairs
)

COMMON = dict(
    n_warmup       = 1000,
    n_samples      = 5000,
    n_chains       = 4,
    n_blas_threads = 1,
    mu_prior_df    = 30.0,
    mu_prior_scale = 1.0,
)

results_pg = {}
for label, use_phi_correction in [("baseline", False), ("phi_correction", True)]:
    print(f"\n{'=' * 60}")
    print(f"PG-Gibbs  ({label})  4 chains × 5000 samples, 1000 warmup")
    print("=" * 60)
    t0 = time.perf_counter()
    results_pg[label] = sample_posterior(
        L_rel, y_rel,
        **COMMON,
        use_phi_correction=use_phi_correction,
    )
    results_pg[label]["_wall"] = time.perf_counter() - t0

    r = results_pg[label]
    mu_all = r["mu_all"]
    print(f"  Wall time: {r['_wall']:.1f} s")
    print(f"  mu: mean={mu_all.mean():.3f}  median={np.median(mu_all):.3f}  "
          f"sd={mu_all.std():.3f}  "
          f"90%CI=[{np.percentile(mu_all,5):.3f},{np.percentile(mu_all,95):.3f}]")

# ── 4. Plots for each PG-Gibbs condition ─────────────────────────────────────
for label, r in results_pg.items():
    mu_all = r["mu_all"]
    summarize_and_plot(
        r, outfile=f"comparison_posterior_{label}.png",
        mu_true=MU_TRUE,
    )
    plot_trace(
        r, outfile=f"comparison_trace_{label}.png",
        title=f"PG-Gibbs {label}  (simulated data, true μ={MU_TRUE})",
    )
    plot_pairs(
        r, outfile=f"comparison_pairs_{label}.png",
        title=f"PG-Gibbs {label}  (simulated data, true μ={MU_TRUE})",
    )

# ── 5. Optional: DiscreteHMCGibbs ────────────────────────────────────────────
if RUN_HMC:
    print("\n" + "=" * 60)
    print("DiscreteHMCGibbs  (4 chains × 2000 samples, 2000 warmup)")
    print("=" * 60)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import random
    import numpyro
    from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
    import arviz as az
    from geneticinfo_functions import lr_discrete_blockdiag

    NUM_CHAINS_HMC = 4
    numpyro.set_host_device_count(NUM_CHAINS_HMC)

    hmc_sizes  = []
    hmc_L_list = []
    hmc_y_list = []
    for s in sorted(gb.sizes, reverse=True):
        if s < 2:
            continue
        hmc_sizes.append(s)
        hmc_L_list.append(jnp.array(gb.L_by_size[s], dtype=jnp.float64))
        hmc_y_list.append(jnp.array(y_perm[gb.idx_by_size[s]], dtype=jnp.float64))

    p_obs  = float(np.mean(y_sample))
    M_hmc  = sum(L.shape[0] * s for L, s in zip(hmc_L_list, hmc_sizes))
    inner_kernel = NUTS(lr_discrete_blockdiag, max_tree_depth=8)
    kernel = DiscreteHMCGibbs(inner_kernel, modified=True)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000,
                num_chains=NUM_CHAINS_HMC, chain_method="parallel",
                progress_bar=True)

    t0_hmc = time.perf_counter()
    mcmc.run(random.PRNGKey(0),
             L_list=hmc_L_list, y_list=hmc_y_list, sizes=hmc_sizes, p_obs=p_obs)
    t_hmc = time.perf_counter() - t0_hmc

    mu_hmc   = np.array(mcmc.get_samples()["mu"])
    az_hmc   = az.convert_to_inference_data({"mu": mu_hmc.reshape(NUM_CHAINS_HMC, -1)})
    summ_hmc = az.summary(az_hmc, var_names=["mu"])
    ess_hmc  = float(summ_hmc["ess_bulk"].iloc[0])
    rhat_hmc = float(summ_hmc["r_hat"].iloc[0])
    ci90_hmc = (float(np.percentile(mu_hmc, 5)), float(np.percentile(mu_hmc, 95)))

    print(f"\nDiscreteHMCGibbs results:")
    print(f"  Wall time: {t_hmc:.1f} s")
    print(f"  mu: mean={mu_hmc.mean():.3f}  median={float(np.median(mu_hmc)):.3f}  "
          f"sd={mu_hmc.std():.3f}  90%CI=[{ci90_hmc[0]:.3f},{ci90_hmc[1]:.3f}]")
    print(f"  ESS_bulk={ess_hmc:.0f}  r_hat={rhat_hmc:.3f}")

# ── 6. Comparison table ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print(f"  True mu = {MU_TRUE}   K = {K}   M = {M:,}   seed = {SEED_DATA}")
print("=" * 60)
print(f"  {'Algorithm':<28}  {'median':>7}  {'sd':>6}  {'90% CI':>18}  "
      f"{'ESS':>6}  {'r_hat':>6}  {'wall':>8}")

import arviz as az
for label, r in results_pg.items():
    mu_all  = r["mu_all"]
    mu_mat  = np.vstack([d["mu"] for d in r["chain_dicts"]])
    az_data = az.convert_to_inference_data({"mu": mu_mat})
    summ    = az.summary(az_data, var_names=["mu"])
    ess     = float(summ["ess_bulk"].iloc[0])
    rhat    = float(summ["r_hat"].iloc[0])
    ci90    = (float(np.percentile(mu_all, 5)), float(np.percentile(mu_all, 95)))
    wall    = r["_wall"]
    name    = f"PG-Gibbs ({label})"
    print(f"  {name:<28}  {np.median(mu_all):>7.3f}  {mu_all.std():>6.3f}  "
          f"  [{ci90[0]:.3f},{ci90[1]:.3f}]  "
          f"{ess:>6.0f}  {rhat:>6.3f}  {wall:>7.1f}s")

if RUN_HMC:
    print(f"  {'DiscreteHMCGibbs':<28}  {float(np.median(mu_hmc)):>7.3f}  {mu_hmc.std():>6.3f}  "
          f"  [{ci90_hmc[0]:.3f},{ci90_hmc[1]:.3f}]  "
          f"{ess_hmc:>6.0f}  {rhat_hmc:>6.3f}  {t_hmc:>7.1f}s")
