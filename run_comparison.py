"""
Compare PG-Gibbs with and without the ASIS (ancillarity-sufficiency interweaving
strategy) option on the same simulated case-control dataset.

Both conditions use the defaults: half-Cauchy(scale=1) prior, collapsed_phi,
use_cauchy_aux (auto-enabled for df=1).

True mu = 2.0, K = 0.01.  Simulated pedigree: full-sib pairs, full-sib
triplets, half-sib pairs, and unrelated individuals.
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")

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

M_rel = sum(gb.L_by_size[s].shape[0] * s for s in gb.sizes if s >= 2)
print(f"  Related (size>=2): M_rel={M_rel}")

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
    M=M_rel, sizes=rel_sizes,
    L_by_size=new_L_by_size, idx_by_size=new_idx_by_size,
)

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

# ── 3. PG-Gibbs: two conditions ───────────────────────────────────────────────
from geneticinfo import (
    sample_posterior, summarize_and_plot, plot_trace, plot_pairs
)

# Defaults: half-Cauchy(scale=1), collapsed_phi, use_cauchy_aux auto-enabled
COMMON = dict(
    n_warmup       = 1000,
    n_samples      = 5000,
    n_chains       = 4,
    n_blas_threads = 1,
)

CONDITIONS = [
    # (label,        use_asis)
    ("no_asis",      False),
    ("asis",         True),
]

results = {}
for label, use_asis in CONDITIONS:
    print(f"\n{'=' * 60}")
    print(f"PG-Gibbs  ({label})  4 chains × 5000 samples, 1000 warmup")
    print("=" * 60)
    t0 = time.perf_counter()
    results[label] = sample_posterior(
        L_rel, y_rel,
        **COMMON,
        use_asis=use_asis,
    )
    results[label]["_wall"] = time.perf_counter() - t0

    mu_all = results[label]["mu_all"]
    print(f"  Wall time: {results[label]['_wall']:.1f} s")
    print(f"  mu: mean={mu_all.mean():.3f}  median={np.median(mu_all):.3f}  "
          f"sd={mu_all.std():.3f}  "
          f"90%CI=[{np.percentile(mu_all,5):.3f},{np.percentile(mu_all,95):.3f}]")

# ── 4. Plots ──────────────────────────────────────────────────────────────────
for label, r in results.items():
    summarize_and_plot(
        r, outfile=f"comparison_posterior_{label}.png",
        mu_true=MU_TRUE,
    )
    plot_trace(
        r, outfile=f"comparison_trace_{label}.png",
        title=f"PG-Gibbs {label}  (half-Cauchy prior, true μ={MU_TRUE})",
    )
    plot_pairs(
        r, outfile=f"comparison_pairs_{label}.png",
        title=f"PG-Gibbs {label}  (half-Cauchy prior, true μ={MU_TRUE})",
    )

# ── 5. Comparison table ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print(f"  True mu = {MU_TRUE}   K = {K}   M_rel = {M_rel}   seed = {SEED_DATA}")
print(f"  Prior: half-Cauchy(scale=1)   Algorithm: collapsed_phi + use_cauchy_aux")
print("=" * 60)
print(f"  {'Algorithm':<28}  {'median':>7}  {'sd':>6}  {'90% CI':>18}  "
      f"{'ESS':>6}  {'r_hat':>6}  {'wall':>8}")

import arviz as az
for label, r in results.items():
    mu_all  = r["mu_all"]
    mu_mat  = np.vstack([d["mu"] for d in r["chain_dicts"]])
    summ    = az.summary(az.convert_to_inference_data({"mu": mu_mat}), var_names=["mu"])
    ess     = float(summ["ess_bulk"].iloc[0])
    rhat    = float(summ["r_hat"].iloc[0])
    ci90    = (float(np.percentile(mu_all, 5)), float(np.percentile(mu_all, 95)))
    wall    = r["_wall"]
    name    = f"PG-Gibbs ({label})"
    print(f"  {name:<28}  {np.median(mu_all):>7.3f}  {mu_all.std():>6.3f}  "
          f"  [{ci90[0]:.3f},{ci90[1]:.3f}]  "
          f"{ess:>6.0f}  {rhat:>6.3f}  {wall:>7.1f}s")
