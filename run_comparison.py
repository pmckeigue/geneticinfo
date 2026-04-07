"""
Compare PG-Gibbs with and without the ASIS (ancillarity-sufficiency interweaving
strategy) option on the same simulated case-control dataset.

Both conditions use the defaults: half-Cauchy(scale=1) prior, collapsed_phi,
use_cauchy_aux (auto-enabled), omit_singletons=True.

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

L_np = np.asarray(L_sample, dtype=np.float64)
y_np = np.asarray(y_sample, dtype=np.float64)

# ── 2. Build block structure (singletons retained) ───────────────────────────
from geneticinfo import (
    build_blocks, sample_posterior, summarize_and_plot, plot_trace, plot_pairs
)

blocks = build_blocks(L_np, y_np)

# ── 3. PG-Gibbs: two conditions ───────────────────────────────────────────────
# Defaults: half-Cauchy(scale=1), collapsed_phi, use_cauchy_aux auto-enabled,
# omit_singletons=True (handled inside sample_posterior).
COMMON = dict(
    n_warmup       = 1000,
    n_samples      = 5000,
    n_chains       = 4,
    n_blas_threads = 1,
    blocks         = blocks,
)

CONDITIONS = [
    # (label,    use_asis)
    ("no_asis",  False),
    ("asis",     True),
]

results = {}
for label, use_asis in CONDITIONS:
    print(f"\n{'=' * 60}")
    print(f"PG-Gibbs  ({label})  4 chains × 5000 samples, 1000 warmup")
    print("=" * 60)
    t0 = time.perf_counter()
    results[label] = sample_posterior(L_np, y_np, **COMMON, use_asis=use_asis)
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
print(f"  True mu = {MU_TRUE}   K = {K}   seed = {SEED_DATA}")
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
