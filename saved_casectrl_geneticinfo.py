#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys

# Phenotype name and slice sampler width from command-line arguments
PHENO            = sys.argv[1] if len(sys.argv) > 1 else "unknown"
slice_w          = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
jags_adapt       = (sys.argv[3].lower() == "true") if len(sys.argv) > 3 else False
n_warmup_phase1  = int(sys.argv[4]) if len(sys.argv) > 4 else 0
use_asis          = (sys.argv[5].lower() == "true") if len(sys.argv) > 5 else False
use_alpha_reparam = (sys.argv[6].lower() == "true") if len(sys.argv) > 6 else False

# Suppress BLAS threading contention when running multiple chains via fork.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Write and run the setup shell script
with open("run.sh", "w") as rsh:
    rsh.write("""\
#!/bin/bash
set -euo pipefail
dx download -f setupscripts/ukbb_utils.py
git clone https://github.com/pmckeigue/geneticinfo.git
dx download -f ancestry_kinship/pg_gibbs_clean.py -o geneticinfo/pg_gibbs_clean.py
pip install -U --force-reinstall polyagamma
dx download -f genotype_data/grm/RA_data.npz
export PYTHONPATH=/opt/notebooks/geneticinfo:$PYTHONPATH
""")

subprocess.run(["bash", "run.sh"], check=True)

sys.path.insert(0, "/opt/notebooks/geneticinfo")

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geneticinfo import build_blocks, sample_posterior, summarize_and_plot

RA_data = np.load("RA_data.npz")
L = RA_data["L"]
y = RA_data["y"]
print("L.shape =", L.shape, "  y.shape =", y.shape)

blocks = build_blocks(L, y, corr_threshold=0.03)

bi = blocks["block_info"]
print(f"Block structure: M={bi['M']}  n_blocks={bi['n_blocks']}  "
      f"corr_threshold={bi['corr_threshold']}  "
      f"max_block_size={max(bi['sizes_summary'])}  "
      f"n_relatives={sum(s*c for s,c in bi['sizes_summary'].items() if s>=2)}")
print("  " + "  ".join(f"size-{s}:{c}" for s, c in sorted(bi["sizes_summary"].items())))

t0 = time.time()
result = sample_posterior(
    L, y,
    blocks=blocks,
    n_warmup=500,
    n_samples=1000,
    n_chains=8,
    n_blas_threads=1,
    mu_prior_df=30.0,
    slice_w=slice_w,
    jags_adapt=jags_adapt,
    n_warmup_phase1=n_warmup_phase1,
    use_asis=use_asis,
    use_alpha_reparam=use_alpha_reparam,
)
wall_time = time.time() - t0
n_iter_per_chain = result["cfg"].n_warmup + result["cfg"].n_samples
print(f"Wall time: {wall_time:.1f}s  ({wall_time / n_iter_per_chain:.4f}s/iter per chain)")

suffix = f"sw{slice_w}_adapted" if jags_adapt else f"sw{slice_w}"
if use_asis:
    suffix += "_asis"
if use_alpha_reparam:
    suffix += "_ar"
outfile = f"posterior_{PHENO}_{suffix}.png"
summary = summarize_and_plot(result, outfile=outfile)

# Slice sampler diagnostics
shrinks  = [d["mean_theta_shrink"]     for d in result["chain_dicts"]]
steps    = [d["mean_theta_step"]       for d in result["chain_dicts"]]
w_thetas = [d["adapted_slice_w_theta"] for d in result["chain_dicts"]]
w_phis   = [d["adapted_slice_w_phi"]   for d in result["chain_dicts"]]
print(f"Slice diagnostics (slice_w={slice_w}, jags_adapt={jags_adapt}):")
print(f"  mean shrinkage steps/iter: {np.mean(shrinks):.2f}  (sd {np.std(shrinks):.2f})")
print(f"  mean accepted |Δtheta|:    {np.mean(steps):.4f}  (sd {np.std(steps):.4f})")
if jags_adapt:
    print(f"  final adapted slice_w_theta: {np.mean(w_thetas):.4f}  (sd {np.std(w_thetas):.4f})")
    print(f"  final adapted slice_w_phi:   {np.mean(w_phis):.4f}  (sd {np.std(w_phis):.4f})")

if use_asis:
    asis_shrinks = [d["mean_asis_shrink"] for d in result["chain_dicts"]]
    asis_steps   = [d["mean_asis_step"]   for d in result["chain_dicts"]]
    print(f"ASIS diagnostics:")
    print(f"  mean shrinkage steps/iter: {np.mean(asis_shrinks):.2f}  (sd {np.std(asis_shrinks):.2f})")
    print(f"  mean accepted |Δtheta|:    {np.mean(asis_steps):.4f}  (sd {np.std(asis_steps):.4f})")

# Save results: posterior samples + summary statistics + block metadata.
# The relationship matrix (L) and per-block L sub-matrices are not saved.
block_info = blocks["block_info"]

# Block size counts as parallel arrays (size, count) for metadata
sizes  = np.array(sorted(block_info["sizes_summary"].keys()), dtype=np.int32)
counts = np.array([block_info["sizes_summary"][s] for s in sizes], dtype=np.int32)

results_file = f"results_{PHENO}_{suffix}.npz"
np.savez_compressed(
    results_file,
    # Posterior samples
    mu_samples    = np.concatenate([d["mu"]    for d in result["chain_dicts"]]),
    beta0_samples = np.concatenate([d["beta0"] for d in result["chain_dicts"]]),
    p_samples     = np.concatenate([d["p"]     for d in result["chain_dicts"]]),
    # Warmup traces (n_chains x n_warmup) for convergence diagnostics
    mu_warmup     = np.vstack([d["mu_warmup"] for d in result["chain_dicts"]]),
    ll_warmup     = np.vstack([d["ll_warmup"] for d in result["chain_dicts"]]),
    # Posterior summary
    mu_mean    = np.float64(summary["mean"]),
    mu_median  = np.float64(summary["median"]),
    mu_sd      = np.float64(summary["sd"]),
    mu_ci90_lo = np.float64(summary["ci90_lo"]),
    mu_ci90_hi = np.float64(summary["ci90_hi"]),
    ess_bulk   = np.float64(summary["ess_bulk"]),
    r_hat      = np.float64(summary["r_hat"]),
    # Slice sampler diagnostics
    slice_w_initial       = np.float64(slice_w),
    jags_adapt            = np.bool_(jags_adapt),
    mean_theta_shrink     = np.float64(np.mean(shrinks)),
    mean_theta_step       = np.float64(np.mean(steps)),
    adapted_slice_w_theta = np.float64(np.mean(w_thetas)),
    adapted_slice_w_phi   = np.float64(np.mean(w_phis)),
    use_asis              = np.bool_(use_asis),
    mean_asis_shrink      = np.float64(np.mean([d["mean_asis_shrink"] for d in result["chain_dicts"]])),
    mean_asis_step        = np.float64(np.mean([d["mean_asis_step"]   for d in result["chain_dicts"]])),
    # Dataset metadata
    K              = np.float64(np.mean(y)),
    M              = np.int32(len(y)),
    n_blocks       = np.int32(block_info["n_blocks"]),
    n_block_sizes  = np.int32(len(block_info["sizes_summary"])),
    corr_threshold = np.float64(block_info["corr_threshold"]),
    max_block_size = np.int32(max(block_info["sizes_summary"].keys())),
    n_relatives    = np.int32(sum(s*c for s,c in block_info["sizes_summary"].items() if s>=2)),
    block_sizes    = sizes,
    block_counts   = counts,
    # Timing
    wall_time_s    = np.float64(wall_time),
    # Sampler settings
    mu_prior_scale = np.float64(result["mu_prior_scale"]),
    mu_prior_df    = np.float64(result["mu_prior_df"]),
    n_warmup       = np.int32(result["cfg"].n_warmup),
    n_samples      = np.int32(result["cfg"].n_samples),
    n_chains       = np.int32(len(result["chain_dicts"])),
)
print(f"Results saved to {results_file}")

# Warmup trace plot (mu and log-likelihood vs iteration, one line per chain)
trace_file = f"trace_{PHENO}_{suffix}.png"
mu_wu = np.vstack([d["mu_warmup"] for d in result["chain_dicts"]])
ll_wu = np.vstack([d["ll_warmup"] for d in result["chain_dicts"]])
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for c in range(mu_wu.shape[0]):
    axes[0].plot(mu_wu[c], alpha=0.6, lw=0.7)
axes[0].set_ylabel("mu")
axes[0].set_title(f"Warmup trace — {PHENO}  {suffix}  (n_warmup={result['cfg'].n_warmup})")
for c in range(ll_wu.shape[0]):
    axes[1].plot(ll_wu[c], alpha=0.6, lw=0.7)
axes[1].set_ylabel("log-likelihood")
axes[1].set_xlabel("warmup iteration")
if n_warmup_phase1 > 0:
    for ax in axes:
        ax.axvline(n_warmup_phase1, color="k", ls="--", lw=1,
                   label=f"adaptation start (iter {n_warmup_phase1})")
    axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(trace_file, dpi=100)
plt.close()
print(f"Warmup trace saved to {trace_file}")

# Remove all files except outputs so dxjupyterlab does not upload
# the cloned repo, input data, and setup files to the project file store.
import shutil
_keep = {results_file, outfile, trace_file}
for _name in os.listdir("."):
    if _name not in _keep:
        if os.path.isdir(_name):
            shutil.rmtree(_name)
        else:
            os.remove(_name)
print("Cleanup done; uploading:", sorted(_keep))
