#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys

# Phenotype name passed as first argument from the submission script
PHENO = sys.argv[1] if len(sys.argv) > 1 else "unknown"

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
pip install -U --force-reinstall polyagamma
dx download -f genotype_data/grm/RA_data.npz
export PYTHONPATH=/opt/notebooks/geneticinfo:$PYTHONPATH
""")

subprocess.run(["bash", "run.sh"], check=True)

sys.path.insert(0, "/opt/notebooks/geneticinfo")

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

result = sample_posterior(
    L, y,
    blocks=blocks,
    n_warmup=2000,
    n_samples=10000,
    n_chains=8,
    n_blas_threads=1,
)

summary = summarize_and_plot(result, outfile=f"posterior_{PHENO}.png")

# Save results: posterior samples + summary statistics + block metadata.
# The relationship matrix (L) and per-block L sub-matrices are not saved.
gb         = blocks["gb"]
block_info = blocks["block_info"]

# Block size counts as parallel arrays (size, count) for metadata
sizes  = np.array(sorted(block_info["sizes_summary"].keys()), dtype=np.int32)
counts = np.array([block_info["sizes_summary"][s] for s in sizes], dtype=np.int32)

np.savez_compressed(
    f"results_{PHENO}.npz",
    # Posterior samples
    mu_samples    = np.concatenate([d["mu"]    for d in result["chain_dicts"]]),
    beta0_samples = np.concatenate([d["beta0"] for d in result["chain_dicts"]]),
    p_samples     = np.concatenate([d["p"]     for d in result["chain_dicts"]]),
    # Posterior summary
    mu_mean    = np.float64(summary["mean"]),
    mu_median  = np.float64(summary["median"]),
    mu_sd      = np.float64(summary["sd"]),
    mu_ci90_lo = np.float64(summary["ci90_lo"]),
    mu_ci90_hi = np.float64(summary["ci90_hi"]),
    ess_bulk   = np.float64(summary["ess_bulk"]),
    r_hat      = np.float64(summary["r_hat"]),
    # Dataset metadata
    K              = np.float64(np.mean(y)),
    M              = np.int32(len(y)),
    n_blocks       = np.int32(len(block_info["sizes_summary"])),
    block_sizes    = sizes,
    block_counts   = counts,
    # Sampler settings
    mu_prior_scale = np.float64(result["mu_prior_scale"]),
    mu_prior_df    = np.float64(result["mu_prior_df"]),
    n_warmup       = np.int32(result["n_warmup"]),
    n_samples      = np.int32(result["n_samples"]),
    n_chains       = np.int32(len(result["chain_dicts"])),
)
print(f"Results saved to results_{PHENO}.npz")
