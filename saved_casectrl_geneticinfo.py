#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys

# Suppress BLAS threading contention when running multiple chains via fork.
# geneticinfo.py also calls _set_blas_threads(1) via threadpoolctl at import,
# but setting env vars here before any import is an extra safeguard.
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

from geneticinfo import build_blocks, sample_posterior, summarize_and_plot, compress

RA_data = np.load("RA_data.npz")
L = RA_data["L"]
y = RA_data["y"]
print("L.shape =", L.shape, "  y.shape =", y.shape)

blocks = build_blocks(
    L,
    y,
    corr_threshold=0.03,
)

result = sample_posterior(
    L, y,
    blocks=blocks,
    n_warmup=2000,
    n_samples=10000,
    n_chains=8,
    n_blas_threads=1,
)

summarize_and_plot(result, outfile="posterior_RA.png")

compress(L, y, result, "compressed_model.npz", corr_threshold=0.03)
