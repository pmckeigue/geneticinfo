#!/bin/bash
# n_warmup=5000 with JAGS adaptation in last 1000 iters; n_samples=5000
set -euo pipefail

export PHENO="RA"

# Two-phase: 4000 iters fixed w=0.5, then 1000 iters JAGS adaptation
export SLICE_W="0.5" JAGS_ADAPT="true" N_WARMUP_PHASE1="4000" USE_ASIS="false"
bash run_dnanexus_job.sh casectrl_geneticinfo.cfg
