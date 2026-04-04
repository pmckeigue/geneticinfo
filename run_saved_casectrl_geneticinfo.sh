#!/bin/bash
# Submit geneticinfo jobs and download results.
# Fixed slice widths (0.5 and 1.0) plus one JAGS-adapted run.

set -euo pipefail

export PHENO="RA"

# Fixed-width runs for comparison
for SLICE_W in 0.5 1.0; do
    export SLICE_W JAGS_ADAPT="false"
    bash run_dnanexus_job.sh casectrl_geneticinfo.cfg
done

# JAGS-adaptive run: phase 1 (1000 iters, fixed w=0.5), phase 2 (1000 iters, adapt)
export SLICE_W="0.5" JAGS_ADAPT="true" N_WARMUP_PHASE1="1000"
bash run_dnanexus_job.sh casectrl_geneticinfo.cfg
