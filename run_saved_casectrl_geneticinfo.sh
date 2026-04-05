#!/bin/bash
# phi correction comparison: same settings as best run, add USE_PHI_CORRECTION=true
set -euo pipefail

export PHENO="RA"

export SLICE_W="0.5" JAGS_ADAPT="true" N_WARMUP_PHASE1="4000" USE_ASIS="false" USE_ALPHA_REPARAM="false" USE_PHI_CORRECTION="true"
bash run_dnanexus_job.sh casectrl_geneticinfo.cfg
