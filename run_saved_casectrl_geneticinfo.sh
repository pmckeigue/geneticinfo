#!/bin/bash
# Submit ASIS comparison job.
set -euo pipefail

export PHENO="RA"

# ASIS run: ancillarity-sufficiency interweaving for theta, fixed w=0.5
export SLICE_W="0.5" JAGS_ADAPT="false" N_WARMUP_PHASE1="0" USE_ASIS="true"
bash run_dnanexus_job.sh casectrl_geneticinfo.cfg
