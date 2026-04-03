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

# JAGS-adaptive run starting from slice_w=1.0
export SLICE_W="1.0" JAGS_ADAPT="true"
bash run_dnanexus_job.sh casectrl_geneticinfo.cfg
