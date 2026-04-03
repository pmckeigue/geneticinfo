#!/bin/bash
# Submit geneticinfo jobs for two slice_w values and download results.
# Uses the generic run_dnanexus_job.sh runner with casectrl_geneticinfo.cfg.

set -euo pipefail

export PHENO="RA"

for SLICE_W in 0.5 1.0; do
    export SLICE_W
    bash run_dnanexus_job.sh casectrl_geneticinfo.cfg
done
