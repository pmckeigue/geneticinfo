#!/bin/bash

# Run saved_casectrl_geneticinfo.py  in Swiss Army Knife app with retry logic
# Results saved to ./summarylevel_results/{PHENO}/

set -o pipefail

SCRIPT_NAME="saved_casectrl_geneticinfo.py"
SCRIPT_LOCAL="./${SCRIPT_NAME}"
REMOTE_FOLDER="ancestry_kinship"
PROJECT_ID="project-Gx64538Jzb269Xvj5bffpBg1"
MAX_RETRIES=2
INSTANCE_TYPE="mem2_ssd1_v2_x16"

# Get phenotype from environment)
PHENO="RA" 
#REGRESSION_METHOD=${REGRESSION_METHOD:-sklearn}
OUTPUT_DIR="/summarylevel_results"
#REMOTE_PHENO_DIR="${OUTPUT_DIR}/${PHENO}"
REMOTE_PHENO_DIR="${OUTPUT_DIR}"

# Upload script using dx_upload
upload_files() {
    echo "Uploading scripts to DNAnexus..."
    python3 -c "
import sys
sys.path.insert(0, '.')
from ukbb_utils import dx_upload
dx_upload('${SCRIPT_LOCAL}', '/${REMOTE_FOLDER}/')
dx_upload('./pg_gibbs_clean.py', '/${REMOTE_FOLDER}/')
"
    echo "Upload complete"
}

# Check if results already exist
check_existing_results() {
    count=$(dx find data --name "results_${PHENO}.npz" --path ${REMOTE_PHENO_DIR} --brief 2>/dev/null | wc -l)
    if [ "$count" -gt 1 ]; then
        echo "Warning: Found $count duplicates, will download"
        return 1
    elif dx ls "${REMOTE_PHENO_DIR}/results_${PHENO}.npz" >/dev/null 2>&1; then
        echo "Results already exist for ${PHENO} in ${REMOTE_PHENO_DIR}"
        return 0
    fi
    return 1
}

# Run one job for a given slice_w value
run_job() {
    local priority=$1
    local attempt=$2
    local sw=$3

    echo "Running job (attempt $attempt, priority: $priority, slice_w: $sw)..."
    echo "  PHENO: ${PHENO}"
    echo "  Remote output: ${REMOTE_PHENO_DIR}/"

    JOB_OUTPUT=$(dx run dxjupyterlab \
        -iin="${REMOTE_FOLDER}/${SCRIPT_NAME}" \
        -isnapshot=".Notebook_snapshots/snapshot-jupyterlab-pheno.tar.gz" \
        -icmd="python3 ${SCRIPT_NAME} ${PHENO} ${sw}" \
        --yes --brief --destination="${REMOTE_PHENO_DIR}/" \
        --instance-type=${INSTANCE_TYPE} \
        --priority=${priority} \
        --name="genetic info ${PHENO} sw${sw} (attempt $attempt)")

    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'job-[A-Za-z0-9]+' | head -1)

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Could not get job ID"
        return 1
    fi

    echo "Started job: ${JOB_ID}"

    if dx wait ${JOB_ID}; then
        echo "Job completed successfully"
        return 0
    else
        echo "Job failed"
        return 1
    fi
}

# Run one slice_w value with retry logic; download result on success
run_slice_w() {
    local sw=$1
    local attempt=1
    local success=false

    while [ $attempt -le $MAX_RETRIES ]; do
        if [ $attempt -gt 1 ]; then
            priority="high"
            echo "Retrying slice_w=${sw} with high priority (attempt $attempt)..."
        else
            priority="normal"
        fi

        if run_job $priority $attempt $sw; then
            success=true
            break
        else
            attempt=$((attempt + 1))
            if [ $attempt -le $MAX_RETRIES ]; then
                echo "Job failed. Will retry..."
                sleep 5
            fi
        fi
    done

    if [ "$success" = false ]; then
        echo "ERROR: slice_w=${sw} job failed after $MAX_RETRIES attempts"
        return 1
    fi

    echo "Downloading results for slice_w=${sw}..."
    mkdir -p summarylevel_results/${PHENO}
    dx download "${REMOTE_PHENO_DIR}/results_${PHENO}_sw${sw}.npz"   -o summarylevel_results/${PHENO} -f
    dx download "${REMOTE_PHENO_DIR}/posterior_${PHENO}_sw${sw}.png" -o summarylevel_results/${PHENO} -f
    echo "Done: results_${PHENO}_sw${sw}.npz saved to ./summarylevel_results/${PHENO}/"
}

main() {
    echo "=========================================="
    echo "Running saved_casectrl_geneticinfo.py"
    echo "  PHENO: ${PHENO}"
    echo "=========================================="

    upload_files

    run_slice_w 1.0
    run_slice_w 1.5

    echo "All jobs complete."
}

main "$@"
