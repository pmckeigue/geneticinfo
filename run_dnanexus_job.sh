#!/bin/bash
# Generic DNAnexus job runner.
#
# Uploads a Python script (and any extra files) to DNAnexus project file
# storage, runs it via dxjupyterlab, waits for completion with retry logic,
# then downloads the result files to a local directory.
#
# Usage:
#   ./run_dnanexus_job.sh <config_file>
#
# The config file is sourced as bash.  Job-specific variables (e.g. PHENO,
# SLICE_W) should be exported in the environment before calling this script,
# because the config file is sourced after those exports and may reference them.
#
# Required config variables:
#   SCRIPT_NAME        local Python script to upload and run
#   JOB_CMD            command to execute inside the job
#   REMOTE_OUTPUT_DIR  DNAnexus folder where the job writes its output files
#   RESULT_FILES       bash array of filenames to download from REMOTE_OUTPUT_DIR
#   LOCAL_OUTPUT_DIR   local directory to download results into
#
# Optional config variables (defaults shown):
#   EXTRA_UPLOADS=()        additional local files to upload to REMOTE_FOLDER
#   REMOTE_FOLDER="ancestry_kinship"
#   SNAPSHOT=".Notebook_snapshots/snapshot-jupyterlab-pheno.tar.gz"
#   INSTANCE_TYPE="mem2_ssd1_v2_x16"
#   MAX_RETRIES=2
#   JOB_NAME="${SCRIPT_NAME}"   human-readable job name shown in DNAnexus UI
#
# Example config (casectrl_geneticinfo.cfg):
#   SCRIPT_NAME="saved_casectrl_geneticinfo.py"
#   EXTRA_UPLOADS=(pg_gibbs_clean.py)
#   REMOTE_OUTPUT_DIR="/summarylevel_results"
#   JOB_CMD="python3 ${SCRIPT_NAME} ${PHENO} ${SLICE_W}"
#   JOB_NAME="genetic info ${PHENO} sw${SLICE_W}"
#   RESULT_FILES=("results_${PHENO}_sw${SLICE_W}.npz" "posterior_${PHENO}_sw${SLICE_W}.png")
#   LOCAL_OUTPUT_DIR="summarylevel_results/${PHENO}"
#
# Example invocation:
#   export PHENO=RA SLICE_W=1.0
#   ./run_dnanexus_job.sh casectrl_geneticinfo.cfg

set -euo pipefail

CONFIG_FILE="${1:?Usage: $0 <config_file>}"

# ── Defaults (overridable by config) ─────────────────────────────────────────
EXTRA_UPLOADS=()
REMOTE_FOLDER="ancestry_kinship"
SNAPSHOT=".Notebook_snapshots/snapshot-jupyterlab-pheno.tar.gz"
INSTANCE_TYPE="mem2_ssd1_v2_x16"
MAX_RETRIES=2

# ── Source config ─────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

# Apply JOB_NAME default after config (config may set it, else fall back)
JOB_NAME="${JOB_NAME:-${SCRIPT_NAME}}"

# Validate required variables
: "${SCRIPT_NAME:?Config must set SCRIPT_NAME}"
: "${JOB_CMD:?Config must set JOB_CMD}"
: "${REMOTE_OUTPUT_DIR:?Config must set REMOTE_OUTPUT_DIR}"
: "${LOCAL_OUTPUT_DIR:?Config must set LOCAL_OUTPUT_DIR}"
if [[ ${#RESULT_FILES[@]} -eq 0 ]]; then
    echo "ERROR: Config must set RESULT_FILES as a non-empty bash array" >&2
    exit 1
fi

# ── Upload ────────────────────────────────────────────────────────────────────

upload_files() {
    local all_files=("${SCRIPT_NAME}")
    if [[ ${#EXTRA_UPLOADS[@]} -gt 0 ]]; then
        all_files+=("${EXTRA_UPLOADS[@]}")
    fi

    echo "Uploading ${#all_files[@]} file(s) to DNAnexus /${REMOTE_FOLDER}/..."
    for f in "${all_files[@]}"; do
        python3 -c "
import sys; sys.path.insert(0, '.')
from ukbb_utils import dx_upload
dx_upload('./${f}', '/${REMOTE_FOLDER}/')
"
    done
    echo "Upload complete"
}

# ── Job submission ────────────────────────────────────────────────────────────

run_job() {
    local priority=$1
    local attempt=$2

    echo "Submitting job (attempt ${attempt}, priority: ${priority})..."
    echo "  name: ${JOB_NAME}"
    echo "  cmd:  ${JOB_CMD}"
    echo "  dest: ${REMOTE_OUTPUT_DIR}/"

    local job_output
    job_output=$(dx run dxjupyterlab \
        -iin="${REMOTE_FOLDER}/${SCRIPT_NAME}" \
        -isnapshot="${SNAPSHOT}" \
        -icmd="${JOB_CMD}" \
        --yes --brief \
        --destination="${REMOTE_OUTPUT_DIR}/" \
        --instance-type="${INSTANCE_TYPE}" \
        --priority="${priority}" \
        --name="${JOB_NAME} (attempt ${attempt})")

    local job_id
    job_id=$(echo "${job_output}" | grep -oP 'job-[A-Za-z0-9]+' | head -1)

    if [[ -z "${job_id}" ]]; then
        echo "ERROR: Could not parse job ID from dx run output" >&2
        return 1
    fi

    echo "Started job: ${job_id}"

    if dx wait "${job_id}"; then
        echo "Job completed successfully"
        return 0
    else
        echo "Job failed"
        return 1
    fi
}

# ── Download ──────────────────────────────────────────────────────────────────

download_results() {
    echo "Downloading ${#RESULT_FILES[@]} result file(s) to ${LOCAL_OUTPUT_DIR}/..."
    mkdir -p "${LOCAL_OUTPUT_DIR}"
    for f in "${RESULT_FILES[@]}"; do
        # When duplicate remote files exist (from earlier runs), pick the most
        # recently created one by file ID (dx find data returns oldest first).
        local file_id
        file_id=$(dx find data --brief --name "${f}" --path "${REMOTE_OUTPUT_DIR}" 2>/dev/null | tail -1)
        if [[ -z "${file_id}" ]]; then
            echo "ERROR: remote file not found: ${REMOTE_OUTPUT_DIR}/${f}" >&2
            return 1
        fi
        dx download "${file_id}" -o "${LOCAL_OUTPUT_DIR}/${f}" -f
        echo "  Downloaded: ${f}"
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "================================================"
echo "DNAnexus job: ${JOB_NAME}"
echo "  script:   ${SCRIPT_NAME}"
echo "  instance: ${INSTANCE_TYPE}"
echo "================================================"

upload_files

attempt=1
success=false

while [[ ${attempt} -le ${MAX_RETRIES} ]]; do
    if [[ ${attempt} -gt 1 ]]; then
        priority="high"
        echo "Retrying with high priority (attempt ${attempt})..."
    else
        priority="normal"
    fi

    if run_job "${priority}" "${attempt}"; then
        success=true
        break
    fi

    attempt=$((attempt + 1))
    if [[ ${attempt} -le ${MAX_RETRIES} ]]; then
        echo "Job failed. Retrying in 5s..."
        sleep 5
    fi
done

if [[ "${success}" != "true" ]]; then
    echo "ERROR: Job failed after ${MAX_RETRIES} attempts" >&2
    exit 1
fi

download_results
echo "Done. Results saved to ${LOCAL_OUTPUT_DIR}/"
