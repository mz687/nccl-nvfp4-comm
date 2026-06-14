#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKS_LIST="${RANKS_LIST:-4 8 16 32 64}"
export DTYPES="${DTYPES:-fp32,bf16,fp16,fp8}"
export MESSAGE_BYTES="${MESSAGE_BYTES:-128MiB,256MiB,512MiB,1GiB}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
REPEATS="${REPEATS:-3}"
RUN_TAG="${RUN_TAG:-$(date +dtype_scale_%Y%m%d_%H%M%S)}"
ACCOUNT="${ACCOUNT:-CCR26006}"
PARTITION="${PARTITION:-gh}"

export NCCL_ALGO="${NCCL_ALGO:-RING}"
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"

declare -a JOB_IDS=()

extract_job_id() {
  grep -Eo '[0-9]+' | tail -n 1
}

for nranks in ${RANKS_LIST}; do
  sbatch_output="$(sbatch \
    -A "${ACCOUNT}" \
    -p "${PARTITION}" \
    -J "dtype-scale-${nranks}g" \
    -N "${nranks}" \
    -n "${nranks}" \
    --ntasks-per-node=1 \
    --export=ALL,NRANKS="${nranks}",WARMUP="${WARMUP}",ITERS="${ITERS}",REPEATS="${REPEATS}",RUN_TAG="${RUN_TAG}" \
    "${SCRIPT_DIR}/dtype_allreduce_scaling_gh.slurm" 2>&1)"
  job_id="$(printf '%s\n' "${sbatch_output}" | extract_job_id)"
  if [[ -z "${job_id}" ]]; then
    printf '%s\n' "${sbatch_output}" >&2
    echo "failed to parse sbatch job ID for nranks=${nranks}" >&2
    exit 1
  fi
  JOB_IDS+=("${job_id}")
  echo "submitted dtype scaling job: nranks=${nranks} job_id=${job_id}"
done

dependency="$(IFS=:; echo "${JOB_IDS[*]}")"
job_id_list="$(IFS=,; echo "${JOB_IDS[*]}")"
plot_output="$(sbatch \
  -A "${ACCOUNT}" \
  -p "${PARTITION}" \
  -J "dtype-scale-plot" \
  --dependency="afterok:${dependency}" \
  --export=ALL,RUN_TAG="${RUN_TAG}",JOB_IDS="${job_id_list}",WARMUP="${WARMUP}",ITERS="${ITERS}",REPEATS="${REPEATS}" \
  "${SCRIPT_DIR}/dtype_allreduce_scaling_plot_gh.slurm" 2>&1)"
plot_job_id="$(printf '%s\n' "${plot_output}" | extract_job_id)"
if [[ -z "${plot_job_id}" ]]; then
  printf '%s\n' "${plot_output}" >&2
  echo "failed to parse plot job ID" >&2
  exit 1
fi

echo "submitted dtype scaling plot job: job_id=${plot_job_id} run_tag=${RUN_TAG} dependency=${dependency}"
echo "scaling_jobs=${JOB_IDS[*]}"
echo "plot_job=${plot_job_id}"
echo "run_tag=${RUN_TAG}"
echo "setting=NCCL_ALGO=${NCCL_ALGO} NCCL_PROTO=${NCCL_PROTO} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}"
