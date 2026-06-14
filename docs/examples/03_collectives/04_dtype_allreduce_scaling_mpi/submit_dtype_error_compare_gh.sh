#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKS_LIST="${RANKS_LIST:-4 8 16 32 64}"
VALUE_COUNTS_LIST="${VALUE_COUNTS_LIST:-128M 256M 512M 1G}"
RUN_TAG="${RUN_TAG:-$(date +dtype_error_compare_%Y%m%d_%H%M%S)}"
ACCOUNT="${ACCOUNT:-CCR26006}"
PARTITION="${PARTITION:-gh}"
TIME_LIMIT="${TIME_LIMIT:-00:05:00}"

export NCCL_NVFP4_ALLREDUCE_IMPL="${NCCL_NVFP4_ALLREDUCE_IMPL:-native}"
export NCCL_ALGO="${NCCL_ALGO:-RING}"
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"

RESULTS_DIR=/work/09308/zhengmk/nccl/dtype_error_compare_results
mkdir -p "${RESULTS_DIR}"
job_file="${RESULTS_DIR}/${RUN_TAG}_jobs.csv"
: > "${job_file}"
echo "job_id,nranks,value_count" > "${job_file}"

for nranks in ${RANKS_LIST}; do
  for value_count in ${VALUE_COUNTS_LIST}; do
    job_name="err-${nranks}g-${value_count}"
    sbatch_output="$(sbatch \
      -A "${ACCOUNT}" \
      -p "${PARTITION}" \
      -J "${job_name}" \
      -N "${nranks}" \
      -n "${nranks}" \
      --ntasks-per-node=1 \
      -t "${TIME_LIMIT}" \
      --export=ALL,NRANKS="${nranks}",VALUE_COUNT="${value_count}",RUN_TAG="${RUN_TAG}" \
      "${SCRIPT_DIR}/dtype_error_compare_gh.slurm" 2>&1)"
    job_id="$(printf '%s\n' "${sbatch_output}" | grep -Eo '[0-9]+' | tail -n 1)"
    if [[ -z "${job_id}" ]]; then
      printf '%s\n' "${sbatch_output}" >&2
      echo "failed to parse sbatch job ID for nranks=${nranks} value_count=${value_count}" >&2
      exit 1
    fi
    echo "${job_id},${nranks},${value_count}" | tee -a "${job_file}"
  done
done

echo "run_tag=${RUN_TAG}"
echo "job_file=${job_file}"
echo "setting=ACCOUNT=${ACCOUNT} PARTITION=${PARTITION} TIME_LIMIT=${TIME_LIMIT} NCCL_NVFP4_ALLREDUCE_IMPL=${NCCL_NVFP4_ALLREDUCE_IMPL} NCCL_ALGO=${NCCL_ALGO} NCCL_PROTO=${NCCL_PROTO} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS} NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-default}"
