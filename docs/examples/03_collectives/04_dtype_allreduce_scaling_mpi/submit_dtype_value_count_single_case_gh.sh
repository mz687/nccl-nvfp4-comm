#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKS_LIST="${RANKS_LIST:-4 8 16 32 64}"
DTYPES_LIST="${DTYPES_LIST:-fp32 bf16 fp16 fp8 nvfp4}"
VALUE_COUNTS_LIST="${VALUE_COUNTS_LIST:-128M 256M 512M 1G 2G}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURED_STEPS="${MEASURED_STEPS:-50}"
RUN_TAG="${RUN_TAG:-$(date +dtype_single_case_%Y%m%d_%H%M%S)}"
ACCOUNT="${ACCOUNT:-CCR26006}"
PARTITION="${PARTITION:-gh}"

export NCCL_ALGO="${NCCL_ALGO:-RING}"
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"

RESULTS_DIR=/work/09308/zhengmk/nccl/dtype_value_count_single_case_results
mkdir -p "${RESULTS_DIR}"
job_file="${RESULTS_DIR}/${RUN_TAG}_jobs.csv"
: > "${job_file}"
echo "job_id,nranks,dtype,value_count" > "${job_file}"

for nranks in ${RANKS_LIST}; do
  for dtype in ${DTYPES_LIST}; do
    for value_count in ${VALUE_COUNTS_LIST}; do
      job_name="one-${nranks}g-${dtype}-${value_count}"
      sbatch_output="$(sbatch \
        -A "${ACCOUNT}" \
        -p "${PARTITION}" \
        -J "${job_name}" \
        -N "${nranks}" \
        -n "${nranks}" \
        --ntasks-per-node=1 \
        --export=ALL,NRANKS="${nranks}",DTYPE="${dtype}",VALUE_COUNT="${value_count}",WARMUP_STEPS="${WARMUP_STEPS}",MEASURED_STEPS="${MEASURED_STEPS}",RUN_TAG="${RUN_TAG}" \
        "${SCRIPT_DIR}/dtype_value_count_single_case_gh.slurm" 2>&1)"
      job_id="$(printf '%s\n' "${sbatch_output}" | grep -Eo '[0-9]+' | tail -n 1)"
      if [[ -z "${job_id}" ]]; then
        printf '%s\n' "${sbatch_output}" >&2
        echo "failed to parse sbatch job ID for nranks=${nranks} dtype=${dtype} value_count=${value_count}" >&2
        exit 1
      fi
      echo "${job_id},${nranks},${dtype},${value_count}" | tee -a "${job_file}"
    done
  done
done

echo "run_tag=${RUN_TAG}"
echo "job_file=${job_file}"
echo "setting=NCCL_ALGO=${NCCL_ALGO} NCCL_PROTO=${NCCL_PROTO} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS} NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-default}"
echo "warmup_steps=${WARMUP_STEPS} measured_steps=${MEASURED_STEPS}"
