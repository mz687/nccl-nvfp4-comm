#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKS_LIST="${RANKS_LIST:-64}"
VALUE_COUNTS_LIST="${VALUE_COUNTS_LIST:-128M 256M 512M 1G}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURED_STEPS="${MEASURED_STEPS:-50}"
RUN_TAG="${RUN_TAG:-$(date +nvfp4_tune_%Y%m%d_%H%M%S)}"
ACCOUNT="${ACCOUNT:-CCR24047}"
PARTITION="${PARTITION:-gh}"

# Format: label|algo|proto|min_channels|max_channels
# Empty algo/proto/channel fields leave NCCL's auto-selection in control.
TUNING_CONFIGS="${TUNING_CONFIGS:-auto|||| ring_simple_16|RING|SIMPLE|16|16 ring_simple_32|RING|SIMPLE|32|32 ring_simple_48|RING|SIMPLE|48|48 ring_simple_64|RING|SIMPLE|64|64}"

RESULTS_DIR=/work/09308/zhengmk/nccl/dtype_value_count_single_case_results
mkdir -p "${RESULTS_DIR}"
job_file="${RESULTS_DIR}/${RUN_TAG}_jobs.csv"
: > "${job_file}"
echo "job_id,nranks,dtype,value_count,config,nccl_algo,nccl_proto,nccl_min_nchannels,nccl_max_nchannels" > "${job_file}"

for nranks in ${RANKS_LIST}; do
  for value_count in ${VALUE_COUNTS_LIST}; do
    for config in ${TUNING_CONFIGS}; do
      IFS='|' read -r label algo proto min_channels max_channels <<< "${config}"
      config_tag="${RUN_TAG}_${label}"
      job_name="tune-${nranks}g-nvfp4-${value_count}-${label}"
      export_args=(
        ALL
        "NRANKS=${nranks}"
        "DTYPE=nvfp4"
        "VALUE_COUNT=${value_count}"
        "WARMUP_STEPS=${WARMUP_STEPS}"
        "MEASURED_STEPS=${MEASURED_STEPS}"
        "RUN_TAG=${config_tag}"
      )
      if [[ -n "${algo}" ]]; then
        export_args+=("NCCL_ALGO=${algo}")
      else
        export_args+=("NCCL_AUTO_TUNE=1")
      fi
      if [[ -n "${proto}" ]]; then
        export_args+=("NCCL_PROTO=${proto}")
      else
        export_args+=("NCCL_PROTO")
      fi
      if [[ -n "${min_channels}" ]]; then
        export_args+=("NCCL_MIN_NCHANNELS=${min_channels}")
      else
        export_args+=("NCCL_MIN_NCHANNELS")
      fi
      if [[ -n "${max_channels}" ]]; then
        export_args+=("NCCL_MAX_NCHANNELS=${max_channels}")
      else
        export_args+=("NCCL_MAX_NCHANNELS")
      fi
      export_csv="$(IFS=,; printf '%s' "${export_args[*]}")"

      sbatch_output="$(sbatch \
        -A "${ACCOUNT}" \
        -p "${PARTITION}" \
        -J "${job_name}" \
        -N "${nranks}" \
        -n "${nranks}" \
        --ntasks-per-node=1 \
        --export="${export_csv}" \
        "${SCRIPT_DIR}/dtype_value_count_single_case_gh.slurm" 2>&1)"
      job_id="$(printf '%s\n' "${sbatch_output}" | grep -Eo '[0-9]+' | tail -n 1)"
      if [[ -z "${job_id}" ]]; then
        printf '%s\n' "${sbatch_output}" >&2
        echo "failed to parse sbatch job ID for nranks=${nranks} value_count=${value_count} config=${label}" >&2
        exit 1
      fi
      echo "${job_id},${nranks},nvfp4,${value_count},${label},${algo},${proto},${min_channels},${max_channels}" | tee -a "${job_file}"
    done
  done
done

echo "run_tag=${RUN_TAG}"
echo "job_file=${job_file}"
echo "warmup_steps=${WARMUP_STEPS} measured_steps=${MEASURED_STEPS}"
