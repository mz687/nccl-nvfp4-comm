#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RANKS_LIST="${RANKS_LIST:-4 8 16 32 64}"
DTYPE="${DTYPE:-fp16}"
SCALING_BYTES="${SCALING_BYTES:-64MB,128MB,256MB,512MB,1GB,2GB,4GB,8GB}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
REPEATS="${REPEATS:-3}"
RUN_TAG="${RUN_TAG:-$(date +nvfp4_scale_%Y%m%d_%H%M%S)}"
METRIC="${METRIC:-algbw_gbps}"
ACCOUNT="${ACCOUNT:-CCR24047}"
PARTITION="${PARTITION:-gh}"

declare -a JOB_IDS=()

extract_job_id() {
  grep -Eo '[0-9]+' | tail -n 1
}

for nranks in ${RANKS_LIST}; do
  sbatch_output="$(sbatch \
    -A "${ACCOUNT}" \
    -p "${PARTITION}" \
    -J "nvfp4-scale-${nranks}g" \
    -N "${nranks}" \
    -n "${nranks}" \
    --ntasks-per-node=1 \
    --export=ALL,NRANKS="${nranks}",DTYPE="${DTYPE}",WARMUP="${WARMUP}",ITERS="${ITERS}",REPEATS="${REPEATS}",RUN_TAG="${RUN_TAG}" \
    "${SCRIPT_DIR}/nvfp4_allreduce_scaling_gh.slurm" 2>&1)"
  job_id="$(printf '%s\n' "${sbatch_output}" | extract_job_id)"
  if [[ -z "${job_id}" ]]; then
    printf '%s\n' "${sbatch_output}" >&2
    echo "failed to parse sbatch job ID for nranks=${nranks}" >&2
    exit 1
  fi
  JOB_IDS+=("${job_id}")
  echo "submitted scaling job: nranks=${nranks} job_id=${job_id}"
done

dependency="$(IFS=:; echo "${JOB_IDS[*]}")"
job_id_list="$(IFS=,; echo "${JOB_IDS[*]}")"
plot_output="$(sbatch \
  -A "${ACCOUNT}" \
  -p "${PARTITION}" \
  -J "nvfp4-plot" \
  --dependency="afterok:${dependency}" \
  --export=ALL,RUN_TAG="${RUN_TAG}",DTYPE="${DTYPE}",WARMUP="${WARMUP}",ITERS="${ITERS}",REPEATS="${REPEATS}",METRIC="${METRIC}" \
  "${SCRIPT_DIR}/nvfp4_allreduce_plot_gh.slurm" 2>&1)"
plot_job_id="$(printf '%s\n' "${plot_output}" | extract_job_id)"
if [[ -z "${plot_job_id}" ]]; then
  printf '%s\n' "${plot_output}" >&2
  echo "failed to parse plot job ID" >&2
  exit 1
fi

echo "submitted plot job: job_id=${plot_job_id} run_tag=${RUN_TAG} dependency=${dependency}"
echo "scaling_jobs=${JOB_IDS[*]}"
echo "plot_job=${plot_job_id}"
echo "run_tag=${RUN_TAG}"
