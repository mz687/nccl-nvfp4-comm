#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

set -euo pipefail

RANKS_LIST="${RANKS_LIST:-4 8 16 32 64}"
SCALING_BYTES="${SCALING_BYTES:-64MB,128MB,256MB,512MB,1GB,2GB,4GB,8GB}"
DTYPE="${DTYPE:-fp16}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
REPEATS="${REPEATS:-1}"
CSV_FLAG="${CSV_FLAG:---csv}"
LAUNCH_TEMPLATE="${LAUNCH_TEMPLATE:-mpirun -np {NRANKS}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${TARGET:-${SCRIPT_DIR}/nvfp4_allreduce_mpi}"

if [[ ! -x "${TARGET}" ]]; then
  echo "missing executable: ${TARGET}" >&2
  echo "build it first with: make -C ${SCRIPT_DIR}" >&2
  exit 1
fi

for nranks in ${RANKS_LIST}; do
  launch_cmd="${LAUNCH_TEMPLATE//\{NRANKS\}/${nranks}}"
  for repeat in $(seq 1 "${REPEATS}"); do
    echo "=== NVFP4 allreduce scaling: nranks=${nranks} repeat=${repeat}/${REPEATS} bytes=${SCALING_BYTES} dtype=${DTYPE} warmup=${WARMUP} iters=${ITERS} ==="
    ${launch_cmd} "${TARGET}" \
      --mode scaling \
      --dtype "${DTYPE}" \
      --scaling-bytes "${SCALING_BYTES}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      ${CSV_FLAG}
  done
done
