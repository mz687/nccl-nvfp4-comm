#!/bin/bash -l
#SBATCH -J dtype_one_case
#SBATCH -A CCR26006
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o /work/09308/zhengmk/nccl/dtype_value_count_single_case_logs/%x_%j.out
#SBATCH -e /work/09308/zhengmk/nccl/dtype_value_count_single_case_logs/%x_%j.err

set -euo pipefail

NRANKS=${NRANKS:?NRANKS must be set via sbatch --export}
DTYPE=${DTYPE:?DTYPE must be set via sbatch --export}
VALUE_COUNT=${VALUE_COUNT:?VALUE_COUNT must be set via sbatch --export}
WARMUP_STEPS=${WARMUP_STEPS:-5}
MEASURED_STEPS=${MEASURED_STEPS:-50}
RUN_TAG=${RUN_TAG:?RUN_TAG must be set via sbatch --export}

EXAMPLE_DIR=/home1/09308/zhengmk/work/nccl/docs/examples/03_collectives/04_dtype_allreduce_scaling_mpi
NCCL_HOME=${NCCL_HOME:-/work/09308/zhengmk/nccl/build-nvfp4}
CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/24.9/cuda/12.6}
LOG_DIR=/work/09308/zhengmk/nccl/dtype_value_count_single_case_logs
RESULTS_DIR=/work/09308/zhengmk/nccl/dtype_value_count_single_case_results

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
ml gcc/15.1.0 cuda/12.6 nvidia/24.9

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PMIX_MCA_psec=native
export NCCL_NVFP4_ALLREDUCE_IMPL="${NCCL_NVFP4_ALLREDUCE_IMPL:-native}"
if [[ "${NCCL_AUTO_TUNE:-0}" == "1" ]]; then
  unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
else
  export NCCL_ALGO="${NCCL_ALGO:-RING}"
  export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
  export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
fi

cd "${EXAMPLE_DIR}"
env OMPI_CXX="$(command -v g++)" make dtype_value_count_single_case_mpi NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}"

safe_count=${VALUE_COUNT//[^A-Za-z0-9]/_}
case_log="${RESULTS_DIR}/single_case_${NRANKS}g_${DTYPE}_${safe_count}_${RUN_TAG}_job${SLURM_JOB_ID}.csv"

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} dtype=${DTYPE} value_count=${VALUE_COUNT} warmup_steps=${WARMUP_STEPS} measured_steps=${MEASURED_STEPS} run_tag=${RUN_TAG} start=$(date -Is) host=$(hostname)"
echo "nvfp4_impl=${NCCL_NVFP4_ALLREDUCE_IMPL:-fallback} nccl_auto_tune=${NCCL_AUTO_TUNE:-0} nccl_algo=${NCCL_ALGO:-auto} nccl_proto=${NCCL_PROTO:-auto} nccl_min_nchannels=${NCCL_MIN_NCHANNELS:-auto} nccl_max_nchannels=${NCCL_MAX_NCHANNELS:-auto} nccl_buffsize=${NCCL_BUFFSIZE:-default} output=${case_log}"

srun --mpi=pmix -N "${NRANKS}" -n "${NRANKS}" --ntasks-per-node=1 ./dtype_value_count_single_case_mpi \
  --dtype "${DTYPE}" \
  --value-count "${VALUE_COUNT}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measured-steps "${MEASURED_STEPS}" \
  --csv | tee "${case_log}"

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} dtype=${DTYPE} value_count=${VALUE_COUNT} done=$(date -Is)"
