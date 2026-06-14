#!/bin/bash -l
#SBATCH -J dtype_nsys
#SBATCH -A CCR26006
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o /work/09308/zhengmk/nccl/nvfp4_profile_logs/%x_%j.out
#SBATCH -e /work/09308/zhengmk/nccl/nvfp4_profile_logs/%x_%j.err

set -euo pipefail

NRANKS=${NRANKS:?NRANKS must be set via sbatch --export}
DTYPE=${DTYPE:-nvfp4}
VALUE_COUNT=${VALUE_COUNT:?VALUE_COUNT must be set via sbatch --export}
WARMUP_STEPS=${WARMUP_STEPS:-1}
MEASURED_STEPS=${MEASURED_STEPS:-3}
RUN_TAG=${RUN_TAG:?RUN_TAG must be set via sbatch --export}
PROFILE_RANKS=${PROFILE_RANKS:-0}

EXAMPLE_DIR=/home1/09308/zhengmk/work/nccl/docs/examples/03_collectives/04_dtype_allreduce_scaling_mpi
NCCL_HOME=${NCCL_HOME:-/work/09308/zhengmk/nccl/build-nvfp4}
CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/24.9/cuda/12.6}
LOG_DIR=/work/09308/zhengmk/nccl/nvfp4_profile_logs
RESULTS_DIR=/work/09308/zhengmk/nccl/nvfp4_profile_results

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
ml gcc/15.1.0 cuda/12.6 nvidia/24.9

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PMIX_MCA_psec=native
export NCCL_NVFP4_ALLREDUCE_IMPL="${NCCL_NVFP4_ALLREDUCE_IMPL:-native}"
export NCCL_ALGO="${NCCL_ALGO:-RING}"
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"

cd "${EXAMPLE_DIR}"
env OMPI_CXX="$(command -v g++)" make dtype_value_count_single_case_mpi NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}"

safe_count=${VALUE_COUNT//[^A-Za-z0-9]/_}
case_log="${RESULTS_DIR}/single_case_profile_${NRANKS}g_${DTYPE}_${safe_count}_${RUN_TAG}_job${SLURM_JOB_ID}.csv"
export PROFILE_BASE="${RESULTS_DIR}/nsys_${RUN_TAG}_${NRANKS}g_${DTYPE}_${safe_count}_rank%q{SLURM_PROCID}_job${SLURM_JOB_ID}"
export BENCH_ARGS="--dtype ${DTYPE} --value-count ${VALUE_COUNT} --warmup-steps ${WARMUP_STEPS} --measured-steps ${MEASURED_STEPS} --csv"
export PROFILE_RANKS

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} dtype=${DTYPE} value_count=${VALUE_COUNT} warmup_steps=${WARMUP_STEPS} measured_steps=${MEASURED_STEPS} run_tag=${RUN_TAG} profile_ranks=${PROFILE_RANKS} start=$(date -Is) host=$(hostname)"
echo "nvfp4_impl=${NCCL_NVFP4_ALLREDUCE_IMPL:-fallback} nccl_algo=${NCCL_ALGO:-auto} nccl_proto=${NCCL_PROTO:-auto} nccl_min_nchannels=${NCCL_MIN_NCHANNELS:-auto} nccl_max_nchannels=${NCCL_MAX_NCHANNELS:-auto} output=${case_log} profile_base=${PROFILE_BASE}"

srun --mpi=pmix -N "${NRANKS}" -n "${NRANKS}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  if [[ "${PROFILE_RANKS}" == "all" || ",${PROFILE_RANKS}," == *",${SLURM_PROCID},"* ]]; then
    nsys profile       --force-overwrite=true       --trace=cuda,nvtx,osrt       --sample=none       --cpuctxsw=none       --stats=true       -o "${PROFILE_BASE}"       ./dtype_value_count_single_case_mpi ${BENCH_ARGS}
  else
    ./dtype_value_count_single_case_mpi ${BENCH_ARGS}
  fi
' | tee "${case_log}"

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} dtype=${DTYPE} value_count=${VALUE_COUNT} done=$(date -Is)"
