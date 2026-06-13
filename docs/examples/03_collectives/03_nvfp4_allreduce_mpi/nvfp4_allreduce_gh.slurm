#!/bin/bash -l
#SBATCH -J nvfp4_ar
#SBATCH -A CCR24047
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o /work/09308/zhengmk/nccl/nvfp4_slurm_logs/%x_%j.out
#SBATCH -e /work/09308/zhengmk/nccl/nvfp4_slurm_logs/%x_%j.err

set -euo pipefail

NRANKS=${NRANKS:?NRANKS must be set via sbatch --export}
DTYPE=${DTYPE:-fp16}
CORRECTNESS_BYTES=${CORRECTNESS_BYTES:-4MB,4194306}
SCALING_BYTES=${SCALING_BYTES:-64MB,128MB,256MB,512MB,1GB,2GB,4GB,8GB}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}

NCCL_DIR=/home1/09308/zhengmk/work/nccl
EXAMPLE_DIR=/home1/09308/zhengmk/work/nccl/docs/examples/03_collectives/03_nvfp4_allreduce_mpi
NCCL_HOME=/work/09308/zhengmk/nccl/build-nvfp4
CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.9/cuda/12.6
LOG_DIR=/work/09308/zhengmk/nccl/nvfp4_slurm_logs
RESULTS_DIR=/work/09308/zhengmk/nccl/nvfp4_allreduce_results

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
ml gcc/15.1.0 cuda/12.6 nvidia/24.9

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PMIX_MCA_psec=native
cd "${EXAMPLE_DIR}"
env OMPI_CXX="$(command -v g++)" make NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}"

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} dtype=${DTYPE} start=$(date -Is) host=$(hostname)"
CORR_LOG="${RESULTS_DIR}/correctness_${DTYPE}_${NRANKS}g_job${SLURM_JOB_ID}.txt"
SCALE_LOG="${RESULTS_DIR}/scaling_${DTYPE}_${NRANKS}g_job${SLURM_JOB_ID}.csv"

srun --mpi=pmix -N "${NRANKS}" -n "${NRANKS}" --ntasks-per-node=1 ./nvfp4_allreduce_mpi --mode correctness --dtype "${DTYPE}" --correctness-bytes "${CORRECTNESS_BYTES}" | tee "${CORR_LOG}"
srun --mpi=pmix -N "${NRANKS}" -n "${NRANKS}" --ntasks-per-node=1 ./nvfp4_allreduce_mpi --mode scaling --dtype "${DTYPE}" --scaling-bytes "${SCALING_BYTES}" --warmup "${WARMUP}" --iters "${ITERS}" --csv | tee "${SCALE_LOG}"

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} done=$(date -Is)"
