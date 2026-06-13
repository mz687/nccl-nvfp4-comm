#!/bin/bash -l
#SBATCH -J nvfp4_ar_scale
#SBATCH -A CCR24047
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -o /work/09308/zhengmk/nccl/nvfp4_slurm_logs/%x_%j.out
#SBATCH -e /work/09308/zhengmk/nccl/nvfp4_slurm_logs/%x_%j.err

set -euo pipefail

NRANKS=${NRANKS:?NRANKS must be set via sbatch --export}
DTYPE=${DTYPE:-fp16}
SCALING_BYTES=${SCALING_BYTES:-64MB,128MB,256MB,512MB,1GB,2GB,4GB,8GB}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
REPEATS=${REPEATS:-3}
RUN_TAG=${RUN_TAG:?RUN_TAG must be set via sbatch --export}

EXAMPLE_DIR=/home1/09308/zhengmk/work/nccl/docs/examples/03_collectives/03_nvfp4_allreduce_mpi
NCCL_HOME=${NCCL_HOME:-/work/09308/zhengmk/nccl/build-nvfp4}
CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/24.9/cuda/12.6}
LOG_DIR=/work/09308/zhengmk/nccl/nvfp4_slurm_logs
RESULTS_DIR=/work/09308/zhengmk/nccl/nvfp4_allreduce_results

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
ml gcc/15.1.0 cuda/12.6 nvidia/24.9

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PMIX_MCA_psec=native
cd "${EXAMPLE_DIR}"
env OMPI_CXX="$(command -v g++)" make NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}"

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} dtype=${DTYPE} repeats=${REPEATS} warmup=${WARMUP} iters=${ITERS} run_tag=${RUN_TAG} start=$(date -Is) host=$(hostname)"

for repeat in $(seq 1 "${REPEATS}"); do
  scale_log="${RESULTS_DIR}/scaling_${DTYPE}_${NRANKS}g_${RUN_TAG}_job${SLURM_JOB_ID}_repeat${repeat}.csv"
  echo "repeat=${repeat} output=${scale_log}"
  srun --mpi=pmix -N "${NRANKS}" -n "${NRANKS}" --ntasks-per-node=1 ./nvfp4_allreduce_mpi \
    --mode scaling \
    --dtype "${DTYPE}" \
    --scaling-bytes "${SCALING_BYTES}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --csv | tee "${scale_log}"
done

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} done=$(date -Is)"
