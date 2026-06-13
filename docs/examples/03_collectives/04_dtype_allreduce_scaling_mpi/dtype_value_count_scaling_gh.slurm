#!/bin/bash -l
#SBATCH -J dtype_val_scale
#SBATCH -A CCR24047
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -o /work/09308/zhengmk/nccl/dtype_value_count_scaling_logs/%x_%j.out
#SBATCH -e /work/09308/zhengmk/nccl/dtype_value_count_scaling_logs/%x_%j.err

set -euo pipefail

NRANKS=${NRANKS:?NRANKS must be set via sbatch --export}
DTYPES=${DTYPES:-fp32,bf16,fp16,fp8,nvfp4}
VALUE_COUNTS=${VALUE_COUNTS:-128M,256M,512M,1G,2G}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
REPEATS=${REPEATS:-1}
SWEEP_REPEATS=${SWEEP_REPEATS:-1}
RUN_TAG=${RUN_TAG:?RUN_TAG must be set via sbatch --export}

EXAMPLE_DIR=/home1/09308/zhengmk/work/nccl/docs/examples/03_collectives/04_dtype_allreduce_scaling_mpi
NCCL_HOME=${NCCL_HOME:-/work/09308/zhengmk/nccl/build-nvfp4}
CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/24.9/cuda/12.6}
LOG_DIR=/work/09308/zhengmk/nccl/dtype_value_count_scaling_logs
RESULTS_DIR=/work/09308/zhengmk/nccl/dtype_value_count_scaling_results

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
ml gcc/15.1.0 cuda/12.6 nvidia/24.9

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PMIX_MCA_psec=native
export NCCL_ALGO="${NCCL_ALGO:-RING}"
export NCCL_PROTO="${NCCL_PROTO:-SIMPLE}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"

cd "${EXAMPLE_DIR}"
env OMPI_CXX="$(command -v g++)" make dtype_value_count_scaling_mpi NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}"

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} dtypes=${DTYPES} value_counts=${VALUE_COUNTS} repeats=${REPEATS} sweep_repeats=${SWEEP_REPEATS} warmup=${WARMUP} iters=${ITERS} run_tag=${RUN_TAG} start=$(date -Is) host=$(hostname)"
echo "nccl_algo=${NCCL_ALGO} nccl_proto=${NCCL_PROTO} nccl_min_nchannels=${NCCL_MIN_NCHANNELS} nccl_max_nchannels=${NCCL_MAX_NCHANNELS}"

for repeat in $(seq 1 "${REPEATS}"); do
  scale_log="${RESULTS_DIR}/value_count_scaling_${NRANKS}g_${RUN_TAG}_job${SLURM_JOB_ID}_repeat${repeat}.csv"
  echo "repeat=${repeat} output=${scale_log}"
  srun --mpi=pmix -N "${NRANKS}" -n "${NRANKS}" --ntasks-per-node=1 ./dtype_value_count_scaling_mpi \
    --dtypes "${DTYPES}" \
    --value-counts "${VALUE_COUNTS}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --sweep-repeats "${SWEEP_REPEATS}" \
    --csv | tee "${scale_log}"
done

echo "job_id=${SLURM_JOB_ID} rank_count=${NRANKS} done=$(date -Is)"
