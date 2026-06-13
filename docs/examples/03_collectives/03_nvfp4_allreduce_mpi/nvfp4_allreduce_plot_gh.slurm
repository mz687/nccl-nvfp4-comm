#!/bin/bash -l
#SBATCH -J nvfp4_ar_plot
#SBATCH -A CCR24047
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o /work/09308/zhengmk/nccl/nvfp4_slurm_logs/%x_%j.out
#SBATCH -e /work/09308/zhengmk/nccl/nvfp4_slurm_logs/%x_%j.err

set -euo pipefail

RUN_TAG=${RUN_TAG:-}
JOB_IDS=${JOB_IDS:-}
DTYPE=${DTYPE:-fp16}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
REPEATS=${REPEATS:-3}
METRIC=${METRIC:-algbw_gbps}

if [[ -z "${RUN_TAG}" && -z "${JOB_IDS}" ]]; then
  echo "set RUN_TAG or JOB_IDS before launching the plot job" >&2
  exit 1
fi

EXAMPLE_DIR=/home1/09308/zhengmk/work/nccl/docs/examples/03_collectives/03_nvfp4_allreduce_mpi
RESULTS_DIR=/work/09308/zhengmk/nccl/nvfp4_allreduce_results
PLOTS_DIR=/work/09308/zhengmk/nccl/nvfp4_plots

mkdir -p "${RESULTS_DIR}" "${PLOTS_DIR}"
ml gcc/15.1.0 cuda/12.6 nvidia/24.9

cd "${EXAMPLE_DIR}"
source /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate
plot_cmd=(
  python plot_scaling.py
  --results-dir "${RESULTS_DIR}"
  --dtype "${DTYPE}"
  --metric "${METRIC}"
  --output-prefix "${PLOTS_DIR}/nvfp4_allreduce_${DTYPE}_${RUN_TAG:-jobs_${JOB_IDS//,/__}}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --repeats "${REPEATS}"
)
if [[ -n "${RUN_TAG}" ]]; then
  plot_cmd+=(--run-tag "${RUN_TAG}")
fi
if [[ -n "${JOB_IDS}" ]]; then
  plot_cmd+=(--job-ids "${JOB_IDS}")
fi
"${plot_cmd[@]}"
