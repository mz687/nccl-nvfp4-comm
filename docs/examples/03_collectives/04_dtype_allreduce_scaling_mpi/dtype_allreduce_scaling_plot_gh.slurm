#!/bin/bash -l
#SBATCH -J dtype_ar_plot
#SBATCH -A CCR24047
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o /work/09308/zhengmk/nccl/dtype_scaling_logs/%x_%j.out
#SBATCH -e /work/09308/zhengmk/nccl/dtype_scaling_logs/%x_%j.err

set -euo pipefail

RUN_TAG=${RUN_TAG:-}
JOB_IDS=${JOB_IDS:-}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
REPEATS=${REPEATS:-3}

if [[ -z "${RUN_TAG}" && -z "${JOB_IDS}" ]]; then
  echo "set RUN_TAG or JOB_IDS before launching the plot job" >&2
  exit 1
fi

EXAMPLE_DIR=/home1/09308/zhengmk/work/nccl/docs/examples/03_collectives/04_dtype_allreduce_scaling_mpi
RESULTS_DIR=/work/09308/zhengmk/nccl/dtype_scaling_results
PLOTS_DIR=/work/09308/zhengmk/nccl/dtype_scaling_plots

mkdir -p "${RESULTS_DIR}" "${PLOTS_DIR}"
ml gcc/15.1.0 cuda/12.6 nvidia/24.9

if [[ -f /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate ]]; then
  source /work/09308/zhengmk/python_vir_envs/range-topk-vista/bin/activate
fi

cd "${EXAMPLE_DIR}"
plot_cmd=(
  python3 plot_dtype_scaling.py
  --results-dir "${RESULTS_DIR}"
  --output-prefix "${PLOTS_DIR}/dtype_allreduce_scaling_efficiency_${RUN_TAG:-jobs_${JOB_IDS//,/__}}"
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
