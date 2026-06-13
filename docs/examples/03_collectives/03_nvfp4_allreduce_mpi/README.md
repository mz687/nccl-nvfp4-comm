<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NVFP4 AllReduce MPI Harness

This example focuses on `ncclAllReduceNvfp4` only and is meant for multi-node or
large single-node runs with one GPU per MPI rank.

## What It Tests
- Correctness: builds an non-communication host-side NVFP4 reference, compares it against
  `ncclAllReduceNvfp4`, and prints the elementwise max diff plus total absolute error.
- Scalability: benchmarks both native NVFP4 allreduce and NCCL FP8 E4M3 allreduce over the default logical size sweep
  `64MB,128MB,256MB,512MB,1GB,2GB,4GB,8GB`, using the same logical element count for both implementations.
- Multi-scale launch: intended to be run with 4, 8, 16, 32, and 64 GPUs by changing
  the MPI job size.

## Build
```shell
make
```

## Correctness Run
```shell
mpirun -np 4 ./nvfp4_allreduce_mpi --mode correctness --dtype fp16
```

The default correctness sizes are `4MiB` and a tail-block variant (`4MiB + 2B`)


## Scalability Run
```shell
mpirun -np 8 ./nvfp4_allreduce_mpi --mode scaling --dtype fp16
```

Useful flags:
- `--scaling-bytes 64MB,128MB,...`
  These are logical byte sizes used to derive one shared count for both NVFP4 and FP8.
- `--warmup 3`
- `--iters 10`
- `--csv`

## Sweep Script
Use `run_scaling.sh` to launch the default matrix of `4 8 16 32 64` GPUs across
`64MB` through `8GB`.

```shell
LAUNCH_TEMPLATE='mpirun -np {NRANKS}' ./run_scaling.sh
```

For Slurm-style environments you can override the launcher template, for
example:

```shell
LAUNCH_TEMPLATE='srun --ntasks={NRANKS} --gpus-per-task=1' ./run_scaling.sh
```

If you want a quick correctness sanity check before the sweep:

```shell
mpirun -np 4 ./nvfp4_allreduce_mpi --mode correctness --dtype fp16
```

## Notes
- The harness prints `SKIP` when the requested payload is not divisible by the
  datatype width or when the current NVFP4 wrapper's estimated staging memory
  exceeds free device memory.
- The correctness path uses a host-side non-communication NVFP4 reference rather than an NCCL allgather baseline.
- The scaling path compares NVFP4 and FP8 allreduce on the same logical count; `logical_bytes` in the CSV and plots are shared across both implementations, while `buffer_bytes` records the actual per-rank buffer size for each implementation.
