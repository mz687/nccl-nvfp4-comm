<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Collective Communication Examples

## Overview
This directory contains collective examples ranging from the simplest
single-process, single-node walkthroughs to a one-GPU-per-rank MPI harness for
NVFP4 allreduce validation. The focus is clarity, correct resource management,
and practical result checking.

## Examples

### [01_allreduce](01_allreduce/)
**AllReduce Collective Operation**
- **Pattern**: All participants reduce and distribute the result
- **API**: `ncclCommInitAll`, `ncclAllReduce`
- **Use case**: Global reductions in ML and HPC (e.g., gradient averaging)
- **Key features**:
  - Initializes all GPUs in a single process
  - Each GPU contributes its rank value
  - Executes AllReduce sum across all GPUs
  - Verifies the expected global sum

### [02_nvfp4_collectives](02_nvfp4_collectives/)
**NVFP4 AllReduce and AlltoAll Verification**
- **Pattern**: Quantized transport with fp16/bf16 user tensors
- **APIs**: `ncclAllReduceNvfp4`, `ncclAlltoAllNvfp4`
- **Use case**: End-to-end verification of block-scaled NVFP4 collective wrappers
- **Key features**:
  - Covers fp16 and bf16 buffers
  - Checks exact-block and tail-block message sizes
  - Exercises eager launches and CUDA graph capture
  - Includes a few negative API checks

### [03_nvfp4_allreduce_mpi](03_nvfp4_allreduce_mpi/)
**NVFP4 AllReduce MPI Correctness and Scalability Harness**
- **Pattern**: One GPU per MPI rank, suitable for single-node or multi-node runs
- **API**: `ncclAllReduceNvfp4` correctness plus NVFP4-vs-FP8 allreduce scaling
- **Use case**: Quantitative validation and scaling sweeps for NVFP4 allreduce
- **Key features**:
  - Compares NVFP4 allreduce against a host-side non-communication NVFP4 reference
  - Reports max diff, total absolute error, and mean absolute error for each correctness case
  - Sweeps `64MB,128MB,256MB,512MB,1GB,2GB,4GB,8GB` as shared logical sizes for both NVFP4 and FP8
  - Includes repeated scaling launchers for the default `4,8,16,32,64` GPU matrix

## Choosing the Right Pattern

*Scenario* : Parallel training needs efficient global communication
*Addresses* : Most commonly used collective algorithms
*Dependencies* : A functional NCCL library and its dependencies

### Why `ncclCommInitAll` here?
For the simplest single-node collective examples we use `ncclCommInitAll` as it
creates a clique of communicators in one call.

```c
// Initialize all GPUs in one call
ncclComm_t* comms;
int num_gpus;
NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));
```

A more advanced setup using MPI to initialize communicators across multiple
nodes is shown in
[01_communicators/03_one_device_per_process_mpi](../01_communicators/03_one_device_per_process_mpi)
and the NVFP4 allreduce harness in
[03_nvfp4_allreduce_mpi](03_nvfp4_allreduce_mpi/).

## Building

### **Quick Start**
```shell
# Build example by directory name
make 01_allreduce
make 02_nvfp4_collectives
make 03_nvfp4_allreduce_mpi
```

### **Individual Examples**
```shell
# Build and run AllReduce
cd 01_allreduce && make
./allreduce

# Build and run NVFP4 verification
cd 02_nvfp4_collectives && make
./nvfp4_collectives

# Build and run the MPI NVFP4 allreduce harness
cd 03_nvfp4_allreduce_mpi && make
mpirun -np 4 ./nvfp4_allreduce_mpi --mode correctness --dtype fp16
```

## References
- [NCCL User Guide:
  Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)
- [NCCL API
  Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
