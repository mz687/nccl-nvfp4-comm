# NCCL

Optimized primitives for inter-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Experimental NVFP4 Collectives

This branch adds an experimental NVFP4 collective path for reducing the
communication volume of large all-reduce payloads. The implementation does not
depend on a public NCCL FP4 datatype. Instead, it adds a custom
`ncclAllReduceNvfp4` entry point that treats the transported payload as packed
bytes and carries NVFP4 metadata through NCCL's enqueue, scheduling, and device
work paths.

At a high level, the NVFP4 path is organized around three ideas:

1. Pack logical values into NVFP4 blocks before communication. Each NVFP4 block
   stores 16 logical values in 10 bytes: 2 bytes of scale metadata plus 8 bytes
   of 4-bit payload values. The packed byte section is aligned for NCCL
   transport.
2. Communicate packed bytes through NCCL. The collective is enqueued as an
   all-reduce over `ncclUint8`, while `transportCodec =
   ncclTransportCodecNvfp4` tells the scheduler, registration path, and device
   work descriptors that the byte stream has NVFP4 semantics. This keeps the
   network payload small while avoiding the need for a public `ncclFloat4`
   datatype.
3. Decode, reduce, and repack on the device. During the all-reduce, device
   kernels interpret the byte stream as NVFP4 blocks, decode nibbles using their
   block scales, apply the reduction, choose output scales, and pack the reduced
   values back to NVFP4. The native ring implementation uses block-aware send,
   receive/copy, and reduction operators so NCCL chunking can use multiple
   channels without splitting the 10-byte packed blocks.

This design is useful when communication bandwidth is the bottleneck: NVFP4
reduces the transported payload relative to FP8 and FP32, but it adds
quantization, dequantization, scale selection, and packing work. On GH200, the
current implementation uses software kernels for that work; future hardware with
accelerated FP4 arithmetic can reduce this overhead, but the communication path
still needs block-aware semantics for correct all-reduce behavior.

### Example: NVFP4 All-Reduce

The custom API is declared in `nccl.h`:

```c++
ncclResult_t ncclAllReduceNvfp4(const void* sendbuff, void* recvbuff,
                                size_t count, ncclDataType_t reservedDatatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream);
```

`count` is the number of logical values, not the number of packed bytes. The
input and output buffers must already use the internal NVFP4 block layout: 16
logical values per 10-byte block, with the packed section rounded up to a
16-byte boundary. Set `NCCL_NVFP4_ALLREDUCE_IMPL=native` to use the implemented
packed-byte native all-reduce path. The `reservedDatatype` argument is ignored
today; pass the source logical type, such as `ncclFloat16` or `ncclBfloat16`,
so the call site remains clear.

The shortest application-side call flow is:

```c++
const size_t count = 1024ULL * 1024ULL * 1024ULL; // Logical FP16 elements.
const size_t nvfp4Bytes = nvfp4PackedBytes(count);

uint8_t* dPackedSend = nullptr;
uint8_t* dPackedRecv = nullptr;
cudaMalloc(&dPackedSend, nvfp4Bytes);
cudaMalloc(&dPackedRecv, nvfp4Bytes);

// Application or harness code packs dFp16Input into NCCL's 16-value NVFP4
// block layout before communication.
packFp16ToNvfp4(dFp16Input, dPackedSend, count, stream);

setenv("NCCL_NVFP4_ALLREDUCE_IMPL", "native", 0);
ncclAllReduceNvfp4(dPackedSend, dPackedRecv, count, ncclFloat16,
                   ncclSum, comm, stream);

// dPackedRecv contains the reduced tensor in the same packed NVFP4 layout.
unpackNvfp4ToFp16(dPackedRecv, dFp16Output, count, stream);
```

Here `packFp16ToNvfp4` and `unpackNvfp4ToFp16` stand in for the
application or harness packing kernels; the NCCL API itself expects packed
NVFP4 buffers.

```c++
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>
#include <nccl.h>

static size_t nvfp4PackedBytes(size_t logicalCount) {
  const size_t valuesPerBlock = 16;
  const size_t bytesPerBlock = 10;
  const size_t alignment = 16;
  size_t blocks = (logicalCount + valuesPerBlock - 1) / valuesPerBlock;
  size_t bytes = blocks * bytesPerBlock;
  return (bytes + alignment - 1) & ~(alignment - 1);
}

void runNvfp4AllReduce(ncclComm_t comm, cudaStream_t stream,
                       const void* packedInput, void* packedOutput,
                       size_t logicalCount) {
  // Select the native packed-byte NVFP4 communication path.
  // Equivalently set this in the shell before running the program.
  setenv("NCCL_NVFP4_ALLREDUCE_IMPL", "native", 0);

  ncclResult_t result = ncclAllReduceNvfp4(packedInput, packedOutput,
                                           logicalCount, ncclFloat16, ncclSum,
                                           comm, stream);
  if (result != ncclSuccess) std::abort();
}

void allocateExample(size_t logicalCount, uint8_t** dSend, uint8_t** dRecv) {
  size_t packedBytes = nvfp4PackedBytes(logicalCount);
  cudaMalloc(dSend, packedBytes);
  cudaMalloc(dRecv, packedBytes);

  // Fill dSend with packed NVFP4 blocks before calling runNvfp4AllReduce.
  // After the call, dRecv contains the reduced result in the same packed layout.
}
```

A typical one-GPU-per-MPI-rank call sequence is shown below. It assumes the
standard MPI/CUDA/NCCL headers and omits error checks for brevity.

```c++
MPI_Init(&argc, &argv);

int rank = 0;
int nranks = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nranks);

int localRank = 0;
MPI_Comm localComm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                    &localComm);
MPI_Comm_rank(localComm, &localRank);
MPI_Comm_free(&localComm);

cudaSetDevice(localRank);
cudaStream_t stream;
cudaStreamCreate(&stream);

ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
MPI_Bcast(&id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);

ncclComm_t comm;
ncclCommInitRank(&comm, nranks, id, rank);

size_t logicalCount = 1024ULL * 1024ULL * 1024ULL;
size_t packedBytes = nvfp4PackedBytes(logicalCount);
uint8_t* dSend = nullptr;
uint8_t* dRecv = nullptr;
cudaMalloc(&dSend, packedBytes);
cudaMalloc(&dRecv, packedBytes);

// Application code must pack logical FP16 values into dSend first.
setenv("NCCL_NVFP4_ALLREDUCE_IMPL", "native", 0);
ncclAllReduceNvfp4(dSend, dRecv, logicalCount, ncclFloat16, ncclSum, comm, stream);
cudaStreamSynchronize(stream);
// dRecv now holds the all-reduced result in the same packed NVFP4 layout.

cudaFree(dRecv);
cudaFree(dSend);
ncclCommDestroy(comm);
cudaStreamDestroy(stream);
MPI_Finalize();
```

The runnable correctness and scaling harness is in
`docs/examples/03_collectives/03_nvfp4_allreduce_mpi/`. The example below
builds this NCCL tree, links the harness against the local build, and then runs
the implemented native NVFP4 all-reduce path. The byte-size flags are logical
source sizes; the harness derives a shared value count, packs the input to
NVFP4, calls `ncclAllReduceNvfp4`, unpacks the result, and compares against the
reference result.

```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
$ cd docs/examples/03_collectives/03_nvfp4_allreduce_mpi
$ make NCCL_HOME=../../../../build
$ LD_LIBRARY_PATH=../../../../build/lib:$LD_LIBRARY_PATH \
    NCCL_NVFP4_ALLREDUCE_IMPL=native \
    mpirun -np 4 ./nvfp4_allreduce_mpi \
    --mode correctness --dtype fp16 --correctness-bytes 128MiB
$ LD_LIBRARY_PATH=../../../../build/lib:$LD_LIBRARY_PATH \
    NCCL_NVFP4_ALLREDUCE_IMPL=native \
    mpirun -np 8 ./nvfp4_allreduce_mpi \
    --mode scaling --dtype fp16 --scaling-bytes 128MiB,256MiB,512MiB,1GiB \
    --warmup 5 --iters 50 --csv
```

Additional dtype scaling/error harnesses live under
`docs/examples/03_collectives/04_dtype_allreduce_scaling_mpi/`.

For the GH Slurm benchmark path used by the current experiment artifacts,
submit one native NVFP4 all-reduce case like this:

```shell
$ cd docs/examples/03_collectives/04_dtype_allreduce_scaling_mpi
$ ACCOUNT=CCR26006 \
    RANKS_LIST="4" \
    DTYPES_LIST="nvfp4" \
    VALUE_COUNTS_LIST="128M" \
    WARMUP_STEPS=5 \
    MEASURED_STEPS=50 \
    NCCL_HOME=../../../../build \
    NCCL_NVFP4_ALLREDUCE_IMPL=native \
    ./submit_dtype_value_count_single_case_gh.sh
```

This creates one Slurm job for the 4-GPU, 128M-value NVFP4 case. The harness
allocates packed NVFP4 buffers, performs 5 warmup `ncclAllReduceNvfp4` calls,
and reports the average time over 50 measured `ncclAllReduceNvfp4` calls.

For high-rank native NVFP4 benchmarking, keep the NCCL channel ceiling explicit
when reproducing a specific experiment. The conservative baseline used
`NCCL_BUFFSIZE=16777216`, while the newer low-rank, 16-rank, and 32-rank
adaptive sweeps used the default NCCL buffer setting. The adaptive NVFP4 policy
uses the available channel count through 8 ranks, then uses a size-aware 16-rank
cap: packed payloads up to 128MiB use up to 32 channels, and larger 16-rank
payloads use up to 48 channels. It uses up to 24 channels at 32 ranks and caps
larger rank counts at 16 channels. Setting `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=16`
reproduces the conservative baseline, while larger channel ceilings expose the
validated rank-specific fast paths. Set `NCCL_NVFP4_ENABLE_MULTICHANNEL=0` to
force the older one-channel path, or set it to a nonzero value to force the full
NCCL channel count for profiling.

The current-build 4-GPU and 8-GPU sweeps did not justify a lower channel cap: the
no-pin adaptive path tracked the best 64-channel rows from the forced-channel
sweep. The timings below use 5 warmup all-reduces and average 50 measured
all-reduces.

| Values | 4G adaptive, default buffer | 8G adaptive, default buffer |
| --- | ---: | ---: |
| 128M | `2.994761 ms` | `3.545857 ms` |
| 256M | `5.900954 ms` | `6.632248 ms` |
| 512M | `11.617089 ms` | `12.928375 ms` |
| 1G | `22.285235 ms` | `25.616132 ms` |

The 16-GPU measurements below use 5 warmup all-reduces and average 50 measured
all-reduces. The size-aware adaptive path keeps NVFP4-vs-FP32 error in the same
range as the 16-channel path across the checked sizes.

| Values | 16 channels, default buffer | Size-aware adaptive, default buffer |
| --- | ---: | ---: |
| 128M | `5.508090 ms` | `3.776538 ms` |
| 256M | `11.332399 ms` | `6.998161 ms` |
| 512M | `22.438578 ms` | `13.918401 ms` |
| 1G | `44.247183 ms` | `27.103501 ms` |

Nsight Systems profiles of the 16-GPU native path confirm the intended channel
selection. The 128M-value case used `gridX=32` and averaged `3.582698 ms` over
the three measured kernels after one warmup; the 256M-value case used `gridX=48`
and averaged `6.749301 ms` over the three measured kernels after one warmup.

The 32-GPU measurements below use 5 warmup all-reduces and average 50 measured
all-reduces. The 24-channel path keeps NVFP4-vs-FP32 error in the same range as
the 16-channel path across 128M, 256M, 512M, and 1G values.

| Values | 1 channel, 16M buffer | 8 channels, 16M buffer | 16 channels, 16M buffer | 24-channel adaptive, default buffer |
| --- | ---: | ---: | ---: | ---: |
| 128M | `13.903343 ms` | `8.880679 ms` | `5.339069 ms` | `4.859171 ms` |
| 256M | `26.808187 ms` | `18.834121 ms` | `11.416908 ms` | `9.071871 ms` |
| 512M | `53.571650 ms` | `37.932027 ms` | `22.886916 ms` | `18.049898 ms` |
| 1G | `104.732547 ms` | `76.335325 ms` | `45.197148 ms` | `35.015138 ms` |

At 64 GPUs and 128M values, the finite NVFP4-vs-FP32 error range was preserved
with 16 channels (`avg_abs_error=5.0292958377`, `max_abs_error=52.644859314`).
The corresponding average all-reduce time improved from `15.289325 ms` on the
older one-channel path to `9.067697 ms` with 8 channels and `6.598625 ms` with
16 channels.

Nsight Systems profiling of the 64-GPU, 128M-value case shows that the measured
section is dominated by the NCCL all-reduce kernel wait, not host launch
overhead. The profiled steady-state kernel time improved from `13.970080 ms`
with 1 channel to `7.822592 ms` with 8 channels and `6.383509 ms` with 16
channels (`gridX=16`). Nsight Compute was not usable directly on the MPI/NCCL
collective: a rank-0-only NCU probe connected to the process but stalled before
the all-reduce completed, consistent with replay/instrumentation interfering
with collective progress.

A raw `ncclUint8` all-reduce control over the same packed NVFP4 byte count gives
a communication-path lower bound for the current implementation. The remaining
gap is the native NVFP4 decode, reduce, scale selection, and repack work.

| Values | 4G raw packed bytes | 4G native NVFP4, adaptive | 8G raw packed bytes | 8G native NVFP4, adaptive | 16G raw packed bytes | 16G native NVFP4, adaptive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128M | `2.785888 ms` | `2.994761 ms` | `3.329375 ms` | `3.545857 ms` | `3.546049 ms` | `3.776538 ms` |
| 256M | `5.366941 ms` | `5.900954 ms` | `6.345814 ms` | `6.632248 ms` | `6.908599 ms` | `6.998161 ms` |
| 512M | `10.489638 ms` | `11.617089 ms` | `12.239894 ms` | `12.928375 ms` | `13.349176 ms` | `13.918401 ms` |
| 1G | `20.732950 ms` | `22.285235 ms` | `24.130345 ms` | `25.616132 ms` | `26.044618 ms` | `27.103501 ms` |

The tuned native path is within roughly 1-11% of this raw packed-byte lower
bound across the current 4-, 8-, and 16-GPU cases, so the remaining GH200
headroom is mostly in communication scheduling or hardware support rather than
simple local packing/decode micro-optimizations.

| Values | 32G raw packed bytes | 32G native NVFP4, adaptive 24ch | 64G raw packed bytes |
| --- | ---: | ---: | ---: |
| 128M | `3.612623 ms` | `4.859171 ms` | `6.029252 ms` |
| 256M | `7.122061 ms` | `9.071871 ms` | `7.858445 ms` |
| 512M | `13.985784 ms` | `18.049898 ms` | `14.639837 ms` |
| 1G | `27.204669 ms` | `35.015138 ms` | `28.292371 ms` |

The 24-channel policy was validated at 32 GPUs across 128M, 256M, 512M, and 1G
values. A post-build policy smoke measured `4.859171 ms` at 32 GPUs and 128M
values with external channel pins removed and without forcing
`NCCL_NVFP4_ENABLE_MULTICHANNEL`, while the 64-GPU,
128M-value smoke stayed in the validated 16-channel range at `6.664795 ms` and
preserved finite NVFP4-vs-FP32 error (`avg_abs_error=5.0292958377`,
`max_abs_error=52.644859314`). A forced 24-channel 64-GPU probe did not make
progress, so larger rank counts remain capped at 16 channels. Additional
64-GPU, 128M-value buffer-size probes did not justify changing the validated 16M
buffer: 8M measured `6.758852 ms`, 16M measured `6.598625 ms`, and 32M measured
`6.585041 ms`. Current-build `NCCL_NTHREADS=256` and `NCCL_NTHREADS=384`
retests also regressed the tuned adaptive timings. For example, the 4-GPU,
128M-value case moved from `2.994761 ms` by default to `3.521443 ms` at 256
threads and `3.170696 ms` at 384 threads, while the 16-GPU, 128M-value case
moved from `3.776538 ms` by default to `4.430053 ms` at 384 threads. A 16-bit
copy-widening probe preserved correctness but regressed
the 32-GPU, 128M-value timing from `5.339069 ms` to `5.389677 ms`. A full-block
decode-once sum-loop probe also preserved correctness but regressed the same
timing to `5.510118 ms`. Removing the payload-zero precheck likewise preserved
correctness but regressed the same timing to `5.395158 ms`, so the byte-copy
path, original decode loop, and zero precheck remain in place.

The figures below compare NVFP4 and FP8 against FP32 using the current scaling
experiment artifacts. The timing sweep uses 5 warmup runs and averages 50
measured all-reduce runs per configuration. The accuracy figures report average
and maximum absolute error relative to FP32. Rerun the timing sweep after kernel
or channel-scheduling changes before treating the speed chart as final.

![NVFP4 and FP8 speedup compared to FP32](docs/images/nvfp4/speedup_vs_fp32_bar.png)

![NVFP4 and FP8 average absolute error compared to FP32](docs/images/nvfp4/accuracy_avg_abs_error_vs_fp32_bar.png)

![NVFP4 and FP8 maximum absolute error compared to FP32](docs/images/nvfp4/accuracy_max_abs_error_vs_fp32_bar.png)

## Build

Note: the official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl. You can skip the following build steps if you choose to use the official builds.

To build the library :

```shell
$ cd nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## Install

To install NCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```shell
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```shell
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## Tests

Tests for NCCL are maintained separately at https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
