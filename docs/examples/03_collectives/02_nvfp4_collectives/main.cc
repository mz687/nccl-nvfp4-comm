/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = (cmd);                                                  \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "NCCL failure %s:%d '%s'\n", __FILE__, __LINE__,       \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = (cmd);                                                   \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA failure %s:%d '%s'\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace {

constexpr size_t kCounts[] = {32, 37};
constexpr float kHalfTol = 1e-3f;
constexpr float kBfloatTol = 1e-2f;

class ThreadBarrier {
 public:
  explicit ThreadBarrier(int count) : threshold_(count), count_(count), generation_(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int generation = generation_;
    if (--count_ == 0) {
      generation_++;
      count_ = threshold_;
      cond_.notify_all();
      return;
    }
    cond_.wait(lock, [&] { return generation != generation_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cond_;
  int threshold_;
  int count_;
  int generation_;
};

enum class LaunchMode { Eager, Capture };

template <typename T>
T fromFloat(float value);

template <>
__half fromFloat<__half>(float value) {
  return __float2half(value);
}

#if CUDART_VERSION >= 11000
template <>
__nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}
#endif

template <typename T>
float toFloat(T value);

template <>
float toFloat(__half value) {
  return __half2float(value);
}

#if CUDART_VERSION >= 11000
template <>
float toFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
#endif

float generalPattern(int key, size_t idx) {
  static const float pattern[16] = {
      2.0f, -2.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f, 1.0f,
      -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 1.0f, 0.0f, -1.0f};
  size_t lane = idx % 16;
  float value = pattern[lane];
  if (lane == 0) return ((key + static_cast<int>(idx / 16)) & 1) ? -2.0f : 2.0f;
  if ((key + static_cast<int>(lane)) & 1) value = -value;
  return value;
}

float prodPattern(int key, size_t idx) {
  static const float pattern[16] = {
      2.0f, 1.0f, 0.5f, -1.0f, 1.0f, -0.5f, 2.0f, -1.0f,
      1.0f, 0.5f, -1.0f, 2.0f, 0.5f, 1.0f, -2.0f, 1.0f};
  size_t lane = idx % 16;
  float value = pattern[lane];
  if (lane == 0) return ((key + static_cast<int>(idx / 16)) & 1) ? -2.0f : 2.0f;
  if ((key & 1) && (lane % 3 == 0)) value = -value;
  return value;
}

template <typename T>
void fillAllReduceInput(std::vector<T>& host, int rank, ncclRedOp_t op) {
  for (size_t i = 0; i < host.size(); ++i) {
    float value = (op == ncclProd) ? prodPattern(rank, i) : generalPattern(rank, i);
    host[i] = fromFloat<T>(value);
  }
}

template <typename T>
void fillAlltoAllInput(std::vector<T>& host, int rank, int nranks, size_t count) {
  for (int peer = 0; peer < nranks; ++peer) {
    for (size_t i = 0; i < count; ++i) {
      host[peer * count + i] = fromFloat<T>(generalPattern(rank * 7 + peer * 11, i));
    }
  }
}

template <typename Fn>
void runOnRanks(int nranks, Fn fn) {
  std::vector<std::thread> threads;
  threads.reserve(nranks);
  for (int rank = 0; rank < nranks; ++rank) {
    threads.emplace_back([&, rank] { fn(rank); });
  }
  for (auto& thread : threads) thread.join();
}

template <typename Fn>
void launchCollective(int nranks, cudaStream_t* streams, LaunchMode mode, Fn fn) {
  ThreadBarrier barrier(nranks);
  runOnRanks(nranks, [&](int rank) {
    CUDACHECK(cudaSetDevice(rank));
    cudaStream_t stream = streams[rank];
    if (mode == LaunchMode::Eager) {
      barrier.wait();
      fn(rank);
      CUDACHECK(cudaStreamSynchronize(stream));
      return;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    CUDACHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    barrier.wait();
    fn(rank);
    barrier.wait();
    CUDACHECK(cudaStreamEndCapture(stream, &graph));
    CUDACHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    barrier.wait();
    CUDACHECK(cudaGraphLaunch(graphExec, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaGraphExecDestroy(graphExec));
    CUDACHECK(cudaGraphDestroy(graph));
  });
}

template <typename T>
bool compareOutputs(const std::vector<T>& expected, const std::vector<T>& actual,
                    float tolerance, const char* label) {
  for (size_t i = 0; i < expected.size(); ++i) {
    float ref = toFloat(expected[i]);
    float got = toFloat(actual[i]);
    if (std::fabs(ref - got) > tolerance) {
      fprintf(stderr, "%s mismatch at %zu: expected %.6f got %.6f\n", label, i, ref, got);
      return false;
    }
  }
  return true;
}

template <typename T>
bool runAllReduceCase(const char* typeName, ncclDataType_t dtype, int nranks, ncclComm_t* comms,
                      cudaStream_t* streams, T** send, T** baselineRecv, T** testRecv,
                      size_t count, ncclRedOp_t op, LaunchMode mode, float tolerance) {
  for (int rank = 0; rank < nranks; ++rank) {
    std::vector<T> host(count);
    fillAllReduceInput(host, rank, op);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMemcpy(send[rank], host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }

  launchCollective(nranks, streams, LaunchMode::Eager, [&](int rank) {
    NCCLCHECK(ncclAllReduce(send[rank], baselineRecv[rank], count, dtype, op, comms[rank], streams[rank]));
  });

  launchCollective(nranks, streams, mode, [&](int rank) {
    NCCLCHECK(ncclAllReduceNvfp4(send[rank], testRecv[rank], count, dtype, op, comms[rank], streams[rank]));
  });

  bool ok = true;
  for (int rank = 0; rank < nranks; ++rank) {
    std::vector<T> expected(count);
    std::vector<T> actual(count);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMemcpy(expected.data(), baselineRecv[rank], count * sizeof(T), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(actual.data(), testRecv[rank], count * sizeof(T), cudaMemcpyDeviceToHost));
    char label[128];
    snprintf(label, sizeof(label), "AllReduce/%s/%s/count=%zu/rank=%d", typeName,
             mode == LaunchMode::Eager ? "eager" : "capture", count, rank);
    ok &= compareOutputs(expected, actual, tolerance, label);
  }

  if (ok) {
    printf("verified allreduce %-8s op=%d count=%zu mode=%s\n", typeName, (int)op, count,
           mode == LaunchMode::Eager ? "eager" : "capture");
  }
  return ok;
}

template <typename T>
bool runAlltoAllCase(const char* typeName, ncclDataType_t dtype, int nranks, ncclComm_t* comms,
                     cudaStream_t* streams, T** send, T** baselineRecv, T** testRecv,
                     size_t count, LaunchMode mode, float tolerance) {
  size_t totalCount = count * nranks;
  for (int rank = 0; rank < nranks; ++rank) {
    std::vector<T> host(totalCount);
    fillAlltoAllInput(host, rank, nranks, count);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMemcpy(send[rank], host.data(), totalCount * sizeof(T), cudaMemcpyHostToDevice));
  }

  launchCollective(nranks, streams, LaunchMode::Eager, [&](int rank) {
    NCCLCHECK(ncclAlltoAll(send[rank], baselineRecv[rank], count, dtype, comms[rank], streams[rank]));
  });

  launchCollective(nranks, streams, mode, [&](int rank) {
    NCCLCHECK(ncclAlltoAllNvfp4(send[rank], testRecv[rank], count, dtype, comms[rank], streams[rank]));
  });

  bool ok = true;
  for (int rank = 0; rank < nranks; ++rank) {
    std::vector<T> expected(totalCount);
    std::vector<T> actual(totalCount);
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMemcpy(expected.data(), baselineRecv[rank], totalCount * sizeof(T), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(actual.data(), testRecv[rank], totalCount * sizeof(T), cudaMemcpyDeviceToHost));
    char label[128];
    snprintf(label, sizeof(label), "AlltoAll/%s/%s/count=%zu/rank=%d", typeName,
             mode == LaunchMode::Eager ? "eager" : "capture", count, rank);
    ok &= compareOutputs(expected, actual, tolerance, label);
  }

  if (ok) {
    printf("verified alltoall %-8s count=%zu mode=%s\n", typeName, count,
           mode == LaunchMode::Eager ? "eager" : "capture");
  }
  return ok;
}

template <typename T>
bool runNegativeChecks(const char* typeName, ncclDataType_t dtype, ncclComm_t comm,
                       cudaStream_t stream, T* send, T* recv) {
  (void)typeName;
  ncclResult_t badTypeReduce = ncclAllReduceNvfp4(send, recv, 16, ncclFloat32, ncclSum, comm, stream);
  ncclResult_t badTypeAlltoAll = ncclAlltoAllNvfp4(send, recv, 16, ncclFloat32, comm, stream);
  NCCLCHECK(ncclGroupStart());
  ncclResult_t groupedReduce = ncclAllReduceNvfp4(send, recv, 16, dtype, ncclSum, comm, stream);
  ncclResult_t groupedAlltoAll = ncclAlltoAllNvfp4(send, recv, 16, dtype, comm, stream);
  NCCLCHECK(ncclGroupEnd());

  bool ok = badTypeReduce == ncclInvalidArgument &&
            badTypeAlltoAll == ncclInvalidArgument &&
            groupedReduce == ncclInvalidUsage &&
            groupedAlltoAll == ncclInvalidUsage;
  if (!ok) {
    fprintf(stderr, "negative API checks failed\n");
  } else {
    printf("verified negative API checks\n");
  }
  return ok;
}

template <typename T>
bool runTypeSuite(const char* typeName, ncclDataType_t dtype, int nranks, ncclComm_t* comms,
                  cudaStream_t* streams, T** send, T** baselineRecv, T** testRecv,
                  float tolerance) {
  bool ok = runNegativeChecks(typeName, dtype, comms[0], streams[0], send[0], testRecv[0]);

  const ncclRedOp_t ops[] = {ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg};
  for (ncclRedOp_t op : ops) {
    for (size_t count : kCounts) {
      ok &= runAllReduceCase(typeName, dtype, nranks, comms, streams, send, baselineRecv,
                             testRecv, count, op, LaunchMode::Eager, tolerance);
      ok &= runAllReduceCase(typeName, dtype, nranks, comms, streams, send, baselineRecv,
                             testRecv, count, op, LaunchMode::Capture, tolerance);
    }
  }

  for (size_t count : kCounts) {
    ok &= runAlltoAllCase(typeName, dtype, nranks, comms, streams, send, baselineRecv,
                          testRecv, count, LaunchMode::Eager, tolerance);
    ok &= runAlltoAllCase(typeName, dtype, nranks, comms, streams, send, baselineRecv,
                          testRecv, count, LaunchMode::Capture, tolerance);
  }

  return ok;
}

}  // namespace

int main() {
  int available = 0;
  CUDACHECK(cudaGetDeviceCount(&available));
  if (available < 2) {
    fprintf(stderr, "Need at least 2 GPUs for NVFP4 collective verification\n");
    return EXIT_FAILURE;
  }

  int nranks = std::min(available, 4);
  size_t maxCount = 37 * nranks;
  printf("Using %d GPUs for NVFP4 collective verification\n", nranks);

  std::vector<ncclComm_t> comms(nranks);
  std::vector<cudaStream_t> streams(nranks);
  NCCLCHECK(ncclCommInitAll(comms.data(), nranks, nullptr));

  std::vector<__half*> halfSend(nranks), halfBaseline(nranks), halfTest(nranks);
  for (int rank = 0; rank < nranks; ++rank) {
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaStreamCreate(&streams[rank]));
    CUDACHECK(cudaMalloc((void**)&halfSend[rank], maxCount * sizeof(__half)));
    CUDACHECK(cudaMalloc((void**)&halfBaseline[rank], maxCount * sizeof(__half)));
    CUDACHECK(cudaMalloc((void**)&halfTest[rank], maxCount * sizeof(__half)));
  }

  bool ok = runTypeSuite("fp16", ncclFloat16, nranks, comms.data(), streams.data(),
                         halfSend.data(), halfBaseline.data(), halfTest.data(), kHalfTol);

#if CUDART_VERSION >= 11000
  std::vector<__nv_bfloat16*> bf16Send(nranks), bf16Baseline(nranks), bf16Test(nranks);
  for (int rank = 0; rank < nranks; ++rank) {
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaMalloc((void**)&bf16Send[rank], maxCount * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc((void**)&bf16Baseline[rank], maxCount * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc((void**)&bf16Test[rank], maxCount * sizeof(__nv_bfloat16)));
  }

  ok &= runTypeSuite("bf16", ncclBfloat16, nranks, comms.data(), streams.data(),
                     bf16Send.data(), bf16Baseline.data(), bf16Test.data(), kBfloatTol);

  for (int rank = 0; rank < nranks; ++rank) {
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaFree(bf16Send[rank]));
    CUDACHECK(cudaFree(bf16Baseline[rank]));
    CUDACHECK(cudaFree(bf16Test[rank]));
  }
#endif

  for (int rank = 0; rank < nranks; ++rank) {
    NCCLCHECK(ncclCommFinalize(comms[rank]));
    NCCLCHECK(ncclCommDestroy(comms[rank]));
  }

  for (int rank = 0; rank < nranks; ++rank) {
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaFree(halfSend[rank]));
    CUDACHECK(cudaFree(halfBaseline[rank]));
    CUDACHECK(cudaFree(halfTest[rank]));
    CUDACHECK(cudaStreamDestroy(streams[rank]));
  }

  if (!ok) return EXIT_FAILURE;
  printf("NVFP4 collective example completed successfully\n");
  return EXIT_SUCCESS;
}
