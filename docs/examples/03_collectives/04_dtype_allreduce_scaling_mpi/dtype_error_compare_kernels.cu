#include "dtype_error_compare_kernels.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxReductionBlocks = 32768;
constexpr int kNvfp4BlockElts = 16;
constexpr int kNvfp4BlockBytes = 10;
constexpr float kNvfp4MaxFinite = 6.0f;

struct PartialStats {
  double nvfp4AbsSum;
  double fp8AbsSum;
  double nvfp4Fp8AbsSum;
  double nvfp4MaxAbs;
  double fp8MaxAbs;
  double nvfp4Fp8MaxAbs;
};

__host__ __device__ __forceinline__ size_t nvfp4NumBlocks(size_t count) {
  return count / kNvfp4BlockElts + ((count % kNvfp4BlockElts) != 0 ? 1 : 0);
}

__host__ __device__ __forceinline__ float nvfp4Magnitude(int code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x7];
}

__host__ __device__ __forceinline__ float nvfp4Decode(uint8_t nibble) {
  float magnitude = nvfp4Magnitude(nibble & 0x7);
  return (nibble & 0x8) ? -magnitude : magnitude;
}

__host__ __device__ __forceinline__ uint8_t nvfp4Encode(float value) {
  float clamped = fminf(fmaxf(value, -kNvfp4MaxFinite), kNvfp4MaxFinite);
  float absValue = fabsf(clamped);
  int best =
      absValue < 0.25f ? 0 :
      absValue < 0.75f ? 1 :
      absValue < 1.25f ? 2 :
      absValue < 1.75f ? 3 :
      absValue < 2.50f ? 4 :
      absValue < 3.50f ? 5 :
      absValue < 5.00f ? 6 : 7;
  return static_cast<uint8_t>(best | (clamped < 0.0f ? 0x8 : 0x0));
}

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__host__ __device__ __forceinline__ float baseValue(size_t idx) {
  uint32_t x = mix32(static_cast<uint32_t>(idx) ^ static_cast<uint32_t>(idx >> 32));
  float unit = static_cast<float>(x & 0xffffu) * (1.0f / 32767.5f) - 1.0f;
  float blockScale = 0.25f + 1.75f * static_cast<float>((idx >> 4) & 0xfu) * (1.0f / 15.0f);
  return unit * blockScale;
}

__host__ __device__ __forceinline__ float rankFactor(size_t idx) {
  int bucket = static_cast<int>((idx >> 3) & 0x7u) - 3;
  return static_cast<float>(bucket) * (1.0f / 512.0f);
}

__host__ __device__ __forceinline__ float inputValue(int rank, size_t idx) {
  return baseValue(idx) + rankFactor(idx) * static_cast<float>(rank);
}

__host__ __device__ __forceinline__ float referenceSum(int nranks, size_t idx) {
  float rankSum = 0.5f * static_cast<float>(nranks) * static_cast<float>(nranks - 1);
  return static_cast<float>(nranks) * baseValue(idx) + rankFactor(idx) * rankSum;
}

__device__ __forceinline__ float loadNvfp4Value(const uint8_t* packed, size_t idx) {
  size_t block = idx / kNvfp4BlockElts;
  int elt = static_cast<int>(idx % kNvfp4BlockElts);
  const uint8_t* blockPtr = packed + block * kNvfp4BlockBytes;
  float scale = __half2float(reinterpret_cast<const __half*>(blockPtr)[0]);
  uint8_t packedByte = blockPtr[2 + (elt >> 1)];
  uint8_t nibble = (elt & 1) ? static_cast<uint8_t>(packedByte >> 4)
                             : static_cast<uint8_t>(packedByte & 0xf);
  return nvfp4Decode(nibble) * scale;
}

__global__ void initErrorCompareInputsKernel(int rank, size_t count, uint8_t* nvfp4Send,
                                             uint8_t* fp8Send) {
  size_t block = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t nBlocks = nvfp4NumBlocks(count);
  if (block >= nBlocks) return;

  size_t base = block * kNvfp4BlockElts;
  uint8_t* nvBlock = nvfp4Send + block * kNvfp4BlockBytes;
  float values[kNvfp4BlockElts];
  float maxAbs = 0.0f;

  #pragma unroll
  for (int i = 0; i < kNvfp4BlockElts; ++i) {
    size_t idx = base + static_cast<size_t>(i);
    float value = idx < count ? __half2float(__float2half_rn(inputValue(rank, idx))) : 0.0f;
    values[i] = value;
    maxAbs = fmaxf(maxAbs, fabsf(value));
    if (idx < count) {
      reinterpret_cast<__nv_fp8_e4m3*>(fp8Send)[idx] = __nv_fp8_e4m3(inputValue(rank, idx));
    }
  }

  if (maxAbs == 0.0f) {
    #pragma unroll
    for (int i = 0; i < kNvfp4BlockBytes; ++i) nvBlock[i] = 0;
    return;
  }

  __half scaleHalf = __float2half_rn(maxAbs / kNvfp4MaxFinite);
  reinterpret_cast<__half*>(nvBlock)[0] = scaleHalf;
  float storedScale = __half2float(scaleHalf);
  float invScale = storedScale == 0.0f ? 0.0f : (1.0f / storedScale);

  #pragma unroll
  for (int byteIdx = 0; byteIdx < 8; ++byteIdx) {
    uint8_t low = nvfp4Encode(values[byteIdx * 2] * invScale);
    uint8_t high = nvfp4Encode(values[byteIdx * 2 + 1] * invScale);
    nvBlock[2 + byteIdx] = static_cast<uint8_t>(low | (high << 4));
  }
}

__global__ void computeErrorCompareStatsKernel(size_t count, int nranks,
                                               const uint8_t* nvfp4Recv,
                                               const uint8_t* fp8Recv,
                                               PartialStats* partials) {
  __shared__ double nvSum[kThreadsPerBlock];
  __shared__ double fp8Sum[kThreadsPerBlock];
  __shared__ double directSum[kThreadsPerBlock];
  __shared__ double nvMax[kThreadsPerBlock];
  __shared__ double fp8Max[kThreadsPerBlock];
  __shared__ double directMax[kThreadsPerBlock];

  double localNvSum = 0.0;
  double localFp8Sum = 0.0;
  double localDirectSum = 0.0;
  double localNvMax = 0.0;
  double localFp8Max = 0.0;
  double localDirectMax = 0.0;

  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < count; idx += stride) {
    float ref = referenceSum(nranks, idx);
    float nv = loadNvfp4Value(nvfp4Recv, idx);
    float fp8 = static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(fp8Recv)[idx]);
    double nvErr = fabs(static_cast<double>(nv) - static_cast<double>(ref));
    double fp8Err = fabs(static_cast<double>(fp8) - static_cast<double>(ref));
    double directErr = fabs(static_cast<double>(nv) - static_cast<double>(fp8));
    localNvSum += nvErr;
    localFp8Sum += fp8Err;
    localDirectSum += directErr;
    localNvMax = fmax(localNvMax, nvErr);
    localFp8Max = fmax(localFp8Max, fp8Err);
    localDirectMax = fmax(localDirectMax, directErr);
  }

  nvSum[threadIdx.x] = localNvSum;
  fp8Sum[threadIdx.x] = localFp8Sum;
  directSum[threadIdx.x] = localDirectSum;
  nvMax[threadIdx.x] = localNvMax;
  fp8Max[threadIdx.x] = localFp8Max;
  directMax[threadIdx.x] = localDirectMax;
  __syncthreads();

  for (int offset = kThreadsPerBlock / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      nvSum[threadIdx.x] += nvSum[threadIdx.x + offset];
      fp8Sum[threadIdx.x] += fp8Sum[threadIdx.x + offset];
      directSum[threadIdx.x] += directSum[threadIdx.x + offset];
      nvMax[threadIdx.x] = fmax(nvMax[threadIdx.x], nvMax[threadIdx.x + offset]);
      fp8Max[threadIdx.x] = fmax(fp8Max[threadIdx.x], fp8Max[threadIdx.x + offset]);
      directMax[threadIdx.x] = fmax(directMax[threadIdx.x], directMax[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    PartialStats out;
    out.nvfp4AbsSum = nvSum[0];
    out.fp8AbsSum = fp8Sum[0];
    out.nvfp4Fp8AbsSum = directSum[0];
    out.nvfp4MaxAbs = nvMax[0];
    out.fp8MaxAbs = fp8Max[0];
    out.nvfp4Fp8MaxAbs = directMax[0];
    partials[blockIdx.x] = out;
  }
}

int reductionBlocks(size_t count) {
  size_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks == 0) return 1;
  return static_cast<int>(std::min<size_t>(blocks, kMaxReductionBlocks));
}

}  // namespace

extern "C" cudaError_t launchInitErrorCompareInputs(int rank, int nranks, size_t count,
                                                    uint8_t* nvfp4Send, size_t nvfp4SectionBytes,
                                                    uint8_t* fp8Send, cudaStream_t stream) {
  (void)nranks;
  cudaError_t err = cudaMemsetAsync(nvfp4Send, 0, nvfp4SectionBytes, stream);
  if (err != cudaSuccess) return err;
  size_t blocks = nvfp4NumBlocks(count);
  int grid = static_cast<int>((blocks + kThreadsPerBlock - 1) / kThreadsPerBlock);
  initErrorCompareInputsKernel<<<grid, kThreadsPerBlock, 0, stream>>>(rank, count, nvfp4Send, fp8Send);
  return cudaGetLastError();
}

extern "C" cudaError_t launchComputeErrorCompareStats(size_t count, int nranks,
                                                      const uint8_t* nvfp4Recv,
                                                      const uint8_t* fp8Recv,
                                                      DTypeErrorStats* hostStats,
                                                      cudaStream_t stream) {
  int grid = reductionBlocks(count);
  PartialStats* partials = nullptr;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&partials), sizeof(PartialStats) * grid);
  if (err != cudaSuccess) return err;

  computeErrorCompareStatsKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      count, nranks, nvfp4Recv, fp8Recv, partials);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(partials);
    return err;
  }

  std::vector<PartialStats> hostPartials(static_cast<size_t>(grid));
  err = cudaMemcpyAsync(hostPartials.data(), partials, sizeof(PartialStats) * grid,
                        cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  cudaError_t freeErr = cudaFree(partials);
  if (err != cudaSuccess) return err;
  if (freeErr != cudaSuccess) return freeErr;

  DTypeErrorStats stats = {};
  for (const PartialStats& partial : hostPartials) {
    stats.nvfp4AbsSum += partial.nvfp4AbsSum;
    stats.fp8AbsSum += partial.fp8AbsSum;
    stats.nvfp4Fp8AbsSum += partial.nvfp4Fp8AbsSum;
    stats.nvfp4MaxAbs = std::max(stats.nvfp4MaxAbs, partial.nvfp4MaxAbs);
    stats.fp8MaxAbs = std::max(stats.fp8MaxAbs, partial.fp8MaxAbs);
    stats.nvfp4Fp8MaxAbs = std::max(stats.nvfp4Fp8MaxAbs, partial.nvfp4Fp8MaxAbs);
  }
  *hostStats = stats;
  return cudaSuccess;
}
