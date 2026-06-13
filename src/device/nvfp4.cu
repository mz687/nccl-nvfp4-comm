/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kPackedValuesPerByte = 2;
constexpr int kValuesPerNvfp4Block = NCCL_NVFP4_BLOCK_ELTS;
constexpr int kBytesPerNvfp4Block = NCCL_NVFP4_BLOCK_BYTES;
constexpr int kPackedBytesPerBlock = 8;
constexpr float kMaxFiniteNvfp4 = 6.0f;

__device__ __forceinline__ float nvfp4Magnitude(int code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x7];
}

__device__ __forceinline__ float nvfp4Decode(uint8_t nibble) {
  float magnitude = nvfp4Magnitude(nibble & 0x7);
  return (nibble & 0x8) ? -magnitude : magnitude;
}

__device__ __forceinline__ uint8_t nvfp4Encode(float value) {
  float clamped = fminf(fmaxf(value, -kMaxFiniteNvfp4), kMaxFiniteNvfp4);
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

__device__ __forceinline__ float nvfp4DefaultScale(float maxAbs) {
  return maxAbs == 0.0f ? 1.0f : (maxAbs / kMaxFiniteNvfp4);
}

__device__ __forceinline__ bool nvfp4PayloadIsZero(const uint8_t* blockPtr) {
  uint8_t any = 0;
  #pragma unroll
  for (int i = 0; i < kPackedBytesPerBlock; ++i) any |= blockPtr[2 + i];
  return any == 0;
}

__device__ __forceinline__ void nvfp4StoreZeroBlock(uint8_t* blockPtr) {
  #pragma unroll
  for (int i = 0; i < kBytesPerNvfp4Block; ++i) blockPtr[i] = 0;
}

__device__ __forceinline__ float nvfp4ChooseReduceScaleL2(const float* values, int validCount, float fallbackScale) {
  (void)values;
  (void)validCount;
  float scale = __half2float(__float2half_rn(fallbackScale));
  return (scale > 0.0f && isfinite(scale)) ? scale : 1.0f;
}

template <typename T>
__device__ __forceinline__ float loadAsFloat(const T* data, size_t index);

template <>
__device__ __forceinline__ float loadAsFloat<__half>(const __half* data, size_t index) {
  return __half2float(data[index]);
}

template <>
__device__ __forceinline__ float loadAsFloat<__nv_bfloat16>(const __nv_bfloat16* data, size_t index) {
  return __bfloat162float(data[index]);
}

template <typename T>
__device__ __forceinline__ void storeFromFloat(T* data, size_t index, float value);

template <>
__device__ __forceinline__ void storeFromFloat<__half>(__half* data, size_t index, float value) {
  data[index] = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void storeFromFloat<__nv_bfloat16>(__nv_bfloat16* data, size_t index, float value) {
  data[index] = __float2bfloat16(value);
}

template <typename T>
__global__ void nvfp4PackKernel(const T* input, uint8_t* output, size_t sectionElts,
                                size_t blocksPerSection, size_t sections, size_t sectionBytes) {
  size_t globalBlock = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalBlocks = blocksPerSection * sections;
  if (globalBlock >= totalBlocks) return;

  size_t section = globalBlock / blocksPerSection;
  size_t blockInSection = globalBlock % blocksPerSection;
  size_t inputBase = section * sectionElts + blockInSection * kValuesPerNvfp4Block;
  uint8_t* blockPtr = output + section * sectionBytes + blockInSection * kBytesPerNvfp4Block;

  float maxAbs = 0.0f;
  #pragma unroll
  for (int i = 0; i < kValuesPerNvfp4Block; ++i) {
    size_t logicalIndex = blockInSection * kValuesPerNvfp4Block + i;
    if (logicalIndex >= sectionElts) break;
    float value = loadAsFloat(input, inputBase + i);
    maxAbs = fmaxf(maxAbs, fabsf(value));
  }

  if (maxAbs == 0.0f) {
    nvfp4StoreZeroBlock(blockPtr);
    return;
  }

  __half scale = __float2half_rn(maxAbs / kMaxFiniteNvfp4);
  reinterpret_cast<__half*>(blockPtr)[0] = scale;
  float invScale = kMaxFiniteNvfp4 / maxAbs;

  #pragma unroll
  for (int byteIdx = 0; byteIdx < kPackedBytesPerBlock; ++byteIdx) {
    uint8_t packed = 0;
    #pragma unroll
    for (int lane = 0; lane < kPackedValuesPerByte; ++lane) {
      int eltInBlock = byteIdx * kPackedValuesPerByte + lane;
      size_t logicalIndex = blockInSection * kValuesPerNvfp4Block + eltInBlock;
      float value = 0.0f;
      if (logicalIndex < sectionElts) {
        value = loadAsFloat(input, inputBase + eltInBlock) * invScale;
      }
      uint8_t nibble = nvfp4Encode(value);
      packed |= (lane == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
    }
    blockPtr[2 + byteIdx] = packed;
  }
}

template <typename T>
__global__ void nvfp4UnpackKernel(const uint8_t* input, T* output, size_t sectionElts,
                                  size_t sections, size_t sectionBytes) {
  size_t globalElt = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalElts = sectionElts * sections;
  if (globalElt >= totalElts) return;

  size_t section = globalElt / sectionElts;
  size_t eltInSection = globalElt % sectionElts;
  size_t blockInSection = eltInSection / kValuesPerNvfp4Block;
  size_t eltInBlock = eltInSection % kValuesPerNvfp4Block;

  const uint8_t* blockPtr = input + section * sectionBytes + blockInSection * kBytesPerNvfp4Block;
  float scale = __half2float(reinterpret_cast<const __half*>(blockPtr)[0]);
  uint8_t packed = blockPtr[2 + (eltInBlock / 2)];
  uint8_t nibble = (eltInBlock & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0xF);
  storeFromFloat(output, globalElt, nvfp4Decode(nibble) * scale);
}

template <typename T>
__global__ void nvfp4ReduceKernel(const uint8_t* gatheredInput, T* output, size_t sectionElts,
                                  int nRanks, size_t sectionBytes, ncclRedOp_t op) {
  size_t elt = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (elt >= sectionElts) return;

  float acc = 0.0f;
  bool initialized = false;
  for (int rank = 0; rank < nRanks; ++rank) {
    const uint8_t* sectionPtr = gatheredInput + static_cast<size_t>(rank) * sectionBytes;
    size_t blockInSection = elt / kValuesPerNvfp4Block;
    size_t eltInBlock = elt % kValuesPerNvfp4Block;
    const uint8_t* blockPtr = sectionPtr + blockInSection * kBytesPerNvfp4Block;
    float scale = __half2float(reinterpret_cast<const __half*>(blockPtr)[0]);
    uint8_t packed = blockPtr[2 + (eltInBlock / 2)];
    uint8_t nibble = (eltInBlock & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0xF);
    float value = nvfp4Decode(nibble) * scale;

    if (!initialized) {
      acc = value;
      initialized = true;
      continue;
    }

    switch (op) {
    case ncclSum:
    case ncclAvg:
      acc += value;
      break;
    case ncclProd:
      acc *= value;
      break;
    case ncclMin:
      acc = fminf(acc, value);
      break;
    case ncclMax:
      acc = fmaxf(acc, value);
      break;
    default:
      break;
    }
  }

  if (op == ncclAvg && nRanks > 0) acc /= static_cast<float>(nRanks);
  storeFromFloat(output, elt, acc);
}



__global__ void nvfp4ReducePackKernel(const uint8_t* packedInput, const uint8_t* localPackedInput,
                                      uint8_t* packedOutput, size_t sectionElts,
                                      size_t blocksPerSection, ncclRedOp_t op,
                                      int nRanks, bool finalize) {
  size_t blockInSection = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (blockInSection >= blocksPerSection) return;

  const uint8_t* inputBlockPtr = packedInput + blockInSection * kBytesPerNvfp4Block;
  const uint8_t* localBlockPtr = localPackedInput + blockInSection * kBytesPerNvfp4Block;
  uint8_t* outputBlockPtr = packedOutput + blockInSection * kBytesPerNvfp4Block;
  if (nvfp4PayloadIsZero(inputBlockPtr) && nvfp4PayloadIsZero(localBlockPtr)) {
    nvfp4StoreZeroBlock(outputBlockPtr);
    return;
  }
  float inputScale = __half2float(reinterpret_cast<const __half*>(inputBlockPtr)[0]);
  float localScale = __half2float(reinterpret_cast<const __half*>(localBlockPtr)[0]);
  float values[kValuesPerNvfp4Block];
  float maxAbs = 0.0f;

  #pragma unroll
  for (int i = 0; i < kValuesPerNvfp4Block; ++i) {
    size_t logicalIndex = blockInSection * kValuesPerNvfp4Block + i;
    float reducedValue = 0.0f;
    if (logicalIndex < sectionElts) {
      uint8_t inputPacked = inputBlockPtr[2 + (i / 2)];
      uint8_t inputNibble = (i & 1) ? static_cast<uint8_t>(inputPacked >> 4)
                                    : static_cast<uint8_t>(inputPacked & 0xF);
      uint8_t localPacked = localBlockPtr[2 + (i / 2)];
      uint8_t localNibble = (i & 1) ? static_cast<uint8_t>(localPacked >> 4)
                                    : static_cast<uint8_t>(localPacked & 0xF);
      float receivedValue = nvfp4Decode(inputNibble) * inputScale;
      float localValue = nvfp4Decode(localNibble) * localScale;
      switch (op) {
      case ncclSum:
      case ncclAvg:
        reducedValue = receivedValue + localValue;
        break;
      case ncclProd:
        reducedValue = receivedValue * localValue;
        break;
      case ncclMin:
        reducedValue = fminf(receivedValue, localValue);
        break;
      case ncclMax:
        reducedValue = fmaxf(receivedValue, localValue);
        break;
      default:
        reducedValue = receivedValue;
        break;
      }
      if (finalize && op == ncclAvg && nRanks > 0) {
        reducedValue /= static_cast<float>(nRanks);
      }
      maxAbs = fmaxf(maxAbs, fabsf(reducedValue));
    }
    values[i] = reducedValue;
  }

  int validCount = static_cast<int>(sectionElts - blockInSection * kValuesPerNvfp4Block);
  if (validCount > kValuesPerNvfp4Block) validCount = kValuesPerNvfp4Block;
  if (maxAbs == 0.0f) {
    nvfp4StoreZeroBlock(outputBlockPtr);
    return;
  }

  float chosenScale = nvfp4ChooseReduceScaleL2(values, validCount, nvfp4DefaultScale(maxAbs));
  __half outputScale = __float2half_rn(chosenScale);
  reinterpret_cast<__half*>(outputBlockPtr)[0] = outputScale;
  float storedScale = __half2float(outputScale);
  float invScale = storedScale == 0.0f ? 0.0f : (1.0f / storedScale);

  #pragma unroll
  for (int byteIdx = 0; byteIdx < kPackedBytesPerBlock; ++byteIdx) {
    uint8_t packed = 0;
    #pragma unroll
    for (int lane = 0; lane < kPackedValuesPerByte; ++lane) {
      int eltInBlock = byteIdx * kPackedValuesPerByte + lane;
      size_t logicalIndex = blockInSection * kValuesPerNvfp4Block + eltInBlock;
      float value = logicalIndex < sectionElts ? values[eltInBlock] * invScale : 0.0f;
      uint8_t nibble = nvfp4Encode(value);
      packed |= (lane == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
    }
    outputBlockPtr[2 + byteIdx] = packed;
  }
}

__global__ void nvfp4ReducePackSumKernel(const uint8_t* packedInput, const uint8_t* localPackedInput,
                                         uint8_t* packedOutput, size_t sectionElts,
                                         size_t blocksPerSection) {
  size_t blockInSection = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (blockInSection >= blocksPerSection) return;

  const uint8_t* inputBlockPtr = packedInput + blockInSection * kBytesPerNvfp4Block;
  const uint8_t* localBlockPtr = localPackedInput + blockInSection * kBytesPerNvfp4Block;
  uint8_t* outputBlockPtr = packedOutput + blockInSection * kBytesPerNvfp4Block;
  if (nvfp4PayloadIsZero(inputBlockPtr) && nvfp4PayloadIsZero(localBlockPtr)) {
    nvfp4StoreZeroBlock(outputBlockPtr);
    return;
  }

  float inputScale = __half2float(reinterpret_cast<const __half*>(inputBlockPtr)[0]);
  float localScale = __half2float(reinterpret_cast<const __half*>(localBlockPtr)[0]);
  float values[kValuesPerNvfp4Block];
  float maxAbs = 0.0f;

  #pragma unroll
  for (int i = 0; i < kValuesPerNvfp4Block; ++i) {
    size_t logicalIndex = blockInSection * kValuesPerNvfp4Block + i;
    float reducedValue = 0.0f;
    if (logicalIndex < sectionElts) {
      uint8_t inputPacked = inputBlockPtr[2 + (i / 2)];
      uint8_t inputNibble = (i & 1) ? static_cast<uint8_t>(inputPacked >> 4)
                                    : static_cast<uint8_t>(inputPacked & 0xF);
      uint8_t localPacked = localBlockPtr[2 + (i / 2)];
      uint8_t localNibble = (i & 1) ? static_cast<uint8_t>(localPacked >> 4)
                                    : static_cast<uint8_t>(localPacked & 0xF);
      reducedValue = nvfp4Decode(inputNibble) * inputScale + nvfp4Decode(localNibble) * localScale;
      maxAbs = fmaxf(maxAbs, fabsf(reducedValue));
    }
    values[i] = reducedValue;
  }

  int validCount = static_cast<int>(sectionElts - blockInSection * kValuesPerNvfp4Block);
  if (validCount > kValuesPerNvfp4Block) validCount = kValuesPerNvfp4Block;
  if (maxAbs == 0.0f) {
    nvfp4StoreZeroBlock(outputBlockPtr);
    return;
  }

  float chosenScale = nvfp4ChooseReduceScaleL2(values, validCount, nvfp4DefaultScale(maxAbs));
  __half outputScale = __float2half_rn(chosenScale);
  reinterpret_cast<__half*>(outputBlockPtr)[0] = outputScale;
  float storedScale = __half2float(outputScale);
  float invScale = storedScale == 0.0f ? 0.0f : (1.0f / storedScale);

  #pragma unroll
  for (int byteIdx = 0; byteIdx < kPackedBytesPerBlock; ++byteIdx) {
    uint8_t packed = 0;
    #pragma unroll
    for (int lane = 0; lane < kPackedValuesPerByte; ++lane) {
      int eltInBlock = byteIdx * kPackedValuesPerByte + lane;
      size_t logicalIndex = blockInSection * kValuesPerNvfp4Block + eltInBlock;
      float value = logicalIndex < sectionElts ? values[eltInBlock] * invScale : 0.0f;
      uint8_t nibble = nvfp4Encode(value);
      packed |= (lane == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
    }
    outputBlockPtr[2 + byteIdx] = packed;
  }
}


template <typename T>
ncclResult_t launchPack(cudaStream_t stream, const void* input, uint8_t* output,
                        size_t sectionElts, size_t sections) {
  size_t blocksPerSection = ncclNvfp4NumBlocks(sectionElts);
  if (blocksPerSection == 0 || sections == 0) return ncclSuccess;

  size_t sectionBytes = 0;
  if (!ncclNvfp4SectionBytesChecked(sectionElts, &sectionBytes)) return ncclInvalidArgument;
  if (blocksPerSection != 0 && sections > SIZE_MAX / blocksPerSection) return ncclInvalidArgument;
  if (sectionBytes != 0 && sections > SIZE_MAX / sectionBytes) return ncclInvalidArgument;

  size_t totalBlocks = blocksPerSection * sections;
  size_t totalBytes = sectionBytes * sections;
  int grid = static_cast<int>((totalBlocks + kThreadsPerBlock - 1) / kThreadsPerBlock);
  CUDACHECK(cudaMemsetAsync(output, 0, totalBytes, stream));
  nvfp4PackKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      static_cast<const T*>(input), output, sectionElts, blocksPerSection, sections, sectionBytes);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
ncclResult_t launchUnpack(cudaStream_t stream, const uint8_t* input, void* output,
                          size_t sectionElts, size_t sections) {
  size_t sectionBytes = 0;
  if (!ncclNvfp4SectionBytesChecked(sectionElts, &sectionBytes)) return ncclInvalidArgument;
  if (sectionElts != 0 && sections > SIZE_MAX / sectionElts) return ncclInvalidArgument;

  size_t totalElts = sectionElts * sections;
  if (totalElts == 0) return ncclSuccess;
  int grid = static_cast<int>((totalElts + kThreadsPerBlock - 1) / kThreadsPerBlock);
  nvfp4UnpackKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      input, static_cast<T*>(output), sectionElts, sections, sectionBytes);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

template <typename T>
ncclResult_t launchReduce(cudaStream_t stream, const uint8_t* gatheredInput, void* output,
                          size_t sectionElts, int nRanks, ncclRedOp_t op) {
  if (sectionElts == 0 || nRanks == 0) return ncclSuccess;

  size_t sectionBytes = 0;
  if (!ncclNvfp4SectionBytesChecked(sectionElts, &sectionBytes)) return ncclInvalidArgument;

  int grid = static_cast<int>((sectionElts + kThreadsPerBlock - 1) / kThreadsPerBlock);
  nvfp4ReduceKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      gatheredInput, static_cast<T*>(output), sectionElts, nRanks, sectionBytes, op);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t launchReduceAndPack(cudaStream_t stream, const uint8_t* packedInput,
                                 const uint8_t* localPackedInput, uint8_t* packedOutput,
                                 size_t sectionElts, ncclRedOp_t op, int nRanks,
                                 bool finalize) {
  size_t blocksPerSection = ncclNvfp4NumBlocks(sectionElts);
  if (blocksPerSection == 0) return ncclSuccess;

  size_t sectionBytes = 0;
  if (!ncclNvfp4SectionBytesChecked(sectionElts, &sectionBytes)) return ncclInvalidArgument;

  int grid = static_cast<int>((blocksPerSection + kThreadsPerBlock - 1) / kThreadsPerBlock);
  if (op == ncclSum) {
    nvfp4ReducePackSumKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        packedInput, localPackedInput, packedOutput, sectionElts, blocksPerSection);
  } else {
    nvfp4ReducePackKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        packedInput, localPackedInput, packedOutput, sectionElts,
        blocksPerSection, op, nRanks, finalize);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

}  // namespace

ncclResult_t ncclNvfp4Pack(cudaStream_t stream, const void* input, uint8_t* output,
                           size_t sectionElts, size_t sections, ncclDataType_t datatype) {
  switch (datatype) {
  case ncclFloat16:
    return launchPack<__half>(stream, input, output, sectionElts, sections);
  case ncclBfloat16:
    return launchPack<__nv_bfloat16>(stream, input, output, sectionElts, sections);
  default:
    return ncclInvalidArgument;
  }
}

ncclResult_t ncclNvfp4Unpack(cudaStream_t stream, const uint8_t* input, void* output,
                             size_t sectionElts, size_t sections, ncclDataType_t datatype) {
  switch (datatype) {
  case ncclFloat16:
    return launchUnpack<__half>(stream, input, output, sectionElts, sections);
  case ncclBfloat16:
    return launchUnpack<__nv_bfloat16>(stream, input, output, sectionElts, sections);
  default:
    return ncclInvalidArgument;
  }
}

ncclResult_t ncclNvfp4ReduceFromGathered(cudaStream_t stream, const uint8_t* gatheredInput, void* output,
                                         size_t sectionElts, int nRanks, ncclDataType_t datatype, ncclRedOp_t op) {
  switch (op) {
  case ncclSum:
  case ncclProd:
  case ncclMin:
  case ncclMax:
  case ncclAvg:
    break;
  default:
    return ncclInvalidArgument;
  }

  switch (datatype) {
  case ncclFloat16:
    return launchReduce<__half>(stream, gatheredInput, output, sectionElts, nRanks, op);
  case ncclBfloat16:
    return launchReduce<__nv_bfloat16>(stream, gatheredInput, output, sectionElts, nRanks, op);
  default:
    return ncclInvalidArgument;
  }
}

ncclResult_t ncclNvfp4ReduceAndPack(cudaStream_t stream, const uint8_t* packedInput,
                                    const uint8_t* localPackedInput, uint8_t* packedOutput,
                                    size_t sectionElts, ncclRedOp_t op, int nRanks,
                                    bool finalize) {
  switch (op) {
  case ncclSum:
  case ncclProd:
  case ncclMin:
  case ncclMax:
  case ncclAvg:
    break;
  default:
    return ncclInvalidArgument;
  }

  return launchReduceAndPack(stream, packedInput, localPackedInput, packedOutput,
                             sectionElts, op, nRanks, finalize);
}
