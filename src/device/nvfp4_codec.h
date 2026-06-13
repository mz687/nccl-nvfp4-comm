/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_NVFP4_CODEC_H_
#define NCCL_DEVICE_NVFP4_CODEC_H_

#include "nvfp4.h"

#include <cuda_fp16.h>
#include <math.h>

namespace ncclNvfp4Device {

constexpr int kValuesPerBlock = NCCL_NVFP4_BLOCK_ELTS;
constexpr int kBytesPerBlock = NCCL_NVFP4_BLOCK_BYTES;
constexpr int kPackedValuesPerByte = 2;
constexpr int kPackedBytesPerBlock = 8;
constexpr float kMaxFinite = 6.0f;

__host__ __device__ __forceinline__ float magnitude(int code) {
  constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x7];
}

__host__ __device__ __forceinline__ float decode(uint8_t nibble) {
  float mag = magnitude(nibble & 0x7);
  return (nibble & 0x8) ? -mag : mag;
}

__host__ __device__ __forceinline__ uint8_t encode(float value) {
  float clamped = fminf(fmaxf(value, -kMaxFinite), kMaxFinite);
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

__host__ __device__ __forceinline__ float defaultScale(float maxAbs) {
  return maxAbs == 0.0f ? 1.0f : (maxAbs / kMaxFinite);
}

__host__ __device__ __forceinline__ int validCountForBlock(size_t logicalCount, size_t blockIndex) {
  size_t blockStart = blockIndex * kValuesPerBlock;
  if (blockStart >= logicalCount) return 0;
  size_t remaining = logicalCount - blockStart;
  return remaining >= (size_t)kValuesPerBlock ? kValuesPerBlock : (int)remaining;
}

__host__ __device__ __forceinline__ float chooseReduceScaleL2(const float* values, int validCount, float fallbackScale) {
  (void)values;
  (void)validCount;
  float scale = __half2float(__float2half_rn(fallbackScale));
  return (scale > 0.0f && isfinite(scale)) ? scale : 1.0f;
}

__host__ __device__ __forceinline__ float loadScale(const uint8_t* blockPtr) {
  return __half2float(reinterpret_cast<const __half*>(blockPtr)[0]);
}

__host__ __device__ __forceinline__ void storeScale(uint8_t* blockPtr, float scale) {
  reinterpret_cast<__half*>(blockPtr)[0] = __float2half_rn(scale);
}

__host__ __device__ __forceinline__ void storeZeroBlock(uint8_t* blockPtr) {
  #pragma unroll
  for (int i = 0; i < kBytesPerBlock; ++i) blockPtr[i] = 0;
}

__device__ __forceinline__ void storeZeroBlockParallel(uint8_t* blockPtr, int laneInGroup) {
  if (laneInGroup < kBytesPerBlock) blockPtr[laneInGroup] = 0;
}

__device__ __forceinline__ bool payloadsAreZero(const uint8_t* receivedBlock, const uint8_t* localBlock,
                                                int laneInGroup, unsigned groupMask) {
  unsigned any = laneInGroup < kPackedBytesPerBlock
      ? (unsigned)(receivedBlock[2 + laneInGroup] | localBlock[2 + laneInGroup])
      : 0u;
  #pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    any |= __shfl_down_sync(groupMask, any, offset, 16);
  }
  any = __shfl_sync(groupMask, any, 0, 16);
  return any == 0u;
}

__device__ __forceinline__ float warpReduceMax16(float value, unsigned groupMask) {
  #pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(groupMask, value, offset, 16));
  }
  return value;
}

__device__ __forceinline__ float reduceValues(float receivedValue, float localValue, ncclRedOp_t op) {
  switch (op) {
  case ncclSum:
  case ncclAvg:
    return receivedValue + localValue;
  case ncclProd:
    return receivedValue * localValue;
  case ncclMin:
    return fminf(receivedValue, localValue);
  case ncclMax:
    return fmaxf(receivedValue, localValue);
  default:
    return receivedValue;
  }
}

static __device__ __noinline__ void reduceAndPackBlock(const uint8_t* receivedBlock, const uint8_t* localBlock,
                                                   uint8_t* outputBlock, uint8_t* sendBlock, int validCount,
                                                   ncclRedOp_t op, int nRanks, bool finalize,
                                                   int laneInGroup, unsigned groupMask) {
  if (payloadsAreZero(receivedBlock, localBlock, laneInGroup, groupMask)) {
    if (outputBlock != nullptr) storeZeroBlockParallel(outputBlock, laneInGroup);
    if (sendBlock != nullptr && sendBlock != outputBlock) storeZeroBlockParallel(sendBlock, laneInGroup);
    return;
  }

  float receivedScale = laneInGroup == 0 ? loadScale(receivedBlock) : 0.0f;
  float localScale = laneInGroup == 0 ? loadScale(localBlock) : 0.0f;
  receivedScale = __shfl_sync(groupMask, receivedScale, 0, 16);
  localScale = __shfl_sync(groupMask, localScale, 0, 16);

  float reducedValue = 0.0f;
  if (laneInGroup < validCount) {
    uint8_t receivedPacked = receivedBlock[2 + (laneInGroup >> 1)];
    uint8_t receivedNibble = (laneInGroup & 1) ? static_cast<uint8_t>(receivedPacked >> 4)
                                               : static_cast<uint8_t>(receivedPacked & 0xF);
    uint8_t localPacked = localBlock[2 + (laneInGroup >> 1)];
    uint8_t localNibble = (laneInGroup & 1) ? static_cast<uint8_t>(localPacked >> 4)
                                            : static_cast<uint8_t>(localPacked & 0xF);
    reducedValue = reduceValues(decode(receivedNibble) * receivedScale,
                                decode(localNibble) * localScale, op);
    if (finalize && op == ncclAvg && nRanks > 0) reducedValue *= (1.0f / nRanks);
  }

  float maxAbs = laneInGroup < validCount ? fabsf(reducedValue) : 0.0f;
  maxAbs = warpReduceMax16(maxAbs, groupMask);
  maxAbs = __shfl_sync(groupMask, maxAbs, 0, 16);
  if (maxAbs == 0.0f) {
    if (outputBlock != nullptr) storeZeroBlockParallel(outputBlock, laneInGroup);
    if (sendBlock != nullptr && sendBlock != outputBlock) storeZeroBlockParallel(sendBlock, laneInGroup);
    return;
  }

  float chosenScale = laneInGroup == 0 ? chooseReduceScaleL2(nullptr, validCount, defaultScale(maxAbs)) : 0.0f;
  chosenScale = __shfl_sync(groupMask, chosenScale, 0, 16);
  float storedScale = __half2float(__float2half_rn(chosenScale));
  if (laneInGroup == 0) {
    if (outputBlock != nullptr) storeScale(outputBlock, storedScale);
    if (sendBlock != nullptr && sendBlock != outputBlock) storeScale(sendBlock, storedScale);
  }
  float invScale = storedScale == 0.0f ? 0.0f : (1.0f / storedScale);

  int packLane = laneInGroup & 7;
  float lowValue = __shfl_sync(groupMask, reducedValue, packLane * 2, 16);
  float highValue = __shfl_sync(groupMask, reducedValue, packLane * 2 + 1, 16);
  if (laneInGroup < kPackedBytesPerBlock) {
    uint8_t lowNibble = encode(lowValue * invScale);
    uint8_t highNibble = encode(highValue * invScale);
    uint8_t packedByte = static_cast<uint8_t>(lowNibble | (highNibble << 4));
    if (outputBlock != nullptr) outputBlock[2 + laneInGroup] = packedByte;
    if (sendBlock != nullptr && sendBlock != outputBlock) sendBlock[2 + laneInGroup] = packedByte;
  }
}

} // namespace ncclNvfp4Device

#endif
