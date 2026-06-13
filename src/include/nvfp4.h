/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NVFP4_H_
#define NCCL_NVFP4_H_

#include "core.h"

enum ncclTransportCodec_t : uint8_t {
  ncclTransportCodecPlain = 0,
  ncclTransportCodecNvfp4 = 1
};

constexpr size_t NCCL_NVFP4_BLOCK_ELTS = 16;
constexpr size_t NCCL_NVFP4_BLOCK_BYTES = 10;
constexpr size_t NCCL_NVFP4_ALIGNMENT = 16;

struct ncclNvfp4Meta {
  size_t logicalCount;
  size_t logicalBytes;
  size_t sectionBytes;
};

inline bool ncclNvfp4BlockCountChecked(size_t logicalElts, size_t* blocks) {
  *blocks = logicalElts / NCCL_NVFP4_BLOCK_ELTS + ((logicalElts % NCCL_NVFP4_BLOCK_ELTS) != 0 ? 1 : 0);
  return *blocks <= SIZE_MAX / NCCL_NVFP4_BLOCK_BYTES;
}

inline size_t ncclNvfp4NumBlocks(size_t logicalElts) {
  return logicalElts / NCCL_NVFP4_BLOCK_ELTS + ((logicalElts % NCCL_NVFP4_BLOCK_ELTS) != 0 ? 1 : 0);
}

inline bool ncclNvfp4SectionBytesChecked(size_t logicalElts, size_t* bytes) {
  size_t blocks = ncclNvfp4NumBlocks(logicalElts);
  if (blocks > SIZE_MAX / NCCL_NVFP4_BLOCK_BYTES) return false;
  size_t logicalBytes = blocks * NCCL_NVFP4_BLOCK_BYTES;
  if (logicalBytes > SIZE_MAX - (NCCL_NVFP4_ALIGNMENT - 1)) return false;
  *bytes = (logicalBytes + NCCL_NVFP4_ALIGNMENT - 1) & ~(NCCL_NVFP4_ALIGNMENT - 1);
  return true;
}

inline size_t ncclNvfp4LogicalBytes(size_t logicalElts) {
  size_t bytes = 0;
  return ncclNvfp4SectionBytesChecked(logicalElts, &bytes) ? ncclNvfp4NumBlocks(logicalElts) * NCCL_NVFP4_BLOCK_BYTES : 0;
}

inline size_t ncclNvfp4SectionBytes(size_t logicalElts) {
  size_t bytes = 0;
  return ncclNvfp4SectionBytesChecked(logicalElts, &bytes) ? bytes : 0;
}

inline bool ncclNvfp4SupportedIoType(ncclDataType_t datatype) {
  return datatype == ncclFloat16 || datatype == ncclBfloat16;
}

ncclResult_t ncclNvfp4Pack(
    cudaStream_t stream, const void* input, uint8_t* output,
    size_t sectionElts, size_t sections, ncclDataType_t datatype);

ncclResult_t ncclNvfp4Unpack(
    cudaStream_t stream, const uint8_t* input, void* output,
    size_t sectionElts, size_t sections, ncclDataType_t datatype);

ncclResult_t ncclNvfp4ReduceFromGathered(
    cudaStream_t stream, const uint8_t* gatheredInput, void* output,
    size_t sectionElts, int nRanks, ncclDataType_t datatype, ncclRedOp_t op);

ncclResult_t ncclNvfp4ReduceAndPack(
    cudaStream_t stream, const uint8_t* packedInput, const uint8_t* localPackedInput,
    uint8_t* packedOutput, size_t sectionElts, ncclRedOp_t op,
    int nRanks, bool finalize);

#endif
