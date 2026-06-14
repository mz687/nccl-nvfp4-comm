/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#include "nvfp4_codec.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t gridOffset;
    ssize_t channelCount;
    ssize_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nranks * chunkCount;
    ssize_t offset;
    int nelem;
    int chunk;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      ssize_t chunkOffset;

      if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));

      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks - 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directSend(offset, offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j = 2; j < nranks; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvReduceDirectSend(offset, offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j = 1; j < nranks - 1; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvCopyDirectSend(offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);

      prims.directRecv(offset, nelem);
    }
  }

  template<typename Proto>
  __device__ __forceinline__ ssize_t ncclNvfp4StepSizeForChunk(ssize_t chunkCount) {
    constexpr ssize_t slicePerChunk = ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS;
    constexpr ssize_t blockBytes = ncclNvfp4Device::kBytesPerBlock;
    constexpr ssize_t stepPerSlice = ALLREDUCE_SLICESTEPS;
    ssize_t minSliceSize = divUp(max((ssize_t)1, chunkCount / 32), blockBytes) * blockBytes;
    ssize_t alignedSliceSize = divUp(chunkCount, blockBytes * slicePerChunk) * blockBytes;
    ssize_t sliceSize = max(alignedSliceSize, minSliceSize);
    ssize_t stepSize = divUp(sliceSize, stepPerSlice * blockBytes) * blockBytes;
    return stepSize > 0 ? stepSize : 1;
  }

  template<bool CopyToOutput>
  __device__ __forceinline__ void ncclNvfp4StoreByte(uint8_t* outBlock, uint8_t* sendBlock, int byte, uint8_t value) {
    if (CopyToOutput) outBlock[byte] = value;
    if (sendBlock != nullptr && (!CopyToOutput || sendBlock != outBlock)) sendBlock[byte] = value;
  }

  __device__ __forceinline__ void ncclNvfp4StoreZeroBlock(uint8_t* block) {
    uint16_t* halfwords = reinterpret_cast<uint16_t*>(block);
    #pragma unroll
    for (int i = 0; i < ncclNvfp4Device::kBytesPerBlock / (int)sizeof(uint16_t); ++i) {
      halfwords[i] = 0;
    }
  }

  template<bool CopyToOutput>
  __device__ __forceinline__ void ncclNvfp4StoreZero(uint8_t* outBlock, uint8_t* sendBlock) {
    if (CopyToOutput) ncclNvfp4StoreZeroBlock(outBlock);
    if (sendBlock != nullptr && (!CopyToOutput || sendBlock != outBlock)) ncclNvfp4StoreZeroBlock(sendBlock);
  }

  template<bool CopyToOutput>
  __device__ __forceinline__ void ncclNvfp4StoreScale(uint8_t* outBlock, uint8_t* sendBlock, float scale) {
    if (CopyToOutput) ncclNvfp4Device::storeScale(outBlock, scale);
    if (sendBlock != nullptr && (!CopyToOutput || sendBlock != outBlock)) ncclNvfp4Device::storeScale(sendBlock, scale);
  }

  __device__ __forceinline__ bool ncclNvfp4PayloadsAreZero(const uint8_t* recvBlock, const uint8_t* localBlock) {
    uint8_t payloadOr = 0;
    #pragma unroll
    for (int i = 0; i < ncclNvfp4Device::kPackedBytesPerBlock; ++i) {
      payloadOr |= recvBlock[2 + i] | localBlock[2 + i];
    }
    return payloadOr == 0;
  }

  template<bool CopyToOutput>
  struct Nvfp4ReduceSendOp {
    struct ncclDevWorkColl* work;
    size_t chunkOffset;
    size_t chunkBytes;

    __device__ __forceinline__ Nvfp4ReduceSendOp(struct ncclDevWorkColl* work, size_t chunkOffset, size_t chunkBytes)
      : work(work), chunkOffset(chunkOffset), chunkBytes(chunkBytes) {}

    template<int SlicePerChunk, int DirectRecv, int RecvMax, int DirectSend, int SendMax, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int nworkers, int slice, int sliceBytesMax,
        int nrecv, void** srcs, int nsend, void** dsts, int32_t* dstSizes,
        uint32_t, uint32_t) {
      size_t sliceOffset = size_t(slice) * size_t(sliceBytesMax);
      size_t sliceBytes = sliceOffset < chunkBytes ? min((size_t)sliceBytesMax, chunkBytes - sliceOffset) : 0;
      if (tid == 0 && nsend > 0) dstSizes[0] = (int)sliceBytes;

      const uint8_t* recvSlice = nrecv > 0 ? reinterpret_cast<const uint8_t*>(srcs[0]) : nullptr;
      uint8_t* sendSlice = nsend > 0 ? reinterpret_cast<uint8_t*>(dsts[0]) : nullptr;
      if (sliceBytes == 0 || nrecv == 0 || recvSlice == nullptr) return;
      if (nworkers == 0) return;

      const uint8_t* localSlice = reinterpret_cast<const uint8_t*>(work->sendbuff) + chunkOffset + sliceOffset;
      uint8_t* outSlice = reinterpret_cast<uint8_t*>(work->recvbuff) + chunkOffset + sliceOffset;
      size_t blockBase = (chunkOffset + sliceOffset) / ncclNvfp4Device::kBytesPerBlock;
      size_t nBlocks = sliceBytes / ncclNvfp4Device::kBytesPerBlock;
      size_t fullBlocks = work->transportAuxCount / ncclNvfp4Device::kValuesPerBlock;

      for (size_t block = (size_t)tid; block < nBlocks; block += (size_t)nworkers) {
        size_t blockOffset = block * ncclNvfp4Device::kBytesPerBlock;
        size_t globalBlock = blockBase + block;
        int validCount = globalBlock < fullBlocks
            ? ncclNvfp4Device::kValuesPerBlock
            : ncclNvfp4Device::validCountForBlock(work->transportAuxCount, globalBlock);
        const uint8_t* recvBlock = recvSlice + blockOffset;
        const uint8_t* localBlock = localSlice + blockOffset;
        uint8_t* outBlock = CopyToOutput ? outSlice + blockOffset : nullptr;
        uint8_t* sendBlock = sendSlice != nullptr ? sendSlice + blockOffset : nullptr;
        if (!CopyToOutput && sendBlock == nullptr) continue;

        if (validCount == 0 || ncclNvfp4PayloadsAreZero(recvBlock, localBlock)) {
          ncclNvfp4StoreZero<CopyToOutput>(outBlock, sendBlock);
          continue;
        }

        float recvScale = ncclNvfp4Device::loadScale(recvBlock);
        float localScale = ncclNvfp4Device::loadScale(localBlock);
        float values[ncclNvfp4Device::kValuesPerBlock];
        float maxAbs = 0.0f;
        ncclRedOp_t op = (ncclRedOp_t)work->transportOp;

        if (op == ncclSum || op == ncclAvg) {
          float postDiv = CopyToOutput && op == ncclAvg && ncclShmem.comm.nRanks > 0 ? (1.0f / ncclShmem.comm.nRanks) : 1.0f;
          if (validCount == ncclNvfp4Device::kValuesPerBlock) {
            #pragma unroll
            for (int i = 0; i < ncclNvfp4Device::kValuesPerBlock; ++i) {
              uint8_t recvPacked = recvBlock[2 + (i >> 1)];
              uint8_t localPacked = localBlock[2 + (i >> 1)];
              uint8_t recvNibble = (i & 1) ? static_cast<uint8_t>(recvPacked >> 4) : static_cast<uint8_t>(recvPacked & 0xF);
              uint8_t localNibble = (i & 1) ? static_cast<uint8_t>(localPacked >> 4) : static_cast<uint8_t>(localPacked & 0xF);
              float reducedValue = (ncclNvfp4Device::decode(recvNibble) * recvScale +
                                    ncclNvfp4Device::decode(localNibble) * localScale) * postDiv;
              values[i] = reducedValue;
              maxAbs = fmaxf(maxAbs, fabsf(reducedValue));
            }
          } else {
            #pragma unroll
            for (int i = 0; i < ncclNvfp4Device::kValuesPerBlock; ++i) {
              float reducedValue = 0.0f;
              if (i < validCount) {
                uint8_t recvPacked = recvBlock[2 + (i >> 1)];
                uint8_t localPacked = localBlock[2 + (i >> 1)];
                uint8_t recvNibble = (i & 1) ? static_cast<uint8_t>(recvPacked >> 4) : static_cast<uint8_t>(recvPacked & 0xF);
                uint8_t localNibble = (i & 1) ? static_cast<uint8_t>(localPacked >> 4) : static_cast<uint8_t>(localPacked & 0xF);
                reducedValue = (ncclNvfp4Device::decode(recvNibble) * recvScale +
                                ncclNvfp4Device::decode(localNibble) * localScale) * postDiv;
                maxAbs = fmaxf(maxAbs, fabsf(reducedValue));
              }
              values[i] = reducedValue;
            }
          }
        } else {
          #pragma unroll
          for (int i = 0; i < ncclNvfp4Device::kValuesPerBlock; ++i) {
            float reducedValue = 0.0f;
            if (i < validCount) {
              uint8_t recvPacked = recvBlock[2 + (i >> 1)];
              uint8_t localPacked = localBlock[2 + (i >> 1)];
              uint8_t recvNibble = (i & 1) ? static_cast<uint8_t>(recvPacked >> 4) : static_cast<uint8_t>(recvPacked & 0xF);
              uint8_t localNibble = (i & 1) ? static_cast<uint8_t>(localPacked >> 4) : static_cast<uint8_t>(localPacked & 0xF);
              float recvValue = ncclNvfp4Device::decode(recvNibble) * recvScale;
              float localValue = ncclNvfp4Device::decode(localNibble) * localScale;
              switch (op) {
              case ncclProd:
                reducedValue = recvValue * localValue;
                break;
              case ncclMin:
                reducedValue = fminf(recvValue, localValue);
                break;
              case ncclMax:
                reducedValue = fmaxf(recvValue, localValue);
                break;
              default:
                reducedValue = recvValue;
                break;
              }
              maxAbs = fmaxf(maxAbs, fabsf(reducedValue));
            }
            values[i] = reducedValue;
          }
        }

        if (maxAbs == 0.0f) {
          ncclNvfp4StoreZero<CopyToOutput>(outBlock, sendBlock);
        } else {
          float chosenScale = ncclNvfp4Device::chooseReduceScaleL2(values, validCount, ncclNvfp4Device::defaultScale(maxAbs));
          float storedScale = __half2float(__float2half_rn(chosenScale));
          ncclNvfp4StoreScale<CopyToOutput>(outBlock, sendBlock, storedScale);
          float invScale = storedScale == 0.0f ? 0.0f : (1.0f / storedScale);
          #pragma unroll
          for (int byteIdx = 0; byteIdx < ncclNvfp4Device::kPackedBytesPerBlock; ++byteIdx) {
            uint8_t lowNibble = ncclNvfp4Device::encode(values[byteIdx * 2] * invScale);
            uint8_t highNibble = ncclNvfp4Device::encode(values[byteIdx * 2 + 1] * invScale);
            ncclNvfp4StoreByte<CopyToOutput>(
                outBlock, sendBlock, 2 + byteIdx, static_cast<uint8_t>(lowNibble | (highNibble << 4)));
          }
        }
      }
    }
  };

  struct Nvfp4LocalSendOp {
    struct ncclDevWorkColl* work;
    size_t chunkOffset;
    size_t chunkBytes;

    __device__ __forceinline__ Nvfp4LocalSendOp(struct ncclDevWorkColl* work, size_t chunkOffset, size_t chunkBytes)
      : work(work), chunkOffset(chunkOffset), chunkBytes(chunkBytes) {}

    template<int SlicePerChunk, int DirectRecv, int RecvMax, int DirectSend, int SendMax, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int nworkers, int slice, int sliceBytesMax,
        int, void**, int nsend, void** dsts, int32_t* dstSizes,
        uint32_t, uint32_t) {
      size_t sliceOffset = size_t(slice) * size_t(sliceBytesMax);
      size_t sliceBytes = sliceOffset < chunkBytes ? min((size_t)sliceBytesMax, chunkBytes - sliceOffset) : 0;
      if (tid == 0 && nsend > 0) dstSizes[0] = (int)sliceBytes;
      if (sliceBytes == 0 || nsend == 0 || dsts[0] == nullptr) return;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(work->sendbuff) + chunkOffset + sliceOffset;
      uint8_t* dst = reinterpret_cast<uint8_t*>(dsts[0]);
      for (size_t i = (size_t)tid; i < sliceBytes; i += (size_t)nworkers) dst[i] = src[i];
    }
  };

  template<bool ForwardSend>
  struct Nvfp4RecvCopyOp {
    struct ncclDevWorkColl* work;
    size_t chunkOffset;
    size_t chunkBytes;

    __device__ __forceinline__ Nvfp4RecvCopyOp(struct ncclDevWorkColl* work, size_t chunkOffset, size_t chunkBytes)
      : work(work), chunkOffset(chunkOffset), chunkBytes(chunkBytes) {}

    template<int SlicePerChunk, int DirectRecv, int RecvMax, int DirectSend, int SendMax, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int nworkers, int slice, int sliceBytesMax,
        int nrecv, void** srcs, int nsend, void** dsts, int32_t* dstSizes,
        uint32_t, uint32_t) {
      size_t sliceOffset = size_t(slice) * size_t(sliceBytesMax);
      size_t sliceBytes = sliceOffset < chunkBytes ? min((size_t)sliceBytesMax, chunkBytes - sliceOffset) : 0;
      if (tid == 0 && ForwardSend && nsend > 0) dstSizes[0] = (int)sliceBytes;
      if (sliceBytes == 0 || nrecv == 0 || srcs[0] == nullptr) return;
      const uint8_t* src = reinterpret_cast<const uint8_t*>(srcs[0]);
      uint8_t* out = reinterpret_cast<uint8_t*>(work->recvbuff) + chunkOffset + sliceOffset;
      uint8_t* send = ForwardSend && nsend > 0 && dsts[0] != nullptr ? reinterpret_cast<uint8_t*>(dsts[0]) : nullptr;
      for (size_t i = (size_t)tid; i < sliceBytes; i += (size_t)nworkers) {
        uint8_t v = src[i];
        out[i] = v;
        if (send != nullptr && send != out) send[i] = v;
      }
    }
  };

  __device__ __forceinline__ void ncclNvfp4ThreadBlockSync(int nthreads) {
    if (nthreads == WARP_SIZE) __syncwarp();
    else __syncthreads();
  }

  template<bool Finalize>
  __device__ __forceinline__ void ncclNvfp4ReduceChunkInPlace(
      int tid, int nthreads, struct ncclDevWorkColl* work, size_t chunkOffset, size_t chunkBytes) {
    if (chunkBytes == 0) return;
    size_t blockBase = chunkOffset / ncclNvfp4Device::kBytesPerBlock;
    size_t nBlocks = chunkBytes / ncclNvfp4Device::kBytesPerBlock;
    size_t fullBlocks = work->transportAuxCount / ncclNvfp4Device::kValuesPerBlock;
    int laneInWarp = tid & 31;
    int laneInGroup = laneInWarp & 15;
    unsigned groupMask = (laneInWarp & 16) ? 0xffff0000u : 0x0000ffffu;
    int groupIndex = tid >> 4;
    int nGroups = nthreads >> 4;
    if (nGroups == 0) return;

    const uint8_t* localBase = reinterpret_cast<const uint8_t*>(work->sendbuff) + chunkOffset;
    uint8_t* accumBase = reinterpret_cast<uint8_t*>(work->recvbuff) + chunkOffset;

    for (size_t block = (size_t)groupIndex; block < nBlocks; block += (size_t)nGroups) {
      size_t globalBlock = blockBase + block;
      int validCount = globalBlock < fullBlocks
          ? ncclNvfp4Device::kValuesPerBlock
          : ncclNvfp4Device::validCountForBlock(work->transportAuxCount, globalBlock);
      uint8_t* accumBlock = accumBase + block * ncclNvfp4Device::kBytesPerBlock;
      ncclNvfp4Device::reduceAndPackBlock(
          accumBlock,
          localBase + block * ncclNvfp4Device::kBytesPerBlock,
          accumBlock,
          nullptr,
          validCount,
          (ncclRedOp_t)work->transportOp,
          ncclShmem.comm.nRanks,
          Finalize,
          laneInGroup,
          groupMask);
    }
  }

  template<typename Proto>
  __device__ __noinline__ void runRingNvfp4(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t gridOffset;
    ssize_t channelCount;
    ssize_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(uint8_t), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    constexpr ssize_t blockBytes = (ssize_t)ncclNvfp4Device::kBytesPerBlock;
    ssize_t blockStart = gridOffset / blockBytes;
    ssize_t blockEnd = divUp(gridOffset + channelCount, blockBytes);
    gridOffset = blockStart * blockBytes;
    channelCount = (blockEnd > blockStart ? blockEnd - blockStart : 0) * blockBytes;
    if (channelCount == 0) return;
    chunkCount = divUp(chunkCount, blockBytes) * blockBytes;
    const ssize_t fullChunkCount = chunkCount;
    const ssize_t loopCount = nranks * fullChunkCount;
    ssize_t offset;
    int nelem;
    int chunk;

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      ssize_t chunkOffset;
      ssize_t chunkCountThisLoop = fullChunkCount;
      bool variableChunks = false;
      ssize_t baseBlocksThisLoop = 0;
      ssize_t remBlocksThisLoop = 0;

      if (remCount < loopCount) {
        ssize_t blocksThisLoop = divUp(remCount, blockBytes);
        baseBlocksThisLoop = blocksThisLoop / nranks;
        remBlocksThisLoop = blocksThisLoop - baseBlocksThisLoop * nranks;
        chunkCountThisLoop = (baseBlocksThisLoop + (remBlocksThisLoop != 0 ? 1 : 0)) * blockBytes;
        variableChunks = remBlocksThisLoop != 0;
      }

      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      auto chunkOffsetFor = [&]__device__(int curChunk)->ssize_t {
        if (!variableChunks) return (ssize_t)curChunk * chunkCountThisLoop;
        ssize_t blocksBefore = (ssize_t)curChunk * baseBlocksThisLoop + min((ssize_t)curChunk, remBlocksThisLoop);
        return blocksBefore * blockBytes;
      };

      auto elemCountForChunk = [&]__device__(int curChunk, ssize_t curChunkOffset)->int {
        if (variableChunks) {
          ssize_t blocks = baseBlocksThisLoop + ((ssize_t)curChunk < remBlocksThisLoop ? 1 : 0);
          return (int)(blocks * blockBytes);
        }
        ssize_t available = remCount - curChunkOffset;
        if (available <= 0) return 0;
        return (int)min(chunkCountThisLoop, available);
      };

      Primitives<uint8_t, FuncCopy<uint8_t>, FanSymmetric<1>, 0, Proto, 0> prims(
        tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg,
        0, 0, 0, work, nullptr, (int)ncclNvfp4StepSizeForChunk<Proto>(chunkCountThisLoop));

      if (!variableChunks) {
        chunk = modRanks(ringIx + nranks - 1);
        chunkOffset = chunkOffsetFor(chunk);
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = elemCountForChunk(chunk, chunkOffset);
        Nvfp4LocalSendOp localSend(work, offset, nelem);
        prims.template process<0, 1>(localSend);

        for (int j = 2; j < nranks; ++j) {
          chunk = modRanks(ringIx + nranks - j);
          chunkOffset = chunkOffsetFor(chunk);
          offset = gridOffset + elemOffset + chunkOffset;
          nelem = elemCountForChunk(chunk, chunkOffset);
          Nvfp4ReduceSendOp<false> reduceForward(work, offset, nelem);
          prims.template process<1, 1>(reduceForward);
        }

        chunk = ringIx + 0;
        chunkOffset = chunkOffsetFor(chunk);
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = elemCountForChunk(chunk, chunkOffset);
        Nvfp4ReduceSendOp<true> reduceStore(work, offset, nelem);
        prims.template process<1, 1>(reduceStore);

        for (int j = 1; j < nranks - 1; ++j) {
          chunk = modRanks(ringIx + nranks - j);
          chunkOffset = chunkOffsetFor(chunk);
          offset = gridOffset + elemOffset + chunkOffset;
          nelem = elemCountForChunk(chunk, chunkOffset);
          Nvfp4RecvCopyOp<true> copyForward(work, offset, nelem);
          prims.template process<1, 1>(copyForward);
        }

        chunk = modRanks(ringIx + 1);
        chunkOffset = chunkOffsetFor(chunk);
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = elemCountForChunk(chunk, chunkOffset);
        Nvfp4RecvCopyOp<false> copyRecv(work, offset, nelem);
        prims.template process<1, 0>(copyRecv);
      } else {
        chunk = modRanks(ringIx + nranks - 1);
        chunkOffset = chunkOffsetFor(chunk);
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = elemCountForChunk(chunk, chunkOffset);
        Nvfp4LocalSendOp localSend(work, offset, nelem);
        prims.template process<0, 1>(localSend);

        for (int j = 2; j < nranks; ++j) {
          chunk = modRanks(ringIx + nranks - j);
          chunkOffset = chunkOffsetFor(chunk);
          offset = gridOffset + elemOffset + chunkOffset;
          nelem = elemCountForChunk(chunk, chunkOffset);
          Nvfp4ReduceSendOp<false> reduceForward(work, offset, nelem);
          prims.template process<1, 1>(reduceForward);
        }

        chunk = ringIx + 0;
        chunkOffset = chunkOffsetFor(chunk);
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = elemCountForChunk(chunk, chunkOffset);
        Nvfp4ReduceSendOp<true> reduceStore(work, offset, nelem);
        prims.template process<1, 1>(reduceStore);

        for (int j = 1; j < nranks - 1; ++j) {
          chunk = modRanks(ringIx + nranks - j);
          chunkOffset = chunkOffsetFor(chunk);
          offset = gridOffset + elemOffset + chunkOffset;
          nelem = elemCountForChunk(chunk, chunkOffset);
          Nvfp4RecvCopyOp<true> copyForward(work, offset, nelem);
          prims.template process<1, 1>(copyForward);
        }

        chunk = modRanks(ringIx + 1);
        chunkOffset = chunkOffsetFor(chunk);
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = elemCountForChunk(chunk, chunkOffset);
        Nvfp4RecvCopyOp<false> copyRecv(work, offset, nelem);
        prims.template process<1, 0>(copyRecv);
      }
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeUpDown(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }

    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, &tree->up, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSendFromOutput(offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

    if (tree->up == -1) {
      // Reduce and broadcast. Max number of recv is 2, max number of send is 2
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreads, tree->down, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*doPost=*/true);
      }
    }
    else if (tid < nthreadsSplit) {
      /* Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local).
       * Why Direct=1????
       * Answer: Because despite not performing any direct operations, the ctor
       * must assume Direct so that it can exchange direct pointers with remote ctors
       * that are Direct, otherwise it hangs. A cleaner solution would be to seperate
       * into DirectRecv and DirectSend capabilities, this ctor would have both=0,
       * but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.
       */
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<n, 1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreadsSplit, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }
    else {
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<1, n>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, work->sendbuff, work->recvbuff,
            work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    if (work->transportCodec == ncclTransportCodecNvfp4) {
      runRingNvfp4<Proto>(tid, nthreads, work);
    } else {
      runRing<T, RedOp, Proto>(tid, nthreads, work);
    }
  }
};

template<>
struct RunWorkColl<ncclFuncAllReduce, uint8_t, FuncSumPostDiv<uint8_t>, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<uint8_t, FuncSumPostDiv<uint8_t>, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    #if CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800
      runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #else
      runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #endif
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    static constexpr int COLLNET_COPY_THREADS = 96;
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    const ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t size = work->collnet.count;
    const ssize_t loopSize = nChannels*direct->nHeads*chunkSize;

    const int hasUp = (direct->up[0] >= 0) ? 1 : 0;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    const int nThreadsScatter = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 3*COLLNET_COPY_THREADS : 0);
    const int nThreadsGather  =             ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2*COLLNET_COPY_THREADS : 0);
    const int nThreadsBcast   = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 0 : 2*COLLNET_COPY_THREADS);
    const int nThreadsReduce = work->nWarps*WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
    const int tidStartBcast = nThreadsGather;
    const int tidStartScatter = tidStartBcast + nThreadsBcast;
    const int tidStartReduce = tidStartScatter + nThreadsScatter;
    using Proto = ProtoSimple<1, 1>;

    if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
      // Scatter
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid-tidStartScatter, nThreadsScatter, NULL, direct->up, work->sendbuff, work->recvbuff,
           work->redOpArg, 2*Proto::MaxGroupWidth, 1, 1, work);
      ssize_t offsetBase, peerOffset;
      ssize_t maxNelems;
      if (work->netRegUsed) {
        offsetBase = bid * chunkSize;
        maxNelems = size;  // never be the min
        peerOffset = nChannels * chunkSize;
      } else {
        offsetBase = bid * direct->nHeads * chunkSize;
        maxNelems = direct->nHeads * chunkSize;
        peerOffset = chunkSize;
      }
      // For collnet UB case, we need to organize buffers differently for contiguous buffer access
      // across channels. This access pattern should be consistent with code in coll_net.cc
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + offsetBase;
        ssize_t nelem = min(maxNelems, size - offset);
        prims.scatter(offset, nelem, chunkSize, peerOffset, direct->headRank, direct->shift);
      }
      // Coverity complains about a possible overrun inside the destructor of "prims", but that's actually
      // a false positive.
      // coverity[overrun-call:FALSE]
    } else if (tid >= tidStartReduce && direct->out != -1) {
      if (hasDn) {
        // Reduce, send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, direct->down, &direct->out, work->sendbuff, work->recvbuff,
             work->redOpArg, 3*Proto::MaxGroupWidth, 1, 1, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize
                                    : gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.recvReduceDirectSend(offset, offset, nelem);
        }
      } else {
        // Directly send to network
        if (work->netRegUsed) {
          if (tid == tidStartReduce) {
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, 1);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, nullptr, &direct->out, work->sendbuff, work->recvbuff,
             work->redOpArg, 3*Proto::MaxGroupWidth, 1, 1);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.send(offset, nelem);
          }
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // Gather
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsGather, direct->up, NULL, work->sendbuff, work->recvbuff,
           work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
      ssize_t offsetBase, peerOffset;
      ssize_t maxNelems;
      if (work->netRegUsed) {
        offsetBase = bid * chunkSize;
        maxNelems = size;  // never be the min
        peerOffset = nChannels * chunkSize;
      } else {
        offsetBase = bid * direct->nHeads * chunkSize;
        maxNelems = direct->nHeads * chunkSize;
        peerOffset = chunkSize;
      }
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + offsetBase;
        ssize_t nelem = min(maxNelems, size - offset);
        prims.directGather(offset, nelem, chunkSize, peerOffset, direct->headRank, direct->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
      if (hasDn) {
        // Recv from network, broadcast
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &direct->out, direct->down, work->sendbuff, work->recvbuff,
             work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize
                                            : gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
        }
      } else {
        if (work->netRegUsed) {
          if (tid == tidStartBcast) {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, 1);
          }
          __syncwarp();
        } else {
          // Recv from network (no post thread needed)
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(tid - tidStartBcast, nThreadsBcast, &direct->out, nullptr, work->sendbuff, work->recvbuff,
              work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.recv(offset, nelem, /*postOp=*/true);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const bool hasOut = nvls->out != -1;
    const int nranks = ncclShmem.comm.nRanks;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasOut ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 2) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasOut ? 3 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (work->oneNode) {
      ssize_t gridOffset, channelCount, chunkSize;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkSize);
      const ssize_t loopCount = nvls->nHeads * chunkSize;
      int remCount = channelCount%(nvls->nHeads*chunkSize);
      int lastChunkSize = alignUp(divUp(remCount, nvls->nHeads), 16384/sizeof(T));

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          ssize_t offset = gridOffset + elemOffset;
          int nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          ssize_t offset = gridOffset + elemOffset;
          int nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        // Reduce, broadcast through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->down, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset, offset;
          int nelem;
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          chunkOffset = elemOffset + nvls->headRank * chunkSize;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkSize, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else {
      const int bid = ncclShmem.channelId - work->channelLo;
      const int nChannels = work->channelHi - work->channelLo + 1;
      const ssize_t chunkSize = work->collnet.chunkCount;
      const ssize_t loopSize = nChannels * nvls->nHeads * chunkSize;
      const ssize_t size = work->collnet.count;

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
        // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        // Reduce, send to network
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
        prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->out, NULL, work->recvbuff,
          work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 1, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->regUsed && work->netRegUsed ? gridOffset + (nvls->headRank * nChannels + bid) * chunkSize
                                                             : gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else if (tid < tidEndBcast && nvls->headRank != -1) {
        // Recv from network, broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndReduce, nThreadsBcast, &nvls->out, &nvls->down, NULL, work->recvbuff,
            work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->regUsed && work->netRegUsed ? gridOffset + (nvls->headRank * nChannels + bid) * chunkSize
                                                             : gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const int treeUp = nvls->treeUp;
    const int* treeDown = nvls->treeDown;
    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nvls->nHeads * chunkCount;
    const int nranks = ncclShmem.comm.nRanks;
    const bool hasUp = treeUp != -1;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasUp ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 4) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasUp ? 5 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;
    ssize_t offset;
    int nelem;
    int remCount = channelCount%(nvls->nHeads*chunkCount);
    int lastChunkCount = alignUp(divUp(remCount, nvls->nHeads), 16/sizeof(T));

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (tid < tidEndScatter) {
      // Scatter
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
          work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.scatter(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndGather) {
      // Gather
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
          work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.gather(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndReduce && nvls->headRank != -1) {
      if (!hasUp) {
        // Reduce and Broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<3>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, treeDown, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else {
        // Reduce, send to network
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        // Coverity reports that the callee treats &treeUp as an array.  However, due to the use of
        // FanAsymmetric<3, 1>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanAsymmetric<3, 1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, &treeUp, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else if (tid < tidEndBcast && nvls->headRank != -1) {
      // Recv from network, broadcast
      using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
      // Coverity reports that the callee treats &treeUp as an array.  However, due to the use of
      // FanAsymmetric<1, 3>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<1, 3>, /*Direct=*/1, Proto, 0>
        prims(tid - tidEndReduce, nThreadsBcast, &treeUp, treeDown, NULL, NULL,
          work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        ssize_t chunkOffset;
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        chunkOffset = elemOffset + nvls->headRank * chunkCount;
        offset = gridOffset + chunkOffset;
        nelem = min(chunkCount, channelCount - chunkOffset);
        prims.directRecvDirectSend(offset, offset, nelem);
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    ncclTree *tree = &ncclShmem.channel.collnetChain;
    ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t loopSize = int(nChannels*chunkSize);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t size = work->collnet.count;

    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;

    int group, connIndex, send, recv, groupTid, groupNthreads;
    using Proto = ProtoSimple<1, 1>;
    if (tid < nthreadsSplit) {
      // Reduce up the chain
      group = 0;
      connIndex = 1;
      recv = tree->down[0];
      send = tree->up;
      groupTid = tid;
      groupNthreads = nthreadsSplit;
    } else {
      // Broadcast down the chain
      group = 1;
      connIndex = 0;
      recv = tree->up;
      send = tree->down[0];
      groupTid = tid - nthreadsSplit;
      groupNthreads = nthreads-nthreadsSplit;
    }

    if (tid < nthreadsSplit) {
      if (recv == -1) {
        if (work->netRegUsed) {
          if (groupTid == 0) {
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::sendPeerNotify(send, connIndex, 1);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
              work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            // coverity[overrun-call] => Coverity think prims.index can be greater than 1
            prims.directSend(offset, offset, nelem);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
        }
      } else {
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
            work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * int(chunkSize);
          int nelem = min(chunkSize, size - offset);
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
        // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      }
    }
    else {
      if (recv == nranks) {
        // I'm the first in the broadcast chain, I need to perform the division (postOp)
        if (send == -1) {
          if (work->netRegUsed) {
            if (groupTid == 0) {
              Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::recvPeerNotify(recv, connIndex, 1);
            }
            __syncwarp();
          } else {
            // Coverity reports that the callee treats &send as an array.  However, due to the use of
            // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
            // coverity[callee_ptr_arith:FALSE]
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
              prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
                work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
            for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
              ssize_t offset = gridOffset + bid * int(chunkSize);
              int nelem = min(chunkSize, size - offset);
              prims.directRecv(offset, nelem, /*postOp*/true);
            }
          }
        } else {
          // Coverity reports that the callee treats &send as an array.  However, due to the use of
          // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
          // coverity[callee_ptr_arith:FALSE]
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
              work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            prims.directRecvCopyDirectSend(offset, offset, nelem, /*postOp*/true);
          }
        }
      } else {
        // Coverity reports that the callee treats &send as an array.  However, due to the use of
        // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
            work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        if (send == -1) {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.directRecv(offset, nelem);
          }
        } else {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.directRecvCopyDirectSend(offset, offset, nelem);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
