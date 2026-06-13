/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "checks.h"
#include "collectives.h"
#include "enqueue.h"
#include "group.h"
#include "nccl.h"
#include "nvfp4.h"
#include "nvtx_payload_schemas.h"

#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <vector>

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncAlltoAll: return "AlltoAll";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncGather: return "Gather";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncScatter: return "Scatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  case ncclFuncPutSignal: return "PutSignal";
  case ncclFuncSignal: return "Signal";
  case ncclFuncWaitSignal: return "WaitSignal";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclBfloat16: return "ncclBfloat16";
  case ncclFloat8e4m3: return "ncclFloat8e4m3";
  case ncclFloat8e5m2: return "ncclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

namespace {

bool ncclNvfp4AllReduceOpSupported(ncclRedOp_t op) {
  switch (op) {
  case ncclSum:
  case ncclProd:
  case ncclMin:
  case ncclMax:
  case ncclAvg:
    return true;
  default:
    return false;
  }
}

bool ncclNvfp4MulOverflow(size_t a, size_t b, size_t* result) {
  if (a != 0 && b > SIZE_MAX / a) return true;
  *result = a * b;
  return false;
}


ncclResult_t ncclNvfp4ValidateCommon(const char* opname, const void* sendbuff, void* recvbuff,
                                     size_t count, ncclDataType_t reservedDatatype, ncclComm_t comm) {
  (void)reservedDatatype;
  NCCLCHECK(CommCheck(comm, opname, "comm"));

  // These wrappers launch CUDA kernels immediately around inner NCCL collectives,
  // so they must not be nested inside an outer NCCL group until the staging work
  // is promoted into the planner.
  if (ncclGroupEnabled()) {
    WARN("%s does not support ncclGroupStart/ncclGroupEnd yet", opname);
    return ncclInvalidUsage;
  }

  if (count != 0) {
    if (sendbuff == nullptr || recvbuff == nullptr) {
      WARN("%s requires non-null sendbuff/recvbuff when count is non-zero", opname);
      return ncclInvalidArgument;
    }
    NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", opname));
    NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", opname));
  }

  NCCLCHECK(ncclCommEnsureReady(comm));
  return ncclSuccess;
}

static bool ncclNvfp4EnvAllows(const char* envValue, const char* token) {
  if (envValue == nullptr) return true;

  bool sawInclude = false;
  bool tokenIncluded = false;
  const char* p = envValue;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t' || *p == ',') ++p;
    if (*p == '\0') break;

    bool exclude = false;
    if (*p == '^') {
      exclude = true;
      ++p;
    }

    const char* start = p;
    while (*p != '\0' && *p != ',') ++p;
    const char* end = p;
    while (start < end && (end[-1] == ' ' || end[-1] == '\t')) --end;
    if (start == end) continue;

    size_t len = (size_t)(end - start);
    if (!exclude) sawInclude = true;
    if (strlen(token) == len && strncasecmp(start, token, len) == 0) {
      if (exclude) return false;
      tokenIncluded = true;
    }
  }

  return sawInclude ? tokenIncluded : true;
}

ncclResult_t ncclNvfp4CheckAllReduceSelection(const char* opname) {
  const char* algo = getenv("NCCL_ALGO");
  if (!ncclNvfp4EnvAllows(algo, "RING")) {
    WARN("%s requires NCCL_ALGO to allow RING; got '%s'", opname, algo);
    return ncclInvalidUsage;
  }

  const char* proto = getenv("NCCL_PROTO");
  if (!ncclNvfp4EnvAllows(proto, "SIMPLE")) {
    WARN("%s requires NCCL_PROTO to allow SIMPLE; got '%s'", opname, proto);
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

ncclResult_t ncclNvfp4Alloc(cudaStream_t stream, cudaMemPool_t memPool, size_t bytes, void** ptr) {
  *ptr = nullptr;
  if (bytes == 0) return ncclSuccess;
  CUDACHECK(cudaMallocFromPoolAsync(ptr, bytes, memPool, stream));
  return ncclSuccess;
}

ncclResult_t ncclNvfp4CopyPackedIfNeeded(const void* sendbuff, void* recvbuff, size_t bytes,
                                         cudaStream_t stream) {
  if (sendbuff == recvbuff || bytes == 0) return ncclSuccess;
  CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));
  return ncclSuccess;
}

struct ncclNvfp4ChunkPlan {
  std::vector<size_t> eltOffsets;
  std::vector<size_t> eltCounts;
  std::vector<size_t> packedByteOffsets;
  std::vector<size_t> logicalPackedBytes;
  std::vector<size_t> sectionBytes;
  size_t maxSectionBytes;
  size_t totalLogicalPackedBytes;
  size_t totalSectionBytes;
};

int ncclNvfp4ModRank(int value, int nranks) {
  value %= nranks;
  return value < 0 ? value + nranks : value;
}

ncclResult_t ncclNvfp4BuildChunkPlan(size_t count, int nranks, struct ncclNvfp4ChunkPlan* plan) {
  plan->eltOffsets.resize((size_t)nranks);
  plan->eltCounts.resize((size_t)nranks);
  plan->packedByteOffsets.resize((size_t)nranks);
  plan->logicalPackedBytes.resize((size_t)nranks);
  plan->sectionBytes.resize((size_t)nranks);
  plan->maxSectionBytes = 0;
  plan->totalLogicalPackedBytes = ncclNvfp4LogicalBytes(count);
  if (!ncclNvfp4SectionBytesChecked(count, &plan->totalSectionBytes)) {
    WARN("NVFP4 section size overflow for count %zu", count);
    return ncclInvalidArgument;
  }

  size_t totalBlocks = ncclNvfp4NumBlocks(count);
  size_t baseBlocks = nranks == 0 ? 0 : totalBlocks / (size_t)nranks;
  size_t remainderBlocks = nranks == 0 ? 0 : totalBlocks % (size_t)nranks;
  size_t blockOffset = 0;
  for (int chunk = 0; chunk < nranks; ++chunk) {
    size_t blockCount = baseBlocks + ((size_t)chunk < remainderBlocks ? 1 : 0);
    size_t eltOffset = blockOffset * NCCL_NVFP4_BLOCK_ELTS;
    size_t eltCount = 0;
    if (eltOffset < count) {
      eltCount = count - eltOffset;
      size_t chunkCapacity = blockCount * NCCL_NVFP4_BLOCK_ELTS;
      if (eltCount > chunkCapacity) eltCount = chunkCapacity;
    }

    size_t chunkSectionBytes = 0;
    if (!ncclNvfp4SectionBytesChecked(eltCount, &chunkSectionBytes)) {
      WARN("NVFP4 chunk size overflow for count %zu and chunk %d", count, chunk);
      return ncclInvalidArgument;
    }

    plan->eltOffsets[(size_t)chunk] = eltOffset;
    plan->eltCounts[(size_t)chunk] = eltCount;
    plan->packedByteOffsets[(size_t)chunk] = blockOffset * NCCL_NVFP4_BLOCK_BYTES;
    plan->logicalPackedBytes[(size_t)chunk] = blockCount * NCCL_NVFP4_BLOCK_BYTES;
    plan->sectionBytes[(size_t)chunk] = chunkSectionBytes;
    if (chunkSectionBytes > plan->maxSectionBytes) plan->maxSectionBytes = chunkSectionBytes;
    blockOffset += blockCount;
  }

  return ncclSuccess;
}

const void* ncclNvfp4OffsetConst(const void* base, size_t byteOffset) {
  return (const void*)(static_cast<const char*>(base) + byteOffset);
}

void* ncclNvfp4OffsetMut(void* base, size_t byteOffset) {
  return (void*)(static_cast<char*>(base) + byteOffset);
}

ncclResult_t ncclNvfp4ExchangePacked(const uint8_t* sendPtr, size_t sendBytes,
                                     uint8_t* recvPtr, size_t recvBytes,
                                     int nextRank, int prevRank,
                                     ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t groupRet = ncclGroupStart();
  if (groupRet != ncclSuccess) return groupRet;
  ncclResult_t sendRet = ncclSend(sendPtr, sendBytes, ncclInt8, nextRank, comm, stream);
  ncclResult_t recvRet = ncclRecv(recvPtr, recvBytes, ncclInt8, prevRank, comm, stream);
  ncclResult_t endRet = ncclGroupEnd();
  if (sendRet != ncclSuccess) return sendRet;
  if (recvRet != ncclSuccess) return recvRet;
  return endRet;
}

}  // namespace

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));

  struct ncclInfo info = { ncclFuncAlltoAll, "AlltoAll",
    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceNvfp4, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t reservedDatatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceNvfp4(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t reservedDatatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  size_t logicalBytes = ncclNvfp4LogicalBytes(count);
  size_t sectionBytes = ncclNvfp4SectionBytes(count);
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, logicalBytes, op));

  if (!ncclNvfp4AllReduceOpSupported(op)) {
    WARN("%s only supports ncclSum, ncclProd, ncclMin, ncclMax, and ncclAvg", __func__);
    return ncclInvalidArgument;
  }

  NCCLCHECK(ncclNvfp4ValidateCommon(__func__, sendbuff, recvbuff, count, reservedDatatype, comm));
  NCCLCHECK(ncclNvfp4CheckAllReduceSelection(__func__));
  (void)reservedDatatype;
  if (count == 0) return ncclSuccess;
  if (logicalBytes == 0) {
    WARN("%s logical packed byte size overflow for count %zu", __func__, count);
    return ncclInvalidArgument;
  }

  const char* implEnv = getenv("NCCL_NVFP4_ALLREDUCE_IMPL");
  bool useNativePath = implEnv != nullptr &&
    (strcasecmp(implEnv, "native") == 0 || strcasecmp(implEnv, "device") == 0 || strcmp(implEnv, "1") == 0);
  if (useNativePath) {
    struct ncclInfo info = { ncclFuncAllReduce, "AllReduceNvfp4",
      sendbuff, recvbuff, logicalBytes, ncclUint8, op, 0, comm, stream, /* Args */
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    info.transportCodec = ncclTransportCodecNvfp4;
    info.nvfp4.logicalCount = count;
    info.nvfp4.logicalBytes = logicalBytes;
    info.nvfp4.sectionBytes = sectionBytes;
    ncclResult_t enqueueRet = ncclEnqueueCheck(&info);
    if (enqueueRet != ncclSuccess) return enqueueRet;
    if (sectionBytes > logicalBytes) {
      CUDACHECK(cudaMemsetAsync(static_cast<uint8_t*>(recvbuff) + logicalBytes, 0, sectionBytes - logicalBytes, stream));
    }
    return ncclSuccess;
  }

  ncclResult_t ret = ncclSuccess;
  int saveDev = -1;
  uint8_t* currentPacked = nullptr;
  uint8_t* recvPacked = nullptr;
  struct ncclNvfp4ChunkPlan plan;
  int rank = comm->rank;
  int nextRank = ncclNvfp4ModRank(rank + 1, comm->nRanks);
  int prevRank = ncclNvfp4ModRank(rank - 1, comm->nRanks);
  const uint8_t* sendBytesBase = static_cast<const uint8_t*>(sendbuff);
  uint8_t* recvBytesBase = static_cast<uint8_t*>(recvbuff);

  CUDACHECK(cudaGetDevice(&saveDev));
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  NCCLCHECKGOTO(ncclNvfp4BuildChunkPlan(count, comm->nRanks, &plan), ret, fail);

  if (comm->nRanks == 1) {
    NCCLCHECKGOTO(ncclNvfp4CopyPackedIfNeeded(sendbuff, recvbuff, sectionBytes, stream), ret, fail);
    goto exit;
  }

  NCCLCHECKGOTO(ncclNvfp4Alloc(stream, comm->memPool, plan.maxSectionBytes, (void**)&currentPacked), ret, fail);
  NCCLCHECKGOTO(ncclNvfp4Alloc(stream, comm->memPool, plan.maxSectionBytes, (void**)&recvPacked), ret, fail);

  for (int step = 0; step < comm->nRanks - 1; ++step) {
    int sendChunk = ncclNvfp4ModRank(rank - step - 1, comm->nRanks);
    int recvChunk = ncclNvfp4ModRank(rank - step - 2, comm->nRanks);
    size_t sendLogicalBytes = plan.logicalPackedBytes[(size_t)sendChunk];
    size_t recvLogicalBytes = plan.logicalPackedBytes[(size_t)recvChunk];
    const uint8_t* sendPtr = step == 0
      ? sendBytesBase + plan.packedByteOffsets[(size_t)sendChunk]
      : currentPacked;

    NCCLCHECKGOTO(
      ncclNvfp4ExchangePacked(
        sendPtr, sendLogicalBytes, recvPacked, recvLogicalBytes,
        nextRank, prevRank, comm, stream),
      ret, fail);

    if (recvLogicalBytes == 0) continue;

    bool finalReduceScatterStep = step == comm->nRanks - 2;
    uint8_t* reduceOutput = finalReduceScatterStep
      ? recvBytesBase + plan.packedByteOffsets[(size_t)recvChunk]
      : currentPacked;
    NCCLCHECKGOTO(
      ncclNvfp4ReduceAndPack(
        stream,
        recvPacked,
        sendBytesBase + plan.packedByteOffsets[(size_t)recvChunk],
        reduceOutput,
        plan.eltCounts[(size_t)recvChunk],
        op, comm->nRanks,
        finalReduceScatterStep),
      ret, fail);
  }

  for (int step = 0; step < comm->nRanks - 1; ++step) {
    int sendChunk = ncclNvfp4ModRank(rank - step, comm->nRanks);
    int recvChunk = ncclNvfp4ModRank(rank - step - 1, comm->nRanks);
    size_t sendLogicalBytes = plan.logicalPackedBytes[(size_t)sendChunk];
    size_t recvLogicalBytes = plan.logicalPackedBytes[(size_t)recvChunk];
    NCCLCHECKGOTO(
      ncclNvfp4ExchangePacked(
        recvBytesBase + plan.packedByteOffsets[(size_t)sendChunk], sendLogicalBytes,
        recvBytesBase + plan.packedByteOffsets[(size_t)recvChunk], recvLogicalBytes,
        nextRank, prevRank, comm, stream),
      ret, fail);
  }

  if (sectionBytes > logicalBytes) {
    CUDACHECKGOTO(
      cudaMemsetAsync(recvBytesBase + logicalBytes, 0, sectionBytes - logicalBytes, stream),
      ret, fail);
  }

exit:
  if (recvPacked != nullptr) CUDACHECKIGNORE(cudaFreeAsync(recvPacked, stream));
  if (currentPacked != nullptr) CUDACHECKIGNORE(cudaFreeAsync(currentPacked, stream));
  if (saveDev != -1) CUDACHECKIGNORE(cudaSetDevice(saveDev));
  return ret;

fail:
  goto exit;
}
NCCL_API(ncclResult_t, ncclAlltoAllNvfp4, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t reservedDatatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAlltoAllNvfp4(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t reservedDatatype, ncclComm* comm, cudaStream_t stream) {
  size_t sectionBytes = ncclNvfp4SectionBytes(count);
  NVTX3_FUNC_WITH_PARAMS(AlltoAll, NcclNvtxParamsAlltoAll,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sectionBytes));

  (void)reservedDatatype;
  NCCLCHECK(ncclNvfp4ValidateCommon(__func__, sendbuff, recvbuff, count, reservedDatatype, comm));
  if (count == 0) return ncclSuccess;

  size_t totalBytes = 0;
  if (ncclNvfp4MulOverflow(sectionBytes, (size_t)comm->nRanks, &totalBytes)) {
    WARN("%s staging size overflow for count %zu and nranks %d", __func__, count, comm->nRanks);
    return ncclInvalidArgument;
  }

  ncclResult_t ret = ncclSuccess;
  int saveDev = -1;
  uint8_t* packedTmp = nullptr;

  CUDACHECK(cudaGetDevice(&saveDev));
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  if (comm->nRanks == 1) {
    NCCLCHECKGOTO(ncclNvfp4CopyPackedIfNeeded(sendbuff, recvbuff, totalBytes, stream), ret, fail);
    goto exit;
  }

  if (sendbuff == recvbuff) {
    NCCLCHECKGOTO(ncclNvfp4Alloc(stream, comm->memPool, totalBytes, (void**)&packedTmp), ret, fail);
    NCCLCHECKGOTO(ncclNvfp4CopyPackedIfNeeded(sendbuff, packedTmp, totalBytes, stream), ret, fail);
    NCCLCHECKGOTO(ncclAlltoAll(packedTmp, recvbuff, sectionBytes, ncclUint8, comm, stream), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclAlltoAll(sendbuff, recvbuff, sectionBytes, ncclInt8, comm, stream), ret, fail);
  }

exit:
  if (packedTmp != nullptr) CUDACHECKIGNORE(cudaFreeAsync(packedTmp, stream));
  if (saveDev != -1) CUDACHECKIGNORE(cudaSetDevice(saveDev));
  return ret;

fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Gather, NcclNvtxParamsGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncGather, "Gather",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    GATHER_CHUNKSTEPS, GATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Scatter, NcclNvtxParamsScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncScatter, "Scatter",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclPutSignal, const void* localbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclPutSignal(const void* localbuff, size_t count, ncclDataType_t datatype,
  int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) {
NVTX3_FUNC_WITH_PARAMS(PutSignal, NcclNvtxParamsPut,
  NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, ctx));

struct ncclInfo info = { ncclFuncPutSignal, "PutSignal",
  localbuff, NULL, count, datatype, ncclSum, peer, comm, stream, /* Args */
  1, 1, ncclTransportCodecPlain, {}, /* chunkSteps, sliceSteps, transport */
  peerWinOffset, peerWin, sigIdx, ctx, flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
  0, NULL }; /* nDesc, signalDescs */
return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSignal, int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm, cudaStream_t stream) {
NVTX3_FUNC_WITH_PARAMS(Signal, NcclNvtxParamsSignal,
  NVTX3_PAYLOAD(comm ? comm->commHash : 0, peer, ctx));

struct ncclInfo info = { ncclFuncSignal, "Signal",
  NULL, NULL, 0, ncclInt8, ncclSum, peer, comm, stream, /* Args */
  1, 1, ncclTransportCodecPlain, {}, /* chunkSteps, sliceSteps, transport */
  0, NULL, sigIdx, ctx, flags, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
  0, NULL }; /* nDesc, signalDescs */
return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclWaitSignal, int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs, ncclComm_t comm, cudaStream_t stream) {
NVTX3_FUNC_WITH_PARAMS(WaitSignal, NcclNvtxParamsWaitSignal,
  NVTX3_PAYLOAD(comm ? comm->commHash : 0, nDesc, 0));

struct ncclInfo info = { ncclFuncWaitSignal, "WaitSignal",
  NULL, NULL, 0, ncclInt32, ncclSum, 0, comm, stream, /* Args */
  1, 1, ncclTransportCodecPlain, {}, /* chunkSteps, sliceSteps, transport */
  0, NULL, 0, 0, 0, /* peerWinOffset, peerWin, sigIdx, ctx, flags */
  nDesc, signalDescs }; /* nDesc, signalDescs */
return ncclEnqueueCheck(&info);
}
