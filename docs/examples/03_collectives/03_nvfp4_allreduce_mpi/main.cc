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
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

#define MPI_CHECK(cmd)                                                         \
  do {                                                                         \
    int err = (cmd);                                                           \
    if (err != MPI_SUCCESS) {                                                  \
      char errStr[MPI_MAX_ERROR_STRING];                                       \
      int len = 0;                                                             \
      MPI_Error_string(err, errStr, &len);                                     \
      fprintf(stderr, "MPI failure %s:%d '%.*s'\n", __FILE__, __LINE__, len, \
              errStr);                                                         \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = (cmd);                                                  \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "NCCL failure %s:%d '%s'\n", __FILE__, __LINE__,      \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                 \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = (cmd);                                                   \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA failure %s:%d '%s'\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                 \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

namespace {

constexpr size_t kNvfp4BlockElts = 16;
constexpr size_t kNvfp4BlockBytes = 10;
constexpr size_t kNvfp4Alignment = 16;
constexpr size_t kDefaultCorrectnessBytes[] = {
    4ULL << 20,
    (4ULL << 20) + sizeof(__half),
};
constexpr size_t kDefaultScalingBytes[] = {
    64ULL << 20,
    128ULL << 20,
    256ULL << 20,
    512ULL << 20,
    1ULL << 30,
    2ULL << 30,
    4ULL << 30,
    8ULL << 30,
};
constexpr const char* kNvfp4ImplName = "nvfp4";
constexpr const char* kFp8ImplName = "fp8e4m3";

struct Options {
  enum Mode { kCorrectness, kScaling, kBoth } mode = kBoth;
  enum DType { kFp16, kBf16 };

  std::vector<DType> dtypes = {kFp16};
  std::vector<size_t> correctnessBytes = std::vector<size_t>(
      kDefaultCorrectnessBytes,
      kDefaultCorrectnessBytes + sizeof(kDefaultCorrectnessBytes) / sizeof(kDefaultCorrectnessBytes[0]));
  std::vector<size_t> scalingBytes = std::vector<size_t>(
      kDefaultScalingBytes,
      kDefaultScalingBytes + sizeof(kDefaultScalingBytes) / sizeof(kDefaultScalingBytes[0]));
  int warmupIters = 3;
  int benchIters = 10;
  bool csv = false;
  double maxDiffThreshold = -1.0;
};

int getLocalRank(MPI_Comm comm) {
  int worldRank = 0;
  MPI_CHECK(MPI_Comm_rank(comm, &worldRank));

  MPI_Comm nodeComm;
  MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));

  int localRank = 0;
  MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
  MPI_CHECK(MPI_Comm_free(&nodeComm));
  return localRank;
}

std::string toLower(std::string value) {
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] >= 'A' && value[i] <= 'Z') value[i] = static_cast<char>(value[i] - 'A' + 'a');
  }
  return value;
}

bool endsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

size_t parseByteValue(const std::string& text) {
  std::string value = toLower(text);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
  size_t start = 0;
  while (start < value.size() && (value[start] == ' ' || value[start] == '\t')) ++start;
  value = value.substr(start);
  if (value.empty()) {
    fprintf(stderr, "empty byte value\n");
    exit(EXIT_FAILURE);
  }

  unsigned long long multiplier = 1;
  if (endsWith(value, "gib")) {
    multiplier = 1ULL << 30;
    value.resize(value.size() - 3);
  } else if (endsWith(value, "mib")) {
    multiplier = 1ULL << 20;
    value.resize(value.size() - 3);
  } else if (endsWith(value, "kib")) {
    multiplier = 1ULL << 10;
    value.resize(value.size() - 3);
  } else if (endsWith(value, "gb")) {
    multiplier = 1ULL << 30;
    value.resize(value.size() - 2);
  } else if (endsWith(value, "mb")) {
    multiplier = 1ULL << 20;
    value.resize(value.size() - 2);
  } else if (endsWith(value, "kb")) {
    multiplier = 1ULL << 10;
    value.resize(value.size() - 2);
  } else if (endsWith(value, "g")) {
    multiplier = 1ULL << 30;
    value.resize(value.size() - 1);
  } else if (endsWith(value, "m")) {
    multiplier = 1ULL << 20;
    value.resize(value.size() - 1);
  } else if (endsWith(value, "k")) {
    multiplier = 1ULL << 10;
    value.resize(value.size() - 1);
  } else if (endsWith(value, "b")) {
    value.resize(value.size() - 1);
  }

  char* end = nullptr;
  double number = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0' || number <= 0.0) {
    fprintf(stderr, "invalid byte value '%s'\n", text.c_str());
    exit(EXIT_FAILURE);
  }

  long double bytes = static_cast<long double>(number) * static_cast<long double>(multiplier);
  if (bytes <= 0.0L || bytes > static_cast<long double>(~0ULL)) {
    fprintf(stderr, "byte value overflow for '%s'\n", text.c_str());
    exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(bytes);
}

std::vector<size_t> parseByteList(const std::string& text) {
  std::vector<size_t> bytes;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) bytes.push_back(parseByteValue(item));
  }
  if (bytes.empty()) {
    fprintf(stderr, "byte list must not be empty\n");
    exit(EXIT_FAILURE);
  }
  return bytes;
}

const char* modeName(Options::Mode mode) {
  switch (mode) {
    case Options::kCorrectness: return "correctness";
    case Options::kScaling: return "scaling";
    case Options::kBoth: return "both";
  }
  return "unknown";
}

const char* dtypeName(Options::DType dtype) {
  switch (dtype) {
    case Options::kFp16: return "fp16";
    case Options::kBf16: return "bf16";
  }
  return "unknown";
}

ncclDataType_t ncclType(Options::DType dtype) {
  return dtype == Options::kFp16 ? ncclFloat16 : ncclBfloat16;
}

size_t dtypeSize(Options::DType dtype) {
  return dtype == Options::kFp16 ? sizeof(__half) : sizeof(__nv_bfloat16);
}

std::string formatBytes(size_t bytes) {
  char buffer[64];
  if (bytes >= (1ULL << 30)) {
    std::snprintf(buffer, sizeof(buffer), "%.2f GiB", static_cast<double>(bytes) / static_cast<double>(1ULL << 30));
  } else if (bytes >= (1ULL << 20)) {
    std::snprintf(buffer, sizeof(buffer), "%.2f MiB", static_cast<double>(bytes) / static_cast<double>(1ULL << 20));
  } else if (bytes >= (1ULL << 10)) {
    std::snprintf(buffer, sizeof(buffer), "%.2f KiB", static_cast<double>(bytes) / static_cast<double>(1ULL << 10));
  } else {
    std::snprintf(buffer, sizeof(buffer), "%zu B", bytes);
  }
  return std::string(buffer);
}

bool fp8AllReduceSupported() {
#if CUDART_VERSION >= 11080
  return true;
#else
  return false;
#endif
}

size_t nvfp4SectionBytes(size_t count) {
  size_t blocks = count / kNvfp4BlockElts + ((count % kNvfp4BlockElts) != 0 ? 1 : 0);
  size_t logicalBytes = blocks * kNvfp4BlockBytes;
  return (logicalBytes + kNvfp4Alignment - 1) & ~(kNvfp4Alignment - 1);
}

bool estimateNvfp4AllReduceMemoryFromCount(size_t count, size_t* scratchBytesOut,
                                           size_t* totalBytesOut) {
  size_t sectionBytes = nvfp4SectionBytes(count);
  if (sectionBytes > SIZE_MAX / 4) return false;
  *scratchBytesOut = sectionBytes * 2;
  *totalBytesOut = sectionBytes * 4;
  return true;
}

bool estimateFp8AllReduceMemoryFromCount(size_t count, size_t* totalBytesOut) {
  if (count > SIZE_MAX / 2) return false;
  *totalBytesOut = count * 2;
  return true;
}

void printUsage(const char* prog) {
  std::fprintf(stderr,
      "Usage: %s [options]\n"
      "  --mode correctness|scaling|both\n"
      "  --dtype fp16|bf16|both\n"
      "  --correctness-bytes list   Comma-separated logical byte sizes (default: 4MiB and tail-block variant)\n"
      "  --scaling-bytes list       Comma-separated logical byte sizes used to derive one shared count for NVFP4 and FP8 (default: 64MB,128MB,256MB,512MB,1GB,2GB,4GB,8GB)\n"
      "  --warmup N                 Warmup iterations for scaling (default: 3)\n"
      "  --iters N                  Timed iterations for scaling (default: 10)\n"
      "  --max-diff-threshold X     Optional correctness failure threshold\n"
      "  --csv                      Emit CSV rows for scaling results\n",
      prog);
}

Options parseOptions(int argc, char* argv[]) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto requireValue = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--mode") {
      std::string mode = toLower(requireValue("--mode"));
      if (mode == "correctness") opts.mode = Options::kCorrectness;
      else if (mode == "scaling") opts.mode = Options::kScaling;
      else if (mode == "both") opts.mode = Options::kBoth;
      else {
        std::fprintf(stderr, "invalid mode '%s'\n", mode.c_str());
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--dtype") {
      std::string dtype = toLower(requireValue("--dtype"));
      opts.dtypes.clear();
      if (dtype == "fp16") {
        opts.dtypes.push_back(Options::kFp16);
      } else if (dtype == "bf16") {
#if CUDART_VERSION >= 11000
        opts.dtypes.push_back(Options::kBf16);
#else
        std::fprintf(stderr, "bf16 requires CUDA 11 or newer\n");
        std::exit(EXIT_FAILURE);
#endif
      } else if (dtype == "both") {
        opts.dtypes.push_back(Options::kFp16);
#if CUDART_VERSION >= 11000
        opts.dtypes.push_back(Options::kBf16);
#else
        std::fprintf(stderr, "bf16 requires CUDA 11 or newer\n");
        std::exit(EXIT_FAILURE);
#endif
      } else {
        std::fprintf(stderr, "invalid dtype '%s'\n", dtype.c_str());
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--correctness-bytes") {
      opts.correctnessBytes = parseByteList(requireValue("--correctness-bytes"));
    } else if (arg == "--scaling-bytes") {
      opts.scalingBytes = parseByteList(requireValue("--scaling-bytes"));
    } else if (arg == "--warmup") {
      opts.warmupIters = std::atoi(requireValue("--warmup"));
    } else if (arg == "--iters") {
      opts.benchIters = std::atoi(requireValue("--iters"));
    } else if (arg == "--max-diff-threshold") {
      opts.maxDiffThreshold = std::atof(requireValue("--max-diff-threshold"));
    } else if (arg == "--csv") {
      opts.csv = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
      printUsage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (opts.warmupIters < 0 || opts.benchIters <= 0) {
    std::fprintf(stderr, "warmup must be >= 0 and iters must be > 0\n");
    std::exit(EXIT_FAILURE);
  }
  return opts;
}

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

float inputPattern(int worldRank, size_t idx) {
  static const float pattern[16] = {
      2.0f, -2.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f, 1.0f,
      -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 1.0f, 0.0f, -1.0f};
  size_t lane = idx % 16;
  float value = pattern[lane];
  if (((worldRank + static_cast<int>(idx / 16)) & 1) != 0) value = -value;
  return value + 0.125f * static_cast<float>(worldRank);
}

template <typename T>
void fillInput(std::vector<T>& host, int worldRank) {
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = fromFloat<T>(inputPattern(worldRank, i));
  }
}

float nvfp4Magnitude(int code) {
  static const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return lut[code & 0x7];
}

float nvfp4Decode(uint8_t nibble) {
  float magnitude = nvfp4Magnitude(nibble & 0x7);
  return (nibble & 0x8) ? -magnitude : magnitude;
}

uint8_t nvfp4Encode(float value) {
  float clamped = std::fmax(-6.0f, std::fmin(6.0f, value));
  float absValue = std::fabs(clamped);
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

template <typename T>
void packNvfp4Section(const std::vector<T>& logical, std::vector<uint8_t>* packed) {
  packed->assign(nvfp4SectionBytes(logical.size()), 0);
  size_t numBlocks = logical.size() / kNvfp4BlockElts + ((logical.size() % kNvfp4BlockElts) != 0 ? 1 : 0);
  for (size_t block = 0; block < numBlocks; ++block) {
    uint8_t* blockPtr = packed->data() + block * kNvfp4BlockBytes;
    size_t baseElt = block * kNvfp4BlockElts;
    size_t validCount = logical.size() > baseElt ? logical.size() - baseElt : 0;
    if (validCount > kNvfp4BlockElts) validCount = kNvfp4BlockElts;

    float maxAbs = 0.0f;
    for (size_t i = 0; i < validCount; ++i) {
      maxAbs = std::max(maxAbs, std::fabs(toFloat(logical[baseElt + i])));
    }

    if (maxAbs == 0.0f) {
      std::memset(blockPtr, 0, kNvfp4BlockBytes);
      continue;
    }

    __half scale = __float2half(maxAbs / 6.0f);
    std::memcpy(blockPtr, &scale, sizeof(scale));
    float invScale = 6.0f / maxAbs;

    for (int byteIdx = 0; byteIdx < 8; ++byteIdx) {
      uint8_t packedByte = 0;
      for (int lane = 0; lane < 2; ++lane) {
        int eltInBlock = byteIdx * 2 + lane;
        float value = 0.0f;
        if (static_cast<size_t>(eltInBlock) < validCount) {
          value = toFloat(logical[baseElt + static_cast<size_t>(eltInBlock)]) * invScale;
        }
        uint8_t nibble = nvfp4Encode(value);
        packedByte |= lane == 0 ? nibble : static_cast<uint8_t>(nibble << 4);
      }
      blockPtr[2 + byteIdx] = packedByte;
    }
  }
}

float unpackNvfp4SectionValue(const uint8_t* section, size_t eltIndex) {
  size_t block = eltIndex / kNvfp4BlockElts;
  size_t eltInBlock = eltIndex % kNvfp4BlockElts;
  const uint8_t* blockPtr = section + block * kNvfp4BlockBytes;
  __half scaleHalf;
  std::memcpy(&scaleHalf, blockPtr, sizeof(scaleHalf));
  float scale = __half2float(scaleHalf);
  uint8_t packedByte = blockPtr[2 + (eltInBlock / 2)];
  uint8_t nibble = (eltInBlock & 1) ? static_cast<uint8_t>(packedByte >> 4)
                                    : static_cast<uint8_t>(packedByte & 0xF);
  return nvfp4Decode(nibble) * scale;
}

struct CorrectnessStats {
  double maxDiff;
  double totalAbsError;
  unsigned long long packedMismatchBytes;
};

size_t nvfp4NumBlocks(size_t count) {
  return count / kNvfp4BlockElts + ((count % kNvfp4BlockElts) != 0 ? 1 : 0);
}

float nvfp4DefaultScale(float maxAbs) {
  return maxAbs == 0.0f ? 1.0f : (maxAbs / 6.0f);
}

bool nvfp4PayloadIsZero(const uint8_t* blockPtr) {
  uint8_t any = 0;
  for (int i = 0; i < 8; ++i) any |= blockPtr[2 + i];
  return any == 0;
}

float nvfp4ChooseReduceScaleL2(const float* values, int validCount, float fallbackScale) {
  (void)values;
  (void)validCount;
  float scale = __half2float(__float2half(fallbackScale));
  return (scale > 0.0f && std::isfinite(scale)) ? scale : 1.0f;
}

struct Nvfp4ChunkPlan {
  std::vector<size_t> eltCounts;
  std::vector<size_t> packedByteOffsets;
  std::vector<size_t> logicalPackedBytes;
  std::vector<size_t> sectionBytes;
};

Nvfp4ChunkPlan buildNvfp4ChunkPlan(size_t count, int nranks) {
  Nvfp4ChunkPlan plan;
  plan.eltCounts.resize(static_cast<size_t>(nranks));
  plan.packedByteOffsets.resize(static_cast<size_t>(nranks));
  plan.logicalPackedBytes.resize(static_cast<size_t>(nranks));
  plan.sectionBytes.resize(static_cast<size_t>(nranks));

  size_t totalBlocks = nvfp4NumBlocks(count);
  size_t baseBlocks = nranks == 0 ? 0 : totalBlocks / static_cast<size_t>(nranks);
  size_t remainderBlocks = nranks == 0 ? 0 : totalBlocks % static_cast<size_t>(nranks);
  size_t blockOffset = 0;
  for (int chunk = 0; chunk < nranks; ++chunk) {
    size_t blockCount = baseBlocks + (static_cast<size_t>(chunk) < remainderBlocks ? 1 : 0);
    size_t eltOffset = blockOffset * kNvfp4BlockElts;
    size_t eltCount = 0;
    if (eltOffset < count) {
      eltCount = count - eltOffset;
      size_t chunkCapacity = blockCount * kNvfp4BlockElts;
      if (eltCount > chunkCapacity) eltCount = chunkCapacity;
    }

    plan.eltCounts[static_cast<size_t>(chunk)] = eltCount;
    plan.packedByteOffsets[static_cast<size_t>(chunk)] = blockOffset * kNvfp4BlockBytes;
    plan.logicalPackedBytes[static_cast<size_t>(chunk)] = blockCount * kNvfp4BlockBytes;
    plan.sectionBytes[static_cast<size_t>(chunk)] = nvfp4SectionBytes(eltCount);
    blockOffset += blockCount;
  }

  return plan;
}

void nvfp4ReduceAndPackChunkCpu(const uint8_t* packedInput, const uint8_t* localPackedInput,
                                uint8_t* packedOutput, size_t sectionElts,
                                ncclRedOp_t op, int nRanks, bool finalize) {
  size_t blocksPerSection = nvfp4NumBlocks(sectionElts);
  size_t sectionBytes = nvfp4SectionBytes(sectionElts);
  std::fill(packedOutput, packedOutput + sectionBytes, 0);

  for (size_t blockInSection = 0; blockInSection < blocksPerSection; ++blockInSection) {
    const uint8_t* inputBlockPtr = packedInput + blockInSection * kNvfp4BlockBytes;
    const uint8_t* localBlockPtr = localPackedInput + blockInSection * kNvfp4BlockBytes;
    uint8_t* outputBlockPtr = packedOutput + blockInSection * kNvfp4BlockBytes;
    if (nvfp4PayloadIsZero(inputBlockPtr) && nvfp4PayloadIsZero(localBlockPtr)) {
      std::memset(outputBlockPtr, 0, kNvfp4BlockBytes);
      continue;
    }
    __half inputScaleHalf;
    __half localScaleHalf;
    std::memcpy(&inputScaleHalf, inputBlockPtr, sizeof(inputScaleHalf));
    std::memcpy(&localScaleHalf, localBlockPtr, sizeof(localScaleHalf));
    float inputScale = __half2float(inputScaleHalf);
    float localScale = __half2float(localScaleHalf);
    float values[kNvfp4BlockElts] = {0.0f};
    float maxAbs = 0.0f;

    int validCount = static_cast<int>(sectionElts - blockInSection * kNvfp4BlockElts);
    if (validCount > static_cast<int>(kNvfp4BlockElts)) validCount = static_cast<int>(kNvfp4BlockElts);
    if (validCount < 0) validCount = 0;

    for (int i = 0; i < validCount; ++i) {
      uint8_t inputPacked = inputBlockPtr[2 + (i / 2)];
      uint8_t inputNibble = (i & 1) ? static_cast<uint8_t>(inputPacked >> 4)
                                    : static_cast<uint8_t>(inputPacked & 0xF);
      uint8_t localPacked = localBlockPtr[2 + (i / 2)];
      uint8_t localNibble = (i & 1) ? static_cast<uint8_t>(localPacked >> 4)
                                    : static_cast<uint8_t>(localPacked & 0xF);
      float receivedValue = nvfp4Decode(inputNibble) * inputScale;
      float localValue = nvfp4Decode(localNibble) * localScale;
      float reducedValue = 0.0f;
      switch (op) {
        case ncclSum:
        case ncclAvg:
          reducedValue = receivedValue + localValue;
          break;
        case ncclProd:
          reducedValue = receivedValue * localValue;
          break;
        case ncclMin:
          reducedValue = std::min(receivedValue, localValue);
          break;
        case ncclMax:
          reducedValue = std::max(receivedValue, localValue);
          break;
        default:
          reducedValue = receivedValue;
          break;
      }
      if (finalize && op == ncclAvg && nRanks > 0) {
        reducedValue /= static_cast<float>(nRanks);
      }
      values[i] = reducedValue;
      maxAbs = std::max(maxAbs, std::fabs(reducedValue));
    }

    if (maxAbs == 0.0f) {
      std::memset(outputBlockPtr, 0, kNvfp4BlockBytes);
      continue;
    }

    float chosenScale = nvfp4ChooseReduceScaleL2(values, validCount, nvfp4DefaultScale(maxAbs));
    __half outputScale = __float2half(chosenScale);
    std::memcpy(outputBlockPtr, &outputScale, sizeof(outputScale));
    float storedScale = __half2float(outputScale);
    float invScale = storedScale == 0.0f ? 0.0f : (1.0f / storedScale);

    for (int byteIdx = 0; byteIdx < 8; ++byteIdx) {
      uint8_t packedByte = 0;
      for (int lane = 0; lane < 2; ++lane) {
        int eltInBlock = byteIdx * 2 + lane;
        size_t logicalIndex = blockInSection * kNvfp4BlockElts + static_cast<size_t>(eltInBlock);
        float value = logicalIndex < sectionElts ? values[eltInBlock] * invScale : 0.0f;
        uint8_t nibble = nvfp4Encode(value);
        packedByte |= lane == 0 ? nibble : static_cast<uint8_t>(nibble << 4);
      }
      outputBlockPtr[2 + byteIdx] = packedByte;
    }
  }
}

template <typename T>
std::vector<uint8_t> buildPackedInputForRank(int rank, size_t count) {
  std::vector<T> logical(count);
  fillInput(logical, rank);
  std::vector<uint8_t> packed;
  packNvfp4Section(logical, &packed);
  return packed;
}

template <typename T>
std::vector<uint8_t> buildNonCommunicationReference(size_t count, int nranks, ncclRedOp_t op) {
  std::vector<std::vector<uint8_t>> packedInputs(static_cast<size_t>(nranks));
  for (int rank = 0; rank < nranks; ++rank) {
    packedInputs[static_cast<size_t>(rank)] = buildPackedInputForRank<T>(rank, count);
  }

  Nvfp4ChunkPlan plan = buildNvfp4ChunkPlan(count, nranks);
  std::vector<uint8_t> reference(nvfp4SectionBytes(count), 0);
  if (nranks == 0) return reference;
  if (nranks == 1) return packedInputs[0];

  for (int chunk = 0; chunk < nranks; ++chunk) {
    size_t sectionElts = plan.eltCounts[static_cast<size_t>(chunk)];
    size_t logicalPackedBytes = plan.logicalPackedBytes[static_cast<size_t>(chunk)];
    size_t packedByteOffset = plan.packedByteOffsets[static_cast<size_t>(chunk)];
    size_t chunkSectionBytes = plan.sectionBytes[static_cast<size_t>(chunk)];
    if (logicalPackedBytes == 0) continue;

    std::vector<uint8_t> accum(chunkSectionBytes, 0);
    std::vector<uint8_t> next(chunkSectionBytes, 0);
    int initialRank = (chunk + 1) % nranks;
    std::memcpy(accum.data(), packedInputs[static_cast<size_t>(initialRank)].data() + packedByteOffset,
                logicalPackedBytes);
    for (int step = 1; step < nranks; ++step) {
      int reducingRank = (chunk + step + 1) % nranks;
      nvfp4ReduceAndPackChunkCpu(
          accum.data(),
          packedInputs[static_cast<size_t>(reducingRank)].data() + packedByteOffset,
          next.data(), sectionElts, op, nranks, step == nranks - 1);
      accum.swap(next);
      std::fill(next.begin(), next.end(), 0);
    }

    std::memcpy(reference.data() + packedByteOffset, accum.data(), logicalPackedBytes);
  }

  return reference;
}

CorrectnessStats computeCorrectnessStatsFromPackedOutputs(const std::vector<uint8_t>& expected,
                                                          const std::vector<uint8_t>& actual,
                                                          size_t count, size_t sectionBytes) {
  CorrectnessStats stats = {0.0, 0.0, 0};
  for (size_t byte = 0; byte < sectionBytes; ++byte) {
    if (expected[byte] != actual[byte]) ++stats.packedMismatchBytes;
  }
  for (size_t i = 0; i < count; ++i) {
    double baseline = static_cast<double>(unpackNvfp4SectionValue(expected.data(), i));
    double got = static_cast<double>(unpackNvfp4SectionValue(actual.data(), i));
    double absDiff = std::fabs(baseline - got);
    stats.maxDiff = std::max(stats.maxDiff, absDiff);
    stats.totalAbsError += absDiff;
  }
  return stats;
}

template <typename T>
bool runCorrectnessCase(const Options& opts, int worldRank, int worldSize, ncclComm_t comm,
                        cudaStream_t stream, size_t logicalBytes, const char* dtypeLabel,
                        ncclDataType_t dtype) {
  const size_t typeBytes = sizeof(T);
  if (logicalBytes % typeBytes != 0) {
    if (worldRank == 0) {
      std::printf("correctness SKIP dtype=%s logical_bytes=%zu reason=logical_bytes_not_divisible_by_type\n",
                  dtypeLabel, logicalBytes);
    }
    return true;
  }

  size_t count = logicalBytes / typeBytes;
  size_t sectionBytes = nvfp4SectionBytes(count);

  std::vector<T> hostInput(count);
  fillInput(hostInput, worldRank);
  std::vector<uint8_t> hostPackedSend;
  packNvfp4Section(hostInput, &hostPackedSend);
  std::vector<uint8_t> hostReference = buildNonCommunicationReference<T>(count, worldSize, ncclSum);

  const char* callTraceEnv = getenv("NVFP4_HARNESS_CALL_TRACE");
  bool traceCall = callTraceEnv != nullptr && callTraceEnv[0] != '\0' && strcmp(callTraceEnv, "0") != 0;

  uint8_t* send = nullptr;
  uint8_t* recv = nullptr;
  CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&send), sectionBytes));
  CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&recv), sectionBytes));
  CUDACHECK(cudaMemcpy(send, hostPackedSend.data(), sectionBytes, cudaMemcpyHostToDevice));

  if (traceCall && worldRank == 0) {
    std::printf("call_trace phase=before_allreduce count=%zu\n", count);
    std::fflush(stdout);
  }
  NCCLCHECK(ncclAllReduceNvfp4(send, recv, count, dtype, ncclSum, comm, stream));
  if (traceCall && worldRank == 0) {
    std::printf("call_trace phase=after_allreduce_return\n");
    std::fflush(stdout);
  }
  CUDACHECK(cudaStreamSynchronize(stream));
  if (traceCall && worldRank == 0) {
    std::printf("call_trace phase=after_sync\n");
    std::fflush(stdout);
  }

  std::vector<uint8_t> hostReduced(sectionBytes);
  CUDACHECK(cudaMemcpy(hostReduced.data(), recv, sectionBytes, cudaMemcpyDeviceToHost));

  CorrectnessStats localStats =
      computeCorrectnessStatsFromPackedOutputs(hostReference, hostReduced, count, sectionBytes);
  double globalMaxDiff = 0.0;
  double globalTotalAbsError = 0.0;
  unsigned long long globalPackedMismatchBytes = 0;
  MPI_CHECK(MPI_Allreduce(&localStats.maxDiff, &globalMaxDiff, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allreduce(&localStats.totalAbsError, &globalTotalAbsError, 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allreduce(&localStats.packedMismatchBytes, &globalPackedMismatchBytes, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
  double meanAbsError = count == 0 ? 0.0 : (globalTotalAbsError / static_cast<double>(count));

  bool pass = opts.maxDiffThreshold < 0.0 || globalMaxDiff <= opts.maxDiffThreshold;
  if (worldRank == 0) {
    std::printf(
        "correctness dtype=%s nranks=%d logical_bytes=%zu (%s) count=%zu packed_bytes=%zu baseline=non_communication_nvfp4 packed_mismatch_bytes=%llu max_diff=%.8f total_abs_error=%.8f mean_abs_error=%.8f%s\n",
        dtypeLabel, worldSize, logicalBytes, formatBytes(logicalBytes).c_str(), count,
        sectionBytes, globalPackedMismatchBytes, globalMaxDiff, globalTotalAbsError,
        meanAbsError, pass ? "" : " threshold_exceeded");
    if (count >= 8) {
      std::printf("debug first8 expected=");
      for (size_t i = 0; i < 8; ++i) std::printf(i == 0 ? "%.5f" : " %.5f", unpackNvfp4SectionValue(hostReference.data(), i));
      std::printf(" actual=");
      for (size_t i = 0; i < 8; ++i) std::printf(i == 0 ? "%.5f" : " %.5f", unpackNvfp4SectionValue(hostReduced.data(), i));
      std::printf("\n");
    }
    std::fflush(stdout);
  }

  CUDACHECK(cudaFree(recv));
  CUDACHECK(cudaFree(send));
  return pass;
}

void printScalingCsvRow(bool csv, const char* impl, const char* dtypeLabel, int nranks,
                        size_t count, size_t logicalBytes, size_t bufferBytes, double avgMs,
                        double algBw, double busBw, size_t scratchBytes) {
  if (csv) {
    std::printf("scaling,%s,%s,%d,%zu,%zu,%zu,%.6f,%.6f,%.6f,%zu\n", impl, dtypeLabel,
                nranks, count, logicalBytes, bufferBytes, avgMs, algBw, busBw, scratchBytes);
  } else {
    std::printf(
        "scaling impl=%s dtype=%s nranks=%d count=%zu logical_bytes=%zu (%s) buffer_bytes=%zu avg_ms=%.6f algbw_gbps=%.6f busbw_gbps=%.6f scratch_bytes=%zu\n",
        impl, dtypeLabel, nranks, count, logicalBytes, formatBytes(logicalBytes).c_str(),
        bufferBytes, avgMs, algBw, busBw, scratchBytes);
  }
  std::fflush(stdout);
}

template <typename T>
bool runNvfp4ScalingCase(const Options& opts, int worldRank, int worldSize, ncclComm_t comm,
                         cudaStream_t stream, size_t logicalBytes, const char* dtypeLabel,
                         ncclDataType_t logicalDtype) {
  const size_t typeBytes = sizeof(T);
  if (logicalBytes % typeBytes != 0) {
    if (worldRank == 0) {
      std::printf("scaling SKIP impl=%s dtype=%s nranks=%d logical_bytes=%zu reason=logical_bytes_not_divisible_by_type\n",
                  kNvfp4ImplName, dtypeLabel, worldSize, logicalBytes);
    }
    return false;
  }

  size_t count = logicalBytes / typeBytes;
  size_t scratchBytes = 0;
  size_t totalBytes = 0;
  if (!estimateNvfp4AllReduceMemoryFromCount(count, &scratchBytes, &totalBytes)) {
    if (worldRank == 0) {
      std::printf("scaling SKIP impl=%s dtype=%s nranks=%d logical_bytes=%zu reason=memory_estimate_overflow\n",
                  kNvfp4ImplName, dtypeLabel, worldSize, logicalBytes);
    }
    return false;
  }

  size_t freeBytes = 0;
  size_t deviceBytes = 0;
  CUDACHECK(cudaMemGetInfo(&freeBytes, &deviceBytes));
  if (totalBytes > freeBytes) {
    if (worldRank == 0) {
      std::printf(
          "scaling SKIP impl=%s dtype=%s nranks=%d logical_bytes=%zu (%s) required_bytes=%zu free_bytes=%zu scratch_bytes=%zu\n",
          kNvfp4ImplName, dtypeLabel, worldSize, logicalBytes,
          formatBytes(logicalBytes).c_str(), totalBytes, freeBytes, scratchBytes);
    }
    return false;
  }

  size_t sectionBytes = nvfp4SectionBytes(count);
  uint8_t* send = nullptr;
  uint8_t* recv = nullptr;
  CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&send), sectionBytes));
  CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&recv), sectionBytes));
  CUDACHECK(cudaMemsetAsync(send, 0, sectionBytes, stream));
  CUDACHECK(cudaMemsetAsync(recv, 0, sectionBytes, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  for (int iter = 0; iter < opts.warmupIters; ++iter) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NCCLCHECK(ncclAllReduceNvfp4(send, recv, count, logicalDtype, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  CUDACHECK(cudaEventRecord(start, stream));
  for (int iter = 0; iter < opts.benchIters; ++iter) {
    NCCLCHECK(ncclAllReduceNvfp4(send, recv, count, logicalDtype, ncclSum, comm, stream));
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaEventSynchronize(stop));

  float totalMs = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&totalMs, start, stop));
  double avgMs = static_cast<double>(totalMs) / static_cast<double>(opts.benchIters);
  double globalMs = 0.0;
  MPI_CHECK(MPI_Allreduce(&avgMs, &globalMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

  double algBw = static_cast<double>(logicalBytes) / 1.0e9 / (globalMs / 1.0e3);
  double busBw = algBw * (2.0 * static_cast<double>(worldSize - 1) / static_cast<double>(worldSize));
  if (worldRank == 0) {
    printScalingCsvRow(opts.csv, kNvfp4ImplName, dtypeLabel, worldSize, count, logicalBytes,
                       sectionBytes, globalMs, algBw, busBw, scratchBytes);
  }

  CUDACHECK(cudaEventDestroy(stop));
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaFree(recv));
  CUDACHECK(cudaFree(send));
  return true;
}

template <typename T>
bool runFp8ScalingCase(const Options& opts, int worldRank, int worldSize, ncclComm_t comm,
                       cudaStream_t stream, size_t logicalBytes, const char* dtypeLabel) {
  if (!fp8AllReduceSupported()) {
    if (worldRank == 0) {
      std::printf("scaling SKIP impl=%s dtype=%s nranks=%d logical_bytes=%zu reason=fp8_requires_cuda_11_8_or_newer\n",
                  kFp8ImplName, dtypeLabel, worldSize, logicalBytes);
    }
    return false;
  }

  const size_t typeBytes = sizeof(T);
  if (logicalBytes % typeBytes != 0) {
    if (worldRank == 0) {
      std::printf("scaling SKIP impl=%s dtype=%s nranks=%d logical_bytes=%zu reason=logical_bytes_not_divisible_by_type\n",
                  kFp8ImplName, dtypeLabel, worldSize, logicalBytes);
    }
    return false;
  }

  size_t count = logicalBytes / typeBytes;
  size_t totalBytes = 0;
  if (!estimateFp8AllReduceMemoryFromCount(count, &totalBytes)) {
    if (worldRank == 0) {
      std::printf("scaling SKIP impl=%s dtype=%s nranks=%d logical_bytes=%zu reason=memory_estimate_overflow\n",
                  kFp8ImplName, dtypeLabel, worldSize, logicalBytes);
    }
    return false;
  }

  size_t freeBytes = 0;
  size_t deviceBytes = 0;
  CUDACHECK(cudaMemGetInfo(&freeBytes, &deviceBytes));
  if (totalBytes > freeBytes) {
    if (worldRank == 0) {
      std::printf(
          "scaling SKIP impl=%s dtype=%s nranks=%d logical_bytes=%zu (%s) required_bytes=%zu free_bytes=%zu scratch_bytes=0\n",
          kFp8ImplName, dtypeLabel, worldSize, logicalBytes,
          formatBytes(logicalBytes).c_str(), totalBytes, freeBytes);
    }
    return false;
  }

  size_t bufferBytes = count;
  uint8_t* send = nullptr;
  uint8_t* recv = nullptr;
  CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&send), bufferBytes));
  CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&recv), bufferBytes));
  CUDACHECK(cudaMemsetAsync(send, 0, bufferBytes, stream));
  CUDACHECK(cudaMemsetAsync(recv, 0, bufferBytes, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  for (int iter = 0; iter < opts.warmupIters; ++iter) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NCCLCHECK(ncclAllReduce(send, recv, count, ncclFloat8e4m3, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  CUDACHECK(cudaEventRecord(start, stream));
  for (int iter = 0; iter < opts.benchIters; ++iter) {
    NCCLCHECK(ncclAllReduce(send, recv, count, ncclFloat8e4m3, ncclSum, comm, stream));
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaEventSynchronize(stop));

  float totalMs = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&totalMs, start, stop));
  double avgMs = static_cast<double>(totalMs) / static_cast<double>(opts.benchIters);
  double globalMs = 0.0;
  MPI_CHECK(MPI_Allreduce(&avgMs, &globalMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

  double algBw = static_cast<double>(logicalBytes) / 1.0e9 / (globalMs / 1.0e3);
  double busBw = algBw * (2.0 * static_cast<double>(worldSize - 1) / static_cast<double>(worldSize));
  if (worldRank == 0) {
    printScalingCsvRow(opts.csv, kFp8ImplName, dtypeLabel, worldSize, count, logicalBytes,
                       bufferBytes, globalMs, algBw, busBw, 0);
  }

  CUDACHECK(cudaEventDestroy(stop));
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaFree(recv));
  CUDACHECK(cudaFree(send));
  return true;
}

template <typename T>
bool runType(const Options& opts, int worldRank, int worldSize, ncclComm_t comm,
             cudaStream_t stream, Options::DType dtypeChoice) {
  const char* dtypeLabel = dtypeName(dtypeChoice);
  ncclDataType_t dtype = ncclType(dtypeChoice);
  bool ok = true;

  if (opts.mode == Options::kCorrectness || opts.mode == Options::kBoth) {
    for (size_t logicalBytes : opts.correctnessBytes) {
      ok &= runCorrectnessCase<T>(opts, worldRank, worldSize, comm, stream, logicalBytes,
                                  dtypeLabel, dtype);
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  }

  if (opts.mode == Options::kScaling || opts.mode == Options::kBoth) {
    if (worldRank == 0 && opts.csv) {
      std::printf("kind,impl,dtype,nranks,count,logical_bytes,buffer_bytes,avg_ms,algbw_gbps,busbw_gbps,scratch_bytes\n");
    }
    for (size_t logicalBytes : opts.scalingBytes) {
      (void)runNvfp4ScalingCase<T>(opts, worldRank, worldSize, comm, stream, logicalBytes,
                                   dtypeLabel, dtype);
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      (void)runFp8ScalingCase<T>(opts, worldRank, worldSize, comm, stream, logicalBytes,
                                 dtypeLabel);
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  }

  return ok;
}

}  // namespace

int main(int argc, char* argv[]) {
  MPI_CHECK(MPI_Init(&argc, &argv));

  int worldRank = 0;
  int worldSize = 0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

  Options opts = parseOptions(argc, argv);
  int localRank = getLocalRank(MPI_COMM_WORLD);
  const char* initTraceEnv = getenv("NVFP4_HARNESS_INIT_TRACE");
  bool traceInit = initTraceEnv != nullptr && initTraceEnv[0] != '\0' && strcmp(initTraceEnv, "0") != 0;
  if (traceInit && worldRank == 0) {
    std::printf("init_trace phase=after_mpi world_size=%d\n", worldSize);
    std::fflush(stdout);
  }

  int deviceCount = 0;
  CUDACHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    if (worldRank == 0) std::fprintf(stderr, "No CUDA devices found\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (localRank >= deviceCount) {
    std::fprintf(stderr,
                 "Rank %d requested local GPU %d but only %d device(s) are visible on this node\n",
                 worldRank, localRank, deviceCount);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  CUDACHECK(cudaSetDevice(localRank));
  cudaStream_t stream = nullptr;
  CUDACHECK(cudaStreamCreate(&stream));
  if (traceInit && worldRank == 0) {
    std::printf("init_trace phase=after_cuda local_rank=%d device_count=%d\n", localRank, deviceCount);
    std::fflush(stdout);
  }

  ncclUniqueId uniqueId;
  if (worldRank == 0) NCCLCHECK(ncclGetUniqueId(&uniqueId));
  MPI_CHECK(MPI_Bcast(&uniqueId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD));

  ncclComm_t comm = nullptr;
  if (traceInit && worldRank == 0) {
    std::printf("init_trace phase=before_nccl_init\n");
    std::fflush(stdout);
  }
  NCCLCHECK(ncclCommInitRank(&comm, worldSize, uniqueId, worldRank));
  if (traceInit && worldRank == 0) {
    std::printf("init_trace phase=after_nccl_init\n");
    std::fflush(stdout);
  }

  if (worldRank == 0) {
    std::printf(
        "NVFP4 allreduce MPI harness starting: mode=%s nranks=%d local_gpu_per_rank=1 scaling_compare=%s_vs_%s\n",
        modeName(opts.mode), worldSize, kNvfp4ImplName, kFp8ImplName);
    std::fflush(stdout);
  }

  bool ok = true;
  for (size_t i = 0; i < opts.dtypes.size(); ++i) {
    switch (opts.dtypes[i]) {
      case Options::kFp16:
        ok &= runType<__half>(opts, worldRank, worldSize, comm, stream, Options::kFp16);
        break;
      case Options::kBf16:
#if CUDART_VERSION >= 11000
        ok &= runType<__nv_bfloat16>(opts, worldRank, worldSize, comm, stream, Options::kBf16);
#else
        if (worldRank == 0) std::fprintf(stderr, "bf16 requires CUDA 11 or newer\n");
        ok = false;
#endif
        break;
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  CUDACHECK(cudaStreamSynchronize(stream));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaStreamDestroy(stream));

  if (worldRank == 0) {
    std::printf("NVFP4 allreduce MPI harness %s\n", ok ? "completed" : "failed");
  }
  MPI_CHECK(MPI_Finalize());
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
