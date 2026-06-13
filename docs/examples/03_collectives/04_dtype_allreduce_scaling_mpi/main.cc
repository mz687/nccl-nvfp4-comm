/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#define MPI_CHECK(cmd)                                                         \
  do {                                                                         \
    int err = (cmd);                                                           \
    if (err != MPI_SUCCESS) {                                                  \
      char errStr[MPI_MAX_ERROR_STRING];                                       \
      int len = 0;                                                             \
      MPI_Error_string(err, errStr, &len);                                     \
      std::fprintf(stderr, "MPI failure %s:%d '%.*s'\n", __FILE__, __LINE__,  \
                   len, errStr);                                               \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t res = (cmd);                                                  \
    if (res != ncclSuccess) {                                                  \
      std::fprintf(stderr, "NCCL failure %s:%d '%s'\n", __FILE__, __LINE__,    \
                   ncclGetErrorString(res));                                   \
      std::fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);              \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t err = (cmd);                                                   \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA failure %s:%d '%s'\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                   \
      std::fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);              \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

namespace {

constexpr size_t kDefaultMessageBytes[] = {
    128ULL << 20,
    256ULL << 20,
    512ULL << 20,
    1ULL << 30,
};

struct DTypeCase {
  std::string label;
  ncclDataType_t ncclType;
  size_t typeBytes;
};

struct Options {
  std::vector<std::string> dtypeLabels = {"fp32", "bf16", "fp16", "fp8"};
  std::vector<size_t> messageBytes = std::vector<size_t>(
      kDefaultMessageBytes,
      kDefaultMessageBytes + sizeof(kDefaultMessageBytes) / sizeof(kDefaultMessageBytes[0]));
  int warmupIters = 3;
  int benchIters = 10;
  bool csv = true;
};

std::string toLower(std::string value) {
  for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return value;
}

bool endsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

size_t parseByteValue(const std::string& text) {
  std::string value = toLower(text);
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
  size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) ++start;
  value = value.substr(start);
  if (value.empty()) {
    std::fprintf(stderr, "empty byte value\n");
    std::exit(EXIT_FAILURE);
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
    std::fprintf(stderr, "invalid byte value '%s'\n", text.c_str());
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(number * static_cast<double>(multiplier));
}

std::vector<std::string> parseStringList(const std::string& text) {
  std::vector<std::string> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = toLower(item);
    size_t start = 0;
    while (start < item.size() && std::isspace(static_cast<unsigned char>(item[start]))) ++start;
    size_t end = item.size();
    while (end > start && std::isspace(static_cast<unsigned char>(item[end - 1]))) --end;
    if (end > start) values.push_back(item.substr(start, end - start));
  }
  return values;
}

std::vector<size_t> parseByteList(const std::string& text) {
  std::vector<size_t> values;
  for (const std::string& item : parseStringList(text)) values.push_back(parseByteValue(item));
  if (values.empty()) {
    std::fprintf(stderr, "message byte list must not be empty\n");
    std::exit(EXIT_FAILURE);
  }
  return values;
}

void printUsage(const char* prog) {
  std::fprintf(stderr,
      "Usage: %s [--dtypes fp32,bf16,fp16,fp8] [--message-bytes 128MiB,256MiB,512MiB,1GiB] "
      "[--warmup N] [--iters N] [--csv]\n",
      prog);
}

Options parseOptions(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto requireValue = [&](const char* name)->const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        printUsage(argv[0]);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--dtypes") {
      opts.dtypeLabels = parseStringList(requireValue("--dtypes"));
    } else if (arg == "--message-bytes" || arg == "--scaling-bytes") {
      opts.messageBytes = parseByteList(requireValue(arg.c_str()));
    } else if (arg == "--warmup") {
      opts.warmupIters = std::atoi(requireValue("--warmup"));
    } else if (arg == "--iters") {
      opts.benchIters = std::atoi(requireValue("--iters"));
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
  if (opts.dtypeLabels.empty()) {
    std::fprintf(stderr, "dtype list must not be empty\n");
    std::exit(EXIT_FAILURE);
  }
  if (opts.warmupIters < 0 || opts.benchIters <= 0) {
    std::fprintf(stderr, "warmup must be >= 0 and iters must be > 0\n");
    std::exit(EXIT_FAILURE);
  }
  return opts;
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

DTypeCase dtypeCaseFromLabel(const std::string& label) {
  std::string dtype = toLower(label);
  if (dtype == "fp32" || dtype == "float" || dtype == "float32") {
    return {"fp32", ncclFloat32, sizeof(float)};
  }
  if (dtype == "fp16" || dtype == "half" || dtype == "float16") {
    return {"fp16", ncclFloat16, sizeof(__half)};
  }
  if (dtype == "bf16" || dtype == "bfloat16") {
#if CUDART_VERSION >= 11000
    return {"bf16", ncclBfloat16, sizeof(__nv_bfloat16)};
#else
    std::fprintf(stderr, "bf16 requires CUDA 11 or newer\n");
    std::exit(EXIT_FAILURE);
#endif
  }
  if (dtype == "fp8" || dtype == "fp8e4m3" || dtype == "e4m3") {
#if CUDART_VERSION >= 11080
    return {"fp8", ncclFloat8e4m3, 1};
#else
    std::fprintf(stderr, "fp8 requires CUDA 11.8 or newer\n");
    std::exit(EXIT_FAILURE);
#endif
  }
  std::fprintf(stderr, "unsupported dtype '%s'\n", label.c_str());
  std::exit(EXIT_FAILURE);
}

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

void runCase(const DTypeCase& dtypeCase, size_t messageBytes, int worldRank, int worldSize,
             const Options& opts, ncclComm_t comm, cudaStream_t stream) {
  if (messageBytes % dtypeCase.typeBytes != 0) {
    if (worldRank == 0) {
      std::printf("skip dtype=%s nranks=%d message_bytes=%zu reason=not_divisible_by_type_size\n",
                  dtypeCase.label.c_str(), worldSize, messageBytes);
      std::fflush(stdout);
    }
    return;
  }

  size_t freeBytes = 0;
  size_t totalDeviceBytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalDeviceBytes));
  size_t bufferBytes = messageBytes;
  if (bufferBytes > (static_cast<size_t>(-1) / 2) || bufferBytes * 2 > freeBytes) {
    if (worldRank == 0) {
      std::printf("skip dtype=%s nranks=%d message_bytes=%zu required_bytes=%zu free_bytes=%zu\n",
                  dtypeCase.label.c_str(), worldSize, messageBytes, bufferBytes * 2, freeBytes);
      std::fflush(stdout);
    }
    return;
  }

  void* send = nullptr;
  void* recv = nullptr;
  CUDA_CHECK(cudaMalloc(&send, bufferBytes));
  CUDA_CHECK(cudaMalloc(&recv, bufferBytes));
  CUDA_CHECK(cudaMemsetAsync(send, 0, bufferBytes, stream));
  CUDA_CHECK(cudaMemsetAsync(recv, 0, bufferBytes, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t count = messageBytes / dtypeCase.typeBytes;
  for (int iter = 0; iter < opts.warmupIters; ++iter) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NCCL_CHECK(ncclAllReduce(send, recv, count, dtypeCase.ncclType, ncclSum, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iter = 0; iter < opts.benchIters; ++iter) {
    NCCL_CHECK(ncclAllReduce(send, recv, count, dtypeCase.ncclType, ncclSum, comm, stream));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float totalMs = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
  double avgMs = static_cast<double>(totalMs) / static_cast<double>(opts.benchIters);
  double globalMs = 0.0;
  MPI_CHECK(MPI_Allreduce(&avgMs, &globalMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

  double algBw = static_cast<double>(messageBytes) / 1.0e9 / (globalMs / 1.0e3);
  double busBw = algBw * (2.0 * static_cast<double>(worldSize - 1) / static_cast<double>(worldSize));
  if (worldRank == 0) {
    if (opts.csv) {
      std::printf("scaling,%s,%d,%zu,%zu,%zu,%.6f,%.6f,%.6f\n",
                  dtypeCase.label.c_str(), worldSize, count, messageBytes, bufferBytes,
                  globalMs, algBw, busBw);
    } else {
      std::printf("scaling dtype=%s nranks=%d message_bytes=%zu (%s) count=%zu avg_ms=%.6f algbw_gbps=%.6f busbw_gbps=%.6f\n",
                  dtypeCase.label.c_str(), worldSize, messageBytes, formatBytes(messageBytes).c_str(),
                  count, globalMs, algBw, busBw);
    }
    std::fflush(stdout);
  }

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaFree(recv));
  CUDA_CHECK(cudaFree(send));
}

}  // namespace

int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));

  int worldRank = 0;
  int worldSize = 0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

  Options opts = parseOptions(argc, argv);
  int localRank = getLocalRank(MPI_COMM_WORLD);

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    if (worldRank == 0) std::fprintf(stderr, "No CUDA devices found\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (localRank >= deviceCount) {
    std::fprintf(stderr, "Rank %d requested local GPU %d but only %d device(s) are visible\n",
                 worldRank, localRank, deviceCount);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  CUDA_CHECK(cudaSetDevice(localRank));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  ncclUniqueId uniqueId;
  if (worldRank == 0) NCCL_CHECK(ncclGetUniqueId(&uniqueId));
  MPI_CHECK(MPI_Bcast(&uniqueId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD));

  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, worldSize, uniqueId, worldRank));

  std::vector<DTypeCase> dtypeCases;
  for (const std::string& label : opts.dtypeLabels) dtypeCases.push_back(dtypeCaseFromLabel(label));

  if (worldRank == 0) {
    std::printf("dtype allreduce scaling MPI harness starting: nranks=%d local_gpu_per_rank=1 warmup=%d iters=%d\n",
                worldSize, opts.warmupIters, opts.benchIters);
    if (opts.csv) {
      std::printf("kind,dtype,nranks,count,message_bytes,buffer_bytes,avg_ms,algbw_gbps,busbw_gbps\n");
    }
    std::fflush(stdout);
  }

  for (const DTypeCase& dtypeCase : dtypeCases) {
    for (size_t messageBytes : opts.messageBytes) {
      runCase(dtypeCase, messageBytes, worldRank, worldSize, opts, comm, stream);
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (worldRank == 0) {
    std::printf("dtype allreduce scaling MPI harness completed\n");
    std::fflush(stdout);
  }

  MPI_CHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}
