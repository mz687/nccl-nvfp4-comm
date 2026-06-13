/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dtype_error_compare_kernels.h"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

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

constexpr size_t kNvfp4BlockElts = 16;
constexpr size_t kNvfp4BlockBytes = 10;
constexpr size_t kNvfp4Alignment = 16;

struct Options {
  size_t valueCount = 128ULL << 20;
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

size_t parseScaledCount(const std::string& text) {
  std::string value = toLower(text);
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
  size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) ++start;
  value = value.substr(start);
  if (value.empty()) {
    std::fprintf(stderr, "empty value count\n");
    std::exit(EXIT_FAILURE);
  }

  unsigned long long multiplier = 1;
  if (endsWith(value, "g")) {
    multiplier = 1ULL << 30;
    value.resize(value.size() - 1);
  } else if (endsWith(value, "m")) {
    multiplier = 1ULL << 20;
    value.resize(value.size() - 1);
  } else if (endsWith(value, "k")) {
    multiplier = 1ULL << 10;
    value.resize(value.size() - 1);
  }

  char* end = nullptr;
  double number = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0' || number <= 0.0) {
    std::fprintf(stderr, "invalid value count '%s'\n", text.c_str());
    std::exit(EXIT_FAILURE);
  }
  long double count = static_cast<long double>(number) * static_cast<long double>(multiplier);
  if (count <= 0.0L || count > static_cast<long double>(~0ULL)) {
    std::fprintf(stderr, "value count overflow for '%s'\n", text.c_str());
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(count);
}

std::string formatCount(size_t count) {
  char buffer[64];
  if (count >= (1ULL << 30)) {
    double value = static_cast<double>(count) / static_cast<double>(1ULL << 30);
    std::snprintf(buffer, sizeof(buffer), (count % (1ULL << 30)) == 0 ? "%.0fG" : "%.2fG", value);
  } else if (count >= (1ULL << 20)) {
    double value = static_cast<double>(count) / static_cast<double>(1ULL << 20);
    std::snprintf(buffer, sizeof(buffer), (count % (1ULL << 20)) == 0 ? "%.0fM" : "%.2fM", value);
  } else if (count >= (1ULL << 10)) {
    double value = static_cast<double>(count) / static_cast<double>(1ULL << 10);
    std::snprintf(buffer, sizeof(buffer), (count % (1ULL << 10)) == 0 ? "%.0fK" : "%.2fK", value);
  } else {
    std::snprintf(buffer, sizeof(buffer), "%zu", count);
  }
  return std::string(buffer);
}

size_t nvfp4SectionBytes(size_t count) {
  size_t blocks = count / kNvfp4BlockElts + ((count % kNvfp4BlockElts) != 0 ? 1 : 0);
  size_t logicalBytes = blocks * kNvfp4BlockBytes;
  return (logicalBytes + kNvfp4Alignment - 1) & ~(kNvfp4Alignment - 1);
}

void printUsage(const char* prog) {
  std::fprintf(stderr,
      "Usage: %s --value-count 128M|256M|512M|1G [--csv]\n",
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
    if (arg == "--value-count" || arg == "--count") {
      opts.valueCount = parseScaledCount(requireValue(arg.c_str()));
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
  return opts;
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

void checkMemory(size_t requiredBytes, int worldRank, int worldSize, size_t valueCount) {
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
  if (requiredBytes > freeBytes) {
    if (worldRank == 0) {
      std::fprintf(stderr,
          "insufficient GPU memory nranks=%d value_count=%zu required_bytes=%zu free_bytes=%zu\n",
          worldSize, valueCount, requiredBytes, freeBytes);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
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

  size_t count = opts.valueCount;
  size_t nvfp4Bytes = nvfp4SectionBytes(count);
  size_t fp8Bytes = count;
  checkMemory(nvfp4Bytes * 2 + fp8Bytes * 2, worldRank, worldSize, count);

  uint8_t* nvfp4Send = nullptr;
  uint8_t* nvfp4Recv = nullptr;
  uint8_t* fp8Send = nullptr;
  uint8_t* fp8Recv = nullptr;
  CUDA_CHECK(cudaMalloc(&nvfp4Send, nvfp4Bytes));
  CUDA_CHECK(cudaMalloc(&nvfp4Recv, nvfp4Bytes));
  CUDA_CHECK(cudaMalloc(&fp8Send, fp8Bytes));
  CUDA_CHECK(cudaMalloc(&fp8Recv, fp8Bytes));
  CUDA_CHECK(cudaMemsetAsync(nvfp4Recv, 0, nvfp4Bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(fp8Recv, 0, fp8Bytes, stream));
  CUDA_CHECK(launchInitErrorCompareInputs(worldRank, worldSize, count, nvfp4Send, nvfp4Bytes,
                                          fp8Send, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (worldRank == 0) {
    std::printf("dtype-error-compare starting: nranks=%d value_count=%zu label=%s nvfp4_bytes=%zu fp8_bytes=%zu\n",
                worldSize, count, formatCount(count).c_str(), nvfp4Bytes, fp8Bytes);
    std::printf("kind,nranks,value_count,value_count_label,nvfp4_buffer_bytes,fp8_buffer_bytes,nvfp4_avg_abs_error,nvfp4_max_abs_error,fp8_avg_abs_error,fp8_max_abs_error,nvfp4_fp8_avg_abs_diff,nvfp4_fp8_max_abs_diff\n");
    std::fflush(stdout);
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  NCCL_CHECK(ncclAllReduceNvfp4(nvfp4Send, nvfp4Recv, count, ncclFloat16, ncclSum, comm, stream));
  NCCL_CHECK(ncclAllReduce(fp8Send, fp8Recv, count, ncclFloat8e4m3, ncclSum, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  DTypeErrorStats localStats = {};
  CUDA_CHECK(launchComputeErrorCompareStats(count, worldSize, nvfp4Recv, fp8Recv, &localStats, stream));

  double localSums[3] = {
      localStats.nvfp4AbsSum,
      localStats.fp8AbsSum,
      localStats.nvfp4Fp8AbsSum,
  };
  double globalSums[3] = {};
  double localMaxes[3] = {
      localStats.nvfp4MaxAbs,
      localStats.fp8MaxAbs,
      localStats.nvfp4Fp8MaxAbs,
  };
  double globalMaxes[3] = {};
  MPI_CHECK(MPI_Allreduce(localSums, globalSums, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allreduce(localMaxes, globalMaxes, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

  double denominator = static_cast<double>(count) * static_cast<double>(worldSize);
  if (worldRank == 0) {
    std::printf("error_summary,%d,%zu,%s,%zu,%zu,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                worldSize, count, formatCount(count).c_str(), nvfp4Bytes, fp8Bytes,
                globalSums[0] / denominator, globalMaxes[0],
                globalSums[1] / denominator, globalMaxes[1],
                globalSums[2] / denominator, globalMaxes[2]);
    std::printf("dtype-error-compare completed\n");
    std::fflush(stdout);
  }

  MPI_CHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}
