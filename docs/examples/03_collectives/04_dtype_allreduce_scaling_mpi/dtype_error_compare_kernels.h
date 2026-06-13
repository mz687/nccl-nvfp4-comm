#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

struct DTypeErrorStats {
  double nvfp4AbsSum;
  double fp8AbsSum;
  double nvfp4Fp8AbsSum;
  double nvfp4MaxAbs;
  double fp8MaxAbs;
  double nvfp4Fp8MaxAbs;
};

extern "C" cudaError_t launchInitErrorCompareInputs(int rank, int nranks, size_t count,
                                                    uint8_t* nvfp4Send, size_t nvfp4SectionBytes,
                                                    uint8_t* fp8Send, cudaStream_t stream);

extern "C" cudaError_t launchComputeErrorCompareStats(size_t count, int nranks,
                                                      const uint8_t* nvfp4Recv,
                                                      const uint8_t* fp8Recv,
                                                      DTypeErrorStats* hostStats,
                                                      cudaStream_t stream);
