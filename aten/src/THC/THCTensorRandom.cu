#include "THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorRandom.cuh"
#include "ATen/Config.h"

#include <thrust/functional.h>

#define MAX_NUM_BLOCKS 200
#define BLOCK_SIZE 256

__device__ inline at::Half half_uniform_scale_and_shift(float x, double a, double b) {
  at::Half width = ScalarConvert<double, at::Half>::to(b - a);
  at::Half start = ScalarConvert<double, at::Half>::to(a);
  at::Half scaled = THCNumerics<at::Half>::mul(ScalarConvert<float, at::Half>::to(x), width);
  return THCNumerics<at::Half>::add(scaled, start);
}

#define GENERATE_KERNEL1(NAME, T, ARG1, RAND_T, RAND_FUNC, TRANSFORM)      \
__global__ void NAME(std::pair<uint64_t, uint64_t> seeds, int size, T *result, ARG1)      \
{                                                                              \
  int idx = blockIdx.x * blockDim.x + threadIdx.x;                             \
  at::cuda::Philox4_32_10 engine(seeds.first, idx, seeds.second);              \
  int rounded_size = ((size - 1)/(blockDim.x * gridDim.x)+1) * blockDim.x * gridDim.x;                \
  for (int i = idx; i < rounded_size; i += gridDim.x * blockDim.x) {      \
    RAND_T x = static_cast<RAND_T>(RAND_FUNC);                         \
    if (i < size) {                                                            \
      T y = TRANSFORM;                                                         \
      result[i] = y;                                                           \
    }                                                                          \
  }                                                                            \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, RAND_T, RAND_FUNC, TRANSFORM)      \
__global__ void NAME(std::pair<uint64_t, uint64_t> seeds, int size, T *result, ARG1, ARG2)      \
{                                                                                               \
  int idx = blockIdx.x * blockDim.x + threadIdx.x;                                              \
  at::cuda::Philox4_32_10 engine(seeds.first, idx, seeds.second);                               \
  int rounded_size = ((size - 1)/(blockDim.x * gridDim.x)+1) * blockDim.x * gridDim.x;                                 \
  for (int i = idx; i < rounded_size; i += gridDim.x * blockDim.x) {                       \
    RAND_T x = static_cast<RAND_T>(RAND_FUNC);                                          \
    if (i < size) {                                                                             \
      T y = TRANSFORM;                                                                          \
      result[i] = y;                                                                            \
    }                                                                                           \
  }                                                                                             \
}

#define GENERATE_KERNEL3(NAME, T, ARG1, ARG2, RAND_T, RAND_FUNC, TRANSFORM)      \
__global__ void NAME(std::pair<uint64_t, uint64_t> seeds, int size, T *result, ARG1, ARG2)      \
{                                                                                               \
  int idx = blockIdx.x * blockDim.x + threadIdx.x;                                              \
  at::cuda::Philox4_32_10 engine(seeds.first, idx, seeds.second);                               \
  int UNROLL = 2;                                                                        \
  int rounded_size = ((size - 1)/(blockDim.x * gridDim.x * UNROLL)+1) *                         \
        blockDim.x * gridDim.x * UNROLL;                                 \
  for (int i = idx; i < rounded_size; i += gridDim.x * blockDim.x*UNROLL) {                       \
    RAND_T dist_vals = static_cast<RAND_T>(RAND_FUNC);                                                                                    \
    for (int ii = 0; ii < UNROLL; ii++) {                                                       \
      int li = i + blockDim.x * gridDim.x * ii;                                                 \
      if (li < size) {                                                                          \
        if(ii == 0) { \
          T x = dist_vals.x; \
          T y = TRANSFORM; \
          result[li] = y; \
        }else{ \
          T x = dist_vals.y; \
          T y = TRANSFORM; \
          result[li] = y; \
        }                                                                             \
      }                                                                                         \
    }                                                                                           \
    __syncthreads(); \
  }                                                                                             \
}

GENERATE_KERNEL2(generate_uniform, float, float a, float b, float, at::cuda::standard_uniform_distribution(engine), x * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, float, double a, double b, float, at::cuda::standard_uniform_distribution(engine), x * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, double, double a, double b, float, at::cuda::standard_uniform_distribution(engine), x * (b-a) + a)

GENERATE_KERNEL3(generate_normal, float, double mean, double stdv, float2, at::cuda::normal_distribution(engine), (x * stdv) + mean)
GENERATE_KERNEL3(generate_normal, double, double mean, double stdv, float2, at::cuda::normal_distribution(engine), (x * stdv) + mean)

GENERATE_KERNEL1(generate_exponential, float, double lambda, float, at::cuda::standard_uniform_distribution(engine), (float)(-1. / lambda * log(x)))
GENERATE_KERNEL1(generate_exponential, double, double lambda, float, at::cuda::standard_uniform_distribution(engine), (double)(-1. / lambda * log(x)))

GENERATE_KERNEL2(generate_cauchy, float, double median, double sigma, float, at::cuda::standard_uniform_distribution(engine), (float)(median + sigma * tan(M_PI*(x-0.5))))
GENERATE_KERNEL2(generate_cauchy, double, double median, double sigma, float, at::cuda::standard_uniform_distribution(engine), (double)(median + sigma * tan(M_PI*(x-0.5))))

GENERATE_KERNEL2(generate_uniform, at::Half, double a, double b, float, at::cuda::standard_uniform_distribution(engine), (half_uniform_scale_and_shift(x, a, b)))
GENERATE_KERNEL3(generate_normal, at::Half, double mean, double stdv, float2, at::cuda::normal_distribution(engine), (ScalarConvert<float, at::Half>::to((x * stdv) + mean)))
GENERATE_KERNEL1(generate_exponential, at::Half, double lambda, float, at::cuda::standard_uniform_distribution(engine), (ScalarConvert<float, at::Half>::to((float)(-1. / lambda * log(x)))))
GENERATE_KERNEL2(generate_cauchy, at::Half, double median, double sigma, float, at::cuda::standard_uniform_distribution(engine), (ScalarConvert<float, at::Half>::to((float)(median + sigma * tan(M_PI*(x-0.5))))))

#include "generic/THCTensorRandom.cu"
#include "THCGenerateAllTypes.h"

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2
#undef GENERATE_KERNEL3
