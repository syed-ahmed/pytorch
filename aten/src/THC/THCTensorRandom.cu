#include "THCDeviceUtils.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorRandom.cuh"
#include "THCGenerator.hpp"
#include "ATen/Config.h"

#include "ATen/cuda/CUDADistributions.cuh"
#include "ATen/core/Generator.h"
#include "ATen/CheckGenerator.h"

#include <thrust/functional.h>

#define MAX_NUM_BLOCKS 200
#define BLOCK_SIZE 256

// __host__ void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
// {
//   at::Generator* gen = &at::globalContext().getDefaultGenerator(at::kCUDA);
//   std::lock_guard<std::mutex> lock(gen->mutex);

//   // The RNG state comprises the MTPG32 states, the seed, and an offset used for Philox
//   static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
//   static const size_t seed_size = sizeof(gen->state.initial_seed);
//   static const size_t offset_size = sizeof(gen->state.philox_seed_offset);
//   static const size_t total_size = states_size + seed_size + offset_size;
//   THByteTensor_resize1d(rng_state, total_size);
//   THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
//   THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
//   THCudaCheck(cudaMemcpy(THByteTensor_data(rng_state), gen->state.gen_states,
//                          states_size, cudaMemcpyDeviceToHost));
//   memcpy(THByteTensor_data(rng_state) + states_size, &gen->state.initial_seed, seed_size);
//   memcpy(THByteTensor_data(rng_state) + states_size + seed_size, &gen->state.philox_seed_offset, offset_size);
// }

// __global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
// {
//   state[threadIdx.x].k = kernel;
// }

// __host__ void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
// {
//   THCGenerator* gen = THCRandom_getGenerator(state);
//   std::lock_guard<std::mutex> lock(gen->mutex);

//   static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
//   static const size_t seed_size = sizeof(gen->state.initial_seed);
//   static const size_t offset_size = sizeof(gen->state.philox_seed_offset);
//   static const size_t total_size = states_size + seed_size + offset_size;
//   bool no_philox_seed = false;
//   if (THByteTensor_nElement(rng_state) == total_size - offset_size) {
//     no_philox_seed = true;
//   }
//   else {
//     THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
//   }
//   THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

//   THCudaCheck(cudaMemcpy(gen->state.gen_states, THByteTensor_data(rng_state),
//                          states_size, cudaMemcpyHostToDevice));
//   set_rngstate_kernel<<<1, MAX_NUM_BLOCKS, 0, THCState_getCurrentStream(state)>>>(
//       gen->state.gen_states, gen->state.kernel_params);
//   memcpy(&gen->state.initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
//   if (!no_philox_seed) {
//     memcpy(&gen->state.philox_seed_offset, THByteTensor_data(rng_state) + states_size + seed_size, offset_size);
//   }
//   else {
//     gen->state.philox_seed_offset = 0;
//   }
// }

__device__ inline at::Half half_uniform_scale_and_shift(float x, double a, double b) {
  at::Half width = ScalarConvert<double, at::Half>::to(b - a);
  at::Half start = ScalarConvert<double, at::Half>::to(a);
  return THCNumerics<at::Half>::add(width, start);
}

#define GENERATE_KERNEL1(NAME, T, ARG1, RAND_T, RAND_FUNC, TRANSFORM)                \
__global__ void NAME(at::Generator* _generator, int size, T *result, ARG1)           \
{                                                                                    \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                   \
  auto seeds = _generator->next_philox_seed(10);                                     \
  _generator->getState()->engine.init_engine(seeds.first, idx, seeds.second);      \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                      \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    RAND_T x = RAND_FUNC;                                                \
    if (i < size) {                                                                  \
      T y = TRANSFORM;                                                               \
      result[i] = y;                                                                 \
    }                                                                                \
  }                                                                                  \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, RAND_T, RAND_FUNC, TRANSFORM)          \
__global__ void NAME(at::Generator* _generator, int size, T *result, ARG1, ARG2)     \
{                                                                                    \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                   \
  auto seeds = _generator->next_philox_seed(10);                                     \
  _generator->getState()->engine.init_engine(seeds.first, idx, seeds.second);      \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                      \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    RAND_T x = RAND_FUNC;                                                \
    if (i < size) {                                                                  \
      T y = TRANSFORM;                                                               \
      result[i] = y;                                                                 \
    }                                                                                \
  }                                                                                  \
}

GENERATE_KERNEL2(generate_uniform, float, float a, float b, float, standard_uniform_distribution(_generator), x * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, float, double a, double b, float, standard_uniform_distribution(_generator), x * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, double, double a, double b, double, standard_uniform_distribution(_generator), x * (b-a) + a)

GENERATE_KERNEL2(generate_normal, float, double mean, double stdv, float, normal_distribution(_generator), (x * stdv) + mean)
GENERATE_KERNEL2(generate_normal, double, double mean, double stdv, double, normal_distribution(_generator), (x * stdv) + mean)

GENERATE_KERNEL1(generate_exponential, float, double lambda, float, standard_uniform_distribution(_generator), (float)(-1. / lambda * log(x)))
GENERATE_KERNEL1(generate_exponential, double, double lambda, double, standard_uniform_distribution(_generator), (double)(-1. / lambda * log(x)))

GENERATE_KERNEL2(generate_cauchy, float, double median, double sigma, float, standard_uniform_distribution(_generator), (float)(median + sigma * tan(M_PI*(x-0.5))))
GENERATE_KERNEL2(generate_cauchy, double, double median, double sigma, double, standard_uniform_distribution(_generator), (double)(median + sigma * tan(M_PI*(x-0.5))))

GENERATE_KERNEL2(generate_uniform, at::Half, double a, double b, float, standard_uniform_distribution(_generator), (half_uniform_scale_and_shift(x, a, b)))
GENERATE_KERNEL2(generate_normal, at::Half, double mean, double stdv, float, normal_distribution(_generator), (ScalarConvert<float, at::Half>::to((x * stdv) + mean)))
GENERATE_KERNEL1(generate_exponential, at::Half, double lambda, float, standard_uniform_distribution(_generator), (ScalarConvert<float, at::Half>::to((float)(-1. / lambda * log(x)))))
GENERATE_KERNEL2(generate_cauchy, at::Half, double median, double sigma, float, standard_uniform_distribution(_generator), (ScalarConvert<float, at::Half>::to((float)(median + sigma * tan(M_PI*(x-0.5))))))

#include "generic/THCTensorRandom.cu"
#include "THCGenerateAllTypes.h"

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2
