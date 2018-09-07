#pragma once

#include <stdint.h>
#include <memory>
#include <mutex>
#include <random>
#include <deque>
#include <atomic>
#include <typeinfo>
#include <utility>

#include "ATen/core/ATenGeneral.h"
#include "ATen/core/Error.h"
#include "ATen/core/Macros.h"

#ifdef __CUDACC__
#include "ATen/PhiloxRNGEngine.h"
#endif

/*
* A Generator interface. Specific implementations in ATen/CPUGenerator.* and
* ATen/cuda/CUDAGenerator.*
*/

/*
* Generator note.
* A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm to 
* generate a seemingly random sequence of numbers, that may be later be used in creating 
* a random distribution. Such an engine almost always maintains a state and requires a
* seed to start off the creation of random numbers. Often times, users have
* encountered that it could be beneficial to be able to create, retain, and destroy 
* generator states and also be able to have control over the seed value.
*
* A Generator in ATen gives the ability to manage the lifecycle of a PRNG engine. In
* addition it provides the user with the ability to control the seeds in a PRNG engine.
*
* By default, there is one generator state per device, and a device's generator state is 
* lazily created. A user can use the torch.Generator() api to create their own generator.
*/

/*
* A Generator State object that contains a generator engine and other state variables.
*/

struct GeneratorState {
  GeneratorState () = default;

  std::mutex mutex;
  int64_t device = -1;
  uint32_t starting_seed; // track the starting seed for a random sequence

  // 32 bit random number generator engines
  #ifdef __CUDACC__
    at::cuda::Philox4_32_10 engine;
    std::atomic<uint32_t> philox_seed_offset;
    // variables for normal distribution
    double normal_x;
    double normal_rho;
    int normal_is_valid;
  #else
    std::mt19937 engine;
  #endif
};

namespace at {

struct CAFFE2_API Generator {
  
  // Getter and Setter
  virtual AT_HOST_DEVICE GeneratorState* getState() = 0;
  virtual AT_HOST_DEVICE void setState(GeneratorState* state_in) = 0;

  // Methods
  virtual AT_HOST_DEVICE uint32_t seed() = 0;
  virtual AT_HOST_DEVICE uint32_t getStartingSeed() = 0;
  virtual AT_HOST_DEVICE void manualSeed(uint32_t seed) = 0;
  virtual AT_HOST_DEVICE uint32_t random() = 0;
  virtual AT_HOST_DEVICE uint64_t random64() = 0;

  // CUDA specific methods
  virtual AT_HOST_DEVICE void setNormalDistState(double x, double rho);
  virtual AT_HOST_DEVICE void setNormalDistValid(int valid_flag);
  virtual AT_HOST_DEVICE uint32_t seedAll();
  virtual AT_HOST_DEVICE void manualSeedAll(uint32_t seed);
  virtual AT_HOST_DEVICE std::pair<uint64_t, uint64_t> next_philox_seed(uint64_t increment);

};

} // namespace at
