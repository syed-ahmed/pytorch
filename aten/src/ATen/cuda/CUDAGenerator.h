#pragma once

// need this for THC random tensors. 
// should be removed when moving random tensors to ATen
#include <THC/THC.h>

#include "ATen/core/Generator.h"
#include "ATen/cuda/ATenCUDAGeneral.h"
#include "ATen/DeviceGuard.h"
#include "ATen/cuda/CUDAContext.h"

namespace at {

namespace detail {

// API (for internal use)
AT_CUDA_API Generator& CUDAGenerator_getDefaultGenerator(int64_t device = -1);
AT_CUDA_API Generator& CUDAGenerator_createGenerator(int64_t device = -1);

} // namespace detail

struct AT_CUDA_API CUDAGenerator : public Generator {

  AT_HOST_DEVICE CUDAGenerator(GeneratorState* state_in)
  : state_{state_in} { }

  // Getter and Setter
  virtual AT_HOST_DEVICE GeneratorState* getState() override;
  virtual AT_HOST_DEVICE void setState(GeneratorState* state_in) override;

  // Methods
  virtual AT_HOST_DEVICE uint32_t seed() override;
  virtual AT_HOST_DEVICE uint32_t getStartingSeed() override;
  virtual AT_HOST_DEVICE void manualSeed(uint32_t seed) override;
  virtual AT_HOST_DEVICE uint32_t random() override;
  virtual AT_HOST_DEVICE uint64_t random64() override;
  
  // CUDA Generator specific methods
  virtual AT_HOST_DEVICE void setNormalDistState(double x, double rho) override;
  virtual AT_HOST_DEVICE void setNormalDistValid(int valid_flag) override;
  virtual AT_HOST_DEVICE uint32_t seedAll() override;
  virtual AT_HOST_DEVICE void manualSeedAll(uint32_t seed) override;
  virtual AT_HOST_DEVICE std::pair<uint64_t, uint64_t> next_philox_seed(uint64_t increment) override;

private:
  GeneratorState* state_ = nullptr;
};

} // namespace at
