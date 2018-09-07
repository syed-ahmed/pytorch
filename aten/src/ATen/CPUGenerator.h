#pragma once

#include "ATen/core/Generator.h"
#include <ATen/core/ATenGeneral.h>
//#include <TH/TH.h>

namespace at {

namespace detail {

// API (for internal use)
CAFFE2_API Generator& CPUGenerator_getDefaultGenerator();
CAFFE2_API Generator& CPUGenerator_createGenerator();

} // namespace detail

struct CPUGenerator : public Generator {

  AT_HOST_DEVICE CPUGenerator(GeneratorState* state_in)
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

private:
  GeneratorState* state_ = nullptr;
};

} // namespace at
