#pragma once

#include "ATen/ATen.h"

void manual_seed(uint64_t seed, at::DeviceType backend) {
  if (backend == at::kCPU) {
    at::Generator & cpu_gen = at::globalContext().getDefaultGenerator(at::kCPU);
    cpu_gen.manualSeed(seed);
  } else if (backend == at::kCUDA && at::hasCUDA()) {
    at::Generator & cuda_gen = at::globalContext().getDefaultGenerator(at::kCUDA);
    cuda_gen.manualSeed(seed);
  }
}
