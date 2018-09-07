#include <ATen/CPUTypeDefault.h>

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>

namespace at {

Allocator* CPUTypeDefault::allocator() const {
  return getCPUAllocator();
}

Device CPUTypeDefault::getDeviceFromPtr(void * data) const {
  return DeviceType::CPU;
}

} // namespace at
