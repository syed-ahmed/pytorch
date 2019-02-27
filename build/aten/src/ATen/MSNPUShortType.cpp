#include <ATen/MSNPUShortType.h>

namespace at {

MSNPUShortType::MSNPUShortType() : MSNPUType() {}

ScalarType MSNPUShortType::scalarType() const {
  return ScalarType::Short;
}

caffe2::TypeMeta MSNPUShortType::typeMeta() const {
    return caffe2::TypeMeta::Make<int16_t>();
}

const char * MSNPUShortType::toString() const {
  return "MSNPUShortType";
}

TypeID MSNPUShortType::ID() const {
  return TypeID::MSNPUShort;
}

} // namespace at
