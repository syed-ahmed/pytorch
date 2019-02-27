#include <ATen/MSNPUHalfType.h>

namespace at {

MSNPUHalfType::MSNPUHalfType() : MSNPUType() {}

ScalarType MSNPUHalfType::scalarType() const {
  return ScalarType::Half;
}

caffe2::TypeMeta MSNPUHalfType::typeMeta() const {
    return caffe2::TypeMeta::Make<Half>();
}

const char * MSNPUHalfType::toString() const {
  return "MSNPUHalfType";
}

TypeID MSNPUHalfType::ID() const {
  return TypeID::MSNPUHalf;
}

} // namespace at
