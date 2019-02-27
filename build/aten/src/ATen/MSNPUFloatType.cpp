#include <ATen/MSNPUFloatType.h>

namespace at {

MSNPUFloatType::MSNPUFloatType() : MSNPUType() {}

ScalarType MSNPUFloatType::scalarType() const {
  return ScalarType::Float;
}

caffe2::TypeMeta MSNPUFloatType::typeMeta() const {
    return caffe2::TypeMeta::Make<float>();
}

const char * MSNPUFloatType::toString() const {
  return "MSNPUFloatType";
}

TypeID MSNPUFloatType::ID() const {
  return TypeID::MSNPUFloat;
}

} // namespace at
