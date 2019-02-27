#include <ATen/MSNPUBoolType.h>

namespace at {

MSNPUBoolType::MSNPUBoolType() : MSNPUType() {}

ScalarType MSNPUBoolType::scalarType() const {
  return ScalarType::Bool;
}

caffe2::TypeMeta MSNPUBoolType::typeMeta() const {
    return caffe2::TypeMeta::Make<uint8_t>();
}

const char * MSNPUBoolType::toString() const {
  return "MSNPUBoolType";
}

TypeID MSNPUBoolType::ID() const {
  return TypeID::MSNPUBool;
}

} // namespace at
