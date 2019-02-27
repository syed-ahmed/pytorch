#include <ATen/MSNPUIntType.h>

namespace at {

MSNPUIntType::MSNPUIntType() : MSNPUType() {}

ScalarType MSNPUIntType::scalarType() const {
  return ScalarType::Int;
}

caffe2::TypeMeta MSNPUIntType::typeMeta() const {
    return caffe2::TypeMeta::Make<int>();
}

const char * MSNPUIntType::toString() const {
  return "MSNPUIntType";
}

TypeID MSNPUIntType::ID() const {
  return TypeID::MSNPUInt;
}

} // namespace at
