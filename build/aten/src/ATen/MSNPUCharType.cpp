#include <ATen/MSNPUCharType.h>

namespace at {

MSNPUCharType::MSNPUCharType() : MSNPUType() {}

ScalarType MSNPUCharType::scalarType() const {
  return ScalarType::Char;
}

caffe2::TypeMeta MSNPUCharType::typeMeta() const {
    return caffe2::TypeMeta::Make<int8_t>();
}

const char * MSNPUCharType::toString() const {
  return "MSNPUCharType";
}

TypeID MSNPUCharType::ID() const {
  return TypeID::MSNPUChar;
}

} // namespace at
