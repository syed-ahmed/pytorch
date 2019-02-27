#include <ATen/MSNPUByteType.h>

namespace at {

MSNPUByteType::MSNPUByteType() : MSNPUType() {}

ScalarType MSNPUByteType::scalarType() const {
  return ScalarType::Byte;
}

caffe2::TypeMeta MSNPUByteType::typeMeta() const {
    return caffe2::TypeMeta::Make<uint8_t>();
}

const char * MSNPUByteType::toString() const {
  return "MSNPUByteType";
}

TypeID MSNPUByteType::ID() const {
  return TypeID::MSNPUByte;
}

} // namespace at
