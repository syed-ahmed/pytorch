#include <ATen/XLAByteType.h>

namespace at {

XLAByteType::XLAByteType() : XLAType() {}

ScalarType XLAByteType::scalarType() const {
  return ScalarType::Byte;
}

caffe2::TypeMeta XLAByteType::typeMeta() const {
    return caffe2::TypeMeta::Make<uint8_t>();
}

const char * XLAByteType::toString() const {
  return "XLAByteType";
}

TypeID XLAByteType::ID() const {
  return TypeID::XLAByte;
}

} // namespace at
