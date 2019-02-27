#include <ATen/XLABoolType.h>

namespace at {

XLABoolType::XLABoolType() : XLAType() {}

ScalarType XLABoolType::scalarType() const {
  return ScalarType::Bool;
}

caffe2::TypeMeta XLABoolType::typeMeta() const {
    return caffe2::TypeMeta::Make<uint8_t>();
}

const char * XLABoolType::toString() const {
  return "XLABoolType";
}

TypeID XLABoolType::ID() const {
  return TypeID::XLABool;
}

} // namespace at
