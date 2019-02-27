#include <ATen/XLAIntType.h>

namespace at {

XLAIntType::XLAIntType() : XLAType() {}

ScalarType XLAIntType::scalarType() const {
  return ScalarType::Int;
}

caffe2::TypeMeta XLAIntType::typeMeta() const {
    return caffe2::TypeMeta::Make<int>();
}

const char * XLAIntType::toString() const {
  return "XLAIntType";
}

TypeID XLAIntType::ID() const {
  return TypeID::XLAInt;
}

} // namespace at
