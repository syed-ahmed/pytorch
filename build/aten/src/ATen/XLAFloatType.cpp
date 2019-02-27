#include <ATen/XLAFloatType.h>

namespace at {

XLAFloatType::XLAFloatType() : XLAType() {}

ScalarType XLAFloatType::scalarType() const {
  return ScalarType::Float;
}

caffe2::TypeMeta XLAFloatType::typeMeta() const {
    return caffe2::TypeMeta::Make<float>();
}

const char * XLAFloatType::toString() const {
  return "XLAFloatType";
}

TypeID XLAFloatType::ID() const {
  return TypeID::XLAFloat;
}

} // namespace at
