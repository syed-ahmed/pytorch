#include <ATen/XLAHalfType.h>

namespace at {

XLAHalfType::XLAHalfType() : XLAType() {}

ScalarType XLAHalfType::scalarType() const {
  return ScalarType::Half;
}

caffe2::TypeMeta XLAHalfType::typeMeta() const {
    return caffe2::TypeMeta::Make<Half>();
}

const char * XLAHalfType::toString() const {
  return "XLAHalfType";
}

TypeID XLAHalfType::ID() const {
  return TypeID::XLAHalf;
}

} // namespace at
