#include <ATen/XLALongType.h>

namespace at {

XLALongType::XLALongType() : XLAType() {}

ScalarType XLALongType::scalarType() const {
  return ScalarType::Long;
}

caffe2::TypeMeta XLALongType::typeMeta() const {
    return caffe2::TypeMeta::Make<int64_t>();
}

const char * XLALongType::toString() const {
  return "XLALongType";
}

TypeID XLALongType::ID() const {
  return TypeID::XLALong;
}

} // namespace at
