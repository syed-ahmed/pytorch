#include <ATen/XLAShortType.h>

namespace at {

XLAShortType::XLAShortType() : XLAType() {}

ScalarType XLAShortType::scalarType() const {
  return ScalarType::Short;
}

caffe2::TypeMeta XLAShortType::typeMeta() const {
    return caffe2::TypeMeta::Make<int16_t>();
}

const char * XLAShortType::toString() const {
  return "XLAShortType";
}

TypeID XLAShortType::ID() const {
  return TypeID::XLAShort;
}

} // namespace at
