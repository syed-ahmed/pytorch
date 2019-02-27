#include <ATen/XLACharType.h>

namespace at {

XLACharType::XLACharType() : XLAType() {}

ScalarType XLACharType::scalarType() const {
  return ScalarType::Char;
}

caffe2::TypeMeta XLACharType::typeMeta() const {
    return caffe2::TypeMeta::Make<int8_t>();
}

const char * XLACharType::toString() const {
  return "XLACharType";
}

TypeID XLACharType::ID() const {
  return TypeID::XLAChar;
}

} // namespace at
