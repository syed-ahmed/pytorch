#include <ATen/XLADoubleType.h>

namespace at {

XLADoubleType::XLADoubleType() : XLAType() {}

ScalarType XLADoubleType::scalarType() const {
  return ScalarType::Double;
}

caffe2::TypeMeta XLADoubleType::typeMeta() const {
    return caffe2::TypeMeta::Make<double>();
}

const char * XLADoubleType::toString() const {
  return "XLADoubleType";
}

TypeID XLADoubleType::ID() const {
  return TypeID::XLADouble;
}

} // namespace at
