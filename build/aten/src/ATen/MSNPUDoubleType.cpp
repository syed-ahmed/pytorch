#include <ATen/MSNPUDoubleType.h>

namespace at {

MSNPUDoubleType::MSNPUDoubleType() : MSNPUType() {}

ScalarType MSNPUDoubleType::scalarType() const {
  return ScalarType::Double;
}

caffe2::TypeMeta MSNPUDoubleType::typeMeta() const {
    return caffe2::TypeMeta::Make<double>();
}

const char * MSNPUDoubleType::toString() const {
  return "MSNPUDoubleType";
}

TypeID MSNPUDoubleType::ID() const {
  return TypeID::MSNPUDouble;
}

} // namespace at
