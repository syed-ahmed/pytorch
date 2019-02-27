#include <ATen/MSNPULongType.h>

namespace at {

MSNPULongType::MSNPULongType() : MSNPUType() {}

ScalarType MSNPULongType::scalarType() const {
  return ScalarType::Long;
}

caffe2::TypeMeta MSNPULongType::typeMeta() const {
    return caffe2::TypeMeta::Make<int64_t>();
}

const char * MSNPULongType::toString() const {
  return "MSNPULongType";
}

TypeID MSNPULongType::ID() const {
  return TypeID::MSNPULong;
}

} // namespace at
