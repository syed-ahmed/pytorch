#pragma once

// @generated by aten/src/ATen/gen.py

#include <ATen/CPUTypeDefault.h>
#include <ATen/Context.h>
#include <ATen/Utils.h>



#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct SparseCPUBoolType final : public CPUTypeDefault {
  explicit SparseCPUBoolType();
  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual Backend backend() const override;
  virtual const char * toString() const override;
  virtual size_t elementSizeInBytes() const override;
  virtual TypeID ID() const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;

};

} // namespace at
