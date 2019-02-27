// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include <ATen/SparseCPUShortType.h>

// @generated by aten/src/ATen/gen.py

#include <ATen/CPUGenerator.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>


namespace at {

SparseCPUShortType::SparseCPUShortType()
  : CPUTypeDefault(SparseCPUTensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}
ScalarType SparseCPUShortType::scalarType() const {
  return ScalarType::Short;
}
caffe2::TypeMeta SparseCPUShortType::typeMeta() const {
  return caffe2::TypeMeta::Make<int16_t>();
}
Backend SparseCPUShortType::backend() const {
  return Backend::SparseCPU;
}

const char * SparseCPUShortType::toString() const {
  return "SparseCPUShortType";
}

TypeID SparseCPUShortType::ID() const {
  return TypeID::SparseCPUShort;
}

size_t SparseCPUShortType::elementSizeInBytes() const {
  return sizeof(int16_t);
}

Tensor SparseCPUShortType::empty(IntArrayRef size, const TensorOptions & options) const {
    const DeviceGuard device_guard(options.device());
    return at::native::empty_sparse(/* actuals */ size, options);
}
Tensor & SparseCPUShortType::log1p_(Tensor & self) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::log1p_sparse_(/* actuals */ self);
}
Tensor & SparseCPUShortType::log1p_out(Tensor & out, const Tensor & self) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::log1p_out_sparse(/* actuals */ out, self);
}
Tensor SparseCPUShortType::narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::narrow_copy_sparse(/* actuals */ self, dim, start, length);
}
Tensor & SparseCPUShortType::_sparse_add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::add_out_sparse_cpu(/* actuals */ out, self, other, alpha);
}
Tensor & SparseCPUShortType::_sparse_div_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::div_out_sparse_zerodim(/* actuals */ out, self, other);
}
Tensor & SparseCPUShortType::_sparse_div_scalar_out(Tensor & out, const Tensor & self, Scalar other) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::div_out_sparse_scalar(/* actuals */ out, self, other);
}
Tensor & SparseCPUShortType::_sparse_mul_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mul_out_sparse_cpu(/* actuals */ out, self, other);
}
Tensor & SparseCPUShortType::_sparse_mul_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mul_out_sparse_zerodim(/* actuals */ out, self, other);
}
Tensor & SparseCPUShortType::_sparse_mul_scalar_out(Tensor & out, const Tensor & self, Scalar other) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::mul_out_sparse_scalar(/* actuals */ out, self, other);
}
Tensor & SparseCPUShortType::sspaddmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::_sspaddmm_out_cpu(/* actuals */ out, self, mat1, mat2, beta, alpha);
}
Tensor SparseCPUShortType::native_norm(const Tensor & self, Scalar p) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::norm_sparse(/* actuals */ self, p);
}
Tensor SparseCPUShortType::_sparse_sum_backward(const Tensor & grad, const Tensor & self, IntArrayRef dim) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::_sparse_sum_backward_cpu(/* actuals */ grad, self, dim);
}
Tensor SparseCPUShortType::native_clone(const Tensor & self) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::clone_sparse(/* actuals */ self);
}
Tensor & SparseCPUShortType::native_resize_as_(Tensor & self, const Tensor & the_template) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::resize_as_sparse_(/* actuals */ self, the_template);
}
Tensor & SparseCPUShortType::native_pow_out(Tensor & out, const Tensor & self, Scalar exponent) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::pow_out_sparse_scalar(/* actuals */ out, self, exponent);
}
Tensor SparseCPUShortType::native_pow(const Tensor & self, Scalar exponent) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::pow_sparse_scalar(/* actuals */ self, exponent);
}
Tensor & SparseCPUShortType::native_zero_(Tensor & self) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::zero_sparse_(/* actuals */ self);
}
Tensor SparseCPUShortType::_sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) const {
    const DeviceGuard device_guard(options.device());
    return at::native::new_with_dims_sparse(/* actuals */ sparse_dim, dense_dim, size, options);
}
Tensor SparseCPUShortType::_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) const {
    const DeviceGuard device_guard(options.device());
    return at::native::new_with_dims_and_tensor_sparse(/* actuals */ sparse_dim, dense_dim, size, indices, values, options);
}
Tensor & SparseCPUShortType::sparse_resize_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::sparse_resize_(/* actuals */ self, size, sparse_dim, dense_dim);
}
Tensor & SparseCPUShortType::sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::sparse_resize_and_clear_(/* actuals */ self, size, sparse_dim, dense_dim);
}
Tensor SparseCPUShortType::to_dense(const Tensor & self) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::sparse_to_dense(/* actuals */ self);
}
int64_t SparseCPUShortType::sparse_dim(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::sparse_dim_sparse(/* actuals */ self);
}
int64_t SparseCPUShortType::dense_dim(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::dense_dim_sparse(/* actuals */ self);
}
int64_t SparseCPUShortType::_nnz(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::_nnz_sparse(/* actuals */ self);
}
Tensor SparseCPUShortType::coalesce(const Tensor & self) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::coalesce_sparse_cpu(/* actuals */ self);
}
bool SparseCPUShortType::is_coalesced(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::is_coalesced_sparse(/* actuals */ self);
}
Tensor SparseCPUShortType::_indices(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::_indices_sparse(/* actuals */ self);
}
Tensor SparseCPUShortType::_values(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::_values_sparse(/* actuals */ self);
}
Tensor & SparseCPUShortType::_coalesced_(Tensor & self, bool coalesced) const {
    // DeviceGuard omitted
    return at::native::_coalesced_sparse_(/* actuals */ self, coalesced);
}
Tensor SparseCPUShortType::indices(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::indices_sparse(/* actuals */ self);
}
Tensor SparseCPUShortType::values(const Tensor & self) const {
    // DeviceGuard omitted
    return at::native::values_sparse(/* actuals */ self);
}
Tensor & SparseCPUShortType::hspmm_out(Tensor & out, const Tensor & mat1, const Tensor & mat2) const {
    const OptionalDeviceGuard device_guard(device_of(out));
    return at::native::hspmm_out_sparse_cpu(/* actuals */ out, mat1, mat2);
}
Tensor SparseCPUShortType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    const OptionalDeviceGuard device_guard(device_of(mat1));
    return at::native::hspmm_sparse_cpu(/* actuals */ mat1, mat2);
}
Tensor & SparseCPUShortType::copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking) const {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::copy_sparse_(/* actuals */ self, src, non_blocking);
}

}
