#include <ATen/XLAType.h>

namespace at {

std::unordered_map<std::string, void *>& XLATypeDispatch::get_fn_table() {
  static std::unordered_map<std::string, void *> fn_table;
  return fn_table;
}

XLAType::XLAType()
  : TypeDefault(XLATensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}

Allocator* XLAType::allocator() const {
  AT_ERROR("allocator is not implemented for XLAType");
}

Device XLAType::getDeviceFromPtr(void * data) const {
  return DeviceType::XLA;
}

std::unique_ptr<Generator> XLAType::generator() const {
  AT_ERROR("generator is not implemented for XLAType");
}

Backend XLAType::backend() const {
  return Backend::XLA;
}

const char * XLAType::toString() const {
  return "XLAType";
}

TypeID XLAType::ID() const {
  return TypeID::XLA;
}

Tensor & XLAType::_th_set_(Tensor & self, Storage source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Storage)>("_th_set_(Tensor self, Storage source) -> Tensor")(self, source);
}
Tensor & XLAType::_th_set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>("_th_set_(Tensor self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) -> Tensor")(self, source, storage_offset, size, stride);
}
Tensor & XLAType::_th_set_(Tensor & self, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_set_(Tensor self, Tensor source) -> Tensor")(self, source);
}
Tensor & XLAType::_th_set_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_set_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_fill_(Tensor & self, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_fill_(Tensor self, real value) -> Tensor")(self, value);
}
Tensor & XLAType::_th_fill_(Tensor & self, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_fill_(Tensor self, Tensor value) -> Tensor")(self, value);
}
bool XLAType::_th_is_set_to(const Tensor & self, const Tensor & tensor) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &, const Tensor &)>("_th_is_set_to(Tensor self, Tensor tensor) -> bool")(self, tensor);
}
Tensor & XLAType::s__th_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_masked_fill_(Tensor self, ByteTensor mask, real value) -> Tensor")(self, mask, value);
}
Tensor & XLAType::s__th_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_masked_fill_(Tensor self, ByteTensor mask, Tensor value) -> Tensor")(self, mask, value);
}
Tensor & XLAType::_th_masked_fill_bool_(Tensor & self, const Tensor & mask, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_masked_fill_bool_(Tensor self, BoolTensor mask, real value) -> Tensor")(self, mask, value);
}
Tensor & XLAType::_th_masked_fill_bool_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_masked_fill_bool_(Tensor self, BoolTensor mask, Tensor value) -> Tensor")(self, mask, value);
}
Tensor & XLAType::s__th_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_masked_scatter_(Tensor self, ByteTensor mask, Tensor source) -> Tensor")(self, mask, source);
}
Tensor & XLAType::_th_masked_scatter_bool_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_masked_scatter_bool_(Tensor self, BoolTensor mask, Tensor source) -> Tensor")(self, mask, source);
}
Tensor & XLAType::s__th_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_masked_select_out(Tensor result, Tensor self, ByteTensor mask) -> Tensor")(result, self, mask);
}
Tensor XLAType::s__th_masked_select(const Tensor & self, const Tensor & mask) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_masked_select(Tensor self, ByteTensor mask) -> Tensor")(self, mask);
}
Tensor & XLAType::_th_masked_select_bool_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_masked_select_bool_out(Tensor result, Tensor self, BoolTensor mask) -> Tensor")(result, self, mask);
}
Tensor XLAType::_th_masked_select_bool(const Tensor & self, const Tensor & mask) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_masked_select_bool(Tensor self, BoolTensor mask) -> Tensor")(self, mask);
}
Tensor & XLAType::_th_nonzero_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_nonzero_out(IndexTensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_nonzero(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_nonzero(Tensor self) -> Tensor")(self);
}
Tensor XLAType::_th_clone(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_clone(Tensor self) -> Tensor")(self);
}
Tensor XLAType::_th_view(const Tensor & self, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_th_view(Tensor self, IntArrayRef size) -> Tensor")(self, size);
}
Tensor & XLAType::_th_resize_as_(Tensor & self, const Tensor & the_template) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_resize_as_(Tensor self, Tensor the_template) -> Tensor")(self, the_template);
}
Tensor & XLAType::_th_index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &)>("_th_index_select_out(Tensor result, Tensor self, int64_t dim, IndexTensor index) -> Tensor")(result, self, dim, index);
}
Tensor XLAType::_th_index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &)>("_th_index_select(Tensor self, int64_t dim, IndexTensor index) -> Tensor")(self, dim, index);
}
Tensor & XLAType::_th_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("_th_index_copy_(Tensor self, int64_t dim, IndexTensor index, Tensor source) -> Tensor")(self, dim, index, source);
}
Tensor & XLAType::_th_take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_take_out(Tensor result, Tensor self, IndexTensor index) -> Tensor")(result, self, index);
}
Tensor XLAType::_th_take(const Tensor & self, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_take(Tensor self, IndexTensor index) -> Tensor")(self, index);
}
Tensor & XLAType::_th_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, bool)>("_th_put_(Tensor self, IndexTensor index, Tensor source, bool accumulate) -> Tensor")(self, index, source, accumulate);
}
Tensor & XLAType::_th_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("_th_index_add_(Tensor self, int64_t dim, IndexTensor index, Tensor source) -> Tensor")(self, dim, index, source);
}
Tensor & XLAType::_th_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, Scalar)>("_th_index_fill_(Tensor self, int64_t dim, IndexTensor index, real value) -> Tensor")(self, dim, index, value);
}
Tensor & XLAType::_th_index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("_th_index_fill_(Tensor self, int64_t dim, IndexTensor index, Tensor value) -> Tensor")(self, dim, index, value);
}
Tensor & XLAType::_th_unfold_out(Tensor & result, const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, int64_t, int64_t)>("_th_unfold_out(Tensor result, Tensor self, int64_t dimension, int64_t size, int64_t step) -> Tensor")(result, self, dimension, size, step);
}
Tensor XLAType::_th_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t)>("_th_unfold(Tensor self, int64_t dimension, int64_t size, int64_t step) -> Tensor")(self, dimension, size, step);
}
Tensor & XLAType::_th_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("_th_scatter_(Tensor self, int64_t dim, IndexTensor index, Tensor src) -> Tensor")(self, dim, index, src);
}
Tensor & XLAType::_th_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, Scalar)>("_th_scatter_(Tensor self, int64_t dim, IndexTensor index, real value) -> Tensor")(self, dim, index, value);
}
Tensor & XLAType::_th_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("_th_scatter_add_(Tensor self, int64_t dim, IndexTensor index, Tensor src) -> Tensor")(self, dim, index, src);
}
Tensor & XLAType::_th_gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &)>("_th_gather_out(Tensor result, Tensor self, int64_t dim, IndexTensor index) -> Tensor")(result, self, dim, index);
}
Tensor XLAType::_th_gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &)>("_th_gather(Tensor self, int64_t dim, IndexTensor index) -> Tensor")(self, dim, index);
}
bool XLAType::_th_equal(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &, const Tensor &)>("_th_equal(Tensor self, Tensor other) -> bool")(self, other);
}
Tensor & XLAType::_th_and_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_and_out(Tensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_and(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_and(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_and_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_and_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_and(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_and(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_iand_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_iand_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_iand_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_iand_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_or_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_or_out(Tensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_or(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_or(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_or_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_or_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_or(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_or(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_ior_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_ior_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_ior_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_ior_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_xor_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_xor_out(Tensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_xor(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_xor(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_xor_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_xor_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_xor(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_xor(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_ixor_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_ixor_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_ixor_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_ixor_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_lshift_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_lshift_out(Tensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_lshift(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_lshift(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_lshift_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_lshift_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_lshift(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_lshift(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_ilshift_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_ilshift_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_ilshift_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_ilshift_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_rshift_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_rshift_out(Tensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_rshift(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_rshift(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_rshift_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_rshift_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_rshift(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_rshift(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_irshift_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_irshift_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_irshift_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_irshift_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_lt_out(ByteTensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_lt(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_lt(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_lt_out(ByteTensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_lt(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_lt(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_lt_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_lt_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_lt_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_lt_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_gt_out(ByteTensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_gt(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_gt(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_gt_out(ByteTensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_gt(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_gt(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_gt_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_gt_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_gt_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_gt_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_le_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_le_out(ByteTensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_le(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_le(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_le_out(ByteTensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_le(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_le(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_le_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_le_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_le_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_le_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_ge_out(ByteTensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_ge(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_ge(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_ge_out(ByteTensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_ge(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_ge(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_ge_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_ge_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_ge_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_ge_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_eq_out(ByteTensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_eq(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_eq(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_eq_out(ByteTensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_eq(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_eq(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_eq_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_eq_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_eq_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_eq_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_ne_out(ByteTensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_ne(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_ne(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_ne_out(ByteTensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_ne(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_ne(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_ne_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_ne_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_ne_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_ne_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_min_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_min(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_min(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::_th_min(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_min(Tensor self) -> Tensor")(self);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("_th_min_out(Tensor min, IndexTensor min_indices, Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(min, min_indices, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> XLAType::_th_min(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("_th_min(Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, dim, keepdim);
}
Tensor & XLAType::s__th_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_max_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_max(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_max(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::_th_max(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_max(Tensor self) -> Tensor")(self);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("_th_max_out(Tensor max, IndexTensor max_indices, Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(max, max_indices, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> XLAType::_th_max(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("_th_max(Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("_th_mode_out(Tensor values, IndexTensor indices, Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> XLAType::_th_mode(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("_th_mode(Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("_th_sort_out(Tensor values, IndexTensor indices, Tensor self, int64_t dim, bool descending) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, dim, descending);
}
std::tuple<Tensor,Tensor> XLAType::_th_sort(const Tensor & self, int64_t dim, bool descending) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("_th_sort(Tensor self, int64_t dim, bool descending) -> std::tuple<Tensor,Tensor>")(self, dim, descending);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>("_th_topk_out(Tensor values, IndexTensor indices, Tensor self, int64_t k, int64_t dim, bool largest, bool sorted) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, k, dim, largest, sorted);
}
std::tuple<Tensor,Tensor> XLAType::_th_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, int64_t, bool, bool)>("_th_topk(Tensor self, int64_t k, int64_t dim, bool largest, bool sorted) -> std::tuple<Tensor,Tensor>")(self, k, dim, largest, sorted);
}
Tensor & XLAType::_th_abs_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_abs_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_abs(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_abs(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_sigmoid_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_sigmoid_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_sigmoid(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_sigmoid(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_log_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_log_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_log(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_log(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_log10_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_log10_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_log10(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_log10(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_log1p_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_log1p_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_log1p(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_log1p(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_log2_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_log2_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_log2(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_log2(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_lgamma_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_lgamma_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_lgamma(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_lgamma(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_lgamma_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_lgamma_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_digamma_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_digamma_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_digamma(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_digamma(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_digamma_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_digamma_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_polygamma_out(Tensor & result, int64_t n, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &)>("_th_polygamma_out(Tensor result, int64_t n, Tensor self) -> Tensor")(result, n, self);
}
Tensor XLAType::_th_polygamma(int64_t n, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const Tensor &)>("_th_polygamma(int64_t n, Tensor self) -> Tensor")(n, self);
}
Tensor & XLAType::_th_polygamma_(Tensor & self, int64_t n) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("_th_polygamma_(Tensor self, int64_t n) -> Tensor")(self, n);
}
Tensor & XLAType::_th_exp_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_exp_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_exp(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_exp(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_expm1_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_expm1_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_expm1(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_expm1(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_cos_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_cos_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_cos(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_cos(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_acos_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_acos_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_acos(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_acos(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_cosh_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_cosh_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_cosh(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_cosh(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_sin_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_sin_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_sin(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_sin(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_asin_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_asin_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_asin(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_asin(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_sinh_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_sinh_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_sinh(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_sinh(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_tan_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_tan_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_tan(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_tan(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_atan_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_atan_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_atan(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_atan(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_tanh_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_tanh_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_tanh(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_tanh(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_erf_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_erf_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_erf(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_erf(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_erfc_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_erfc_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_erfc(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_erfc(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_erfinv_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_erfinv_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_erfinv_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_erfinv_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_erfinv(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_erfinv(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_sqrt_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_sqrt_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_sqrt(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_sqrt(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_rsqrt_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_rsqrt_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_rsqrt(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_rsqrt(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_ceil_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_ceil_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_ceil(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_ceil(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_floor_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_floor_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_floor(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_floor(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_round_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_round_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_round(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_round(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_trunc_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_trunc_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_trunc(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_trunc(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_frac_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_frac_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_frac_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_frac_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_frac(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_frac(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool, bool)>("_th_var_out(Tensor result, Tensor self, int64_t dim, bool unbiased, bool keepdim) -> Tensor")(result, self, dim, unbiased, keepdim);
}
Tensor XLAType::_th_var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, bool)>("_th_var(Tensor self, int64_t dim, bool unbiased, bool keepdim) -> Tensor")(self, dim, unbiased, keepdim);
}
Tensor XLAType::_th_var(const Tensor & self, bool unbiased) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_th_var(Tensor self, bool unbiased) -> Tensor")(self, unbiased);
}
Tensor & XLAType::_th_std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool, bool)>("_th_std_out(Tensor result, Tensor self, int64_t dim, bool unbiased, bool keepdim) -> Tensor")(result, self, dim, unbiased, keepdim);
}
Tensor XLAType::_th_std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, bool)>("_th_std(Tensor self, int64_t dim, bool unbiased, bool keepdim) -> Tensor")(self, dim, unbiased, keepdim);
}
Tensor XLAType::_th_std(const Tensor & self, bool unbiased) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_th_std(Tensor self, bool unbiased) -> Tensor")(self, unbiased);
}
Tensor & XLAType::_th_renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, int64_t, Scalar)>("_th_renorm_out(Tensor result, Tensor self, real p, int64_t dim, real maxnorm) -> Tensor")(result, self, p, dim, maxnorm);
}
Tensor XLAType::_th_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, int64_t, Scalar)>("_th_renorm(Tensor self, real p, int64_t dim, real maxnorm) -> Tensor")(self, p, dim, maxnorm);
}
Tensor & XLAType::_th_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, int64_t, Scalar)>("_th_renorm_(Tensor self, real p, int64_t dim, real maxnorm) -> Tensor")(self, p, dim, maxnorm);
}
Tensor XLAType::s__th_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("_th_dist(Tensor self, Tensor other, real p) -> Tensor")(self, other, p);
}
Tensor & XLAType::_th_reciprocal_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_reciprocal_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_reciprocal(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_reciprocal(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_reciprocal_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_reciprocal_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_neg_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_neg_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_neg(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_neg(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_neg_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_neg_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::s__th_atan2_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_atan2_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_atan2(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_atan2(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_atan2_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_atan2_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_pow_out(Tensor & result, const Tensor & self, Scalar exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_pow_out(Tensor result, Tensor self, real exponent) -> Tensor")(result, self, exponent);
}
Tensor XLAType::_th_pow(const Tensor & self, Scalar exponent) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_pow(Tensor self, real exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::s__th_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_pow_out(Tensor result, Tensor self, Tensor exponent) -> Tensor")(result, self, exponent);
}
Tensor XLAType::s__th_pow(const Tensor & self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_pow(Tensor self, Tensor exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::_th_pow_out(Tensor & result, Scalar self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, const Tensor &)>("_th_pow_out(Tensor result, real self, Tensor exponent) -> Tensor")(result, self, exponent);
}
Tensor XLAType::_th_pow(Scalar self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, const Tensor &)>("_th_pow(real self, Tensor exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::_th_pow_(Tensor & self, Scalar exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_pow_(Tensor self, real exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::s__th_pow_(Tensor & self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_pow_(Tensor self, Tensor exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::_th_histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, Scalar, Scalar)>("_th_histc_out(Tensor result, Tensor self, int64_t bins, real min, real max) -> Tensor")(result, self, bins, min, max);
}
Tensor XLAType::_th_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, Scalar, Scalar)>("_th_histc(Tensor self, int64_t bins, real min, real max) -> Tensor")(self, bins, min, max);
}
Tensor & XLAType::_th_zero_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_zero_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("_th_cumsum_out(Tensor result, Tensor self, int64_t dim) -> Tensor")(result, self, dim);
}
Tensor XLAType::_th_cumsum(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("_th_cumsum(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::_th_cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("_th_cumprod_out(Tensor result, Tensor self, int64_t dim) -> Tensor")(result, self, dim);
}
Tensor XLAType::_th_cumprod(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("_th_cumprod(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::_th_sign_out(Tensor & result, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_sign_out(Tensor result, Tensor self) -> Tensor")(result, self);
}
Tensor XLAType::_th_sign(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_sign(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_sign_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("_th_sign_(Tensor self) -> Tensor")(self);
}
Tensor XLAType::_th_trace(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_trace(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_fmod_out(Tensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_fmod(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_fmod(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_fmod_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_fmod(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_fmod(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_fmod_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_fmod_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_fmod_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_fmod_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_remainder_out(Tensor result, Tensor self, real other) -> Tensor")(result, self, other);
}
Tensor XLAType::_th_remainder(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_remainder(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_remainder_out(Tensor result, Tensor self, Tensor other) -> Tensor")(result, self, other);
}
Tensor XLAType::s__th_remainder(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_remainder(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_remainder_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_th_remainder_(Tensor self, real other) -> Tensor")(self, other);
}
Tensor & XLAType::s__th_remainder_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_remainder_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::_th_clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar)>("_th_clamp_out(Tensor result, Tensor self, real min, real max) -> Tensor")(result, self, min, max);
}
Tensor XLAType::_th_clamp(const Tensor & self, Scalar min, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("_th_clamp(Tensor self, real min, real max) -> Tensor")(self, min, max);
}
Tensor & XLAType::_th_clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_clamp_min_out(Tensor result, Tensor self, real min) -> Tensor")(result, self, min);
}
Tensor XLAType::_th_clamp_min(const Tensor & self, Scalar min) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_clamp_min(Tensor self, real min) -> Tensor")(self, min);
}
Tensor & XLAType::_th_clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_th_clamp_max_out(Tensor result, Tensor self, real max) -> Tensor")(result, self, max);
}
Tensor XLAType::_th_clamp_max(const Tensor & self, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_th_clamp_max(Tensor self, real max) -> Tensor")(self, max);
}
Tensor XLAType::_th_dot(const Tensor & self, const Tensor & tensor) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_dot(Tensor self, Tensor tensor) -> Tensor")(self, tensor);
}
Tensor & XLAType::_th_cross_kernel_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("_th_cross_kernel_out(Tensor result, Tensor self, Tensor other, int64_t dim) -> Tensor")(result, self, other, dim);
}
Tensor XLAType::_th_cross_kernel(const Tensor & self, const Tensor & other, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("_th_cross_kernel(Tensor self, Tensor other, int64_t dim) -> Tensor")(self, other, dim);
}
Tensor & XLAType::_th_diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("_th_diag_out(Tensor result, Tensor self, int64_t diagonal) -> Tensor")(result, self, diagonal);
}
Tensor XLAType::_th_diag(const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("_th_diag(Tensor self, int64_t diagonal) -> Tensor")(self, diagonal);
}
Tensor & XLAType::s__th_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addmm_out(Tensor result, Tensor self, Tensor mat1, Tensor mat2, real beta, real alpha) -> Tensor")(result, self, mat1, mat2, beta, alpha);
}
Tensor XLAType::s__th_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addmm(Tensor self, Tensor mat1, Tensor mat2, real beta, real alpha) -> Tensor")(self, mat1, mat2, beta, alpha);
}
Tensor & XLAType::_th_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addmm_(Tensor self, Tensor mat1, Tensor mat2, real beta, real alpha) -> Tensor")(self, mat1, mat2, beta, alpha);
}
Tensor & XLAType::s__th_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addmv_out(Tensor result, Tensor self, Tensor mat, Tensor vec, real beta, real alpha) -> Tensor")(result, self, mat, vec, beta, alpha);
}
Tensor XLAType::s__th_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addmv(Tensor self, Tensor mat, Tensor vec, real beta, real alpha) -> Tensor")(self, mat, vec, beta, alpha);
}
Tensor & XLAType::_th_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addmv_(Tensor self, Tensor mat, Tensor vec, real beta, real alpha) -> Tensor")(self, mat, vec, beta, alpha);
}
Tensor & XLAType::s__th_addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addr_out(Tensor result, Tensor self, Tensor vec1, Tensor vec2, real beta, real alpha) -> Tensor")(result, self, vec1, vec2, beta, alpha);
}
Tensor XLAType::s__th_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addr(Tensor self, Tensor vec1, Tensor vec2, real beta, real alpha) -> Tensor")(self, vec1, vec2, beta, alpha);
}
Tensor & XLAType::_th_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addr_(Tensor self, Tensor vec1, Tensor vec2, real beta, real alpha) -> Tensor")(self, vec1, vec2, beta, alpha);
}
Tensor & XLAType::_th_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_ger_out(Tensor result, Tensor self, Tensor vec2) -> Tensor")(result, self, vec2);
}
Tensor XLAType::_th_ger(const Tensor & self, const Tensor & vec2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_ger(Tensor self, Tensor vec2) -> Tensor")(self, vec2);
}
Tensor & XLAType::_th_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_mv_out(Tensor result, Tensor self, Tensor vec) -> Tensor")(result, self, vec);
}
Tensor XLAType::_th_mv(const Tensor & self, const Tensor & vec) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_mv(Tensor self, Tensor vec) -> Tensor")(self, vec);
}
Tensor & XLAType::_th_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_mm_out(Tensor result, Tensor self, Tensor mat2) -> Tensor")(result, self, mat2);
}
Tensor XLAType::_th_mm(const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_mm(Tensor self, Tensor mat2) -> Tensor")(self, mat2);
}
Tensor & XLAType::_th_bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_bmm_out(Tensor result, Tensor self, Tensor mat2) -> Tensor")(result, self, mat2);
}
Tensor XLAType::_th_bmm(const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_bmm(Tensor self, Tensor mat2) -> Tensor")(self, mat2);
}
Tensor & XLAType::s__th_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addbmm_out(Tensor result, Tensor self, Tensor batch1, Tensor batch2, real beta, real alpha) -> Tensor")(result, self, batch1, batch2, beta, alpha);
}
Tensor XLAType::s__th_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addbmm(Tensor self, Tensor batch1, Tensor batch2, real beta, real alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::_th_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_addbmm_(Tensor self, Tensor batch1, Tensor batch2, real beta, real alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::s__th_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_baddbmm_out(Tensor result, Tensor self, Tensor batch1, Tensor batch2, real beta, real alpha) -> Tensor")(result, self, batch1, batch2, beta, alpha);
}
Tensor XLAType::s__th_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_th_baddbmm(Tensor self, Tensor batch1, Tensor batch2, real beta, real alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::s__th_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>("_th_addcmul_out(Tensor result, Tensor self, Tensor tensor1, Tensor tensor2, real value) -> Tensor")(result, self, tensor1, tensor2, value);
}
Tensor XLAType::s__th_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar)>("_th_addcmul(Tensor self, Tensor tensor1, Tensor tensor2, real value) -> Tensor")(self, tensor1, tensor2, value);
}
Tensor & XLAType::s__th_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("_th_addcmul_(Tensor self, Tensor tensor1, Tensor tensor2, real value) -> Tensor")(self, tensor1, tensor2, value);
}
Tensor & XLAType::s__th_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>("_th_addcdiv_out(Tensor result, Tensor self, Tensor tensor1, Tensor tensor2, real value) -> Tensor")(result, self, tensor1, tensor2, value);
}
Tensor XLAType::s__th_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar)>("_th_addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, real value) -> Tensor")(self, tensor1, tensor2, value);
}
Tensor & XLAType::s__th_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("_th_addcdiv_(Tensor self, Tensor tensor1, Tensor tensor2, real value) -> Tensor")(self, tensor1, tensor2, value);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &)>("_th_gels_out(Tensor res1, Tensor res2, Tensor self, Tensor A) -> std::tuple<Tensor &,Tensor &>")(res1, res2, self, A);
}
std::tuple<Tensor,Tensor> XLAType::_th_gels(const Tensor & self, const Tensor & A) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &)>("_th_gels(Tensor self, Tensor A) -> std::tuple<Tensor,Tensor>")(self, A);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, bool, bool)>("_th_symeig_out(Tensor res1, Tensor res2, Tensor self, bool eigenvectors, bool upper) -> std::tuple<Tensor &,Tensor &>")(res1, res2, self, eigenvectors, upper);
}
std::tuple<Tensor,Tensor> XLAType::_th_symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, bool, bool)>("_th_symeig(Tensor self, bool eigenvectors, bool upper) -> std::tuple<Tensor,Tensor>")(self, eigenvectors, upper);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, bool)>("_th_eig_out(Tensor res1, Tensor res2, Tensor self, bool eigenvectors) -> std::tuple<Tensor &,Tensor &>")(res1, res2, self, eigenvectors);
}
std::tuple<Tensor,Tensor> XLAType::_th_eig(const Tensor & self, bool eigenvectors) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, bool)>("_th_eig(Tensor self, bool eigenvectors) -> std::tuple<Tensor,Tensor>")(self, eigenvectors);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_th_svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some, bool compute_uv) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, bool, bool)>("_th_svd_out(Tensor res1, Tensor res2, Tensor res3, Tensor self, bool some, bool compute_uv) -> std::tuple<Tensor &,Tensor &,Tensor &>")(res1, res2, res3, self, some, compute_uv);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_th_svd(const Tensor & self, bool some, bool compute_uv) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, bool, bool)>("_th_svd(Tensor self, bool some, bool compute_uv) -> std::tuple<Tensor,Tensor,Tensor>")(self, some, compute_uv);
}
Tensor & XLAType::_th_getri_single_out(Tensor & output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_getri_single_out(Tensor output, Tensor self) -> Tensor")(output, self);
}
Tensor XLAType::_th_getri_single(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_getri_single(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_potri_out(Tensor & output, const Tensor & self, bool upper) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, bool)>("_th_potri_out(Tensor output, Tensor self, bool upper) -> Tensor")(output, self, upper);
}
Tensor XLAType::_th_potri(const Tensor & self, bool upper) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_th_potri(Tensor self, bool upper) -> Tensor")(self, upper);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, bool, Scalar)>("_th_pstrf_out(Tensor res1, IntegerTensor res2, Tensor self, bool upper, real tol) -> std::tuple<Tensor &,Tensor &>")(res1, res2, self, upper, tol);
}
std::tuple<Tensor,Tensor> XLAType::_th_pstrf(const Tensor & self, bool upper, Scalar tol) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, bool, Scalar)>("_th_pstrf(Tensor self, bool upper, real tol) -> std::tuple<Tensor,Tensor>")(self, upper, tol);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &)>("_th_qr_out(Tensor res1, Tensor res2, Tensor self) -> std::tuple<Tensor &,Tensor &>")(res1, res2, self);
}
std::tuple<Tensor,Tensor> XLAType::_th_qr(const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("_th_qr(Tensor self) -> std::tuple<Tensor,Tensor>")(self);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &)>("_th_geqrf_out(Tensor res1, Tensor res2, Tensor self) -> std::tuple<Tensor &,Tensor &>")(res1, res2, self);
}
std::tuple<Tensor,Tensor> XLAType::_th_geqrf(const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("_th_geqrf(Tensor self) -> std::tuple<Tensor,Tensor>")(self);
}
Tensor & XLAType::_th_orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_th_orgqr_out(Tensor result, Tensor self, Tensor input2) -> Tensor")(result, self, input2);
}
Tensor XLAType::_th_orgqr(const Tensor & self, const Tensor & input2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_th_orgqr(Tensor self, Tensor input2) -> Tensor")(self, input2);
}
Tensor & XLAType::_th_ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool)>("_th_ormqr_out(Tensor result, Tensor self, Tensor input2, Tensor input3, bool left, bool transpose) -> Tensor")(result, self, input2, input3, left, transpose);
}
Tensor XLAType::_th_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, bool, bool)>("_th_ormqr(Tensor self, Tensor input2, Tensor input3, bool left, bool transpose) -> Tensor")(self, input2, input3, left, transpose);
}
Tensor & XLAType::_th_btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("_th_btrisolve_out(Tensor result, Tensor self, Tensor LU_data, IntegerTensor LU_pivots) -> Tensor")(result, self, LU_data, LU_pivots);
}
Tensor XLAType::_th_btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("_th_btrisolve(Tensor self, Tensor LU_data, IntegerTensor LU_pivots) -> Tensor")(self, LU_data, LU_pivots);
}
Tensor & XLAType::_th_random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, int64_t, Generator *)>("_th_random_(Tensor self, int64_t from, int64_t to, Generator* generator) -> Tensor")(self, from, to, generator);
}
Tensor & XLAType::_th_random_(Tensor & self, int64_t to, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, Generator *)>("_th_random_(Tensor self, int64_t to, Generator* generator) -> Tensor")(self, to, generator);
}
Tensor & XLAType::_th_random_(Tensor & self, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Generator *)>("_th_random_(Tensor self, Generator* generator) -> Tensor")(self, generator);
}
std::tuple<Tensor &,Tensor &> XLAType::_th_multinomial_alias_setup_out(Tensor & J, Tensor & q, const Tensor & probs) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &)>("_th_multinomial_alias_setup_out(IndexTensor J, Tensor q, Tensor probs) -> std::tuple<Tensor &,Tensor &>")(J, q, probs);
}
std::tuple<Tensor,Tensor> XLAType::_th_multinomial_alias_setup(const Tensor & probs) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("_th_multinomial_alias_setup(Tensor probs) -> std::tuple<Tensor,Tensor>")(probs);
}
Tensor & XLAType::_th_multinomial_alias_draw_out(Tensor & result, const Tensor & q, const Tensor & J, int64_t num_samples, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t, Generator *)>("_th_multinomial_alias_draw_out(IndexTensor result, Tensor q, IndexTensor J, int64_t num_samples, Generator* generator) -> Tensor")(result, q, J, num_samples, generator);
}
Tensor XLAType::_th_multinomial_alias_draw(const Tensor & q, const Tensor & J, int64_t num_samples, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, Generator *)>("_th_multinomial_alias_draw(Tensor q, IndexTensor J, int64_t num_samples, Generator* generator) -> Tensor")(q, J, num_samples, generator);
}
Tensor & XLAType::_th_multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool, Generator *)>("_th_multinomial_out(IndexTensor result, Tensor self, int64_t num_samples, bool replacement, Generator* generator) -> Tensor")(result, self, num_samples, replacement, generator);
}
Tensor XLAType::_th_multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, Generator *)>("_th_multinomial(Tensor self, int64_t num_samples, bool replacement, Generator* generator) -> Tensor")(self, num_samples, replacement, generator);
}
Tensor & XLAType::_th_uniform_(Tensor & self, double from, double to, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("_th_uniform_(Tensor self, double from, double to, Generator* generator) -> Tensor")(self, from, to, generator);
}
Tensor & XLAType::_th_normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, double, Generator *)>("_th_normal_out(Tensor output, Tensor mean, double std, Generator* generator) -> Tensor")(output, mean, std, generator);
}
Tensor XLAType::_th_normal(const Tensor & mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, Generator *)>("_th_normal(Tensor mean, double std, Generator* generator) -> Tensor")(mean, std, generator);
}
Tensor & XLAType::_th_normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, const Tensor &, Generator *)>("_th_normal_out(Tensor output, double mean, Tensor std, Generator* generator) -> Tensor")(output, mean, std, generator);
}
Tensor XLAType::_th_normal(double mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(double, const Tensor &, Generator *)>("_th_normal(double mean, Tensor std, Generator* generator) -> Tensor")(mean, std, generator);
}
Tensor & XLAType::_th_normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Generator *)>("_th_normal_out(Tensor output, Tensor mean, Tensor std, Generator* generator) -> Tensor")(output, mean, std, generator);
}
Tensor XLAType::_th_normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Generator *)>("_th_normal(Tensor mean, Tensor std, Generator* generator) -> Tensor")(mean, std, generator);
}
Tensor & XLAType::_th_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("_th_normal_(Tensor self, double mean, double std, Generator* generator) -> Tensor")(self, mean, std, generator);
}
Tensor & XLAType::_th_cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("_th_cauchy_(Tensor self, double median, double sigma, Generator* generator) -> Tensor")(self, median, sigma, generator);
}
Tensor & XLAType::_th_log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("_th_log_normal_(Tensor self, double mean, double std, Generator* generator) -> Tensor")(self, mean, std, generator);
}
Tensor & XLAType::_th_exponential_(Tensor & self, double lambd, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, Generator *)>("_th_exponential_(Tensor self, double lambd, Generator* generator) -> Tensor")(self, lambd, generator);
}
Tensor & XLAType::_th_geometric_(Tensor & self, double p, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, Generator *)>("_th_geometric_(Tensor self, double p, Generator* generator) -> Tensor")(self, p, generator);
}
Tensor & XLAType::_th_dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("_th_dirichlet_grad_out(Tensor output, Tensor x, Tensor alpha, Tensor total) -> Tensor")(output, x, alpha, total);
}
Tensor XLAType::_th_dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("_th_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor")(x, alpha, total);
}
Tensor XLAType::_th_alias(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_th_alias(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_th_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_th_copy_ignoring_overlaps_(Tensor self, Tensor src) -> Tensor")(self, src);
}
Tensor & XLAType::_th_cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, TensorList, int64_t)>("_th_cat_out(Tensor self, TensorList tensors, int64_t dim) -> Tensor")(self, tensors, dim);
}
Tensor XLAType::_th_cat(TensorList tensors, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(TensorList, int64_t)>("_th_cat(TensorList tensors, int64_t dim) -> Tensor")(tensors, dim);
}
Tensor & XLAType::_thnn_binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_binary_cross_entropy_forward_out(Tensor output, Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(output, self, target, weight, reduction);
}
Tensor XLAType::_thnn_binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_binary_cross_entropy_forward(Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(self, target, weight, reduction);
}
Tensor & XLAType::_thnn_binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_binary_cross_entropy_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, weight, reduction);
}
Tensor XLAType::_thnn_binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(grad_output, self, target, weight, reduction);
}
Tensor & XLAType::_thnn_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_l1_loss_forward_out(Tensor output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(output, self, target, reduction);
}
Tensor XLAType::_thnn_l1_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("_thnn_l1_loss_forward(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::_thnn_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_l1_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::_thnn_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::_thnn_mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_mse_loss_forward_out(Tensor output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(output, self, target, reduction);
}
Tensor XLAType::_thnn_mse_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("_thnn_mse_loss_forward(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::_thnn_mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_mse_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::_thnn_mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::_thnn_multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("_thnn_multi_margin_loss_forward_out(Tensor output, Tensor self, IndexTensor target, accreal p, accreal margin, Tensor weight, int64_t reduction) -> Tensor")(output, self, target, p, margin, weight, reduction);
}
Tensor XLAType::_thnn_multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("_thnn_multi_margin_loss_forward(Tensor self, IndexTensor target, accreal p, accreal margin, Tensor weight, int64_t reduction) -> Tensor")(self, target, p, margin, weight, reduction);
}
Tensor & XLAType::_thnn_multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("_thnn_multi_margin_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IndexTensor target, accreal p, accreal margin, Tensor weight, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, p, margin, weight, reduction);
}
Tensor XLAType::_thnn_multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("_thnn_multi_margin_loss_backward(Tensor grad_output, Tensor self, IndexTensor target, accreal p, accreal margin, Tensor weight, int64_t reduction) -> Tensor")(grad_output, self, target, p, margin, weight, reduction);
}
std::tuple<Tensor &,Tensor &> XLAType::_thnn_multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_multilabel_margin_loss_forward_out(Tensor output, Tensor is_target, Tensor self, IndexTensor target, int64_t reduction) -> std::tuple<Tensor &,Tensor &>")(output, is_target, self, target, reduction);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, int64_t)>("_thnn_multilabel_margin_loss_forward(Tensor self, IndexTensor target, int64_t reduction) -> std::tuple<Tensor,Tensor>")(self, target, reduction);
}
Tensor & XLAType::_thnn_multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>("_thnn_multilabel_margin_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IndexTensor target, int64_t reduction, Tensor is_target) -> Tensor")(grad_input, grad_output, self, target, reduction, is_target);
}
Tensor XLAType::_thnn_multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>("_thnn_multilabel_margin_loss_backward(Tensor grad_output, Tensor self, IndexTensor target, int64_t reduction, Tensor is_target) -> Tensor")(grad_output, self, target, reduction, is_target);
}
std::tuple<Tensor &,Tensor &> XLAType::_thnn_nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("_thnn_nll_loss_forward_out(Tensor output, Tensor total_weight, Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor &,Tensor &>")(output, total_weight, self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("_thnn_nll_loss_forward(Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor,Tensor>")(self, target, weight, reduction, ignore_index);
}
Tensor & XLAType::_thnn_nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("_thnn_nll_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor XLAType::_thnn_nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("_thnn_nll_loss_backward(Tensor grad_output, Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
std::tuple<Tensor &,Tensor &> XLAType::_thnn_nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("_thnn_nll_loss2d_forward_out(Tensor output, Tensor total_weight, Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor &,Tensor &>")(output, total_weight, self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("_thnn_nll_loss2d_forward(Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor,Tensor>")(self, target, weight, reduction, ignore_index);
}
Tensor & XLAType::_thnn_nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("_thnn_nll_loss2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor XLAType::_thnn_nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("_thnn_nll_loss2d_backward(Tensor grad_output, Tensor self, IndexTensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor & XLAType::_thnn_smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_smooth_l1_loss_forward_out(Tensor output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(output, self, target, reduction);
}
Tensor XLAType::_thnn_smooth_l1_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("_thnn_smooth_l1_loss_forward(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::_thnn_smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_smooth_l1_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::_thnn_smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::_thnn_soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_soft_margin_loss_forward_out(Tensor output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(output, self, target, reduction);
}
Tensor XLAType::_thnn_soft_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("_thnn_soft_margin_loss_forward(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::_thnn_soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_soft_margin_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::_thnn_soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::_thnn_elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar, Scalar)>("_thnn_elu_forward_out(Tensor output, Tensor self, accreal alpha, accreal scale, accreal input_scale) -> Tensor")(output, self, alpha, scale, input_scale);
}
Tensor XLAType::_thnn_elu_forward(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar, Scalar)>("_thnn_elu_forward(Tensor self, accreal alpha, accreal scale, accreal input_scale) -> Tensor")(self, alpha, scale, input_scale);
}
Tensor & XLAType::_thnn_elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>("_thnn_elu_backward_out(Tensor grad_input, Tensor grad_output, accreal alpha, accreal scale, accreal input_scale, Tensor output) -> Tensor")(grad_input, grad_output, alpha, scale, input_scale, output);
}
Tensor XLAType::_thnn_elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>("_thnn_elu_backward(Tensor grad_output, accreal alpha, accreal scale, accreal input_scale, Tensor output) -> Tensor")(grad_output, alpha, scale, input_scale, output);
}
Tensor & XLAType::_thnn_elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, Scalar)>("_thnn_elu_(Tensor self, accreal alpha, accreal scale, accreal input_scale) -> Tensor")(self, alpha, scale, input_scale);
}
Tensor & XLAType::_thnn_elu_forward_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, Scalar)>("_thnn_elu_forward_(Tensor self, accreal alpha, accreal scale, accreal input_scale) -> Tensor")(self, alpha, scale, input_scale);
}
Tensor & XLAType::_thnn_glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("_thnn_glu_forward_out(Tensor output, Tensor self, int64_t dim) -> Tensor")(output, self, dim);
}
Tensor XLAType::_thnn_glu_forward(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("_thnn_glu_forward(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::_thnn_glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("_thnn_glu_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, int64_t dim) -> Tensor")(grad_input, grad_output, self, dim);
}
Tensor XLAType::_thnn_glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("_thnn_glu_backward(Tensor grad_output, Tensor self, int64_t dim) -> Tensor")(grad_output, self, dim);
}
Tensor & XLAType::_thnn_hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar)>("_thnn_hardtanh_forward_out(Tensor output, Tensor self, accreal min_val, accreal max_val) -> Tensor")(output, self, min_val, max_val);
}
Tensor XLAType::_thnn_hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("_thnn_hardtanh_forward(Tensor self, accreal min_val, accreal max_val) -> Tensor")(self, min_val, max_val);
}
Tensor & XLAType::_thnn_hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_thnn_hardtanh_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, accreal min_val, accreal max_val) -> Tensor")(grad_input, grad_output, self, min_val, max_val);
}
Tensor XLAType::_thnn_hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar)>("_thnn_hardtanh_backward(Tensor grad_output, Tensor self, accreal min_val, accreal max_val) -> Tensor")(grad_output, self, min_val, max_val);
}
Tensor & XLAType::_thnn_hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar)>("_thnn_hardtanh_(Tensor self, accreal min_val, accreal max_val) -> Tensor")(self, min_val, max_val);
}
Tensor & XLAType::_thnn_hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar)>("_thnn_hardtanh_forward_(Tensor self, accreal min_val, accreal max_val) -> Tensor")(self, min_val, max_val);
}
Tensor & XLAType::_thnn_leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_thnn_leaky_relu_forward_out(Tensor output, Tensor self, accreal negative_slope) -> Tensor")(output, self, negative_slope);
}
Tensor XLAType::_thnn_leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_thnn_leaky_relu_forward(Tensor self, accreal negative_slope) -> Tensor")(self, negative_slope);
}
Tensor & XLAType::_thnn_leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("_thnn_leaky_relu_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, accreal negative_slope) -> Tensor")(grad_input, grad_output, self, negative_slope);
}
Tensor XLAType::_thnn_leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("_thnn_leaky_relu_backward(Tensor grad_output, Tensor self, accreal negative_slope) -> Tensor")(grad_output, self, negative_slope);
}
Tensor & XLAType::_thnn_leaky_relu_(Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_thnn_leaky_relu_(Tensor self, accreal negative_slope) -> Tensor")(self, negative_slope);
}
Tensor & XLAType::_thnn_leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("_thnn_leaky_relu_forward_(Tensor self, accreal negative_slope) -> Tensor")(self, negative_slope);
}
std::tuple<Tensor &,Tensor &> XLAType::_thnn_log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &)>("_thnn_log_sigmoid_forward_out(Tensor output, Tensor buffer, Tensor self) -> std::tuple<Tensor &,Tensor &>")(output, buffer, self);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_log_sigmoid_forward(const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("_thnn_log_sigmoid_forward(Tensor self) -> std::tuple<Tensor,Tensor>")(self);
}
Tensor & XLAType::_thnn_log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("_thnn_log_sigmoid_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor buffer) -> Tensor")(grad_input, grad_output, self, buffer);
}
Tensor XLAType::_thnn_log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("_thnn_log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor")(grad_output, self, buffer);
}
Tensor & XLAType::_thnn_rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *)>("_thnn_rrelu_with_noise_forward_out(Tensor output, Tensor self, Tensor noise, accreal lower, accreal upper, bool training, Generator* generator) -> Tensor")(output, self, noise, lower, upper, training, generator);
}
Tensor XLAType::_thnn_rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *)>("_thnn_rrelu_with_noise_forward(Tensor self, Tensor noise, accreal lower, accreal upper, bool training, Generator* generator) -> Tensor")(self, noise, lower, upper, training, generator);
}
Tensor & XLAType::_thnn_rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool)>("_thnn_rrelu_with_noise_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor noise, accreal lower, accreal upper, bool training) -> Tensor")(grad_input, grad_output, self, noise, lower, upper, training);
}
Tensor XLAType::_thnn_rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool)>("_thnn_rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, accreal lower, accreal upper, bool training) -> Tensor")(grad_output, self, noise, lower, upper, training);
}
Tensor & XLAType::_thnn_rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *)>("_thnn_rrelu_with_noise_(Tensor self, Tensor noise, accreal lower, accreal upper, bool training, Generator* generator) -> Tensor")(self, noise, lower, upper, training, generator);
}
Tensor & XLAType::_thnn_rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *)>("_thnn_rrelu_with_noise_forward_(Tensor self, Tensor noise, accreal lower, accreal upper, bool training, Generator* generator) -> Tensor")(self, noise, lower, upper, training, generator);
}
Tensor & XLAType::_thnn_softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar)>("_thnn_softplus_forward_out(Tensor output, Tensor self, accreal beta, accreal threshold) -> Tensor")(output, self, beta, threshold);
}
Tensor XLAType::_thnn_softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("_thnn_softplus_forward(Tensor self, accreal beta, accreal threshold) -> Tensor")(self, beta, threshold);
}
Tensor & XLAType::_thnn_softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>("_thnn_softplus_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, accreal beta, accreal threshold, Tensor output) -> Tensor")(grad_input, grad_output, self, beta, threshold, output);
}
Tensor XLAType::_thnn_softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>("_thnn_softplus_backward(Tensor grad_output, Tensor self, accreal beta, accreal threshold, Tensor output) -> Tensor")(grad_output, self, beta, threshold, output);
}
Tensor & XLAType::_thnn_softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_thnn_softshrink_forward_out(Tensor output, Tensor self, accreal lambd) -> Tensor")(output, self, lambd);
}
Tensor XLAType::_thnn_softshrink_forward(const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("_thnn_softshrink_forward(Tensor self, accreal lambd) -> Tensor")(self, lambd);
}
Tensor & XLAType::_thnn_softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("_thnn_softshrink_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, accreal lambd) -> Tensor")(grad_input, grad_output, self, lambd);
}
Tensor XLAType::_thnn_softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("_thnn_softshrink_backward(Tensor grad_output, Tensor self, accreal lambd) -> Tensor")(grad_output, self, lambd);
}
Tensor & XLAType::_thnn_adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("_thnn_adaptive_avg_pool3d_forward_out(Tensor output, Tensor self, IntArrayRef output_size) -> Tensor")(output, self, output_size);
}
Tensor XLAType::_thnn_adaptive_avg_pool3d_forward(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_thnn_adaptive_avg_pool3d_forward(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::_thnn_adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_thnn_adaptive_avg_pool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self) -> Tensor")(grad_input, grad_output, self);
}
Tensor XLAType::_thnn_adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_thnn_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor")(grad_output, self);
}
Tensor & XLAType::_thnn_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool2d_forward_out(Tensor output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::_thnn_avg_pool2d_forward(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool2d_forward(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & XLAType::_thnn_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::_thnn_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool2d_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & XLAType::_thnn_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool3d_forward_out(Tensor output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::_thnn_avg_pool3d_forward(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool3d_forward(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & XLAType::_thnn_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::_thnn_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("_thnn_avg_pool3d_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
std::tuple<Tensor &,Tensor &> XLAType::_thnn_max_pool2d_with_indices_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("_thnn_max_pool2d_with_indices_forward_out(Tensor output, IndexTensor indices, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor &,Tensor &>")(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_max_pool2d_with_indices_forward(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("_thnn_max_pool2d_with_indices_forward(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor>")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & XLAType::_thnn_max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("_thnn_max_pool2d_with_indices_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, IndexTensor indices) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor XLAType::_thnn_max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("_thnn_max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, IndexTensor indices) -> Tensor")(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
std::tuple<Tensor &,Tensor &> XLAType::_thnn_max_pool3d_with_indices_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("_thnn_max_pool3d_with_indices_forward_out(Tensor output, IndexTensor indices, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor &,Tensor &>")(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_max_pool3d_with_indices_forward(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("_thnn_max_pool3d_with_indices_forward(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor>")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & XLAType::_thnn_max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("_thnn_max_pool3d_with_indices_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, IndexTensor indices) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor XLAType::_thnn_max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("_thnn_max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, IndexTensor indices) -> Tensor")(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor & XLAType::_thnn_max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("_thnn_max_unpool2d_forward_out(Tensor output, Tensor self, IndexTensor indices, IntArrayRef output_size) -> Tensor")(output, self, indices, output_size);
}
Tensor XLAType::_thnn_max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("_thnn_max_unpool2d_forward(Tensor self, IndexTensor indices, IntArrayRef output_size) -> Tensor")(self, indices, output_size);
}
Tensor & XLAType::_thnn_max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("_thnn_max_unpool2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IndexTensor indices, IntArrayRef output_size) -> Tensor")(grad_input, grad_output, self, indices, output_size);
}
Tensor XLAType::_thnn_max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("_thnn_max_unpool2d_backward(Tensor grad_output, Tensor self, IndexTensor indices, IntArrayRef output_size) -> Tensor")(grad_output, self, indices, output_size);
}
Tensor & XLAType::_thnn_max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_max_unpool3d_forward_out(Tensor output, Tensor self, IndexTensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(output, self, indices, output_size, stride, padding);
}
Tensor XLAType::_thnn_max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_max_unpool3d_forward(Tensor self, IndexTensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(self, indices, output_size, stride, padding);
}
Tensor & XLAType::_thnn_max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_max_unpool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IndexTensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(grad_input, grad_output, self, indices, output_size, stride, padding);
}
Tensor XLAType::_thnn_max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_max_unpool3d_backward(Tensor grad_output, Tensor self, IndexTensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(grad_output, self, indices, output_size, stride, padding);
}
Tensor & XLAType::_thnn_upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("_thnn_upsample_linear1d_forward_out(Tensor output, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(output, self, output_size, align_corners);
}
Tensor XLAType::_thnn_upsample_linear1d_forward(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("_thnn_upsample_linear1d_forward(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_linear1d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::_thnn_upsample_linear1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_linear1d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("_thnn_upsample_bilinear2d_forward_out(Tensor output, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(output, self, output_size, align_corners);
}
Tensor XLAType::_thnn_upsample_bilinear2d_forward(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("_thnn_upsample_bilinear2d_forward(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_bilinear2d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::_thnn_upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_bilinear2d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_bicubic2d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("_thnn_upsample_bicubic2d_forward_out(Tensor output, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(output, self, output_size, align_corners);
}
Tensor XLAType::_thnn_upsample_bicubic2d_forward(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("_thnn_upsample_bicubic2d_forward(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_bicubic2d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::_thnn_upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_bicubic2d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("_thnn_upsample_trilinear3d_forward_out(Tensor output, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(output, self, output_size, align_corners);
}
Tensor XLAType::_thnn_upsample_trilinear3d_forward(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("_thnn_upsample_trilinear3d_forward(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_trilinear3d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::_thnn_upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("_thnn_upsample_trilinear3d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::_thnn_upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("_thnn_upsample_nearest1d_forward_out(Tensor output, Tensor self, IntArrayRef output_size) -> Tensor")(output, self, output_size);
}
Tensor XLAType::_thnn_upsample_nearest1d_forward(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_thnn_upsample_nearest1d_forward(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::_thnn_upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_upsample_nearest1d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_input, grad_output, output_size, input_size);
}
Tensor XLAType::_thnn_upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_upsample_nearest1d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_output, output_size, input_size);
}
Tensor & XLAType::_thnn_upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("_thnn_upsample_nearest2d_forward_out(Tensor output, Tensor self, IntArrayRef output_size) -> Tensor")(output, self, output_size);
}
Tensor XLAType::_thnn_upsample_nearest2d_forward(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_thnn_upsample_nearest2d_forward(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::_thnn_upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_upsample_nearest2d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_input, grad_output, output_size, input_size);
}
Tensor XLAType::_thnn_upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_upsample_nearest2d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_output, output_size, input_size);
}
Tensor & XLAType::_thnn_upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("_thnn_upsample_nearest3d_forward_out(Tensor output, Tensor self, IntArrayRef output_size) -> Tensor")(output, self, output_size);
}
Tensor XLAType::_thnn_upsample_nearest3d_forward(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_thnn_upsample_nearest3d_forward(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::_thnn_upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_upsample_nearest3d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_input, grad_output, output_size, input_size);
}
Tensor XLAType::_thnn_upsample_nearest3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_upsample_nearest3d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_output, output_size, input_size);
}
Tensor & XLAType::_thnn_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_thnn_sigmoid_forward_out(Tensor output, Tensor self) -> Tensor")(output, self);
}
Tensor XLAType::_thnn_sigmoid_forward(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_thnn_sigmoid_forward(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_thnn_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_thnn_sigmoid_backward_out(Tensor grad_input, Tensor grad_output, Tensor output) -> Tensor")(grad_input, grad_output, output);
}
Tensor XLAType::_thnn_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_thnn_sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor")(grad_output, output);
}
Tensor & XLAType::_thnn_tanh_forward_out(Tensor & output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("_thnn_tanh_forward_out(Tensor output, Tensor self) -> Tensor")(output, self);
}
Tensor XLAType::_thnn_tanh_forward(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_thnn_tanh_forward(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_thnn_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_thnn_tanh_backward_out(Tensor grad_input, Tensor grad_output, Tensor output) -> Tensor")(grad_input, grad_output, output);
}
Tensor XLAType::_thnn_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_thnn_tanh_backward(Tensor grad_output, Tensor output) -> Tensor")(grad_output, output);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_transpose2d_forward_out(Tensor output, Tensor columns, Tensor ones, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, columns, ones, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_transpose2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("_thnn_conv_transpose2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor columns, Tensor ones) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("_thnn_conv_transpose2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor columns, Tensor ones, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_transpose3d_forward_out(Tensor output, Tensor finput, Tensor fgrad_input, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_transpose3d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("_thnn_conv_transpose3d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor finput, Tensor fgrad_input) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("_thnn_conv_transpose3d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor finput, Tensor fgrad_input, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_conv2d_forward_out(Tensor output, Tensor finput, Tensor fgrad_input, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_conv2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("_thnn_conv2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("_thnn_conv2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
Tensor & XLAType::_thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_depthwise2d_forward_out(Tensor output, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor XLAType::_thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &> XLAType::_thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_depthwise2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &>")(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,2>)>("_thnn_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) -> std::tuple<Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_conv3d_forward_out(Tensor output, Tensor finput, Tensor fgrad_input, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("_thnn_conv3d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("_thnn_conv3d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("_thnn_conv3d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_dilated2d_forward_out(Tensor output, Tensor columns, Tensor ones, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_dilated2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("_thnn_conv_dilated2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("_thnn_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_dilated3d_forward_out(Tensor output, Tensor columns, Tensor ones, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_conv_dilated3d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::_thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("_thnn_conv_dilated3d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("_thnn_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
Tensor & XLAType::_thnn_col2im_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_col2im_forward_out(Tensor output, Tensor self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(output, self, output_size, kernel_size, dilation, padding, stride);
}
Tensor XLAType::_thnn_col2im_forward(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_col2im_forward(Tensor self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(self, output_size, kernel_size, dilation, padding, stride);
}
Tensor & XLAType::_thnn_col2im_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_col2im_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(grad_input, grad_output, kernel_size, dilation, padding, stride);
}
Tensor XLAType::_thnn_col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_col2im_backward(Tensor grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(grad_output, kernel_size, dilation, padding, stride);
}
Tensor & XLAType::_thnn_im2col_forward_out(Tensor & output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_im2col_forward_out(Tensor output, Tensor self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(output, self, kernel_size, dilation, padding, stride);
}
Tensor XLAType::_thnn_im2col_forward(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_im2col_forward(Tensor self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(self, kernel_size, dilation, padding, stride);
}
Tensor & XLAType::_thnn_im2col_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_im2col_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
}
Tensor XLAType::_thnn_im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("_thnn_im2col_backward(Tensor grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(grad_output, input_size, kernel_size, dilation, padding, stride);
}
Tensor XLAType::_cast_Byte(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Byte(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
Tensor XLAType::_cast_Char(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Char(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
Tensor XLAType::_cast_Double(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Double(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
Tensor XLAType::_cast_Float(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Float(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
Tensor XLAType::_cast_Int(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Int(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
Tensor XLAType::_cast_Long(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Long(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
Tensor XLAType::_cast_Short(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Short(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
Tensor XLAType::_cast_Half(const Tensor & self, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cast_Half(Tensor self, bool non_blocking) -> Tensor")(self, non_blocking);
}
std::tuple<Tensor,Tensor> XLAType::_cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("_cudnn_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) -> std::tuple<Tensor,Tensor>")(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
}
Tensor XLAType::_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) const {
    return XLATypeDispatch::get_function<Tensor (*)(TensorList, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool)>("_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) -> Tensor")(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> XLAType::_cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (*)(const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &)>("_cudnn_rnn(Tensor input, TensorList weight, int64_t weight_stride0, Tensor weight_buf, Tensor hx, Tensor cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, Tensor dropout_state) -> std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>")(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> XLAType::_cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> (*)(const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>)>("_cudnn_rnn_backward(Tensor input, TensorList weight, int64_t weight_stride0, Tensor weight_buf, Tensor hx, Tensor cx, Tensor output, Tensor grad_output, Tensor grad_hy, Tensor grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, Tensor dropout_state, Tensor reserve, std::array<bool,4> output_mask) -> std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>>")(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}
Tensor XLAType::_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(double, bool, int64_t, const TensorOptions &)>("_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, TensorOptions options) -> Tensor")(dropout, train, dropout_seed, options);
}
int64_t XLAType::_debug_has_internal_overlap(const Tensor & self) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &)>("_debug_has_internal_overlap(Tensor self) -> int64_t")(self);
}
std::tuple<Tensor,Tensor> XLAType::_fused_dropout(const Tensor & self, double p, Generator * generator) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, double, Generator *)>("_fused_dropout(Tensor self, double p, Generator * generator) -> std::tuple<Tensor,Tensor>")(self, p, generator);
}
Tensor XLAType::_masked_scale(const Tensor & self, const Tensor & mask, double scale) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, double)>("_masked_scale(Tensor self, Tensor mask, double scale) -> Tensor")(self, mask, scale);
}
std::tuple<Tensor,Tensor> XLAType::_sobol_engine_draw(const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, const Tensor &, int64_t, int64_t, c10::optional<ScalarType>)>("_sobol_engine_draw(Tensor quasi, int64_t n, Tensor sobolstate, int64_t dimension, int64_t num_generated, ScalarType dtype) -> std::tuple<Tensor,Tensor>")(quasi, n, sobolstate, dimension, num_generated, dtype);
}
Tensor & XLAType::_sobol_engine_ff_(Tensor & self, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, int64_t, int64_t)>("_sobol_engine_ff_(Tensor self, int64_t n, Tensor sobolstate, int64_t dimension, int64_t num_generated) -> Tensor")(self, n, sobolstate, dimension, num_generated);
}
Tensor & XLAType::_sobol_engine_scramble_(Tensor & self, const Tensor & ltm, int64_t dimension) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("_sobol_engine_scramble_(Tensor self, Tensor ltm, int64_t dimension) -> Tensor")(self, ltm, dimension);
}
Tensor & XLAType::_sobol_engine_initialize_state_(Tensor & self, int64_t dimension) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("_sobol_engine_initialize_state_(Tensor self, int64_t dimension) -> Tensor")(self, dimension);
}
Tensor XLAType::_reshape_from_tensor(const Tensor & self, const Tensor & shape) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor")(self, shape);
}
Tensor XLAType::_shape_as_tensor(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_shape_as_tensor(Tensor self) -> Tensor")(self);
}
Tensor XLAType::dropout(const Tensor & input, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, bool)>("dropout(Tensor input, double p, bool train) -> Tensor")(input, p, train);
}
Tensor & XLAType::dropout_(Tensor & self, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, bool)>("dropout_(Tensor self, double p, bool train) -> Tensor")(self, p, train);
}
Tensor XLAType::feature_dropout(const Tensor & input, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, bool)>("feature_dropout(Tensor input, double p, bool train) -> Tensor")(input, p, train);
}
Tensor & XLAType::feature_dropout_(Tensor & self, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, bool)>("feature_dropout_(Tensor self, double p, bool train) -> Tensor")(self, p, train);
}
Tensor XLAType::alpha_dropout(const Tensor & input, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, bool)>("alpha_dropout(Tensor input, double p, bool train) -> Tensor")(input, p, train);
}
Tensor & XLAType::alpha_dropout_(Tensor & self, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, bool)>("alpha_dropout_(Tensor self, double p, bool train) -> Tensor")(self, p, train);
}
Tensor XLAType::feature_alpha_dropout(const Tensor & input, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, bool)>("feature_alpha_dropout(Tensor input, double p, bool train) -> Tensor")(input, p, train);
}
Tensor & XLAType::feature_alpha_dropout_(Tensor & self, double p, bool train) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, bool)>("feature_alpha_dropout_(Tensor self, double p, bool train) -> Tensor")(self, p, train);
}
Tensor XLAType::abs(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("abs(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::abs_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("abs_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::abs_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("abs_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::acos(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("acos(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::acos_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("acos_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::acos_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("acos_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool1d(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("adaptive_avg_pool1d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
std::tuple<Tensor,Tensor> XLAType::adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef)>("adaptive_max_pool1d(Tensor self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor>")(self, output_size);
}
Tensor XLAType::add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("add(Tensor self, Tensor other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor & XLAType::add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("add_(Tensor self, Tensor other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor & XLAType::add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("add_out(Tensor out, Tensor self, Tensor other, Scalar alpha) -> Tensor")(out, self, other, alpha);
}
Tensor XLAType::add(const Tensor & self, Scalar other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("add(Tensor self, Scalar other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor & XLAType::add_(Tensor & self, Scalar other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar)>("add_(Tensor self, Scalar other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor XLAType::addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addmv(Tensor self, Tensor mat, Tensor vec, Scalar beta, Scalar alpha) -> Tensor")(self, mat, vec, beta, alpha);
}
Tensor & XLAType::addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addmv_(Tensor self, Tensor mat, Tensor vec, Scalar beta, Scalar alpha) -> Tensor")(self, mat, vec, beta, alpha);
}
Tensor & XLAType::addmv_out(Tensor & out, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addmv_out(Tensor out, Tensor self, Tensor mat, Tensor vec, Scalar beta, Scalar alpha) -> Tensor")(out, self, mat, vec, beta, alpha);
}
Tensor XLAType::addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addr(Tensor self, Tensor vec1, Tensor vec2, Scalar beta, Scalar alpha) -> Tensor")(self, vec1, vec2, beta, alpha);
}
Tensor & XLAType::addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addr_(Tensor self, Tensor vec1, Tensor vec2, Scalar beta, Scalar alpha) -> Tensor")(self, vec1, vec2, beta, alpha);
}
Tensor & XLAType::addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addr_out(Tensor out, Tensor self, Tensor vec1, Tensor vec2, Scalar beta, Scalar alpha) -> Tensor")(out, self, vec1, vec2, beta, alpha);
}
Tensor XLAType::affine_grid_generator(const Tensor & theta, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("affine_grid_generator(Tensor theta, IntArrayRef size) -> Tensor")(theta, size);
}
Tensor XLAType::affine_grid_generator_backward(const Tensor & grad, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("affine_grid_generator_backward(Tensor grad, IntArrayRef size) -> Tensor")(grad, size);
}
Tensor XLAType::all(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("all(Tensor self, int64_t dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor & XLAType::all_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool)>("all_out(Tensor out, Tensor self, int64_t dim, bool keepdim) -> Tensor")(out, self, dim, keepdim);
}
bool XLAType::allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &, const Tensor &, double, double, bool)>("allclose(Tensor self, Tensor other, double rtol, double atol, bool equal_nan) -> bool")(self, other, rtol, atol, equal_nan);
}
Tensor XLAType::any(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("any(Tensor self, int64_t dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor & XLAType::any_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool)>("any_out(Tensor out, Tensor self, int64_t dim, bool keepdim) -> Tensor")(out, self, dim, keepdim);
}
Tensor XLAType::arange(Scalar end, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, const TensorOptions &)>("arange(Scalar end, TensorOptions options) -> Tensor")(end, options);
}
Tensor XLAType::arange(Scalar start, Scalar end, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, Scalar, const TensorOptions &)>("arange(Scalar start, Scalar end, TensorOptions options) -> Tensor")(start, end, options);
}
Tensor XLAType::arange(Scalar start, Scalar end, Scalar step, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, Scalar, Scalar, const TensorOptions &)>("arange(Scalar start, Scalar end, Scalar step, TensorOptions options) -> Tensor")(start, end, step, options);
}
Tensor & XLAType::arange_out(Tensor & out, Scalar end) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("arange_out(Tensor out, Scalar end) -> Tensor")(out, end);
}
Tensor & XLAType::arange_out(Tensor & out, Scalar start, Scalar end, Scalar step) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, Scalar)>("arange_out(Tensor out, Scalar start, Scalar end, Scalar step) -> Tensor")(out, start, end, step);
}
Tensor XLAType::_dim_arange(const Tensor & like, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("_dim_arange(Tensor like, int64_t dim) -> Tensor")(like, dim);
}
Tensor XLAType::argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, c10::optional<int64_t>, bool)>("argmax(Tensor self, int64_t dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor XLAType::argmin(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, c10::optional<int64_t>, bool)>("argmin(Tensor self, int64_t dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor XLAType::as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>("as_strided(Tensor self, IntArrayRef size, IntArrayRef stride, int64_t storage_offset) -> Tensor")(self, size, stride, storage_offset);
}
Tensor & XLAType::as_strided_(Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>("as_strided_(Tensor self, IntArrayRef size, IntArrayRef stride, int64_t storage_offset) -> Tensor")(self, size, stride, storage_offset);
}
Tensor XLAType::asin(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("asin(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::asin_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("asin_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::asin_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("asin_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::atan(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("atan(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::atan_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("atan_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::atan_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("atan_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("baddbmm(Tensor self, Tensor batch1, Tensor batch2, Scalar beta, Scalar alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("baddbmm_(Tensor self, Tensor batch1, Tensor batch2, Scalar beta, Scalar alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::_baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_baddbmm_mkl_(Tensor self, Tensor batch1, Tensor batch2, Scalar beta, Scalar alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("baddbmm_out(Tensor out, Tensor self, Tensor batch1, Tensor batch2, Scalar beta, Scalar alpha) -> Tensor")(out, self, batch1, batch2, beta, alpha);
}
Tensor XLAType::bartlett_window(int64_t window_length, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const TensorOptions &)>("bartlett_window(int64_t window_length, TensorOptions options) -> Tensor")(window_length, options);
}
Tensor XLAType::bartlett_window(int64_t window_length, bool periodic, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, bool, const TensorOptions &)>("bartlett_window(int64_t window_length, bool periodic, TensorOptions options) -> Tensor")(window_length, periodic, options);
}
Tensor XLAType::batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>("batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, bool training, double momentum, double eps, bool cudnn_enabled) -> Tensor")(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
std::tuple<Tensor,Tensor,Tensor,int64_t> XLAType::_batch_norm_impl_index(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,int64_t> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>("_batch_norm_impl_index(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, bool training, double momentum, double eps, bool cudnn_enabled) -> std::tuple<Tensor,Tensor,Tensor,int64_t>")(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_batch_norm_impl_index_backward(int64_t impl_index, const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var_transform, bool train, double eps, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>)>("_batch_norm_impl_index_backward(int64_t impl_index, Tensor input, Tensor grad_output, Tensor weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor save_var_transform, bool train, double eps, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask);
}
Tensor XLAType::bernoulli(const Tensor & self, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Generator *)>("bernoulli(Tensor self, Generator * generator) -> Tensor")(self, generator);
}
Tensor & XLAType::bernoulli_out(Tensor & out, const Tensor & self, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Generator *)>("bernoulli_out(Tensor out, Tensor self, Generator * generator) -> Tensor")(out, self, generator);
}
Tensor & XLAType::bernoulli_(Tensor & self, const Tensor & p, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Generator *)>("bernoulli_(Tensor self, Tensor p, Generator * generator) -> Tensor")(self, p, generator);
}
Tensor & XLAType::bernoulli_(Tensor & self, double p, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, Generator *)>("bernoulli_(Tensor self, double p, Generator * generator) -> Tensor")(self, p, generator);
}
Tensor XLAType::bernoulli(const Tensor & self, double p, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, Generator *)>("bernoulli(Tensor self, double p, Generator * generator) -> Tensor")(self, p, generator);
}
Tensor XLAType::bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor bias) -> Tensor")(input1, input2, weight, bias);
}
Tensor XLAType::binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor weight, Tensor pos_weight, int64_t reduction) -> Tensor")(self, target, weight, pos_weight, reduction);
}
Tensor XLAType::binary_cross_entropy_with_logits_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor weight, Tensor pos_weight, int64_t reduction) -> Tensor")(grad_output, self, target, weight, pos_weight, reduction);
}
Tensor XLAType::bincount(const Tensor & self, const Tensor & weights, int64_t minlength) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("bincount(Tensor self, Tensor weights, int64_t minlength) -> Tensor")(self, weights, minlength);
}
Tensor XLAType::blackman_window(int64_t window_length, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const TensorOptions &)>("blackman_window(int64_t window_length, TensorOptions options) -> Tensor")(window_length, options);
}
Tensor XLAType::blackman_window(int64_t window_length, bool periodic, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, bool, const TensorOptions &)>("blackman_window(int64_t window_length, bool periodic, TensorOptions options) -> Tensor")(window_length, periodic, options);
}
Tensor XLAType::bmm(const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("bmm(Tensor self, Tensor mat2) -> Tensor")(self, mat2);
}
Tensor & XLAType::bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("bmm_out(Tensor out, Tensor self, Tensor mat2) -> Tensor")(out, self, mat2);
}
std::vector<Tensor> XLAType::broadcast_tensors(TensorList tensors) const {
    return XLATypeDispatch::get_function<std::vector<Tensor> (*)(TensorList)>("broadcast_tensors(TensorList tensors) -> std::vector<Tensor>")(tensors);
}
Tensor XLAType::cat(TensorList tensors, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(TensorList, int64_t)>("cat(TensorList tensors, int64_t dim) -> Tensor")(tensors, dim);
}
Tensor & XLAType::cat_out(Tensor & out, TensorList tensors, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, TensorList, int64_t)>("cat_out(Tensor out, TensorList tensors, int64_t dim) -> Tensor")(out, tensors, dim);
}
Tensor XLAType::ceil(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("ceil(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::ceil_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("ceil_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::ceil_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("ceil_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::chain_matmul(TensorList matrices) const {
    return XLATypeDispatch::get_function<Tensor (*)(TensorList)>("chain_matmul(TensorList matrices) -> Tensor")(matrices);
}
std::vector<Tensor> XLAType::chunk(const Tensor & self, int64_t chunks, int64_t dim) const {
    return XLATypeDispatch::get_function<std::vector<Tensor> (*)(const Tensor &, int64_t, int64_t)>("chunk(Tensor self, int64_t chunks, int64_t dim) -> std::vector<Tensor>")(self, chunks, dim);
}
Tensor XLAType::clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>("clamp(Tensor self, Scalar min, Scalar max) -> Tensor")(self, min, max);
}
Tensor & XLAType::clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>("clamp_(Tensor self, Scalar min, Scalar max) -> Tensor")(self, min, max);
}
Tensor & XLAType::clamp_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>("clamp_out(Tensor out, Tensor self, Scalar min, Scalar max) -> Tensor")(out, self, min, max);
}
Tensor XLAType::clamp_max(const Tensor & self, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("clamp_max(Tensor self, Scalar max) -> Tensor")(self, max);
}
Tensor & XLAType::clamp_max_(Tensor & self, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("clamp_max_(Tensor self, Scalar max) -> Tensor")(self, max);
}
Tensor & XLAType::clamp_max_out(Tensor & out, const Tensor & self, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("clamp_max_out(Tensor out, Tensor self, Scalar max) -> Tensor")(out, self, max);
}
Tensor XLAType::clamp_min(const Tensor & self, Scalar min) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("clamp_min(Tensor self, Scalar min) -> Tensor")(self, min);
}
Tensor & XLAType::clamp_min_(Tensor & self, Scalar min) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("clamp_min_(Tensor self, Scalar min) -> Tensor")(self, min);
}
Tensor & XLAType::clamp_min_out(Tensor & out, const Tensor & self, Scalar min) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("clamp_min_out(Tensor out, Tensor self, Scalar min) -> Tensor")(out, self, min);
}
bool XLAType::cudnn_is_acceptable(const Tensor & self) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &)>("cudnn_is_acceptable(Tensor self) -> bool")(self);
}
Tensor XLAType::constant_pad_nd(const Tensor & self, IntArrayRef pad, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, Scalar)>("constant_pad_nd(Tensor self, IntArrayRef pad, Scalar value) -> Tensor")(self, pad, value);
}
Tensor XLAType::contiguous(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("contiguous(Tensor self) -> Tensor")(self);
}
Tensor XLAType::convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t)>("convolution(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) -> Tensor")(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
Tensor XLAType::_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool)>("_convolution(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor")(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}
Tensor XLAType::_convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef)>("_convolution_nogroup(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding) -> Tensor")(input, weight, bias, stride, padding, dilation, transposed, output_padding);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, std::array<bool,3>)>("_convolution_double_backward(Tensor ggI, Tensor ggW, Tensor ggb, Tensor gO, Tensor weight, Tensor self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
}
Tensor XLAType::conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>("conv1d(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) -> Tensor")(input, weight, bias, stride, padding, dilation, groups);
}
Tensor XLAType::conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>("conv2d(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) -> Tensor")(input, weight, bias, stride, padding, dilation, groups);
}
Tensor XLAType::conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>("conv3d(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) -> Tensor")(input, weight, bias, stride, padding, dilation, groups);
}
Tensor XLAType::conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("conv_tbc(Tensor self, Tensor weight, Tensor bias, int64_t pad) -> Tensor")(self, weight, bias, pad);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int64_t pad) -> std::tuple<Tensor,Tensor,Tensor>")(self, input, weight, bias, pad);
}
Tensor XLAType::conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>("conv_transpose1d(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) -> Tensor")(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor XLAType::conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>("conv_transpose2d(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) -> Tensor")(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor XLAType::conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>("conv_transpose3d(Tensor input, Tensor weight, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) -> Tensor")(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor & XLAType::copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, bool)>("copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor")(self, src, non_blocking);
}
Tensor & XLAType::s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, bool)>("s_copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor")(self, src, non_blocking);
}
Tensor XLAType::_s_copy_from(const Tensor & self, const Tensor & dst, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, bool)>("_s_copy_from(Tensor self, Tensor dst, bool non_blocking) -> Tensor")(self, dst, non_blocking);
}
void XLAType::_copy_same_type_(Tensor & self, const Tensor & src) const {
    return XLATypeDispatch::get_function<void (*)(Tensor &, const Tensor &)>("_copy_same_type_(Tensor self, Tensor src) -> void")(self, src);
}
Tensor XLAType::cos(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("cos(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::cos_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("cos_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::cos_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("cos_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::cosh(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("cosh(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::cosh_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("cosh_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::cosh_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("cosh_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, double, int64_t)>("cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin, int64_t reduction) -> Tensor")(input1, input2, target, margin, reduction);
}
Tensor XLAType::cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t, int64_t)>("cudnn_affine_grid_generator(Tensor theta, int64_t N, int64_t C, int64_t H, int64_t W) -> Tensor")(theta, N, C, H, W);
}
Tensor XLAType::cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t, int64_t)>("cudnn_affine_grid_generator_backward(Tensor grad, int64_t N, int64_t C, int64_t H, int64_t W) -> Tensor")(grad, N, C, H, W);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>("cudnn_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, bool training, double exponential_average_factor, double epsilon) -> std::tuple<Tensor,Tensor,Tensor>")(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>("cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor save_var, double epsilon) -> std::tuple<Tensor,Tensor,Tensor>")(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}
Tensor XLAType::cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("cudnn_convolution(Tensor self, Tensor weight, Tensor bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::cudnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("cudnn_convolution_backward_input(IntArrayRef self_size, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>("cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor XLAType::cudnn_convolution_backward_bias(const Tensor & grad_output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("cudnn_convolution_backward_bias(Tensor grad_output) -> Tensor")(grad_output);
}
Tensor XLAType::cudnn_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("cudnn_convolution_backward_weight(IntArrayRef weight_size, Tensor grad_output, Tensor self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>("cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor XLAType::cudnn_convolution_transpose_backward_bias(const Tensor & grad_output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("cudnn_convolution_transpose_backward_bias(Tensor grad_output) -> Tensor")(grad_output);
}
Tensor XLAType::cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::cudnn_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("cudnn_convolution_transpose_backward_weight(IntArrayRef weight_size, Tensor grad_output, Tensor self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::cudnn_grid_sampler(const Tensor & self, const Tensor & grid) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor")(self, grid);
}
std::tuple<Tensor,Tensor> XLAType::cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &)>("cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> std::tuple<Tensor,Tensor>")(self, grid, grad_output);
}
Tensor XLAType::cumsum(const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, ScalarType)>("cumsum(Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor XLAType::cumsum(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("cumsum(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::cumsum_out(Tensor & out, const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, ScalarType)>("cumsum_out(Tensor out, Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(out, self, dim, dtype);
}
Tensor & XLAType::cumsum_out(Tensor & out, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("cumsum_out(Tensor out, Tensor self, int64_t dim) -> Tensor")(out, self, dim);
}
Tensor XLAType::cumprod(const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, ScalarType)>("cumprod(Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor XLAType::cumprod(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("cumprod(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::cumprod_out(Tensor & out, const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, ScalarType)>("cumprod_out(Tensor out, Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(out, self, dim, dtype);
}
Tensor & XLAType::cumprod_out(Tensor & out, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("cumprod_out(Tensor out, Tensor self, int64_t dim) -> Tensor")(out, self, dim);
}
Tensor XLAType::ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, int64_t, bool)>("ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) -> Tensor")(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}
Tensor XLAType::ctc_loss(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool)>("ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) -> Tensor")(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}
std::tuple<Tensor,Tensor> XLAType::_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool)>("_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) -> std::tuple<Tensor,Tensor>")(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}
Tensor XLAType::_ctc_loss_backward(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, int64_t, bool)>("_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int64_t blank, bool zero_infinity) -> Tensor")(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
}
Tensor XLAType::det(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("det(Tensor self) -> Tensor")(self);
}
Tensor XLAType::diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t)>("diag_embed(Tensor self, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor")(self, offset, dim1, dim2);
}
Tensor XLAType::diagflat(const Tensor & self, int64_t offset) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("diagflat(Tensor self, int64_t offset) -> Tensor")(self, offset);
}
Tensor XLAType::diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t)>("diagonal(Tensor self, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor")(self, offset, dim1, dim2);
}
Tensor XLAType::div(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("div(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::div_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("div_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::div_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("div_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::div(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("div(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::div_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("div_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor XLAType::dot(const Tensor & self, const Tensor & tensor) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("dot(Tensor self, Tensor tensor) -> Tensor")(self, tensor);
}
Tensor & XLAType::dot_out(Tensor & out, const Tensor & self, const Tensor & tensor) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("dot_out(Tensor out, Tensor self, Tensor tensor) -> Tensor")(out, self, tensor);
}
Tensor XLAType::einsum(std::string equation, TensorList tensors) const {
    return XLATypeDispatch::get_function<Tensor (*)(std::string, TensorList)>("einsum(std::string equation, TensorList tensors) -> Tensor")(equation, tensors);
}
Tensor XLAType::embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, bool, bool)>("embedding(Tensor weight, Tensor indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor")(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}
Tensor XLAType::embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>("embedding_backward(Tensor grad, Tensor indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor")(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}
Tensor XLAType::embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, int64_t, bool)>("embedding_dense_backward(Tensor grad_output, Tensor indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) -> Tensor")(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor & XLAType::embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, double, double)>("embedding_renorm_(Tensor self, Tensor indices, double max_norm, double norm_type) -> Tensor")(self, indices, max_norm, norm_type);
}
Tensor XLAType::embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, int64_t, bool)>("embedding_sparse_backward(Tensor grad, Tensor indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) -> Tensor")(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> XLAType::embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &)>("embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, Tensor per_sample_weights) -> std::tuple<Tensor,Tensor,Tensor,Tensor>")(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> XLAType::_embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &)>("_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, Tensor per_sample_weights) -> std::tuple<Tensor,Tensor,Tensor,Tensor>")(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
}
Tensor XLAType::_embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &)>("_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, Tensor per_sample_weights) -> Tensor")(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
}
Tensor XLAType::_embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &)>("_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, Tensor per_sample_weights) -> Tensor")(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
}
Tensor XLAType::_embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &)>("_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, Tensor per_sample_weights) -> Tensor")(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights);
}
Tensor XLAType::_embedding_bag_per_sample_weights_backward(const Tensor & grad, const Tensor & weight, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, int64_t mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int64_t mode) -> Tensor")(grad, weight, indices, offsets, offset2bag, mode);
}
Tensor XLAType::empty(IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const TensorOptions &)>("empty(IntArrayRef size, TensorOptions options) -> Tensor")(size, options);
}
Tensor XLAType::_empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const TensorOptions &, double, int64_t)>("_empty_affine_quantized(IntArrayRef size, TensorOptions options, double scale, int64_t zero_point) -> Tensor")(size, options, scale, zero_point);
}
Tensor & XLAType::resize_(Tensor & self, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef)>("resize_(Tensor self, IntArrayRef size) -> Tensor")(self, size);
}
Tensor & XLAType::empty_out(Tensor & out, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef)>("empty_out(Tensor out, IntArrayRef size) -> Tensor")(out, size);
}
Tensor XLAType::empty_like(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("empty_like(Tensor self) -> Tensor")(self);
}
Tensor XLAType::empty_like(const Tensor & self, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const TensorOptions &)>("empty_like(Tensor self, TensorOptions options) -> Tensor")(self, options);
}
Tensor XLAType::empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, IntArrayRef, const TensorOptions &)>("empty_strided(IntArrayRef size, IntArrayRef stride, TensorOptions options) -> Tensor")(size, stride, options);
}
Tensor XLAType::erf(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("erf(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::erf_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("erf_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::erf_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("erf_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::erfc(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("erfc(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::erfc_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("erfc_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::erfc_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("erfc_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::exp(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("exp(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::exp_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("exp_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::exp_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("exp_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::expm1(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("expm1(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::expm1_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("expm1_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::expm1_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("expm1_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::expand(const Tensor & self, IntArrayRef size, bool implicit) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("expand(Tensor self, IntArrayRef size, bool implicit) -> Tensor")(self, size, implicit);
}
Tensor XLAType::expand_as(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("expand_as(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::eye(int64_t n, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const TensorOptions &)>("eye(int64_t n, TensorOptions options) -> Tensor")(n, options);
}
Tensor XLAType::eye(int64_t n, int64_t m, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, int64_t, const TensorOptions &)>("eye(int64_t n, int64_t m, TensorOptions options) -> Tensor")(n, m, options);
}
Tensor & XLAType::eye_out(Tensor & out, int64_t n) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("eye_out(Tensor out, int64_t n) -> Tensor")(out, n);
}
Tensor & XLAType::eye_out(Tensor & out, int64_t n, int64_t m) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, int64_t)>("eye_out(Tensor out, int64_t n, int64_t m) -> Tensor")(out, n, m);
}
Tensor XLAType::flatten(const Tensor & self, int64_t start_dim, int64_t end_dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t)>("flatten(Tensor self, int64_t start_dim, int64_t end_dim) -> Tensor")(self, start_dim, end_dim);
}
Tensor & XLAType::fill_(Tensor & self, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("fill_(Tensor self, Scalar value) -> Tensor")(self, value);
}
Tensor & XLAType::fill_(Tensor & self, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("fill_(Tensor self, Tensor value) -> Tensor")(self, value);
}
Tensor XLAType::floor(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("floor(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::floor_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("floor_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::floor_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("floor_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::frac(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("frac(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::frac_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("frac_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::frac_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("frac_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, Scalar, const TensorOptions &)>("full(IntArrayRef size, Scalar fill_value, TensorOptions options) -> Tensor")(size, fill_value, options);
}
Tensor & XLAType::full_out(Tensor & out, IntArrayRef size, Scalar fill_value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef, Scalar)>("full_out(Tensor out, IntArrayRef size, Scalar fill_value) -> Tensor")(out, size, fill_value);
}
Tensor XLAType::full_like(const Tensor & self, Scalar fill_value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("full_like(Tensor self, Scalar fill_value) -> Tensor")(self, fill_value);
}
Tensor XLAType::full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, const TensorOptions &)>("full_like(Tensor self, Scalar fill_value, TensorOptions options) -> Tensor")(self, fill_value, options);
}
Tensor XLAType::from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(std::string, c10::optional<bool>, c10::optional<int64_t>, const TensorOptions &)>("from_file(std::string filename, bool shared, int64_t size, TensorOptions options) -> Tensor")(filename, shared, size, options);
}
Tensor XLAType::grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, int64_t)>("grid_sampler(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode) -> Tensor")(input, grid, interpolation_mode, padding_mode);
}
Tensor XLAType::grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, int64_t)>("grid_sampler_2d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode) -> Tensor")(input, grid, interpolation_mode, padding_mode);
}
std::tuple<Tensor,Tensor> XLAType::grid_sampler_2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode) -> std::tuple<Tensor,Tensor>")(grad_output, input, grid, interpolation_mode, padding_mode);
}
Tensor XLAType::grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, int64_t)>("grid_sampler_3d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode) -> Tensor")(input, grid, interpolation_mode, padding_mode);
}
std::tuple<Tensor,Tensor> XLAType::grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode) -> std::tuple<Tensor,Tensor>")(grad_output, input, grid, interpolation_mode, padding_mode);
}
Tensor XLAType::hann_window(int64_t window_length, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const TensorOptions &)>("hann_window(int64_t window_length, TensorOptions options) -> Tensor")(window_length, options);
}
Tensor XLAType::hann_window(int64_t window_length, bool periodic, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, bool, const TensorOptions &)>("hann_window(int64_t window_length, bool periodic, TensorOptions options) -> Tensor")(window_length, periodic, options);
}
Tensor XLAType::hamming_window(int64_t window_length, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const TensorOptions &)>("hamming_window(int64_t window_length, TensorOptions options) -> Tensor")(window_length, options);
}
Tensor XLAType::hamming_window(int64_t window_length, bool periodic, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, bool, const TensorOptions &)>("hamming_window(int64_t window_length, bool periodic, TensorOptions options) -> Tensor")(window_length, periodic, options);
}
Tensor XLAType::hamming_window(int64_t window_length, bool periodic, double alpha, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, bool, double, const TensorOptions &)>("hamming_window(int64_t window_length, bool periodic, double alpha, TensorOptions options) -> Tensor")(window_length, periodic, alpha, options);
}
Tensor XLAType::hamming_window(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, bool, double, double, const TensorOptions &)>("hamming_window(int64_t window_length, bool periodic, double alpha, double beta, TensorOptions options) -> Tensor")(window_length, periodic, alpha, beta, options);
}
Tensor XLAType::hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, double, int64_t)>("hinge_embedding_loss(Tensor self, Tensor target, double margin, int64_t reduction) -> Tensor")(self, target, margin, reduction);
}
Tensor XLAType::ger(const Tensor & self, const Tensor & vec2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("ger(Tensor self, Tensor vec2) -> Tensor")(self, vec2);
}
Tensor & XLAType::ger_out(Tensor & out, const Tensor & self, const Tensor & vec2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("ger_out(Tensor out, Tensor self, Tensor vec2) -> Tensor")(out, self, vec2);
}
Tensor XLAType::group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool)>("group_norm(Tensor input, int64_t num_groups, Tensor weight, Tensor bias, double eps, bool cudnn_enabled) -> Tensor")(input, num_groups, weight, bias, eps, cudnn_enabled);
}
Tensor XLAType::fft(const Tensor & self, int64_t signal_ndim, bool normalized) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("fft(Tensor self, int64_t signal_ndim, bool normalized) -> Tensor")(self, signal_ndim, normalized);
}
Tensor XLAType::ifft(const Tensor & self, int64_t signal_ndim, bool normalized) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("ifft(Tensor self, int64_t signal_ndim, bool normalized) -> Tensor")(self, signal_ndim, normalized);
}
Tensor XLAType::rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, bool)>("rfft(Tensor self, int64_t signal_ndim, bool normalized, bool onesided) -> Tensor")(self, signal_ndim, normalized, onesided);
}
Tensor XLAType::irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, bool, IntArrayRef)>("irfft(Tensor self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) -> Tensor")(self, signal_ndim, normalized, onesided, signal_sizes);
}
Tensor XLAType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, bool, bool, IntArrayRef, bool, bool, IntArrayRef)>("_fft_with_size(Tensor self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes) -> Tensor")(self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
int64_t XLAType::_cufft_get_plan_cache_size(int64_t device_index) const {
    return XLATypeDispatch::get_function<int64_t (*)(int64_t)>("_cufft_get_plan_cache_size(int64_t device_index) -> int64_t")(device_index);
}
int64_t XLAType::_cufft_get_plan_cache_max_size(int64_t device_index) const {
    return XLATypeDispatch::get_function<int64_t (*)(int64_t)>("_cufft_get_plan_cache_max_size(int64_t device_index) -> int64_t")(device_index);
}
void XLAType::_cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) const {
    return XLATypeDispatch::get_function<void (*)(int64_t, int64_t)>("_cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) -> void")(device_index, max_size);
}
void XLAType::_cufft_clear_plan_cache(int64_t device_index) const {
    return XLATypeDispatch::get_function<void (*)(int64_t)>("_cufft_clear_plan_cache(int64_t device_index) -> void")(device_index);
}
Tensor XLAType::index(const Tensor & self, TensorList indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, TensorList)>("index(Tensor self, TensorList indices) -> Tensor")(self, indices);
}
Tensor & XLAType::index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("index_copy_(Tensor self, int64_t dim, Tensor index, Tensor source) -> Tensor")(self, dim, index, source);
}
Tensor XLAType::index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &)>("index_copy(Tensor self, int64_t dim, Tensor index, Tensor source) -> Tensor")(self, dim, index, source);
}
Tensor & XLAType::index_put_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, TensorList, const Tensor &, bool)>("index_put_(Tensor self, TensorList indices, Tensor values, bool accumulate) -> Tensor")(self, indices, values, accumulate);
}
Tensor XLAType::index_put(const Tensor & self, TensorList indices, const Tensor & values, bool accumulate) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, TensorList, const Tensor &, bool)>("index_put(Tensor self, TensorList indices, Tensor values, bool accumulate) -> Tensor")(self, indices, values, accumulate);
}
Tensor XLAType::instance_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>("instance_norm(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) -> Tensor")(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
}
Tensor XLAType::inverse(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("inverse(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::inverse_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("inverse_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::_inverse_helper(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_inverse_helper(Tensor self) -> Tensor")(self);
}
Tensor XLAType::isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, double, double, bool)>("isclose(Tensor self, Tensor other, double rtol, double atol, bool equal_nan) -> Tensor")(self, other, rtol, atol, equal_nan);
}
Tensor XLAType::isnan(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("isnan(Tensor self) -> Tensor")(self);
}
bool XLAType::is_distributed(const Tensor & self) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &)>("is_distributed(Tensor self) -> bool")(self);
}
bool XLAType::is_floating_point(const Tensor & self) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &)>("is_floating_point(Tensor self) -> bool")(self);
}
bool XLAType::is_complex(const Tensor & self) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &)>("is_complex(Tensor self) -> bool")(self);
}
bool XLAType::is_nonzero(const Tensor & self) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &)>("is_nonzero(Tensor self) -> bool")(self);
}
bool XLAType::is_same_size(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &, const Tensor &)>("is_same_size(Tensor self, Tensor other) -> bool")(self, other);
}
bool XLAType::is_signed(const Tensor & self) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &)>("is_signed(Tensor self) -> bool")(self);
}
Tensor XLAType::kl_div(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("kl_div(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor XLAType::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
std::tuple<Tensor,Tensor> XLAType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, int64_t, bool)>("kthvalue(Tensor self, int64_t k, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, k, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> XLAType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool)>("kthvalue_out(Tensor values, Tensor indices, Tensor self, int64_t k, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, k, dim, keepdim);
}
Tensor XLAType::layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool)>("layer_norm(Tensor input, IntArrayRef normalized_shape, Tensor weight, Tensor bias, double eps, bool cudnn_enable) -> Tensor")(input, normalized_shape, weight, bias, eps, cudnn_enable);
}
Tensor XLAType::linear(const Tensor & input, const Tensor & weight, const Tensor & bias) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("linear(Tensor input, Tensor weight, Tensor bias) -> Tensor")(input, weight, bias);
}
Tensor XLAType::mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("mkldnn_linear(Tensor input, Tensor weight, Tensor bias) -> Tensor")(input, weight, bias);
}
Tensor XLAType::fbgemm_linear_int8_weight(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>("fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor")(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}
std::tuple<Tensor,Tensor,double,int64_t> XLAType::fbgemm_linear_quantize_weight(const Tensor & input) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,double,int64_t> (*)(const Tensor &)>("fbgemm_linear_quantize_weight(Tensor input) -> std::tuple<Tensor,Tensor,double,int64_t>")(input);
}
Tensor XLAType::fbgemm_pack_quantized_matrix(const Tensor & input, int64_t K, int64_t N) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t)>("fbgemm_pack_quantized_matrix(Tensor input, int64_t K, int64_t N) -> Tensor")(input, K, N);
}
bool XLAType::fbgemm_is_cpu_supported() const {
    return XLATypeDispatch::get_function<bool (*)()>("fbgemm_is_cpu_supported() -> bool")();
}
Tensor XLAType::linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, Scalar, int64_t, const TensorOptions &)>("linspace(Scalar start, Scalar end, int64_t steps, TensorOptions options) -> Tensor")(start, end, steps, options);
}
Tensor & XLAType::linspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, int64_t)>("linspace_out(Tensor out, Scalar start, Scalar end, int64_t steps) -> Tensor")(out, start, end, steps);
}
Tensor XLAType::log(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("log(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("log_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("log_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::log10(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("log10(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log10_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("log10_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log10_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("log10_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::log1p(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("log1p(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log1p_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("log1p_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log1p_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("log1p_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::log2(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("log2(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log2_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("log2_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::log2_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("log2_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::logdet(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("logdet(Tensor self) -> Tensor")(self);
}
Tensor XLAType::logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, Scalar, int64_t, double, const TensorOptions &)>("logspace(Scalar start, Scalar end, int64_t steps, double base, TensorOptions options) -> Tensor")(start, end, steps, base, options);
}
Tensor & XLAType::logspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, int64_t, double)>("logspace_out(Tensor out, Scalar start, Scalar end, int64_t steps, double base) -> Tensor")(out, start, end, steps, base);
}
Tensor XLAType::log_softmax(const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, ScalarType)>("log_softmax(Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor XLAType::log_softmax(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("log_softmax(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor XLAType::_log_softmax(const Tensor & self, int64_t dim, bool half_to_float) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("_log_softmax(Tensor self, int64_t dim, bool half_to_float) -> Tensor")(self, dim, half_to_float);
}
Tensor XLAType::_log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, const Tensor &)>("_log_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor self) -> Tensor")(grad_output, output, dim, self);
}
Tensor XLAType::logsumexp(const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("logsumexp(Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor & XLAType::logsumexp_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("logsumexp_out(Tensor out, Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(out, self, dim, keepdim);
}
Tensor XLAType::margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, double, int64_t)>("margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, double margin, int64_t reduction) -> Tensor")(input1, input2, target, margin, reduction);
}
Tensor XLAType::matmul(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("matmul(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::matmul_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("matmul_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::matrix_rank(const Tensor & self, double tol, bool symmetric) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, bool)>("matrix_rank(Tensor self, double tol, bool symmetric) -> Tensor")(self, tol, symmetric);
}
Tensor XLAType::matrix_rank(const Tensor & self, bool symmetric) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("matrix_rank(Tensor self, bool symmetric) -> Tensor")(self, symmetric);
}
Tensor XLAType::matrix_power(const Tensor & self, int64_t n) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("matrix_power(Tensor self, int64_t n) -> Tensor")(self, n);
}
std::tuple<Tensor,Tensor> XLAType::max(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("max(Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> XLAType::max_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("max_out(Tensor max, Tensor max_values, Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(max, max_values, self, dim, keepdim);
}
Tensor XLAType::max_values(const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("max_values(Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
std::tuple<Tensor,Tensor> XLAType::max_pool1d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool1d_with_indices(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor>")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor XLAType::max_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool1d(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor XLAType::max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool2d(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor XLAType::mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("mkldnn_max_pool2d(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor XLAType::max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool3d(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> Tensor")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor XLAType::mean(const Tensor & self, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, ScalarType)>("mean(Tensor self, ScalarType dtype) -> Tensor")(self, dtype);
}
Tensor XLAType::mean(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("mean(Tensor self) -> Tensor")(self);
}
Tensor XLAType::mean(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool, ScalarType)>("mean(Tensor self, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor")(self, dim, keepdim, dtype);
}
Tensor XLAType::mean(const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("mean(Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor XLAType::mean(const Tensor & self, IntArrayRef dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, ScalarType)>("mean(Tensor self, IntArrayRef dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor & XLAType::mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool, ScalarType)>("mean_out(Tensor out, Tensor self, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor")(out, self, dim, keepdim, dtype);
}
Tensor & XLAType::mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("mean_out(Tensor out, Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(out, self, dim, keepdim);
}
Tensor & XLAType::mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, ScalarType)>("mean_out(Tensor out, Tensor self, IntArrayRef dim, ScalarType dtype) -> Tensor")(out, self, dim, dtype);
}
std::tuple<Tensor,Tensor> XLAType::median(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("median(Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> XLAType::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("median_out(Tensor values, Tensor indices, Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> XLAType::min(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("min(Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> XLAType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("min_out(Tensor min, Tensor min_indices, Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(min, min_indices, self, dim, keepdim);
}
Tensor XLAType::min_values(const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("min_values(Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor XLAType::mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>("mkldnn_convolution(Tensor self, Tensor weight, Tensor bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) -> Tensor")(self, weight, bias, padding, stride, dilation, groups);
}
Tensor XLAType::mkldnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool)>("mkldnn_convolution_backward_input(IntArrayRef self_size, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) -> Tensor")(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
}
std::tuple<Tensor,Tensor> XLAType::mkldnn_convolution_backward_weights(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool)>("mkldnn_convolution_backward_weights(IntArrayRef weight_size, Tensor grad_output, Tensor self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) -> std::tuple<Tensor,Tensor>")(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::mkldnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, std::array<bool,3>)>("mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::miopen_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>("miopen_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, bool training, double exponential_average_factor, double epsilon) -> std::tuple<Tensor,Tensor,Tensor>")(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::miopen_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>("miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor save_var, double epsilon) -> std::tuple<Tensor,Tensor,Tensor>")(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}
Tensor XLAType::miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_convolution(Tensor self, Tensor weight, Tensor bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::miopen_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_convolution_backward_input(IntArrayRef self_size, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::miopen_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>("miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor XLAType::miopen_convolution_backward_bias(const Tensor & grad_output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("miopen_convolution_backward_bias(Tensor grad_output) -> Tensor")(grad_output);
}
Tensor XLAType::miopen_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_convolution_backward_weight(IntArrayRef weight_size, Tensor grad_output, Tensor self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_convolution_transpose(Tensor self, Tensor weight, Tensor bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::miopen_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>("miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor XLAType::miopen_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::miopen_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_convolution_transpose_backward_weight(IntArrayRef weight_size, Tensor grad_output, Tensor self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::miopen_depthwise_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::miopen_depthwise_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_depthwise_convolution_backward_input(IntArrayRef self_size, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::miopen_depthwise_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>("miopen_depthwise_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor XLAType::miopen_depthwise_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("miopen_depthwise_convolution_backward_weight(IntArrayRef weight_size, Tensor grad_output, Tensor self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) -> Tensor")(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor XLAType::mm(const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("mm(Tensor self, Tensor mat2) -> Tensor")(self, mat2);
}
Tensor & XLAType::mm_out(Tensor & out, const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("mm_out(Tensor out, Tensor self, Tensor mat2) -> Tensor")(out, self, mat2);
}
Tensor XLAType::_sparse_mm(const Tensor & sparse, const Tensor & dense) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_sparse_mm(Tensor sparse, Tensor dense) -> Tensor")(sparse, dense);
}
std::tuple<Tensor,Tensor> XLAType::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("mode(Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>")(self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> XLAType::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("mode_out(Tensor values, Tensor indices, Tensor self, int64_t dim, bool keepdim) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, dim, keepdim);
}
Tensor XLAType::mul(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("mul(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::mul_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("mul_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::mul_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("mul_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::mul(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("mul(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::mul_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("mul_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor XLAType::mv(const Tensor & self, const Tensor & vec) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("mv(Tensor self, Tensor vec) -> Tensor")(self, vec);
}
Tensor & XLAType::mv_out(Tensor & out, const Tensor & self, const Tensor & vec) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("mv_out(Tensor out, Tensor self, Tensor vec) -> Tensor")(out, self, vec);
}
Tensor XLAType::mvlgamma(const Tensor & self, int64_t p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("mvlgamma(Tensor self, int64_t p) -> Tensor")(self, p);
}
Tensor & XLAType::mvlgamma_(Tensor & self, int64_t p) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("mvlgamma_(Tensor self, int64_t p) -> Tensor")(self, p);
}
Tensor XLAType::narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t)>("narrow_copy(Tensor self, int64_t dim, int64_t start, int64_t length) -> Tensor")(self, dim, start, length);
}
Tensor XLAType::narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t)>("narrow(Tensor self, int64_t dim, int64_t start, int64_t length) -> Tensor")(self, dim, start, length);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>("native_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, bool training, double momentum, double eps) -> std::tuple<Tensor,Tensor,Tensor>")(input, weight, bias, running_mean, running_var, training, momentum, eps);
}
std::tuple<Tensor,Tensor> XLAType::batch_norm_stats(const Tensor & input, double eps) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, double)>("batch_norm_stats(Tensor input, double eps) -> std::tuple<Tensor,Tensor>")(input, eps);
}
Tensor XLAType::batch_norm_elemt(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>("batch_norm_elemt(Tensor input, Tensor weight, Tensor bias, Tensor mean, Tensor invstd, double eps) -> Tensor")(input, weight, bias, mean, invstd, eps);
}
std::tuple<Tensor,Tensor> XLAType::batch_norm_gather_stats(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, int64_t count) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>("batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor running_mean, Tensor running_var, double momentum, double eps, int64_t count) -> std::tuple<Tensor,Tensor>")(input, mean, invstd, running_mean, running_var, momentum, eps, count);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::native_batch_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_invstd, bool train, double eps, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>)>("native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor save_invstd, bool train, double eps, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> XLAType::batch_norm_backward_reduce(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, bool input_g, bool weight_g, bool bias_g) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool)>("batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, bool input_g, bool weight_g, bool bias_g) -> std::tuple<Tensor,Tensor,Tensor,Tensor>")(grad_out, input, mean, invstd, input_g, weight_g, bias_g);
}
Tensor XLAType::batch_norm_backward_elemt(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor weight, Tensor mean_dy, Tensor mean_dy_xmu) -> Tensor")(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
}
std::tuple<Tensor,Tensor> XLAType::batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, double)>("batch_norm_update_stats(Tensor input, Tensor running_mean, Tensor running_var, double momentum) -> std::tuple<Tensor,Tensor>")(input, running_mean, running_var, momentum);
}
bool XLAType::_nnpack_available() const {
    return XLATypeDispatch::get_function<bool (*)()>("_nnpack_available() -> bool")();
}
Tensor XLAType::_nnpack_spatial_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor bias, IntArrayRef padding) -> Tensor")(input, weight, bias, padding);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_nnpack_spatial_convolution_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, std::array<bool,3>)>("_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, IntArrayRef padding, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(input, grad_output, weight, padding, output_mask);
}
Tensor XLAType::_nnpack_spatial_convolution_backward_input(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, IntArrayRef padding) -> Tensor")(input, grad_output, weight, padding);
}
Tensor XLAType::_nnpack_spatial_convolution_backward_weight(const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, const Tensor &, IntArrayRef)>("_nnpack_spatial_convolution_backward_weight(Tensor input, IntArrayRef weightsize, Tensor grad_output, IntArrayRef padding) -> Tensor")(input, weightsize, grad_output, padding);
}
Tensor XLAType::ones(IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const TensorOptions &)>("ones(IntArrayRef size, TensorOptions options) -> Tensor")(size, options);
}
Tensor & XLAType::ones_out(Tensor & out, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef)>("ones_out(Tensor out, IntArrayRef size) -> Tensor")(out, size);
}
Tensor XLAType::ones_like(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("ones_like(Tensor self) -> Tensor")(self);
}
Tensor XLAType::ones_like(const Tensor & self, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const TensorOptions &)>("ones_like(Tensor self, TensorOptions options) -> Tensor")(self, options);
}
Tensor XLAType::pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, double, double, bool)>("pairwise_distance(Tensor x1, Tensor x2, double p, double eps, bool keepdim) -> Tensor")(x1, x2, p, eps, keepdim);
}
Tensor XLAType::cdist(const Tensor & x1, const Tensor & x2, double p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, double)>("cdist(Tensor x1, Tensor x2, double p) -> Tensor")(x1, x2, p);
}
Tensor XLAType::_cdist_backward(const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, double, const Tensor &)>("_cdist_backward(Tensor grad, Tensor x1, Tensor x2, double p, Tensor cdist) -> Tensor")(grad, x1, x2, p, cdist);
}
Tensor XLAType::pdist(const Tensor & self, double p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double)>("pdist(Tensor self, double p) -> Tensor")(self, p);
}
Tensor XLAType::_pdist_forward(const Tensor & self, double p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double)>("_pdist_forward(Tensor self, double p) -> Tensor")(self, p);
}
Tensor XLAType::_pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, double, const Tensor &)>("_pdist_backward(Tensor grad, Tensor self, double p, Tensor pdist) -> Tensor")(grad, self, p, pdist);
}
Tensor XLAType::cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, double)>("cosine_similarity(Tensor x1, Tensor x2, int64_t dim, double eps) -> Tensor")(x1, x2, dim, eps);
}
Tensor XLAType::permute(const Tensor & self, IntArrayRef dims) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("permute(Tensor self, IntArrayRef dims) -> Tensor")(self, dims);
}
Tensor XLAType::pixel_shuffle(const Tensor & self, int64_t upscale_factor) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("pixel_shuffle(Tensor self, int64_t upscale_factor) -> Tensor")(self, upscale_factor);
}
Tensor XLAType::pin_memory(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("pin_memory(Tensor self) -> Tensor")(self);
}
Tensor XLAType::pinverse(const Tensor & self, double rcond) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double)>("pinverse(Tensor self, double rcond) -> Tensor")(self, rcond);
}
Tensor XLAType::scalar_tensor(Scalar s, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, const TensorOptions &)>("scalar_tensor(Scalar s, TensorOptions options) -> Tensor")(s, options);
}
Tensor XLAType::rand(IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const TensorOptions &)>("rand(IntArrayRef size, TensorOptions options) -> Tensor")(size, options);
}
Tensor XLAType::rand(IntArrayRef size, Generator * generator, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, Generator *, const TensorOptions &)>("rand(IntArrayRef size, Generator * generator, TensorOptions options) -> Tensor")(size, generator, options);
}
Tensor & XLAType::rand_out(Tensor & out, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef)>("rand_out(Tensor out, IntArrayRef size) -> Tensor")(out, size);
}
Tensor & XLAType::rand_out(Tensor & out, IntArrayRef size, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef, Generator *)>("rand_out(Tensor out, IntArrayRef size, Generator * generator) -> Tensor")(out, size, generator);
}
Tensor XLAType::rand_like(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("rand_like(Tensor self) -> Tensor")(self);
}
Tensor XLAType::rand_like(const Tensor & self, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const TensorOptions &)>("rand_like(Tensor self, TensorOptions options) -> Tensor")(self, options);
}
Tensor XLAType::randint(int64_t high, IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, IntArrayRef, const TensorOptions &)>("randint(int64_t high, IntArrayRef size, TensorOptions options) -> Tensor")(high, size, options);
}
Tensor XLAType::randint(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, IntArrayRef, Generator *, const TensorOptions &)>("randint(int64_t high, IntArrayRef size, Generator * generator, TensorOptions options) -> Tensor")(high, size, generator, options);
}
Tensor XLAType::randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, int64_t, IntArrayRef, const TensorOptions &)>("randint(int64_t low, int64_t high, IntArrayRef size, TensorOptions options) -> Tensor")(low, high, size, options);
}
Tensor XLAType::randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, int64_t, IntArrayRef, Generator *, const TensorOptions &)>("randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, TensorOptions options) -> Tensor")(low, high, size, generator, options);
}
Tensor & XLAType::randint_out(Tensor & out, int64_t high, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, IntArrayRef)>("randint_out(Tensor out, int64_t high, IntArrayRef size) -> Tensor")(out, high, size);
}
Tensor & XLAType::randint_out(Tensor & out, int64_t high, IntArrayRef size, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, IntArrayRef, Generator *)>("randint_out(Tensor out, int64_t high, IntArrayRef size, Generator * generator) -> Tensor")(out, high, size, generator);
}
Tensor & XLAType::randint_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, int64_t, IntArrayRef)>("randint_out(Tensor out, int64_t low, int64_t high, IntArrayRef size) -> Tensor")(out, low, high, size);
}
Tensor & XLAType::randint_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, int64_t, IntArrayRef, Generator *)>("randint_out(Tensor out, int64_t low, int64_t high, IntArrayRef size, Generator * generator) -> Tensor")(out, low, high, size, generator);
}
Tensor XLAType::randint_like(const Tensor & self, int64_t high) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("randint_like(Tensor self, int64_t high) -> Tensor")(self, high);
}
Tensor XLAType::randint_like(const Tensor & self, int64_t low, int64_t high) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t)>("randint_like(Tensor self, int64_t low, int64_t high) -> Tensor")(self, low, high);
}
Tensor XLAType::randint_like(const Tensor & self, int64_t high, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const TensorOptions &)>("randint_like(Tensor self, int64_t high, TensorOptions options) -> Tensor")(self, high, options);
}
Tensor XLAType::randint_like(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, const TensorOptions &)>("randint_like(Tensor self, int64_t low, int64_t high, TensorOptions options) -> Tensor")(self, low, high, options);
}
Tensor XLAType::randn(IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const TensorOptions &)>("randn(IntArrayRef size, TensorOptions options) -> Tensor")(size, options);
}
Tensor XLAType::randn(IntArrayRef size, Generator * generator, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, Generator *, const TensorOptions &)>("randn(IntArrayRef size, Generator * generator, TensorOptions options) -> Tensor")(size, generator, options);
}
Tensor & XLAType::randn_out(Tensor & out, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef)>("randn_out(Tensor out, IntArrayRef size) -> Tensor")(out, size);
}
Tensor & XLAType::randn_out(Tensor & out, IntArrayRef size, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef, Generator *)>("randn_out(Tensor out, IntArrayRef size, Generator * generator) -> Tensor")(out, size, generator);
}
Tensor XLAType::randn_like(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("randn_like(Tensor self) -> Tensor")(self);
}
Tensor XLAType::randn_like(const Tensor & self, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const TensorOptions &)>("randn_like(Tensor self, TensorOptions options) -> Tensor")(self, options);
}
Tensor XLAType::randperm(int64_t n, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const TensorOptions &)>("randperm(int64_t n, TensorOptions options) -> Tensor")(n, options);
}
Tensor XLAType::randperm(int64_t n, Generator * generator, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, Generator *, const TensorOptions &)>("randperm(int64_t n, Generator * generator, TensorOptions options) -> Tensor")(n, generator, options);
}
Tensor & XLAType::randperm_out(Tensor & out, int64_t n) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("randperm_out(Tensor out, int64_t n) -> Tensor")(out, n);
}
Tensor & XLAType::randperm_out(Tensor & out, int64_t n, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, Generator *)>("randperm_out(Tensor out, int64_t n, Generator * generator) -> Tensor")(out, n, generator);
}
Tensor XLAType::range(Scalar start, Scalar end, Scalar step, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, Scalar, Scalar, const TensorOptions &)>("range(Scalar start, Scalar end, Scalar step, TensorOptions options) -> Tensor")(start, end, step, options);
}
Tensor XLAType::range(Scalar start, Scalar end, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, Scalar, const TensorOptions &)>("range(Scalar start, Scalar end, TensorOptions options) -> Tensor")(start, end, options);
}
Tensor & XLAType::range_out(Tensor & out, Scalar start, Scalar end, Scalar step) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, Scalar)>("range_out(Tensor out, Scalar start, Scalar end, Scalar step) -> Tensor")(out, start, end, step);
}
Tensor XLAType::reciprocal(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("reciprocal(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::reciprocal_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("reciprocal_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::reciprocal_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("reciprocal_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::neg(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("neg(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::neg_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("neg_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::neg_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("neg_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::repeat(const Tensor & self, IntArrayRef repeats) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("repeat(Tensor self, IntArrayRef repeats) -> Tensor")(self, repeats);
}
Tensor XLAType::repeat_interleave(const Tensor & repeats) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("repeat_interleave(Tensor repeats) -> Tensor")(repeats);
}
Tensor XLAType::repeat_interleave(const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, c10::optional<int64_t>)>("repeat_interleave(Tensor self, Tensor repeats, int64_t dim) -> Tensor")(self, repeats, dim);
}
Tensor XLAType::repeat_interleave(const Tensor & self, int64_t repeats, c10::optional<int64_t> dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, c10::optional<int64_t>)>("repeat_interleave(Tensor self, int64_t repeats, int64_t dim) -> Tensor")(self, repeats, dim);
}
Tensor XLAType::reshape(const Tensor & self, IntArrayRef shape) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("reshape(Tensor self, IntArrayRef shape) -> Tensor")(self, shape);
}
Tensor XLAType::mkldnn_reshape(const Tensor & self, IntArrayRef shape) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("mkldnn_reshape(Tensor self, IntArrayRef shape) -> Tensor")(self, shape);
}
Tensor XLAType::reshape_as(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("reshape_as(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::round(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("round(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::round_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("round_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::round_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("round_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar, bool, Generator *)>("rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor")(self, lower, upper, training, generator);
}
Tensor & XLAType::rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, bool, Generator *)>("rrelu_(Tensor self, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor")(self, lower, upper, training, generator);
}
Tensor XLAType::relu(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("relu(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::relu_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("relu_(Tensor self) -> Tensor")(self);
}
Tensor XLAType::prelu(const Tensor & self, const Tensor & weight) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("prelu(Tensor self, Tensor weight) -> Tensor")(self, weight);
}
std::tuple<Tensor,Tensor> XLAType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &)>("prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> std::tuple<Tensor,Tensor>")(grad_output, self, weight);
}
Tensor XLAType::hardshrink(const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("hardshrink(Tensor self, Scalar lambd) -> Tensor")(self, lambd);
}
Tensor XLAType::hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor")(grad_out, self, lambd);
}
Tensor XLAType::rsqrt(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("rsqrt(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::rsqrt_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("rsqrt_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::rsqrt_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("rsqrt_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::select(const Tensor & self, int64_t dim, int64_t index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t)>("select(Tensor self, int64_t dim, int64_t index) -> Tensor")(self, dim, index);
}
Tensor XLAType::selu(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("selu(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::selu_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("selu_(Tensor self) -> Tensor")(self);
}
Tensor XLAType::celu(const Tensor & self, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("celu(Tensor self, Scalar alpha) -> Tensor")(self, alpha);
}
Tensor & XLAType::celu_(Tensor & self, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("celu_(Tensor self, Scalar alpha) -> Tensor")(self, alpha);
}
Tensor XLAType::sigmoid(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("sigmoid(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sigmoid_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("sigmoid_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sigmoid_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("sigmoid_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::sin(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("sin(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sin_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("sin_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sin_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("sin_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::sinh(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("sinh(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sinh_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("sinh_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sinh_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("sinh_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::detach(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("detach(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::detach_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("detach_(Tensor self) -> Tensor")(self);
}
int64_t XLAType::size(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &, int64_t)>("size(Tensor self, int64_t dim) -> int64_t")(self, dim);
}
Tensor XLAType::slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t, int64_t)>("slice(Tensor self, int64_t dim, int64_t start, int64_t end, int64_t step) -> Tensor")(self, dim, start, end, step);
}
std::tuple<Tensor,Tensor> XLAType::slogdet(const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("slogdet(Tensor self) -> std::tuple<Tensor,Tensor>")(self);
}
Tensor XLAType::smm(const Tensor & self, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("smm(Tensor self, Tensor mat2) -> Tensor")(self, mat2);
}
Tensor XLAType::softmax(const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, ScalarType)>("softmax(Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor XLAType::softmax(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("softmax(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor XLAType::_softmax(const Tensor & self, int64_t dim, bool half_to_float) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("_softmax(Tensor self, int64_t dim, bool half_to_float) -> Tensor")(self, dim, half_to_float);
}
Tensor XLAType::_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, const Tensor &)>("_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor self) -> Tensor")(grad_output, output, dim, self);
}
Tensor & XLAType::_sparse_add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("_sparse_add_out(Tensor out, Tensor self, Tensor other, Scalar alpha) -> Tensor")(out, self, other, alpha);
}
Tensor & XLAType::_sparse_dense_add_out(Tensor & out, const Tensor & self, SparseTensorRef other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, SparseTensorRef, Scalar)>("_sparse_dense_add_out(Tensor out, Tensor self, SparseTensorRef other, Scalar alpha) -> Tensor")(out, self, other, alpha);
}
Tensor & XLAType::_sparse_div_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_sparse_div_zerodim_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor & XLAType::_sparse_div_scalar_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_sparse_div_scalar_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor & XLAType::_sparse_mul_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_sparse_mul_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor & XLAType::_sparse_mul_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("_sparse_mul_zerodim_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor & XLAType::_sparse_mul_scalar_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("_sparse_mul_scalar_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
std::vector<Tensor> XLAType::split(const Tensor & self, int64_t split_size, int64_t dim) const {
    return XLATypeDispatch::get_function<std::vector<Tensor> (*)(const Tensor &, int64_t, int64_t)>("split(Tensor self, int64_t split_size, int64_t dim) -> std::vector<Tensor>")(self, split_size, dim);
}
std::vector<Tensor> XLAType::split_with_sizes(const Tensor & self, IntArrayRef split_sizes, int64_t dim) const {
    return XLATypeDispatch::get_function<std::vector<Tensor> (*)(const Tensor &, IntArrayRef, int64_t)>("split_with_sizes(Tensor self, IntArrayRef split_sizes, int64_t dim) -> std::vector<Tensor>")(self, split_sizes, dim);
}
Tensor XLAType::squeeze(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("squeeze(Tensor self) -> Tensor")(self);
}
Tensor XLAType::squeeze(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("squeeze(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::squeeze_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("squeeze_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::squeeze_(Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("squeeze_(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor XLAType::sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("sspaddmm(Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(self, mat1, mat2, beta, alpha);
}
Tensor & XLAType::sspaddmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("sspaddmm_out(Tensor out, Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(out, self, mat1, mat2, beta, alpha);
}
Tensor XLAType::stack(TensorList tensors, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(TensorList, int64_t)>("stack(TensorList tensors, int64_t dim) -> Tensor")(tensors, dim);
}
Tensor & XLAType::stack_out(Tensor & out, TensorList tensors, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, TensorList, int64_t)>("stack_out(Tensor out, TensorList tensors, int64_t dim) -> Tensor")(out, tensors, dim);
}
Tensor XLAType::stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>("stft(Tensor self, int64_t n_fft, int64_t hop_length, int64_t win_length, Tensor window, bool normalized, bool onesided) -> Tensor")(self, n_fft, hop_length, win_length, window, normalized, onesided);
}
int64_t XLAType::stride(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &, int64_t)>("stride(Tensor self, int64_t dim) -> int64_t")(self, dim);
}
Tensor XLAType::sum(const Tensor & self, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, ScalarType)>("sum(Tensor self, ScalarType dtype) -> Tensor")(self, dtype);
}
Tensor XLAType::sum(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("sum(Tensor self) -> Tensor")(self);
}
Tensor XLAType::sum(const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool, ScalarType)>("sum(Tensor self, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor")(self, dim, keepdim, dtype);
}
Tensor XLAType::sum(const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("sum(Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor XLAType::sum(const Tensor & self, IntArrayRef dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, ScalarType)>("sum(Tensor self, IntArrayRef dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor & XLAType::sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool, ScalarType)>("sum_out(Tensor out, Tensor self, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor")(out, self, dim, keepdim, dtype);
}
Tensor & XLAType::sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("sum_out(Tensor out, Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(out, self, dim, keepdim);
}
Tensor & XLAType::sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, ScalarType)>("sum_out(Tensor out, Tensor self, IntArrayRef dim, ScalarType dtype) -> Tensor")(out, self, dim, dtype);
}
Tensor XLAType::sum_to_size(const Tensor & self, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("sum_to_size(Tensor self, IntArrayRef size) -> Tensor")(self, size);
}
Tensor XLAType::sqrt(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("sqrt(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sqrt_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("sqrt_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sqrt_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("sqrt_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::std(const Tensor & self, bool unbiased) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("std(Tensor self, bool unbiased) -> Tensor")(self, unbiased);
}
Tensor XLAType::std(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool, bool)>("std(Tensor self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor")(self, dim, unbiased, keepdim);
}
Tensor & XLAType::std_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool, bool)>("std_out(Tensor out, Tensor self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor")(out, self, dim, unbiased, keepdim);
}
Tensor XLAType::prod(const Tensor & self, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, ScalarType)>("prod(Tensor self, ScalarType dtype) -> Tensor")(self, dtype);
}
Tensor XLAType::prod(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("prod(Tensor self) -> Tensor")(self);
}
Tensor XLAType::prod(const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, ScalarType)>("prod(Tensor self, int64_t dim, bool keepdim, ScalarType dtype) -> Tensor")(self, dim, keepdim, dtype);
}
Tensor XLAType::prod(const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("prod(Tensor self, int64_t dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor XLAType::prod(const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, ScalarType)>("prod(Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor & XLAType::prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool, ScalarType)>("prod_out(Tensor out, Tensor self, int64_t dim, bool keepdim, ScalarType dtype) -> Tensor")(out, self, dim, keepdim, dtype);
}
Tensor & XLAType::prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool)>("prod_out(Tensor out, Tensor self, int64_t dim, bool keepdim) -> Tensor")(out, self, dim, keepdim);
}
Tensor & XLAType::prod_out(Tensor & out, const Tensor & self, int64_t dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, ScalarType)>("prod_out(Tensor out, Tensor self, int64_t dim, ScalarType dtype) -> Tensor")(out, self, dim, dtype);
}
Tensor XLAType::t(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("t(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::t_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("t_(Tensor self) -> Tensor")(self);
}
Tensor XLAType::tan(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("tan(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::tan_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("tan_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::tan_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("tan_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::tanh(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("tanh(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::tanh_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("tanh_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::tanh_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("tanh_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::tensordot(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>("tensordot(Tensor self, Tensor other, IntArrayRef dims_self, IntArrayRef dims_other) -> Tensor")(self, other, dims_self, dims_other);
}
Tensor XLAType::threshold(const Tensor & self, Scalar threshold, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor")(self, threshold, value);
}
Tensor & XLAType::threshold_(Tensor & self, Scalar threshold, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar)>("threshold_(Tensor self, Scalar threshold, Scalar value) -> Tensor")(self, threshold, value);
}
Tensor & XLAType::threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar)>("threshold_out(Tensor out, Tensor self, Scalar threshold, Scalar value) -> Tensor")(out, self, threshold, value);
}
Tensor XLAType::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor")(grad_output, self, threshold);
}
Tensor XLAType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t)>("transpose(Tensor self, int64_t dim0, int64_t dim1) -> Tensor")(self, dim0, dim1);
}
Tensor & XLAType::transpose_(Tensor & self, int64_t dim0, int64_t dim1) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, int64_t)>("transpose_(Tensor self, int64_t dim0, int64_t dim1) -> Tensor")(self, dim0, dim1);
}
Tensor XLAType::one_hot(const Tensor & self, int64_t num_classes) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("one_hot(Tensor self, int64_t num_classes) -> Tensor")(self, num_classes);
}
Tensor XLAType::flip(const Tensor & self, IntArrayRef dims) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("flip(Tensor self, IntArrayRef dims) -> Tensor")(self, dims);
}
Tensor XLAType::roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef)>("roll(Tensor self, IntArrayRef shifts, IntArrayRef dims) -> Tensor")(self, shifts, dims);
}
Tensor XLAType::rot90(const Tensor & self, int64_t k, IntArrayRef dims) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, IntArrayRef)>("rot90(Tensor self, int64_t k, IntArrayRef dims) -> Tensor")(self, k, dims);
}
Tensor XLAType::_trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>("_trilinear(Tensor i1, Tensor i2, Tensor i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) -> Tensor")(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}
Tensor XLAType::triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t)>("triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin, double p, double eps, bool swap, int64_t reduction) -> Tensor")(anchor, positive, negative, margin, p, eps, swap, reduction);
}
Tensor XLAType::trunc(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("trunc(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::trunc_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("trunc_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::trunc_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("trunc_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::type_as(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("type_as(Tensor self, Tensor other) -> Tensor")(self, other);
}
std::tuple<Tensor,Tensor> XLAType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, bool, bool)>("_unique(Tensor self, bool sorted, bool return_inverse) -> std::tuple<Tensor,Tensor>")(self, sorted, return_inverse);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, int64_t, bool, bool, bool)>("unique_dim(Tensor self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) -> std::tuple<Tensor,Tensor,Tensor>")(self, dim, sorted, return_inverse, return_counts);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, bool, bool, c10::optional<int64_t>)>("unique_consecutive(Tensor self, bool return_inverse, bool return_counts, int64_t dim) -> std::tuple<Tensor,Tensor,Tensor>")(self, return_inverse, return_counts, dim);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::unique_dim_consecutive(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, int64_t, bool, bool)>("unique_dim_consecutive(Tensor self, int64_t dim, bool return_inverse, bool return_counts) -> std::tuple<Tensor,Tensor,Tensor>")(self, dim, return_inverse, return_counts);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_unique2(const Tensor & self, bool sorted, bool return_inverse, bool return_counts) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, bool, bool, bool)>("_unique2(Tensor self, bool sorted, bool return_inverse, bool return_counts) -> std::tuple<Tensor,Tensor,Tensor>")(self, sorted, return_inverse, return_counts);
}
Tensor XLAType::_unsafe_view(const Tensor & self, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_unsafe_view(Tensor self, IntArrayRef size) -> Tensor")(self, size);
}
Tensor XLAType::unsqueeze(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("unsqueeze(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::unsqueeze_(Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("unsqueeze_(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor XLAType::var(const Tensor & self, bool unbiased) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("var(Tensor self, bool unbiased) -> Tensor")(self, unbiased);
}
Tensor XLAType::var(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool, bool)>("var(Tensor self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor")(self, dim, unbiased, keepdim);
}
Tensor & XLAType::var_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool, bool)>("var_out(Tensor out, Tensor self, IntArrayRef dim, bool unbiased, bool keepdim) -> Tensor")(out, self, dim, unbiased, keepdim);
}
Tensor XLAType::view_as(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("view_as(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("where(Tensor condition, Tensor self, Tensor other) -> Tensor")(condition, self, other);
}
Tensor XLAType::_s_where(const Tensor & condition, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor")(condition, self, other);
}
Tensor XLAType::norm_except_dim(const Tensor & v, int64_t pow, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t)>("norm_except_dim(Tensor v, int64_t pow, int64_t dim) -> Tensor")(v, pow, dim);
}
Tensor XLAType::_weight_norm(const Tensor & v, const Tensor & g, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("_weight_norm(Tensor v, Tensor g, int64_t dim) -> Tensor")(v, g, dim);
}
std::tuple<Tensor,Tensor> XLAType::_weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, int64_t)>("_weight_norm_cuda_interface(Tensor v, Tensor g, int64_t dim) -> std::tuple<Tensor,Tensor>")(v, g, dim);
}
std::tuple<Tensor,Tensor> XLAType::_weight_norm_cuda_interface_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int64_t dim) -> std::tuple<Tensor,Tensor>")(grad_w, saved_v, saved_g, saved_norms, dim);
}
std::tuple<Tensor,Tensor> XLAType::_weight_norm_differentiable_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int64_t dim) -> std::tuple<Tensor,Tensor>")(grad_w, saved_v, saved_g, saved_norms, dim);
}
Tensor XLAType::zeros(IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const TensorOptions &)>("zeros(IntArrayRef size, TensorOptions options) -> Tensor")(size, options);
}
Tensor & XLAType::zeros_out(Tensor & out, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef)>("zeros_out(Tensor out, IntArrayRef size) -> Tensor")(out, size);
}
Tensor XLAType::zeros_like(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("zeros_like(Tensor self) -> Tensor")(self);
}
Tensor XLAType::zeros_like(const Tensor & self, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const TensorOptions &)>("zeros_like(Tensor self, TensorOptions options) -> Tensor")(self, options);
}
Tensor XLAType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_standard_gamma_grad(Tensor self, Tensor output) -> Tensor")(self, output);
}
Tensor XLAType::_standard_gamma(const Tensor & self, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Generator *)>("_standard_gamma(Tensor self, Generator * generator) -> Tensor")(self, generator);
}
Tensor XLAType::_sample_dirichlet(const Tensor & self, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Generator *)>("_sample_dirichlet(Tensor self, Generator * generator) -> Tensor")(self, generator);
}
Tensor XLAType::poisson(const Tensor & self, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Generator *)>("poisson(Tensor self, Generator * generator) -> Tensor")(self, generator);
}
Tensor XLAType::native_norm(const Tensor & self, Scalar p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("native_norm(Tensor self, Scalar p) -> Tensor")(self, p);
}
Tensor XLAType::_sparse_sum(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_sparse_sum(Tensor self) -> Tensor")(self);
}
Tensor XLAType::_sparse_sum(const Tensor & self, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, ScalarType)>("_sparse_sum(Tensor self, ScalarType dtype) -> Tensor")(self, dtype);
}
Tensor XLAType::_sparse_sum(const Tensor & self, IntArrayRef dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_sparse_sum(Tensor self, IntArrayRef dim) -> Tensor")(self, dim);
}
Tensor XLAType::_sparse_sum(const Tensor & self, IntArrayRef dim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, ScalarType)>("_sparse_sum(Tensor self, IntArrayRef dim, ScalarType dtype) -> Tensor")(self, dim, dtype);
}
Tensor XLAType::_sparse_sum_backward(const Tensor & grad, const Tensor & self, IntArrayRef dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("_sparse_sum_backward(Tensor grad, Tensor self, IntArrayRef dim) -> Tensor")(grad, self, dim);
}
Tensor XLAType::norm(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, c10::optional<Scalar>, ScalarType)>("norm(Tensor self, Scalar p, ScalarType dtype) -> Tensor")(self, p, dtype);
}
Tensor XLAType::norm(const Tensor & self, Scalar p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("norm(Tensor self, Scalar p) -> Tensor")(self, p);
}
Tensor XLAType::norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>("norm(Tensor self, Scalar p, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor")(self, p, dim, keepdim, dtype);
}
Tensor XLAType::norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>("norm(Tensor self, Scalar p, IntArrayRef dim, bool keepdim) -> Tensor")(self, p, dim, keepdim);
}
Tensor & XLAType::norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>("norm_out(Tensor out, Tensor self, Scalar p, IntArrayRef dim, bool keepdim, ScalarType dtype) -> Tensor")(out, self, p, dim, keepdim, dtype);
}
Tensor & XLAType::norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>("norm_out(Tensor out, Tensor self, Scalar p, IntArrayRef dim, bool keepdim) -> Tensor")(out, self, p, dim, keepdim);
}
Tensor XLAType::frobenius_norm(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("frobenius_norm(Tensor self) -> Tensor")(self);
}
Tensor XLAType::frobenius_norm(const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("frobenius_norm(Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(self, dim, keepdim);
}
Tensor & XLAType::frobenius_norm_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("frobenius_norm_out(Tensor out, Tensor self, IntArrayRef dim, bool keepdim) -> Tensor")(out, self, dim, keepdim);
}
Tensor XLAType::nuclear_norm(const Tensor & self, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("nuclear_norm(Tensor self, bool keepdim) -> Tensor")(self, keepdim);
}
Tensor & XLAType::nuclear_norm_out(Tensor & out, const Tensor & self, bool keepdim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, bool)>("nuclear_norm_out(Tensor out, Tensor self, bool keepdim) -> Tensor")(out, self, keepdim);
}
Tensor XLAType::clone(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("clone(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::resize_as_(Tensor & self, const Tensor & the_template) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("resize_as_(Tensor self, Tensor the_template) -> Tensor")(self, the_template);
}
Tensor & XLAType::pow_out(Tensor & out, const Tensor & self, Scalar exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("pow_out(Tensor out, Tensor self, Scalar exponent) -> Tensor")(out, self, exponent);
}
Tensor XLAType::pow(const Tensor & self, Scalar exponent) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("pow(Tensor self, Scalar exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::zero_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("zero_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::sub_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("sub_out(Tensor out, Tensor self, Tensor other, Scalar alpha) -> Tensor")(out, self, other, alpha);
}
Tensor XLAType::sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("sub(Tensor self, Tensor other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor & XLAType::sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("sub_(Tensor self, Tensor other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor XLAType::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("sub(Tensor self, Scalar other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor & XLAType::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar)>("sub_(Tensor self, Scalar other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor XLAType::rsub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("rsub(Tensor self, Tensor other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor XLAType::rsub(const Tensor & self, Scalar other, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("rsub(Tensor self, Scalar other, Scalar alpha) -> Tensor")(self, other, alpha);
}
Tensor & XLAType::s_native_addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("s_native_addmm_out(Tensor out, Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(out, self, mat1, mat2, beta, alpha);
}
Tensor XLAType::s_native_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("s_native_addmm(Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(self, mat1, mat2, beta, alpha);
}
Tensor & XLAType::s_native_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("s_native_addmm_(Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(self, mat1, mat2, beta, alpha);
}
Tensor XLAType::_sparse_addmm(const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, Scalar beta, Scalar alpha) -> Tensor")(self, sparse, dense, beta, alpha);
}
Tensor & XLAType::addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addmm_out(Tensor out, Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(out, self, mat1, mat2, beta, alpha);
}
Tensor XLAType::addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addmm(Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(self, mat1, mat2, beta, alpha);
}
Tensor & XLAType::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addmm_(Tensor self, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor")(self, mat1, mat2, beta, alpha);
}
Tensor XLAType::sparse_coo_tensor(IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(IntArrayRef, const TensorOptions &)>("sparse_coo_tensor(IntArrayRef size, TensorOptions options) -> Tensor")(size, options);
}
Tensor XLAType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const TensorOptions &)>("sparse_coo_tensor(Tensor indices, Tensor values, TensorOptions options) -> Tensor")(indices, values, options);
}
Tensor XLAType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &)>("sparse_coo_tensor(Tensor indices, Tensor values, IntArrayRef size, TensorOptions options) -> Tensor")(indices, values, size, options);
}
Tensor XLAType::_sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &)>("_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, IntArrayRef size, TensorOptions options) -> Tensor")(indices, values, size, options);
}
Tensor XLAType::_sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, int64_t, IntArrayRef, const TensorOptions &)>("_sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, TensorOptions options) -> Tensor")(sparse_dim, dense_dim, size, options);
}
Tensor XLAType::_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &)>("_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, Tensor indices, Tensor values, TensorOptions options) -> Tensor")(sparse_dim, dense_dim, size, indices, values, options);
}
Tensor & XLAType::sparse_resize_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef, int64_t, int64_t)>("sparse_resize_(Tensor self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) -> Tensor")(self, size, sparse_dim, dense_dim);
}
Tensor & XLAType::sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, IntArrayRef, int64_t, int64_t)>("sparse_resize_and_clear_(Tensor self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) -> Tensor")(self, size, sparse_dim, dense_dim);
}
Tensor XLAType::sparse_mask(const Tensor & self, SparseTensorRef mask) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, SparseTensorRef)>("sparse_mask(Tensor self, SparseTensorRef mask) -> Tensor")(self, mask);
}
Tensor XLAType::to_dense(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("to_dense(Tensor self) -> Tensor")(self);
}
Tensor XLAType::to_dense_backward(const Tensor & grad, const Tensor & input) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("to_dense_backward(Tensor grad, Tensor input) -> Tensor")(grad, input);
}
int64_t XLAType::sparse_dim(const Tensor & self) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &)>("sparse_dim(Tensor self) -> int64_t")(self);
}
int64_t XLAType::_dimI(const Tensor & self) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &)>("_dimI(Tensor self) -> int64_t")(self);
}
int64_t XLAType::dense_dim(const Tensor & self) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &)>("dense_dim(Tensor self) -> int64_t")(self);
}
int64_t XLAType::_dimV(const Tensor & self) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &)>("_dimV(Tensor self) -> int64_t")(self);
}
int64_t XLAType::_nnz(const Tensor & self) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &)>("_nnz(Tensor self) -> int64_t")(self);
}
Tensor XLAType::coalesce(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("coalesce(Tensor self) -> Tensor")(self);
}
bool XLAType::is_coalesced(const Tensor & self) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &)>("is_coalesced(Tensor self) -> bool")(self);
}
Tensor XLAType::_indices(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_indices(Tensor self) -> Tensor")(self);
}
Tensor XLAType::_values(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("_values(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_coalesced_(Tensor & self, bool coalesced) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, bool)>("_coalesced_(Tensor self, bool coalesced) -> Tensor")(self, coalesced);
}
Tensor XLAType::indices(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("indices(Tensor self) -> Tensor")(self);
}
Tensor XLAType::values(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("values(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::hspmm_out(Tensor & out, const Tensor & mat1, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("hspmm_out(Tensor out, Tensor mat1, Tensor mat2) -> Tensor")(out, mat1, mat2);
}
Tensor XLAType::hspmm(const Tensor & mat1, const Tensor & mat2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("hspmm(Tensor mat1, Tensor mat2) -> Tensor")(mat1, mat2);
}
Tensor & XLAType::copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, bool)>("copy_sparse_to_sparse_(Tensor self, Tensor src, bool non_blocking) -> Tensor")(self, src, non_blocking);
}
int64_t XLAType::numel(const Tensor & self) const {
    return XLATypeDispatch::get_function<int64_t (*)(const Tensor &)>("numel(Tensor self) -> int64_t")(self);
}
std::vector<Tensor> XLAType::unbind(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<std::vector<Tensor> (*)(const Tensor &, int64_t)>("unbind(Tensor self, int64_t dim) -> std::vector<Tensor>")(self, dim);
}
Tensor XLAType::to_sparse(const Tensor & self, int64_t sparse_dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("to_sparse(Tensor self, int64_t sparse_dim) -> Tensor")(self, sparse_dim);
}
Tensor XLAType::to_sparse(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("to_sparse(Tensor self) -> Tensor")(self);
}
Tensor XLAType::to_mkldnn(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("to_mkldnn(Tensor self) -> Tensor")(self);
}
Tensor XLAType::mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>("mkldnn_reorder_conv2d_weight(Tensor self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) -> Tensor")(self, padding, stride, dilation, groups);
}
Tensor XLAType::to_mkldnn_backward(const Tensor & grad, const Tensor & input) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor")(grad, input);
}
Tensor XLAType::quantize_linear(const Tensor & self, double scale, int64_t zero_point) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, int64_t)>("quantize_linear(Tensor self, double scale, int64_t zero_point) -> Tensor")(self, scale, zero_point);
}
Tensor XLAType::dequantize(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("dequantize(Tensor self) -> Tensor")(self);
}
Scalar XLAType::q_scale(const Tensor & self) const {
    return XLATypeDispatch::get_function<Scalar (*)(const Tensor &)>("q_scale(Tensor self) -> Scalar")(self);
}
Scalar XLAType::q_zero_point(const Tensor & self) const {
    return XLATypeDispatch::get_function<Scalar (*)(const Tensor &)>("q_zero_point(Tensor self) -> Scalar")(self);
}
Tensor XLAType::int_repr(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("int_repr(Tensor self) -> Tensor")(self);
}
Tensor XLAType::to(const Tensor & self, const TensorOptions & options, bool non_blocking, bool copy) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const TensorOptions &, bool, bool)>("to(Tensor self, TensorOptions options, bool non_blocking, bool copy) -> Tensor")(self, options, non_blocking, copy);
}
Tensor XLAType::to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Device, ScalarType, bool, bool)>("to(Tensor self, Device device, ScalarType dtype, bool non_blocking, bool copy) -> Tensor")(self, device, dtype, non_blocking, copy);
}
Tensor XLAType::to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, ScalarType, bool, bool)>("to(Tensor self, ScalarType dtype, bool non_blocking, bool copy) -> Tensor")(self, dtype, non_blocking, copy);
}
Tensor XLAType::to(const Tensor & self, const Tensor & other, bool non_blocking, bool copy) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, bool, bool)>("to(Tensor self, Tensor other, bool non_blocking, bool copy) -> Tensor")(self, other, non_blocking, copy);
}
std::vector<Tensor> XLAType::meshgrid(TensorList tensors) const {
    return XLATypeDispatch::get_function<std::vector<Tensor> (*)(TensorList)>("meshgrid(TensorList tensors) -> std::vector<Tensor>")(tensors);
}
Tensor XLAType::cartesian_prod(TensorList tensors) const {
    return XLATypeDispatch::get_function<Tensor (*)(TensorList)>("cartesian_prod(TensorList tensors) -> Tensor")(tensors);
}
Tensor XLAType::combinations(const Tensor & self, int64_t r, bool with_replacement) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("combinations(Tensor self, int64_t r, bool with_replacement) -> Tensor")(self, r, with_replacement);
}
Scalar XLAType::item(const Tensor & self) const {
    return XLATypeDispatch::get_function<Scalar (*)(const Tensor &)>("item(Tensor self) -> Scalar")(self);
}
Scalar XLAType::_local_scalar_dense(const Tensor & self) const {
    return XLATypeDispatch::get_function<Scalar (*)(const Tensor &)>("_local_scalar_dense(Tensor self) -> Scalar")(self);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor input_bias, Tensor hidden_bias) -> std::tuple<Tensor,Tensor,Tensor>")(input_gates, hidden_gates, cx, input_bias, hidden_bias);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> XLAType::_thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool)>("_thnn_fused_lstm_cell_backward(Tensor grad_hy, Tensor grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>")(grad_hy, grad_cy, cx, cy, workspace, has_bias);
}
std::tuple<Tensor,Tensor> XLAType::_thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor input_bias, Tensor hidden_bias) -> std::tuple<Tensor,Tensor>")(input_gates, hidden_gates, hx, input_bias, hidden_bias);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> XLAType::_thnn_fused_gru_cell_backward(const Tensor & grad_hy, const Tensor & workspace, bool has_bias) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, bool)>("_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>")(grad_hy, workspace, has_bias);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, TensorList, TensorList, bool, int64_t, double, bool, bool, bool)>("lstm(Tensor input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor,Tensor>")(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::lstm(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, TensorList, TensorList, bool, int64_t, double, bool, bool)>("lstm(Tensor data, Tensor batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor,Tensor>")(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
std::tuple<Tensor,Tensor> XLAType::gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool)>("gru(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor>")(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor> XLAType::gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool)>("gru(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor>")(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
std::tuple<Tensor,Tensor> XLAType::rnn_tanh(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool)>("rnn_tanh(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor>")(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor> XLAType::rnn_tanh(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool)>("rnn_tanh(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor>")(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
std::tuple<Tensor,Tensor> XLAType::rnn_relu(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool)>("rnn_relu(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor>")(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor> XLAType::rnn_relu(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool)>("rnn_relu(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> std::tuple<Tensor,Tensor>")(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
std::tuple<Tensor,Tensor> XLAType::lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("lstm_cell(Tensor input, TensorList hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) -> std::tuple<Tensor,Tensor>")(input, hx, w_ih, w_hh, b_ih, b_hh);
}
Tensor XLAType::gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) -> Tensor")(input, hx, w_ih, w_hh, b_ih, b_hh);
}
Tensor XLAType::rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) -> Tensor")(input, hx, w_ih, w_hh, b_ih, b_hh);
}
Tensor XLAType::rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) -> Tensor")(input, hx, w_ih, w_hh, b_ih, b_hh);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::quantized_lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, TensorList, TensorList, bool, int64_t, double, bool, bool, bool)>("quantized_lstm(Tensor input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> std::tuple<Tensor,Tensor,Tensor>")(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor> XLAType::quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>("quantized_lstm_cell(Tensor input, TensorList hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> std::tuple<Tensor,Tensor>")(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor XLAType::quantized_gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>("quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor")(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor XLAType::quantized_rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>("quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor")(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor XLAType::quantized_rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>("quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor")(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
std::tuple<Tensor,Tensor> XLAType::_pack_padded_sequence(const Tensor & input, const Tensor & lengths, bool batch_first) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, bool)>("_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> std::tuple<Tensor,Tensor>")(input, lengths, batch_first);
}
Tensor XLAType::_pack_padded_sequence_backward(const Tensor & grad, IntArrayRef input_size, const Tensor & batch_sizes, bool batch_first) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, const Tensor &, bool)>("_pack_padded_sequence_backward(Tensor grad, IntArrayRef input_size, Tensor batch_sizes, bool batch_first) -> Tensor")(grad, input_size, batch_sizes, batch_first);
}
std::tuple<Tensor,Tensor> XLAType::_pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, bool, Scalar, int64_t)>("_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) -> std::tuple<Tensor,Tensor>")(data, batch_sizes, batch_first, padding_value, total_length);
}
void* XLAType::data_ptr(const Tensor & self) const {
    return XLATypeDispatch::get_function<void* (*)(const Tensor &)>("data_ptr(Tensor self) -> void*")(self);
}
Tensor & XLAType::set_(Tensor & self, Storage source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Storage)>("set_(Tensor self, Storage source) -> Tensor")(self, source);
}
Tensor & XLAType::set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>("set_(Tensor self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) -> Tensor")(self, source, storage_offset, size, stride);
}
Tensor & XLAType::set_(Tensor & self, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("set_(Tensor self, Tensor source) -> Tensor")(self, source);
}
Tensor & XLAType::set_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("set_(Tensor self) -> Tensor")(self);
}
bool XLAType::is_set_to(const Tensor & self, const Tensor & tensor) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &, const Tensor &)>("is_set_to(Tensor self, Tensor tensor) -> bool")(self, tensor);
}
Tensor & XLAType::masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("masked_fill_(Tensor self, Tensor mask, Scalar value) -> Tensor")(self, mask, value);
}
Tensor XLAType::masked_fill(const Tensor & self, const Tensor & mask, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("masked_fill(Tensor self, Tensor mask, Scalar value) -> Tensor")(self, mask, value);
}
Tensor & XLAType::masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("masked_fill_(Tensor self, Tensor mask, Tensor value) -> Tensor")(self, mask, value);
}
Tensor XLAType::masked_fill(const Tensor & self, const Tensor & mask, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("masked_fill(Tensor self, Tensor mask, Tensor value) -> Tensor")(self, mask, value);
}
Tensor & XLAType::masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("masked_scatter_(Tensor self, Tensor mask, Tensor source) -> Tensor")(self, mask, source);
}
Tensor XLAType::masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor")(self, mask, source);
}
Tensor XLAType::view(const Tensor & self, IntArrayRef size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("view(Tensor self, IntArrayRef size) -> Tensor")(self, size);
}
Tensor & XLAType::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, bool)>("put_(Tensor self, Tensor index, Tensor source, bool accumulate) -> Tensor")(self, index, source, accumulate);
}
Tensor & XLAType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("index_add_(Tensor self, int64_t dim, Tensor index, Tensor source) -> Tensor")(self, dim, index, source);
}
Tensor XLAType::index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &)>("index_add(Tensor self, int64_t dim, Tensor index, Tensor source) -> Tensor")(self, dim, index, source);
}
Tensor & XLAType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, Scalar)>("index_fill_(Tensor self, int64_t dim, Tensor index, Scalar value) -> Tensor")(self, dim, index, value);
}
Tensor XLAType::index_fill(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, Scalar)>("index_fill(Tensor self, int64_t dim, Tensor index, Scalar value) -> Tensor")(self, dim, index, value);
}
Tensor & XLAType::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("index_fill_(Tensor self, int64_t dim, Tensor index, Tensor value) -> Tensor")(self, dim, index, value);
}
Tensor XLAType::index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &)>("index_fill(Tensor self, int64_t dim, Tensor index, Tensor value) -> Tensor")(self, dim, index, value);
}
Tensor & XLAType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("scatter_(Tensor self, int64_t dim, Tensor index, Tensor src) -> Tensor")(self, dim, index, src);
}
Tensor XLAType::scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &)>("scatter(Tensor self, int64_t dim, Tensor index, Tensor src) -> Tensor")(self, dim, index, src);
}
Tensor & XLAType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, Scalar)>("scatter_(Tensor self, int64_t dim, Tensor index, Scalar value) -> Tensor")(self, dim, index, value);
}
Tensor XLAType::scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, Scalar)>("scatter(Tensor self, int64_t dim, Tensor index, Scalar value) -> Tensor")(self, dim, index, value);
}
Tensor & XLAType::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &, const Tensor &)>("scatter_add_(Tensor self, int64_t dim, Tensor index, Tensor src) -> Tensor")(self, dim, index, src);
}
Tensor XLAType::scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &)>("scatter_add(Tensor self, int64_t dim, Tensor index, Tensor src) -> Tensor")(self, dim, index, src);
}
Tensor & XLAType::lt_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("lt_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::lt_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("lt_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::gt_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("gt_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::gt_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("gt_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::le_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("le_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::le_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("le_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::ge_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("ge_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::ge_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("ge_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::eq_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("eq_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::eq_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("eq_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::ne_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("ne_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::ne_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("ne_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::__and__(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("__and__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor XLAType::__and__(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("__and__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::__iand__(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("__iand__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::__iand__(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("__iand__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::__or__(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("__or__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor XLAType::__or__(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("__or__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::__ior__(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("__ior__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::__ior__(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("__ior__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::__xor__(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("__xor__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor XLAType::__xor__(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("__xor__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::__ixor__(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("__ixor__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::__ixor__(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("__ixor__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::__lshift__(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("__lshift__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor XLAType::__lshift__(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("__lshift__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::__ilshift__(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("__ilshift__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::__ilshift__(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("__ilshift__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::__rshift__(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("__rshift__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor XLAType::__rshift__(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("__rshift__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::__irshift__(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("__irshift__(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::__irshift__(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("__irshift__(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::lgamma_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("lgamma_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::atan2_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("atan2_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::tril_(Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("tril_(Tensor self, int64_t diagonal) -> Tensor")(self, diagonal);
}
Tensor & XLAType::triu_(Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("triu_(Tensor self, int64_t diagonal) -> Tensor")(self, diagonal);
}
Tensor & XLAType::digamma_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("digamma_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::polygamma_(Tensor & self, int64_t n) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t)>("polygamma_(Tensor self, int64_t n) -> Tensor")(self, n);
}
Tensor & XLAType::erfinv_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("erfinv_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, int64_t, Scalar)>("renorm_(Tensor self, Scalar p, int64_t dim, Scalar maxnorm) -> Tensor")(self, p, dim, maxnorm);
}
Tensor & XLAType::pow_(Tensor & self, Scalar exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("pow_(Tensor self, Scalar exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::pow_(Tensor & self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("pow_(Tensor self, Tensor exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("lerp_(Tensor self, Tensor end, Scalar weight) -> Tensor")(self, end, weight);
}
Tensor & XLAType::lerp_(Tensor & self, const Tensor & end, const Tensor & weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("lerp_(Tensor self, Tensor end, Tensor weight) -> Tensor")(self, end, weight);
}
Tensor & XLAType::sign_(Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &)>("sign_(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::fmod_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("fmod_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::fmod_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("fmod_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::remainder_(Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("remainder_(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::remainder_(Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("remainder_(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addbmm_(Tensor self, Tensor batch1, Tensor batch2, Scalar beta, Scalar alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::addbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addbmm_out(Tensor out, Tensor self, Tensor batch1, Tensor batch2, Scalar beta, Scalar alpha) -> Tensor")(out, self, batch1, batch2, beta, alpha);
}
Tensor XLAType::addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("addbmm(Tensor self, Tensor batch1, Tensor batch2, Scalar beta, Scalar alpha) -> Tensor")(self, batch1, batch2, beta, alpha);
}
Tensor & XLAType::addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("addcmul_(Tensor self, Tensor tensor1, Tensor tensor2, Scalar value) -> Tensor")(self, tensor1, tensor2, value);
}
Tensor & XLAType::addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("addcdiv_(Tensor self, Tensor tensor1, Tensor tensor2, Scalar value) -> Tensor")(self, tensor1, tensor2, value);
}
Tensor & XLAType::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, int64_t, Generator *)>("random_(Tensor self, int64_t from, int64_t to, Generator * generator) -> Tensor")(self, from, to, generator);
}
Tensor & XLAType::random_(Tensor & self, int64_t to, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, Generator *)>("random_(Tensor self, int64_t to, Generator * generator) -> Tensor")(self, to, generator);
}
Tensor & XLAType::random_(Tensor & self, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Generator *)>("random_(Tensor self, Generator * generator) -> Tensor")(self, generator);
}
Tensor & XLAType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("uniform_(Tensor self, double from, double to, Generator * generator) -> Tensor")(self, from, to, generator);
}
Tensor & XLAType::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("normal_(Tensor self, double mean, double std, Generator * generator) -> Tensor")(self, mean, std, generator);
}
Tensor & XLAType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("cauchy_(Tensor self, double median, double sigma, Generator * generator) -> Tensor")(self, median, sigma, generator);
}
Tensor & XLAType::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, double, Generator *)>("log_normal_(Tensor self, double mean, double std, Generator * generator) -> Tensor")(self, mean, std, generator);
}
Tensor & XLAType::exponential_(Tensor & self, double lambd, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, Generator *)>("exponential_(Tensor self, double lambd, Generator * generator) -> Tensor")(self, lambd, generator);
}
Tensor & XLAType::geometric_(Tensor & self, double p, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, Generator *)>("geometric_(Tensor self, double p, Generator * generator) -> Tensor")(self, p, generator);
}
Tensor & XLAType::diag_out(Tensor & out, const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("diag_out(Tensor out, Tensor self, int64_t diagonal) -> Tensor")(out, self, diagonal);
}
Tensor XLAType::diag(const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("diag(Tensor self, int64_t diagonal) -> Tensor")(self, diagonal);
}
Tensor & XLAType::cross_out(Tensor & out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, c10::optional<int64_t>)>("cross_out(Tensor out, Tensor self, Tensor other, int64_t dim) -> Tensor")(out, self, other, dim);
}
Tensor XLAType::cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, c10::optional<int64_t>)>("cross(Tensor self, Tensor other, int64_t dim) -> Tensor")(self, other, dim);
}
Tensor & XLAType::triu_out(Tensor & out, const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("triu_out(Tensor out, Tensor self, int64_t diagonal) -> Tensor")(out, self, diagonal);
}
Tensor XLAType::triu(const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("triu(Tensor self, int64_t diagonal) -> Tensor")(self, diagonal);
}
Tensor & XLAType::tril_out(Tensor & out, const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("tril_out(Tensor out, Tensor self, int64_t diagonal) -> Tensor")(out, self, diagonal);
}
Tensor XLAType::tril(const Tensor & self, int64_t diagonal) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("tril(Tensor self, int64_t diagonal) -> Tensor")(self, diagonal);
}
Tensor XLAType::tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, int64_t, int64_t, const TensorOptions &)>("tril_indices(int64_t row, int64_t col, int64_t offset, TensorOptions options) -> Tensor")(row, col, offset, options);
}
Tensor XLAType::triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, int64_t, int64_t, const TensorOptions &)>("triu_indices(int64_t row, int64_t col, int64_t offset, TensorOptions options) -> Tensor")(row, col, offset, options);
}
Tensor XLAType::trace(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("trace(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::ne_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("ne_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::ne(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("ne(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::ne_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("ne_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::ne(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("ne(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::eq_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("eq_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::eq(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("eq(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::eq_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("eq_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::eq(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("eq(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::ge_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("ge_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::ge(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("ge(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::ge_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("ge_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::ge(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("ge(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::le_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("le_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::le(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("le(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::le_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("le_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::le(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("le(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::gt_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("gt_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::gt(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("gt(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::gt_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("gt_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::gt(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("gt(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::lt_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("lt_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::lt(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("lt(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::lt_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("lt_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::lt(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("lt(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::take_out(Tensor & out, const Tensor & self, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("take_out(Tensor out, Tensor self, Tensor index) -> Tensor")(out, self, index);
}
Tensor XLAType::take(const Tensor & self, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("take(Tensor self, Tensor index) -> Tensor")(self, index);
}
Tensor & XLAType::index_select_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &)>("index_select_out(Tensor out, Tensor self, int64_t dim, Tensor index) -> Tensor")(out, self, dim, index);
}
Tensor XLAType::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &)>("index_select(Tensor self, int64_t dim, Tensor index) -> Tensor")(self, dim, index);
}
Tensor & XLAType::masked_select_out(Tensor & out, const Tensor & self, const Tensor & mask) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("masked_select_out(Tensor out, Tensor self, Tensor mask) -> Tensor")(out, self, mask);
}
Tensor XLAType::masked_select(const Tensor & self, const Tensor & mask) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("masked_select(Tensor self, Tensor mask) -> Tensor")(self, mask);
}
Tensor & XLAType::nonzero_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("nonzero_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::nonzero(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("nonzero(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::gather_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, const Tensor &, bool)>("gather_out(Tensor out, Tensor self, int64_t dim, Tensor index, bool sparse_grad) -> Tensor")(out, self, dim, index, sparse_grad);
}
Tensor XLAType::gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, bool)>("gather(Tensor self, int64_t dim, Tensor index, bool sparse_grad) -> Tensor")(self, dim, index, sparse_grad);
}
Tensor XLAType::_gather_sparse_backward(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, const Tensor &, const Tensor &)>("_gather_sparse_backward(Tensor self, int64_t dim, Tensor index, Tensor grad) -> Tensor")(self, dim, index, grad);
}
Tensor & XLAType::addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>("addcmul_out(Tensor out, Tensor self, Tensor tensor1, Tensor tensor2, Scalar value) -> Tensor")(out, self, tensor1, tensor2, value);
}
Tensor XLAType::addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar)>("addcmul(Tensor self, Tensor tensor1, Tensor tensor2, Scalar value) -> Tensor")(self, tensor1, tensor2, value);
}
Tensor & XLAType::addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>("addcdiv_out(Tensor out, Tensor self, Tensor tensor1, Tensor tensor2, Scalar value) -> Tensor")(out, self, tensor1, tensor2, value);
}
Tensor XLAType::addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar)>("addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, Scalar value) -> Tensor")(self, tensor1, tensor2, value);
}
std::tuple<Tensor &,Tensor &> XLAType::gels_out(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &)>("gels_out(Tensor X, Tensor qr, Tensor self, Tensor A) -> std::tuple<Tensor &,Tensor &>")(X, qr, self, A);
}
std::tuple<Tensor,Tensor> XLAType::gels(const Tensor & self, const Tensor & A) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &)>("gels(Tensor self, Tensor A) -> std::tuple<Tensor,Tensor>")(self, A);
}
std::tuple<Tensor &,Tensor &> XLAType::triangular_solve_out(Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool, bool)>("triangular_solve_out(Tensor X, Tensor M, Tensor self, Tensor A, bool upper, bool transpose, bool unitriangular) -> std::tuple<Tensor &,Tensor &>")(X, M, self, A, upper, transpose, unitriangular);
}
std::tuple<Tensor,Tensor> XLAType::triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, bool, bool, bool)>("triangular_solve(Tensor self, Tensor A, bool upper, bool transpose, bool unitriangular) -> std::tuple<Tensor,Tensor>")(self, A, upper, transpose, unitriangular);
}
std::tuple<Tensor,Tensor> XLAType::_triangular_solve_helper(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, bool, bool, bool)>("_triangular_solve_helper(Tensor self, Tensor A, bool upper, bool transpose, bool unitriangular) -> std::tuple<Tensor,Tensor>")(self, A, upper, transpose, unitriangular);
}
std::tuple<Tensor &,Tensor &> XLAType::symeig_out(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, bool, bool)>("symeig_out(Tensor e, Tensor V, Tensor self, bool eigenvectors, bool upper) -> std::tuple<Tensor &,Tensor &>")(e, V, self, eigenvectors, upper);
}
std::tuple<Tensor,Tensor> XLAType::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, bool, bool)>("symeig(Tensor self, bool eigenvectors, bool upper) -> std::tuple<Tensor,Tensor>")(self, eigenvectors, upper);
}
std::tuple<Tensor &,Tensor &> XLAType::eig_out(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, bool)>("eig_out(Tensor e, Tensor v, Tensor self, bool eigenvectors) -> std::tuple<Tensor &,Tensor &>")(e, v, self, eigenvectors);
}
std::tuple<Tensor,Tensor> XLAType::eig(const Tensor & self, bool eigenvectors) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, bool)>("eig(Tensor self, bool eigenvectors) -> std::tuple<Tensor,Tensor>")(self, eigenvectors);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::svd_out(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, bool, bool)>("svd_out(Tensor U, Tensor S, Tensor V, Tensor self, bool some, bool compute_uv) -> std::tuple<Tensor &,Tensor &,Tensor &>")(U, S, V, self, some, compute_uv);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::svd(const Tensor & self, bool some, bool compute_uv) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, bool, bool)>("svd(Tensor self, bool some, bool compute_uv) -> std::tuple<Tensor,Tensor,Tensor>")(self, some, compute_uv);
}
Tensor & XLAType::cholesky_out(Tensor & out, const Tensor & self, bool upper) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, bool)>("cholesky_out(Tensor out, Tensor self, bool upper) -> Tensor")(out, self, upper);
}
Tensor XLAType::cholesky(const Tensor & self, bool upper) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("cholesky(Tensor self, bool upper) -> Tensor")(self, upper);
}
Tensor XLAType::_cholesky_helper(const Tensor & self, bool upper) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("_cholesky_helper(Tensor self, bool upper) -> Tensor")(self, upper);
}
Tensor & XLAType::cholesky_solve_out(Tensor & out, const Tensor & self, const Tensor & input2, bool upper) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, bool)>("cholesky_solve_out(Tensor out, Tensor self, Tensor input2, bool upper) -> Tensor")(out, self, input2, upper);
}
Tensor XLAType::cholesky_solve(const Tensor & self, const Tensor & input2, bool upper) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, bool)>("cholesky_solve(Tensor self, Tensor input2, bool upper) -> Tensor")(self, input2, upper);
}
Tensor XLAType::_cholesky_solve_helper(const Tensor & self, const Tensor & A, bool upper) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, bool)>("_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor")(self, A, upper);
}
std::tuple<Tensor,Tensor> XLAType::solve(const Tensor & self, const Tensor & A) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &)>("solve(Tensor self, Tensor A) -> std::tuple<Tensor,Tensor>")(self, A);
}
std::tuple<Tensor &,Tensor &> XLAType::solve_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &)>("solve_out(Tensor solution, Tensor lu, Tensor self, Tensor A) -> std::tuple<Tensor &,Tensor &>")(solution, lu, self, A);
}
std::tuple<Tensor,Tensor> XLAType::_solve_helper(const Tensor & self, const Tensor & A) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &)>("_solve_helper(Tensor self, Tensor A) -> std::tuple<Tensor,Tensor>")(self, A);
}
Tensor & XLAType::cholesky_inverse_out(Tensor & out, const Tensor & self, bool upper) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, bool)>("cholesky_inverse_out(Tensor out, Tensor self, bool upper) -> Tensor")(out, self, upper);
}
Tensor XLAType::cholesky_inverse(const Tensor & self, bool upper) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, bool)>("cholesky_inverse(Tensor self, bool upper) -> Tensor")(self, upper);
}
std::tuple<Tensor &,Tensor &> XLAType::pstrf_out(Tensor & u, Tensor & pivot, const Tensor & self, bool upper, Scalar tol) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, bool, Scalar)>("pstrf_out(Tensor u, Tensor pivot, Tensor self, bool upper, Scalar tol) -> std::tuple<Tensor &,Tensor &>")(u, pivot, self, upper, tol);
}
std::tuple<Tensor,Tensor> XLAType::pstrf(const Tensor & self, bool upper, Scalar tol) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, bool, Scalar)>("pstrf(Tensor self, bool upper, Scalar tol) -> std::tuple<Tensor,Tensor>")(self, upper, tol);
}
std::tuple<Tensor &,Tensor &> XLAType::qr_out(Tensor & Q, Tensor & R, const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &)>("qr_out(Tensor Q, Tensor R, Tensor self) -> std::tuple<Tensor &,Tensor &>")(Q, R, self);
}
std::tuple<Tensor,Tensor> XLAType::qr(const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("qr(Tensor self) -> std::tuple<Tensor,Tensor>")(self);
}
std::tuple<Tensor &,Tensor &> XLAType::geqrf_out(Tensor & a, Tensor & tau, const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &)>("geqrf_out(Tensor a, Tensor tau, Tensor self) -> std::tuple<Tensor &,Tensor &>")(a, tau, self);
}
std::tuple<Tensor,Tensor> XLAType::geqrf(const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("geqrf(Tensor self) -> std::tuple<Tensor,Tensor>")(self);
}
Tensor & XLAType::orgqr_out(Tensor & out, const Tensor & self, const Tensor & input2) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("orgqr_out(Tensor out, Tensor self, Tensor input2) -> Tensor")(out, self, input2);
}
Tensor XLAType::orgqr(const Tensor & self, const Tensor & input2) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("orgqr(Tensor self, Tensor input2) -> Tensor")(self, input2);
}
Tensor & XLAType::ormqr_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool)>("ormqr_out(Tensor out, Tensor self, Tensor input2, Tensor input3, bool left, bool transpose) -> Tensor")(out, self, input2, input3, left, transpose);
}
Tensor XLAType::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, bool, bool)>("ormqr(Tensor self, Tensor input2, Tensor input3, bool left, bool transpose) -> Tensor")(self, input2, input3, left, transpose);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::_lu_with_info(const Tensor & self, bool pivot, bool check_errors) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, bool, bool)>("_lu_with_info(Tensor self, bool pivot, bool check_errors) -> std::tuple<Tensor,Tensor,Tensor>")(self, pivot, check_errors);
}
Tensor & XLAType::lu_solve_out(Tensor & out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("lu_solve_out(Tensor out, Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor")(out, self, LU_data, LU_pivots);
}
Tensor XLAType::lu_solve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor")(self, LU_data, LU_pivots);
}
Tensor & XLAType::multinomial_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, bool, Generator *)>("multinomial_out(Tensor out, Tensor self, int64_t num_samples, bool replacement, Generator * generator) -> Tensor")(out, self, num_samples, replacement, generator);
}
Tensor XLAType::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool, Generator *)>("multinomial(Tensor self, int64_t num_samples, bool replacement, Generator * generator) -> Tensor")(self, num_samples, replacement, generator);
}
std::tuple<Tensor,Tensor> XLAType::_multinomial_alias_setup(const Tensor & probs) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("_multinomial_alias_setup(Tensor probs) -> std::tuple<Tensor,Tensor>")(probs);
}
Tensor XLAType::_multinomial_alias_draw(const Tensor & J, const Tensor & q, int64_t num_samples, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t, Generator *)>("_multinomial_alias_draw(Tensor J, Tensor q, int64_t num_samples, Generator * generator) -> Tensor")(J, q, num_samples, generator);
}
Tensor & XLAType::lgamma_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("lgamma_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::lgamma(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("lgamma(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::digamma_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("digamma_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::digamma(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("digamma(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::polygamma_out(Tensor & out, int64_t n, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, int64_t, const Tensor &)>("polygamma_out(Tensor out, int64_t n, Tensor self) -> Tensor")(out, n, self);
}
Tensor XLAType::polygamma(int64_t n, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(int64_t, const Tensor &)>("polygamma(int64_t n, Tensor self) -> Tensor")(n, self);
}
Tensor & XLAType::erfinv_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("erfinv_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::erfinv(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("erfinv(Tensor self) -> Tensor")(self);
}
Tensor XLAType::dist(const Tensor & self, const Tensor & other, Scalar p) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("dist(Tensor self, Tensor other, Scalar p) -> Tensor")(self, other, p);
}
Tensor & XLAType::atan2_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("atan2_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::atan2(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("atan2(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::lerp_out(Tensor & out, const Tensor & self, const Tensor & end, Scalar weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("lerp_out(Tensor out, Tensor self, Tensor end, Scalar weight) -> Tensor")(out, self, end, weight);
}
Tensor & XLAType::lerp_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("lerp_out(Tensor out, Tensor self, Tensor end, Tensor weight) -> Tensor")(out, self, end, weight);
}
Tensor XLAType::lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("lerp(Tensor self, Tensor end, Scalar weight) -> Tensor")(self, end, weight);
}
Tensor XLAType::lerp(const Tensor & self, const Tensor & end, const Tensor & weight) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("lerp(Tensor self, Tensor end, Tensor weight) -> Tensor")(self, end, weight);
}
Tensor & XLAType::histc_out(Tensor & out, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t, Scalar, Scalar)>("histc_out(Tensor out, Tensor self, int64_t bins, Scalar min, Scalar max) -> Tensor")(out, self, bins, min, max);
}
Tensor XLAType::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, Scalar, Scalar)>("histc(Tensor self, int64_t bins, Scalar min, Scalar max) -> Tensor")(self, bins, min, max);
}
Tensor & XLAType::sign_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("sign_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::sign(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("sign(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::fmod_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("fmod_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::fmod(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("fmod(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::fmod_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("fmod_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::fmod(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("fmod(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::remainder_out(Tensor & out, const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("remainder_out(Tensor out, Tensor self, Scalar other) -> Tensor")(out, self, other);
}
Tensor XLAType::remainder(const Tensor & self, Scalar other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("remainder(Tensor self, Scalar other) -> Tensor")(self, other);
}
Tensor & XLAType::remainder_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("remainder_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::remainder(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("remainder(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor & XLAType::min_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("min_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::min(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("min(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::min(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("min(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::max_out(Tensor & out, const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("max_out(Tensor out, Tensor self, Tensor other) -> Tensor")(out, self, other);
}
Tensor XLAType::max(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("max(Tensor self, Tensor other) -> Tensor")(self, other);
}
Tensor XLAType::max(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("max(Tensor self) -> Tensor")(self);
}
Tensor XLAType::median(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("median(Tensor self) -> Tensor")(self);
}
std::tuple<Tensor &,Tensor &> XLAType::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, bool)>("sort_out(Tensor values, Tensor indices, Tensor self, int64_t dim, bool descending) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, dim, descending);
}
std::tuple<Tensor,Tensor> XLAType::sort(const Tensor & self, int64_t dim, bool descending) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, bool)>("sort(Tensor self, int64_t dim, bool descending) -> std::tuple<Tensor,Tensor>")(self, dim, descending);
}
Tensor XLAType::argsort(const Tensor & self, int64_t dim, bool descending) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, bool)>("argsort(Tensor self, int64_t dim, bool descending) -> Tensor")(self, dim, descending);
}
std::tuple<Tensor &,Tensor &> XLAType::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>("topk_out(Tensor values, Tensor indices, Tensor self, int64_t k, int64_t dim, bool largest, bool sorted) -> std::tuple<Tensor &,Tensor &>")(values, indices, self, k, dim, largest, sorted);
}
std::tuple<Tensor,Tensor> XLAType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, int64_t, int64_t, bool, bool)>("topk(Tensor self, int64_t k, int64_t dim, bool largest, bool sorted) -> std::tuple<Tensor,Tensor>")(self, k, dim, largest, sorted);
}
Tensor XLAType::all(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("all(Tensor self) -> Tensor")(self);
}
Tensor XLAType::any(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("any(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::renorm_out(Tensor & out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, int64_t, Scalar)>("renorm_out(Tensor out, Tensor self, Scalar p, int64_t dim, Scalar maxnorm) -> Tensor")(out, self, p, dim, maxnorm);
}
Tensor XLAType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, int64_t, Scalar)>("renorm(Tensor self, Scalar p, int64_t dim, Scalar maxnorm) -> Tensor")(self, p, dim, maxnorm);
}
Tensor XLAType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t, int64_t, int64_t)>("unfold(Tensor self, int64_t dimension, int64_t size, int64_t step) -> Tensor")(self, dimension, size, step);
}
bool XLAType::equal(const Tensor & self, const Tensor & other) const {
    return XLATypeDispatch::get_function<bool (*)(const Tensor &, const Tensor &)>("equal(Tensor self, Tensor other) -> bool")(self, other);
}
Tensor & XLAType::pow_out(Tensor & out, const Tensor & self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("pow_out(Tensor out, Tensor self, Tensor exponent) -> Tensor")(out, self, exponent);
}
Tensor XLAType::pow(const Tensor & self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("pow(Tensor self, Tensor exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::pow_out(Tensor & out, Scalar self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, const Tensor &)>("pow_out(Tensor out, Scalar self, Tensor exponent) -> Tensor")(out, self, exponent);
}
Tensor XLAType::pow(Scalar self, const Tensor & exponent) const {
    return XLATypeDispatch::get_function<Tensor (*)(Scalar, const Tensor &)>("pow(Scalar self, Tensor exponent) -> Tensor")(self, exponent);
}
Tensor & XLAType::normal_out(Tensor & out, const Tensor & mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, double, Generator *)>("normal_out(Tensor out, Tensor mean, double std, Generator * generator) -> Tensor")(out, mean, std, generator);
}
Tensor XLAType::normal(const Tensor & mean, double std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, double, Generator *)>("normal(Tensor mean, double std, Generator * generator) -> Tensor")(mean, std, generator);
}
Tensor & XLAType::normal_out(Tensor & out, double mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, double, const Tensor &, Generator *)>("normal_out(Tensor out, double mean, Tensor std, Generator * generator) -> Tensor")(out, mean, std, generator);
}
Tensor XLAType::normal(double mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(double, const Tensor &, Generator *)>("normal(double mean, Tensor std, Generator * generator) -> Tensor")(mean, std, generator);
}
Tensor & XLAType::normal_out(Tensor & out, const Tensor & mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Generator *)>("normal_out(Tensor out, Tensor mean, Tensor std, Generator * generator) -> Tensor")(out, mean, std, generator);
}
Tensor XLAType::normal(const Tensor & mean, const Tensor & std, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Generator *)>("normal(Tensor mean, Tensor std, Generator * generator) -> Tensor")(mean, std, generator);
}
Tensor XLAType::alias(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("alias(Tensor self) -> Tensor")(self);
}
Tensor & XLAType::_dirichlet_grad_out(Tensor & out, const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("_dirichlet_grad_out(Tensor out, Tensor x, Tensor alpha, Tensor total) -> Tensor")(out, x, alpha, total);
}
Tensor XLAType::_dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor")(x, alpha, total);
}
Tensor & XLAType::binary_cross_entropy_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("binary_cross_entropy_out(Tensor out, Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(out, self, target, weight, reduction);
}
Tensor XLAType::binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("binary_cross_entropy(Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(self, target, weight, reduction);
}
Tensor & XLAType::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("binary_cross_entropy_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, weight, reduction);
}
Tensor XLAType::binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction) -> Tensor")(grad_output, self, target, weight, reduction);
}
Tensor & XLAType::mse_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("mse_loss_out(Tensor out, Tensor self, Tensor target, int64_t reduction) -> Tensor")(out, self, target, reduction);
}
Tensor XLAType::mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("mse_loss(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("mse_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("l1_loss_out(Tensor out, Tensor self, Tensor target, int64_t reduction) -> Tensor")(out, self, target, reduction);
}
Tensor XLAType::l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("l1_loss(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("l1_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::multi_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("multi_margin_loss_out(Tensor out, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor weight, int64_t reduction) -> Tensor")(out, self, target, p, margin, weight, reduction);
}
Tensor XLAType::multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("multi_margin_loss(Tensor self, Tensor target, Scalar p, Scalar margin, Tensor weight, int64_t reduction) -> Tensor")(self, target, p, margin, weight, reduction);
}
Tensor & XLAType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("multi_margin_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor weight, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, p, margin, weight, reduction);
}
Tensor XLAType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor weight, int64_t reduction) -> Tensor")(grad_output, self, target, p, margin, weight, reduction);
}
Tensor & XLAType::multilabel_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("multilabel_margin_loss_out(Tensor out, Tensor self, Tensor target, int64_t reduction) -> Tensor")(out, self, target, reduction);
}
Tensor XLAType::multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("multilabel_margin_loss(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
std::tuple<Tensor &,Tensor &> XLAType::multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t)>("multilabel_margin_loss_forward_out(Tensor output, Tensor is_target, Tensor self, Tensor target, int64_t reduction) -> std::tuple<Tensor &,Tensor &>")(output, is_target, self, target, reduction);
}
std::tuple<Tensor,Tensor> XLAType::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, int64_t)>("multilabel_margin_loss_forward(Tensor self, Tensor target, int64_t reduction) -> std::tuple<Tensor,Tensor>")(self, target, reduction);
}
Tensor & XLAType::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>("multilabel_margin_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction, Tensor is_target) -> Tensor")(grad_input, grad_output, self, target, reduction, is_target);
}
Tensor XLAType::multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>("multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction, Tensor is_target) -> Tensor")(grad_output, self, target, reduction, is_target);
}
Tensor & XLAType::nll_loss_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss_out(Tensor out, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> Tensor")(out, self, target, weight, reduction, ignore_index);
}
Tensor XLAType::nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss(Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> Tensor")(self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor &,Tensor &> XLAType::nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss_forward_out(Tensor output, Tensor total_weight, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor &,Tensor &>")(output, total_weight, self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor,Tensor> XLAType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss_forward(Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor,Tensor>")(self, target, weight, reduction, ignore_index);
}
Tensor & XLAType::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("nll_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor XLAType::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor & XLAType::nll_loss2d_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss2d_out(Tensor out, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> Tensor")(out, self, target, weight, reduction, ignore_index);
}
Tensor XLAType::nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss2d(Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> Tensor")(self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor &,Tensor &> XLAType::nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss2d_forward_out(Tensor output, Tensor total_weight, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor &,Tensor &>")(output, total_weight, self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor,Tensor> XLAType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("nll_loss2d_forward(Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index) -> std::tuple<Tensor,Tensor>")(self, target, weight, reduction, ignore_index);
}
Tensor & XLAType::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("nll_loss2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor XLAType::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>("nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor weight, int64_t reduction, int64_t ignore_index, Tensor total_weight) -> Tensor")(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor & XLAType::smooth_l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("smooth_l1_loss_out(Tensor out, Tensor self, Tensor target, int64_t reduction) -> Tensor")(out, self, target, reduction);
}
Tensor XLAType::smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("smooth_l1_loss(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("smooth_l1_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::soft_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("soft_margin_loss_out(Tensor out, Tensor self, Tensor target, int64_t reduction) -> Tensor")(out, self, target, reduction);
}
Tensor XLAType::soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("soft_margin_loss(Tensor self, Tensor target, int64_t reduction) -> Tensor")(self, target, reduction);
}
Tensor & XLAType::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("soft_margin_loss_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_input, grad_output, self, target, reduction);
}
Tensor XLAType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t)>("soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction) -> Tensor")(grad_output, self, target, reduction);
}
Tensor & XLAType::elu_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar, Scalar)>("elu_out(Tensor out, Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor")(out, self, alpha, scale, input_scale);
}
Tensor XLAType::elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar, Scalar)>("elu(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor")(self, alpha, scale, input_scale);
}
Tensor & XLAType::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>("elu_backward_out(Tensor grad_input, Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor")(grad_input, grad_output, alpha, scale, input_scale, output);
}
Tensor XLAType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>("elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor")(grad_output, alpha, scale, input_scale, output);
}
Tensor & XLAType::elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar, Scalar)>("elu_(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor")(self, alpha, scale, input_scale);
}
Tensor & XLAType::glu_out(Tensor & out, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, int64_t)>("glu_out(Tensor out, Tensor self, int64_t dim) -> Tensor")(out, self, dim);
}
Tensor XLAType::glu(const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, int64_t)>("glu(Tensor self, int64_t dim) -> Tensor")(self, dim);
}
Tensor & XLAType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, int64_t)>("glu_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, int64_t dim) -> Tensor")(grad_input, grad_output, self, dim);
}
Tensor XLAType::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, int64_t)>("glu_backward(Tensor grad_output, Tensor self, int64_t dim) -> Tensor")(grad_output, self, dim);
}
Tensor & XLAType::hardtanh_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar)>("hardtanh_out(Tensor out, Tensor self, Scalar min_val, Scalar max_val) -> Tensor")(out, self, min_val, max_val);
}
Tensor XLAType::hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor")(self, min_val, max_val);
}
Tensor & XLAType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("hardtanh_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor")(grad_input, grad_output, self, min_val, max_val);
}
Tensor XLAType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar)>("hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor")(grad_output, self, min_val, max_val);
}
Tensor & XLAType::hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar, Scalar)>("hardtanh_(Tensor self, Scalar min_val, Scalar max_val) -> Tensor")(self, min_val, max_val);
}
Tensor & XLAType::leaky_relu_out(Tensor & out, const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("leaky_relu_out(Tensor out, Tensor self, Scalar negative_slope) -> Tensor")(out, self, negative_slope);
}
Tensor XLAType::leaky_relu(const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("leaky_relu(Tensor self, Scalar negative_slope) -> Tensor")(self, negative_slope);
}
Tensor & XLAType::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("leaky_relu_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Scalar negative_slope) -> Tensor")(grad_input, grad_output, self, negative_slope);
}
Tensor XLAType::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope) -> Tensor")(grad_output, self, negative_slope);
}
Tensor & XLAType::leaky_relu_(Tensor & self, Scalar negative_slope) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, Scalar)>("leaky_relu_(Tensor self, Scalar negative_slope) -> Tensor")(self, negative_slope);
}
Tensor & XLAType::log_sigmoid_out(Tensor & out, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &)>("log_sigmoid_out(Tensor out, Tensor self) -> Tensor")(out, self);
}
Tensor XLAType::log_sigmoid(const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &)>("log_sigmoid(Tensor self) -> Tensor")(self);
}
std::tuple<Tensor &,Tensor &> XLAType::log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &)>("log_sigmoid_forward_out(Tensor output, Tensor buffer, Tensor self) -> std::tuple<Tensor &,Tensor &>")(output, buffer, self);
}
std::tuple<Tensor,Tensor> XLAType::log_sigmoid_forward(const Tensor & self) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &)>("log_sigmoid_forward(Tensor self) -> std::tuple<Tensor,Tensor>")(self);
}
Tensor & XLAType::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("log_sigmoid_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor buffer) -> Tensor")(grad_input, grad_output, self, buffer);
}
Tensor XLAType::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor")(grad_output, self, buffer);
}
Tensor & XLAType::rrelu_with_noise_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *)>("rrelu_with_noise_out(Tensor out, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor")(out, self, noise, lower, upper, training, generator);
}
Tensor XLAType::rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *)>("rrelu_with_noise(Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor")(self, noise, lower, upper, training, generator);
}
Tensor & XLAType::rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool)>("rrelu_with_noise_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training) -> Tensor")(grad_input, grad_output, self, noise, lower, upper, training);
}
Tensor XLAType::rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool)>("rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training) -> Tensor")(grad_output, self, noise, lower, upper, training);
}
Tensor & XLAType::rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar, bool, Generator *)>("rrelu_with_noise_(Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator * generator) -> Tensor")(self, noise, lower, upper, training, generator);
}
Tensor & XLAType::softplus_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar, Scalar)>("softplus_out(Tensor out, Tensor self, Scalar beta, Scalar threshold) -> Tensor")(out, self, beta, threshold);
}
Tensor XLAType::softplus(const Tensor & self, Scalar beta, Scalar threshold) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar, Scalar)>("softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor")(self, beta, threshold);
}
Tensor & XLAType::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>("softplus_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor")(grad_input, grad_output, self, beta, threshold, output);
}
Tensor XLAType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>("softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor")(grad_output, self, beta, threshold, output);
}
Tensor & XLAType::softshrink_out(Tensor & out, const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, Scalar)>("softshrink_out(Tensor out, Tensor self, Scalar lambd) -> Tensor")(out, self, lambd);
}
Tensor XLAType::softshrink(const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, Scalar)>("softshrink(Tensor self, Scalar lambd) -> Tensor")(self, lambd);
}
Tensor & XLAType::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, Scalar)>("softshrink_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Scalar lambd) -> Tensor")(grad_input, grad_output, self, lambd);
}
Tensor XLAType::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, Scalar)>("softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor")(grad_output, self, lambd);
}
Tensor & XLAType::adaptive_avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("adaptive_avg_pool2d_out(Tensor out, Tensor self, IntArrayRef output_size) -> Tensor")(out, self, output_size);
}
Tensor XLAType::adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("adaptive_avg_pool2d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor XLAType::mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("mkldnn_adaptive_avg_pool2d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor XLAType::_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("_adaptive_avg_pool2d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor XLAType::_adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor")(grad_output, self);
}
Tensor & XLAType::adaptive_avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("adaptive_avg_pool3d_out(Tensor out, Tensor self, IntArrayRef output_size) -> Tensor")(out, self, output_size);
}
Tensor XLAType::adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("adaptive_avg_pool3d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("adaptive_avg_pool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self) -> Tensor")(grad_input, grad_output, self);
}
Tensor XLAType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor")(grad_output, self);
}
std::tuple<Tensor &,Tensor &> XLAType::adaptive_max_pool2d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef)>("adaptive_max_pool2d_out(Tensor out, Tensor indices, Tensor self, IntArrayRef output_size) -> std::tuple<Tensor &,Tensor &>")(out, indices, self, output_size);
}
std::tuple<Tensor,Tensor> XLAType::adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef)>("adaptive_max_pool2d(Tensor self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor>")(self, output_size);
}
Tensor & XLAType::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("adaptive_max_pool2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor indices) -> Tensor")(grad_input, grad_output, self, indices);
}
Tensor XLAType::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor")(grad_output, self, indices);
}
std::tuple<Tensor &,Tensor &> XLAType::adaptive_max_pool3d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef)>("adaptive_max_pool3d_out(Tensor out, Tensor indices, Tensor self, IntArrayRef output_size) -> std::tuple<Tensor &,Tensor &>")(out, indices, self, output_size);
}
std::tuple<Tensor,Tensor> XLAType::adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef)>("adaptive_max_pool3d(Tensor self, IntArrayRef output_size) -> std::tuple<Tensor,Tensor>")(self, output_size);
}
Tensor & XLAType::adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &)>("adaptive_max_pool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor indices) -> Tensor")(grad_input, grad_output, self, indices);
}
Tensor XLAType::adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &)>("adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor")(grad_output, self, indices);
}
Tensor & XLAType::avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool2d_out(Tensor out, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool2d(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & XLAType::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool2d_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & XLAType::avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool3d_out(Tensor out, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool3d(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & XLAType::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor XLAType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("avg_pool3d_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> Tensor")(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
std::tuple<Tensor &,Tensor &> XLAType::fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool2d_out(Tensor output, Tensor indices, Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor random_samples) -> std::tuple<Tensor &,Tensor &>")(output, indices, self, kernel_size, output_size, random_samples);
}
std::tuple<Tensor,Tensor> XLAType::fractional_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool2d(Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor random_samples) -> std::tuple<Tensor,Tensor>")(self, kernel_size, output_size, random_samples);
}
Tensor & XLAType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor indices) -> Tensor")(grad_input, grad_output, self, kernel_size, output_size, indices);
}
Tensor XLAType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool2d_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor indices) -> Tensor")(grad_output, self, kernel_size, output_size, indices);
}
std::tuple<Tensor &,Tensor &> XLAType::fractional_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool3d_out(Tensor output, Tensor indices, Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor random_samples) -> std::tuple<Tensor &,Tensor &>")(output, indices, self, kernel_size, output_size, random_samples);
}
std::tuple<Tensor,Tensor> XLAType::fractional_max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool3d(Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor random_samples) -> std::tuple<Tensor,Tensor>")(self, kernel_size, output_size, random_samples);
}
Tensor & XLAType::fractional_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor indices) -> Tensor")(grad_input, grad_output, self, kernel_size, output_size, indices);
}
Tensor XLAType::fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("fractional_max_pool3d_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef output_size, Tensor indices) -> Tensor")(grad_output, self, kernel_size, output_size, indices);
}
std::tuple<Tensor &,Tensor &> XLAType::max_pool2d_with_indices_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool2d_with_indices_out(Tensor output, Tensor indices, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor &,Tensor &>")(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> XLAType::max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool2d_with_indices(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor>")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & XLAType::max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("max_pool2d_with_indices_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor indices) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor XLAType::max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor indices) -> Tensor")(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
std::tuple<Tensor &,Tensor &> XLAType::max_pool3d_with_indices_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool3d_with_indices_out(Tensor output, Tensor indices, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor &,Tensor &>")(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> XLAType::max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("max_pool3d_with_indices(Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) -> std::tuple<Tensor,Tensor>")(self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & XLAType::max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("max_pool3d_with_indices_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor indices) -> Tensor")(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor XLAType::max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor indices) -> Tensor")(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor & XLAType::max_unpool2d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("max_unpool2d_out(Tensor out, Tensor self, Tensor indices, IntArrayRef output_size) -> Tensor")(out, self, indices, output_size);
}
Tensor XLAType::max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("max_unpool2d(Tensor self, Tensor indices, IntArrayRef output_size) -> Tensor")(self, indices, output_size);
}
Tensor & XLAType::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("max_unpool2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor indices, IntArrayRef output_size) -> Tensor")(grad_input, grad_output, self, indices, output_size);
}
Tensor XLAType::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, IntArrayRef output_size) -> Tensor")(grad_output, self, indices, output_size);
}
Tensor & XLAType::max_unpool3d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("max_unpool3d_out(Tensor out, Tensor self, Tensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(out, self, indices, output_size, stride, padding);
}
Tensor XLAType::max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("max_unpool3d(Tensor self, Tensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(self, indices, output_size, stride, padding);
}
Tensor & XLAType::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("max_unpool3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, Tensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(grad_input, grad_output, self, indices, output_size, stride, padding);
}
Tensor XLAType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) -> Tensor")(grad_output, self, indices, output_size, stride, padding);
}
Tensor & XLAType::reflection_pad1d_out(Tensor & out, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("reflection_pad1d_out(Tensor out, Tensor self, IntArrayRef padding) -> Tensor")(out, self, padding);
}
Tensor XLAType::reflection_pad1d(const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("reflection_pad1d(Tensor self, IntArrayRef padding) -> Tensor")(self, padding);
}
Tensor & XLAType::reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("reflection_pad1d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_input, grad_output, self, padding);
}
Tensor XLAType::reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("reflection_pad1d_backward(Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_output, self, padding);
}
Tensor & XLAType::reflection_pad2d_out(Tensor & out, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("reflection_pad2d_out(Tensor out, Tensor self, IntArrayRef padding) -> Tensor")(out, self, padding);
}
Tensor XLAType::reflection_pad2d(const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("reflection_pad2d(Tensor self, IntArrayRef padding) -> Tensor")(self, padding);
}
Tensor & XLAType::reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("reflection_pad2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_input, grad_output, self, padding);
}
Tensor XLAType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("reflection_pad2d_backward(Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_output, self, padding);
}
Tensor & XLAType::replication_pad1d_out(Tensor & out, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("replication_pad1d_out(Tensor out, Tensor self, IntArrayRef padding) -> Tensor")(out, self, padding);
}
Tensor XLAType::replication_pad1d(const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("replication_pad1d(Tensor self, IntArrayRef padding) -> Tensor")(self, padding);
}
Tensor & XLAType::replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("replication_pad1d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_input, grad_output, self, padding);
}
Tensor XLAType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("replication_pad1d_backward(Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_output, self, padding);
}
Tensor & XLAType::replication_pad2d_out(Tensor & out, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("replication_pad2d_out(Tensor out, Tensor self, IntArrayRef padding) -> Tensor")(out, self, padding);
}
Tensor XLAType::replication_pad2d(const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("replication_pad2d(Tensor self, IntArrayRef padding) -> Tensor")(self, padding);
}
Tensor & XLAType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("replication_pad2d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_input, grad_output, self, padding);
}
Tensor XLAType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("replication_pad2d_backward(Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_output, self, padding);
}
Tensor & XLAType::replication_pad3d_out(Tensor & out, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("replication_pad3d_out(Tensor out, Tensor self, IntArrayRef padding) -> Tensor")(out, self, padding);
}
Tensor XLAType::replication_pad3d(const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("replication_pad3d(Tensor self, IntArrayRef padding) -> Tensor")(self, padding);
}
Tensor & XLAType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("replication_pad3d_backward_out(Tensor grad_input, Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_input, grad_output, self, padding);
}
Tensor XLAType::replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef)>("replication_pad3d_backward(Tensor grad_output, Tensor self, IntArrayRef padding) -> Tensor")(grad_output, self, padding);
}
Tensor & XLAType::upsample_linear1d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("upsample_linear1d_out(Tensor out, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(out, self, output_size, align_corners);
}
Tensor XLAType::upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("upsample_linear1d(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_linear1d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::upsample_linear1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_linear1d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::upsample_bilinear2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("upsample_bilinear2d_out(Tensor out, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(out, self, output_size, align_corners);
}
Tensor XLAType::upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("upsample_bilinear2d(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_bilinear2d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_bilinear2d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::upsample_bicubic2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("upsample_bicubic2d_out(Tensor out, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(out, self, output_size, align_corners);
}
Tensor XLAType::upsample_bicubic2d(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("upsample_bicubic2d(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_bicubic2d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_bicubic2d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::upsample_trilinear3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, bool)>("upsample_trilinear3d_out(Tensor out, Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(out, self, output_size, align_corners);
}
Tensor XLAType::upsample_trilinear3d(const Tensor & self, IntArrayRef output_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, bool)>("upsample_trilinear3d(Tensor self, IntArrayRef output_size, bool align_corners) -> Tensor")(self, output_size, align_corners);
}
Tensor & XLAType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_trilinear3d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_input, grad_output, output_size, input_size, align_corners);
}
Tensor XLAType::upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, bool)>("upsample_trilinear3d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) -> Tensor")(grad_output, output_size, input_size, align_corners);
}
Tensor & XLAType::upsample_nearest1d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("upsample_nearest1d_out(Tensor out, Tensor self, IntArrayRef output_size) -> Tensor")(out, self, output_size);
}
Tensor XLAType::upsample_nearest1d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("upsample_nearest1d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>("upsample_nearest1d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_input, grad_output, output_size, input_size);
}
Tensor XLAType::upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef)>("upsample_nearest1d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_output, output_size, input_size);
}
Tensor & XLAType::upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("upsample_nearest2d_out(Tensor out, Tensor self, IntArrayRef output_size) -> Tensor")(out, self, output_size);
}
Tensor XLAType::upsample_nearest2d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("upsample_nearest2d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>("upsample_nearest2d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_input, grad_output, output_size, input_size);
}
Tensor XLAType::upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef)>("upsample_nearest2d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_output, output_size, input_size);
}
Tensor & XLAType::upsample_nearest3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef)>("upsample_nearest3d_out(Tensor out, Tensor self, IntArrayRef output_size) -> Tensor")(out, self, output_size);
}
Tensor XLAType::upsample_nearest3d(const Tensor & self, IntArrayRef output_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef)>("upsample_nearest3d(Tensor self, IntArrayRef output_size) -> Tensor")(self, output_size);
}
Tensor & XLAType::upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>("upsample_nearest3d_backward_out(Tensor grad_input, Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_input, grad_output, output_size, input_size);
}
Tensor XLAType::upsample_nearest3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef)>("upsample_nearest3d_backward(Tensor grad_output, IntArrayRef output_size, IntArrayRef input_size) -> Tensor")(grad_output, output_size, input_size);
}
Tensor & XLAType::sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("sigmoid_backward_out(Tensor grad_input, Tensor grad_output, Tensor output) -> Tensor")(grad_input, grad_output, output);
}
Tensor XLAType::sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor")(grad_output, output);
}
Tensor & XLAType::tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &)>("tanh_backward_out(Tensor grad_input, Tensor grad_output, Tensor output) -> Tensor")(grad_input, grad_output, output);
}
Tensor XLAType::tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &)>("tanh_backward(Tensor grad_output, Tensor output) -> Tensor")(grad_output, output);
}
Tensor & XLAType::thnn_conv_transpose2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose2d_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
Tensor XLAType::thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose2d(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor")(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose2d_forward_out(Tensor output, Tensor columns, Tensor ones, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, columns, ones, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("thnn_conv_transpose2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor columns, Tensor ones) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("thnn_conv_transpose2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor columns, Tensor ones, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}
Tensor & XLAType::thnn_conv_transpose3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose3d_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
Tensor XLAType::thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose3d(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> Tensor")(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose3d_forward_out(Tensor output, Tensor finput, Tensor fgrad_input, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_transpose3d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("thnn_conv_transpose3d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor finput, Tensor fgrad_input) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("thnn_conv_transpose3d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, Tensor finput, Tensor fgrad_input, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}
Tensor & XLAType::thnn_conv2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv2d_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding);
}
Tensor XLAType::thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv2d(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> Tensor")(self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv2d_forward_out(Tensor output, Tensor finput, Tensor fgrad_input, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("thnn_conv2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("thnn_conv2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
Tensor & XLAType::thnn_conv_depthwise2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_depthwise2d_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor XLAType::thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_depthwise2d(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor & XLAType::thnn_conv_depthwise2d_forward_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_depthwise2d_forward_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor XLAType::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &> XLAType::thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &> (*)(Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_depthwise2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &>")(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}
std::tuple<Tensor,Tensor> XLAType::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,2>)>("thnn_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) -> std::tuple<Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
Tensor & XLAType::thnn_conv3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv3d_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding);
}
Tensor XLAType::thnn_conv3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv3d(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> Tensor")(self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv3d_forward_out(Tensor output, Tensor finput, Tensor fgrad_input, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("thnn_conv3d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("thnn_conv3d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("thnn_conv3d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, Tensor finput, Tensor fgrad_input, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
Tensor & XLAType::thnn_conv_dilated2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated2d_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor XLAType::thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated2d(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated2d_forward_out(Tensor output, Tensor columns, Tensor ones, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated2d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("thnn_conv_dilated2d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("thnn_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
Tensor & XLAType::thnn_conv_dilated3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor & (*)(Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated3d_out(Tensor out, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(out, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor XLAType::thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated3d(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> Tensor")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated3d_forward_out(Tensor output, Tensor columns, Tensor ones, Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor &,Tensor &,Tensor &>")(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_conv_dilated3d_forward(Tensor self, Tensor weight, IntArrayRef kernel_size, Tensor bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) -> std::tuple<Tensor,Tensor,Tensor>")(self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> XLAType::thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor &,Tensor &,Tensor &> (*)(Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>("thnn_conv_dilated3d_backward_out(Tensor grad_input, Tensor grad_weight, Tensor grad_bias, Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones) -> std::tuple<Tensor &,Tensor &,Tensor &>")(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> XLAType::thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    return XLATypeDispatch::get_function<std::tuple<Tensor,Tensor,Tensor> (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>("thnn_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor columns, Tensor ones, std::array<bool,3> output_mask) -> std::tuple<Tensor,Tensor,Tensor>")(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
Tensor XLAType::thnn_col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_col2im(Tensor self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(self, output_size, kernel_size, dilation, padding, stride);
}
Tensor XLAType::thnn_col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_col2im_backward(Tensor grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(grad_output, kernel_size, dilation, padding, stride);
}
Tensor XLAType::thnn_im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_im2col(Tensor self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(self, kernel_size, dilation, padding, stride);
}
Tensor XLAType::thnn_im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
    return XLATypeDispatch::get_function<Tensor (*)(const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("thnn_im2col_backward(Tensor grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) -> Tensor")(grad_output, input_size, kernel_size, dilation, padding, stride);
}

} // namespace at
