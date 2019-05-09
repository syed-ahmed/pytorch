#include "torch/csrc/autograd/VariableTypeUtils.h"

// @generated from tools/autograd/templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

std::tuple<Tensor,Tensor,Tensor,int64_t> VariableType::_batch_norm_impl_index(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) const {
  RECORD_FUNCTION("_batch_norm_impl_index", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  int64_t result3;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_batch_norm_impl_index");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2, result3) = TypeDefault::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
std::tuple<Tensor,Tensor> VariableType::_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) const {
  RECORD_FUNCTION("_ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Function::peek_at_next_sequence_nr());
  auto& log_probs_ = unpack(log_probs, "log_probs", 0);
  auto& targets_ = unpack(targets, "targets", 1);
  check_no_requires_grad(targets, "targets");
  std::shared_ptr<CtcLossBackward> grad_fn;
  if (compute_requires_grad( log_probs )) {
    grad_fn = std::shared_ptr<CtcLossBackward>(new CtcLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( log_probs ));
    grad_fn->log_probs_ = SavedVariable(log_probs, false);
    grad_fn->targets_ = SavedVariable(targets, false);
    grad_fn->input_lengths = input_lengths.vec();
    grad_fn->target_lengths = target_lengths.vec();
    grad_fn->blank = blank;
    grad_fn->zero_infinity = zero_infinity;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> log_probs__storage_saved =
    log_probs_.has_storage() ? c10::optional<Storage>(log_probs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_probs__impl_saved;
  if (log_probs_.defined()) log_probs__impl_saved = log_probs_.getIntrusivePtr();
  c10::optional<Storage> targets__storage_saved =
    targets_.has_storage() ? c10::optional<Storage>(targets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> targets__impl_saved;
  if (targets_.defined()) targets__impl_saved = targets_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_ctc_loss(log_probs_, targets_, input_lengths, target_lengths, blank, zero_infinity);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (log_probs__storage_saved.has_value())
    AT_ASSERT(log_probs__storage_saved.value().is_alias_of(log_probs_.storage()));
  if (log_probs__impl_saved) AT_ASSERT(log_probs__impl_saved == log_probs_.getIntrusivePtr());
  if (targets__storage_saved.has_value())
    AT_ASSERT(targets__storage_saved.value().is_alias_of(targets_.storage()));
  if (targets__impl_saved) AT_ASSERT(targets__impl_saved == targets_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) const {
  RECORD_FUNCTION("_cudnn_init_dropout_state", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_init_dropout_state");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "dropout_seed", dropout_seed);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_cudnn_init_dropout_state(dropout, train, dropout_seed, options_);
  })();
  auto result = as_variable(tmp);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) const {
  RECORD_FUNCTION("_cudnn_rnn_flatten_weight", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto weight_arr_ = unpack(weight_arr, "weight_arr", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( weight_arr )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cudnn_rnn_flatten_weight"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( weight_arr ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_rnn_flatten_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_arr", weight_arr);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_cudnn_rnn_flatten_weight(weight_arr_, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
  })();
  auto result = as_variable(tmp);
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) const {
  RECORD_FUNCTION("_embedding_bag_dense_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, maximum_indices, per_sample_weights}), Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  auto& offset2bag_ = unpack(offset2bag, "offset2bag", 3);
  auto& bag_size_ = unpack(bag_size, "bag_size", 4);
  auto& maximum_indices_ = unpack(maximum_indices, "maximum_indices", 5);
  auto per_sample_weights_ = unpack_opt(per_sample_weights, "per_sample_weights", 9);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad, per_sample_weights )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_embedding_bag_dense_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad, per_sample_weights ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "maximum_indices", maximum_indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> offsets__storage_saved =
    offsets_.has_storage() ? c10::optional<Storage>(offsets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offsets__impl_saved;
  if (offsets_.defined()) offsets__impl_saved = offsets_.getIntrusivePtr();
  c10::optional<Storage> offset2bag__storage_saved =
    offset2bag_.has_storage() ? c10::optional<Storage>(offset2bag_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offset2bag__impl_saved;
  if (offset2bag_.defined()) offset2bag__impl_saved = offset2bag_.getIntrusivePtr();
  c10::optional<Storage> bag_size__storage_saved =
    bag_size_.has_storage() ? c10::optional<Storage>(bag_size_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bag_size__impl_saved;
  if (bag_size_.defined()) bag_size__impl_saved = bag_size_.getIntrusivePtr();
  c10::optional<Storage> maximum_indices__storage_saved =
    maximum_indices_.has_storage() ? c10::optional<Storage>(maximum_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> maximum_indices__impl_saved;
  if (maximum_indices_.defined()) maximum_indices__impl_saved = maximum_indices_.getIntrusivePtr();
  c10::optional<Storage> per_sample_weights__storage_saved =
    per_sample_weights_.has_storage() ? c10::optional<Storage>(per_sample_weights_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> per_sample_weights__impl_saved;
  if (per_sample_weights_.defined()) per_sample_weights__impl_saved = per_sample_weights_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_embedding_bag_dense_backward(grad_, indices_, offsets_, offset2bag_, bag_size_, maximum_indices_, num_weights, scale_grad_by_freq, mode, per_sample_weights_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (offsets__storage_saved.has_value())
    AT_ASSERT(offsets__storage_saved.value().is_alias_of(offsets_.storage()));
  if (offsets__impl_saved) AT_ASSERT(offsets__impl_saved == offsets_.getIntrusivePtr());
  if (offset2bag__storage_saved.has_value())
    AT_ASSERT(offset2bag__storage_saved.value().is_alias_of(offset2bag_.storage()));
  if (offset2bag__impl_saved) AT_ASSERT(offset2bag__impl_saved == offset2bag_.getIntrusivePtr());
  if (bag_size__storage_saved.has_value())
    AT_ASSERT(bag_size__storage_saved.value().is_alias_of(bag_size_.storage()));
  if (bag_size__impl_saved) AT_ASSERT(bag_size__impl_saved == bag_size_.getIntrusivePtr());
  if (maximum_indices__storage_saved.has_value())
    AT_ASSERT(maximum_indices__storage_saved.value().is_alias_of(maximum_indices_.storage()));
  if (maximum_indices__impl_saved) AT_ASSERT(maximum_indices__impl_saved == maximum_indices_.getIntrusivePtr());
  if (per_sample_weights__storage_saved.has_value())
    AT_ASSERT(per_sample_weights__storage_saved.value().is_alias_of(per_sample_weights_.storage()));
  if (per_sample_weights__impl_saved) AT_ASSERT(per_sample_weights__impl_saved == per_sample_weights_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_embedding_bag_per_sample_weights_backward(const Tensor & grad, const Tensor & weight, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, int64_t mode) const {
  RECORD_FUNCTION("_embedding_bag_per_sample_weights_backward", std::vector<c10::IValue>({grad, weight, indices, offsets, offset2bag}), Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  auto& offsets_ = unpack(offsets, "offsets", 3);
  auto& offset2bag_ = unpack(offset2bag, "offset2bag", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad, weight, indices, offsets, offset2bag )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_embedding_bag_per_sample_weights_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad, weight, indices, offsets, offset2bag ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_per_sample_weights_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "mode", mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> offsets__storage_saved =
    offsets_.has_storage() ? c10::optional<Storage>(offsets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offsets__impl_saved;
  if (offsets_.defined()) offsets__impl_saved = offsets_.getIntrusivePtr();
  c10::optional<Storage> offset2bag__storage_saved =
    offset2bag_.has_storage() ? c10::optional<Storage>(offset2bag_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offset2bag__impl_saved;
  if (offset2bag_.defined()) offset2bag__impl_saved = offset2bag_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_embedding_bag_per_sample_weights_backward(grad_, weight_, indices_, offsets_, offset2bag_, mode);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (offsets__storage_saved.has_value())
    AT_ASSERT(offsets__storage_saved.value().is_alias_of(offsets_.storage()));
  if (offsets__impl_saved) AT_ASSERT(offsets__impl_saved == offsets_.getIntrusivePtr());
  if (offset2bag__storage_saved.has_value())
    AT_ASSERT(offset2bag__storage_saved.value().is_alias_of(offset2bag_.storage()));
  if (offset2bag__impl_saved) AT_ASSERT(offset2bag__impl_saved == offset2bag_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) const {
  RECORD_FUNCTION("_log_softmax_backward_data", std::vector<c10::IValue>({grad_output, output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<LogSoftmaxBackwardDataBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<LogSoftmaxBackwardDataBackward>(new LogSoftmaxBackwardDataBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->dim = dim;
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_log_softmax_backward_data");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_log_softmax_backward_data(grad_output_, output_, dim, self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_multinomial_alias_setup(const Tensor & probs) const {
  RECORD_FUNCTION("_multinomial_alias_setup", std::vector<c10::IValue>({probs}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_multinomial_alias_setup");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "probs", probs);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::_multinomial_alias_setup(probs);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::_pack_padded_sequence_backward(const Tensor & grad, IntArrayRef input_size, const Tensor & batch_sizes, bool batch_first) const {
  RECORD_FUNCTION("_pack_padded_sequence_backward", std::vector<c10::IValue>({grad, batch_sizes}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pack_padded_sequence_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_pack_padded_sequence_backward(grad, input_size, batch_sizes, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_shape_as_tensor(const Tensor & self) const {
  RECORD_FUNCTION("_shape_as_tensor", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_shape_as_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_shape_as_tensor(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_sobol_engine_draw(const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype) const {
  RECORD_FUNCTION("_sobol_engine_draw", std::vector<c10::IValue>({quasi, sobolstate}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sobol_engine_draw");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "quasi", quasi);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "sobolstate", sobolstate);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "num_generated", num_generated);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::_sobol_engine_draw(quasi, n, sobolstate, dimension, num_generated, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::_solve_helper(const Tensor & self, const Tensor & A) const {
  RECORD_FUNCTION("_solve_helper", std::vector<c10::IValue>({self, A}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_solve_helper"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_solve_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_solve_helper(self_, A_);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::_sparse_div_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("_sparse_div_zerodim_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_sparse_div_zerodim");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_sparse_div_zerodim");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_div_zerodim");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_div_zerodim_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_sparse_div_zerodim_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::_standard_gamma_grad(const Tensor & self, const Tensor & output) const {
  RECORD_FUNCTION("_standard_gamma_grad", std::vector<c10::IValue>({self, output}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<StandardGammaGradBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<StandardGammaGradBackward>(new StandardGammaGradBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_standard_gamma_grad");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_standard_gamma_grad(self_, output_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::s__th_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  RECORD_FUNCTION("_th_addbmm", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addbmm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->s__th_addbmm(self_, batch1_, batch2_, beta, alpha);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  RECORD_FUNCTION("_th_addbmm_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_addbmm_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_addbmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_addbmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addbmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_addbmm_(self_, batch1_, batch2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::s__th_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  RECORD_FUNCTION("_th_addcdiv_out", std::vector<c10::IValue>({result, self, tensor1, tensor2, value}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("_th_addcdiv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_addcdiv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addcdiv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_addcdiv_out(result_, self_, tensor1_, tensor2_, value);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  RECORD_FUNCTION("_th_addmv_out", std::vector<c10::IValue>({result, self, mat, vec, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat_ = unpack(mat, "mat", 2);
  auto& vec_ = unpack(vec, "vec", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    throw_error_out_requires_grad("_th_addmv");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_addmv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_addmv_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat__storage_saved =
    mat_.has_storage() ? c10::optional<Storage>(mat_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat__impl_saved;
  if (mat_.defined()) mat__impl_saved = mat_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_addmv_out(result_, self_, mat_, vec_, beta, alpha);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat__storage_saved.has_value())
    AT_ASSERT(mat__storage_saved.value().is_alias_of(mat_.storage()));
  if (mat__impl_saved) AT_ASSERT(mat__impl_saved == mat_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_ceil(const Tensor & self) const {
  RECORD_FUNCTION("_th_ceil", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ceil"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ceil");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_ceil(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
  RECORD_FUNCTION("_th_clamp_min_out", std::vector<c10::IValue>({result, self, min}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_clamp_min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_clamp_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_clamp_min_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_clamp_min_out(result_, self_, min);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_cosh_out(Tensor & result, const Tensor & self) const {
  RECORD_FUNCTION("_th_cosh_out", std::vector<c10::IValue>({result, self}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_cosh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_cosh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_cosh_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_cosh_out(result_, self_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
  RECORD_FUNCTION("_th_eig_out", std::vector<c10::IValue>({res1, res2, self}), Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_eig");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_eig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_eig_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> res1__storage_saved =
    res1_.has_storage() ? c10::optional<Storage>(res1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> res1__impl_saved;
  if (res1_.defined()) res1__impl_saved = res1_.getIntrusivePtr();
  c10::optional<Storage> res2__storage_saved =
    res2_.has_storage() ? c10::optional<Storage>(res2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> res2__impl_saved;
  if (res2_.defined()) res2__impl_saved = res2_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_eig_out(res1_, res2_, self_, eigenvectors);
  }
  #ifndef NDEBUG
  if (res1__storage_saved.has_value())
    AT_ASSERT(res1__storage_saved.value().is_alias_of(res1_.storage()));
  if (res1__impl_saved) AT_ASSERT(res1__impl_saved == res1_.getIntrusivePtr());
  if (res2__storage_saved.has_value())
    AT_ASSERT(res2__storage_saved.value().is_alias_of(res2_.storage()));
  if (res2__impl_saved) AT_ASSERT(res2__impl_saved == res2_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(res1);
  increment_version(res2);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( res1, res2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::forward_as_tuple(res1, res2);
}
Tensor & VariableType::_th_eq_out(Tensor & result, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("_th_eq_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_eq_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_eq_out(result_, self_, other);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("_th_eq_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_eq_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_eq_out(result_, self_, other_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_exp(const Tensor & self) const {
  RECORD_FUNCTION("_th_exp", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_exp"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_exp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_exp(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_fill_(Tensor & self, Scalar value) const {
  RECORD_FUNCTION("_th_fill_", std::vector<c10::IValue>({self, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_fill_(self_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_fill_(Tensor & self, const Tensor & value) const {
  RECORD_FUNCTION("_th_fill_", std::vector<c10::IValue>({self, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& value_ = unpack(value, "value", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_fill_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> value__storage_saved =
    value_.has_storage() ? c10::optional<Storage>(value_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_fill_(self_, value_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (value__storage_saved.has_value())
    AT_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved) AT_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_ge_out(Tensor & result, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("_th_ge_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ge_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_ge_out(result_, self_, other);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("_th_ge_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_ge_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_ge_out(result_, self_, other_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_geqrf(const Tensor & self) const {
  RECORD_FUNCTION("_th_geqrf", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_geqrf"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_geqrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_geqrf(self_);
  })();
  std::tie(res1, res2) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( res1, res2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::make_tuple(std::move(res1), std::move(res2));
}
Tensor VariableType::_th_ger(const Tensor & self, const Tensor & vec2) const {
  RECORD_FUNCTION("_th_ger", std::vector<c10::IValue>({self, vec2}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& vec2_ = unpack(vec2, "vec2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_ger"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, vec2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_ger");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_ger(self_, vec2_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_gt_out(Tensor & result, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("_th_gt_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_gt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_gt_out(result_, self_, other);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("_th_gt_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_gt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_gt_out(result_, self_, other_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  RECORD_FUNCTION("_th_histc_out", std::vector<c10::IValue>({result, self, min, max}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_histc");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_histc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_histc_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_histc_out(result_, self_, bins, min, max);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  RECORD_FUNCTION("_th_index_add_", std::vector<c10::IValue>({self, index, source}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_index_add_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_index_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_index_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_index_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_index_add_(self_, dim, index_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_le_out(Tensor & result, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("_th_le_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_le_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_le_out(result_, self_, other);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("_th_le_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_le_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_le_out(result_, self_, other_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_lgamma(const Tensor & self) const {
  RECORD_FUNCTION("_th_lgamma", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lgamma"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_lgamma(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_lgamma_(Tensor & self) const {
  RECORD_FUNCTION("_th_lgamma_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_lgamma_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_lgamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_lgamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_lgamma_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_log10_out(Tensor & result, const Tensor & self) const {
  RECORD_FUNCTION("_th_log10_out", std::vector<c10::IValue>({result, self}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_log10");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_log10");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_log10_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_log10_out(result_, self_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_log1p(const Tensor & self) const {
  RECORD_FUNCTION("_th_log1p", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_log1p"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_log1p");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_log1p(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_lt_out(Tensor & result, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("_th_lt_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_lt_out(result_, self_, other);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("_th_lt_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_lt_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_lt_out(result_, self_, other_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
  RECORD_FUNCTION("_th_masked_select_out", std::vector<c10::IValue>({result, self, mask}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mask_ = unpack(mask, "mask", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_masked_select");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_masked_select");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_masked_select_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_masked_select_out(result_, self_, mask_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s__th_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("_th_min_out", std::vector<c10::IValue>({result, self, other}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_th_min");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_min_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s__th_min_out(result_, self_, other_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
  RECORD_FUNCTION("_th_min_out", std::vector<c10::IValue>({min, min_indices, self}), Function::peek_at_next_sequence_nr());
  auto& min_ = unpack(min, "min", 0);
  auto& min_indices_ = unpack(min_indices, "min_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_min");
  }
  if (compute_requires_grad( min )) {
    throw_error_out_requires_grad("_th_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "min_indices", min_indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "min", min);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_min_out", min);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> min__storage_saved =
    min_.has_storage() ? c10::optional<Storage>(min_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> min__impl_saved;
  if (min_.defined()) min__impl_saved = min_.getIntrusivePtr();
  c10::optional<Storage> min_indices__storage_saved =
    min_indices_.has_storage() ? c10::optional<Storage>(min_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> min_indices__impl_saved;
  if (min_indices_.defined()) min_indices__impl_saved = min_indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_min_out(min_, min_indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (min__storage_saved.has_value())
    AT_ASSERT(min__storage_saved.value().is_alias_of(min_.storage()));
  if (min__impl_saved) AT_ASSERT(min__impl_saved == min_.getIntrusivePtr());
  if (min_indices__storage_saved.has_value())
    AT_ASSERT(min_indices__storage_saved.value().is_alias_of(min_indices_.storage()));
  if (min_indices__impl_saved) AT_ASSERT(min_indices__impl_saved == min_indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(min);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( min ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, min);
    jit::tracer::addOutput(node, min_indices);
  }
  return std::forward_as_tuple(min, min_indices);
}
Tensor VariableType::_th_mm(const Tensor & self, const Tensor & mat2) const {
  RECORD_FUNCTION("_th_mm", std::vector<c10::IValue>({self, mat2}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_mm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_mm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_mm(self_, mat2_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_th_multinomial_alias_setup(const Tensor & probs) const {
  RECORD_FUNCTION("_th_multinomial_alias_setup", std::vector<c10::IValue>({probs}), Function::peek_at_next_sequence_nr());
  auto& probs_ = unpack(probs, "probs", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( probs )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_multinomial_alias_setup"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( probs ));
  }
  Tensor J;
  Tensor q;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_multinomial_alias_setup");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "probs", probs);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> probs__storage_saved =
    probs_.has_storage() ? c10::optional<Storage>(probs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> probs__impl_saved;
  if (probs_.defined()) probs__impl_saved = probs_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_multinomial_alias_setup(probs_);
  })();
  std::tie(J, q) = as_variable(tmp);
  #ifndef NDEBUG
  if (probs__storage_saved.has_value())
    AT_ASSERT(probs__storage_saved.value().is_alias_of(probs_.storage()));
  if (probs__impl_saved) AT_ASSERT(probs__impl_saved == probs_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( q ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, J);
    jit::tracer::addOutput(node, q);
  }
  return std::make_tuple(std::move(J), std::move(q));
}
Tensor & VariableType::_th_neg_out(Tensor & result, const Tensor & self) const {
  RECORD_FUNCTION("_th_neg_out", std::vector<c10::IValue>({result, self}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_neg");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_neg");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_neg_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_neg_out(result_, self_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_nonzero_out(Tensor & result, const Tensor & self) const {
  RECORD_FUNCTION("_th_nonzero_out", std::vector<c10::IValue>({result, self}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_nonzero_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_nonzero_out(result_, self_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) const {
  RECORD_FUNCTION("_th_normal_out", std::vector<c10::IValue>({output, mean}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean )) {
    throw_error_out_requires_grad("_th_normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_normal_out(output_, mean_, std, generator);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
  RECORD_FUNCTION("_th_normal_out", std::vector<c10::IValue>({output, std}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( std )) {
    throw_error_out_requires_grad("_th_normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_normal_out(output_, mean, std_, generator);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_th_normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) const {
  RECORD_FUNCTION("_th_normal_out", std::vector<c10::IValue>({output, mean, std}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean, std )) {
    throw_error_out_requires_grad("_th_normal");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_th_normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_normal_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_normal_out(output_, mean_, std_, generator);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_th_potri(const Tensor & self, bool upper) const {
  RECORD_FUNCTION("_th_potri", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_potri"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_potri");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_potri(self_, upper);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor &,Tensor &> VariableType::_th_pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
  RECORD_FUNCTION("_th_pstrf_out", std::vector<c10::IValue>({res1, res2, self, tol}), Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_pstrf");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_pstrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_pstrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "tol", tol);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_pstrf_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> res1__storage_saved =
    res1_.has_storage() ? c10::optional<Storage>(res1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> res1__impl_saved;
  if (res1_.defined()) res1__impl_saved = res1_.getIntrusivePtr();
  c10::optional<Storage> res2__storage_saved =
    res2_.has_storage() ? c10::optional<Storage>(res2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> res2__impl_saved;
  if (res2_.defined()) res2__impl_saved = res2_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_pstrf_out(res1_, res2_, self_, upper, tol);
  }
  #ifndef NDEBUG
  if (res1__storage_saved.has_value())
    AT_ASSERT(res1__storage_saved.value().is_alias_of(res1_.storage()));
  if (res1__impl_saved) AT_ASSERT(res1__impl_saved == res1_.getIntrusivePtr());
  if (res2__storage_saved.has_value())
    AT_ASSERT(res2__storage_saved.value().is_alias_of(res2_.storage()));
  if (res2__impl_saved) AT_ASSERT(res2__impl_saved == res2_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(res1);
  increment_version(res2);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( res1, res2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::forward_as_tuple(res1, res2);
}
Tensor & VariableType::_th_renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  RECORD_FUNCTION("_th_renorm_out", std::vector<c10::IValue>({result, self, p, maxnorm}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_renorm");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_renorm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_renorm_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_renorm_out(result_, self_, p, dim, maxnorm);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  RECORD_FUNCTION("_th_scatter_", std::vector<c10::IValue>({self, index, src}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_scatter_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_scatter_(self_, dim, index_, src_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  RECORD_FUNCTION("_th_scatter_", std::vector<c10::IValue>({self, index, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_scatter_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_scatter_(self_, dim, index_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_th_sigmoid(const Tensor & self) const {
  RECORD_FUNCTION("_th_sigmoid", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sigmoid"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_sigmoid(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_sign(const Tensor & self) const {
  RECORD_FUNCTION("_th_sign", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sign"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sign");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_sign(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_sign_(Tensor & self) const {
  RECORD_FUNCTION("_th_sign_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sign_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_sign");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_sign_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_sign_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_sinh_out(Tensor & result, const Tensor & self) const {
  RECORD_FUNCTION("_th_sinh_out", std::vector<c10::IValue>({result, self}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_sinh");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_sinh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_sinh_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_sinh_out(result_, self_);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_sqrt(const Tensor & self) const {
  RECORD_FUNCTION("_th_sqrt", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_sqrt"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_sqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_sqrt(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_th_svd(const Tensor & self, bool some, bool compute_uv) const {
  RECORD_FUNCTION("_th_svd", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_svd"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor res1;
  Tensor res2;
  Tensor res3;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_svd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_svd(self_, some, compute_uv);
  })();
  std::tie(res1, res2, res3) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( res1, res2, res3 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
    jit::tracer::addOutput(node, res3);
  }
  return std::make_tuple(std::move(res1), std::move(res2), std::move(res3));
}
std::tuple<Tensor &,Tensor &> VariableType::_th_symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
  RECORD_FUNCTION("_th_symeig_out", std::vector<c10::IValue>({res1, res2, self}), Function::peek_at_next_sequence_nr());
  auto& res1_ = unpack(res1, "res1", 0);
  auto& res2_ = unpack(res2, "res2", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_symeig");
  }
  if (compute_requires_grad( res1, res2 )) {
    throw_error_out_requires_grad("_th_symeig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "res2", res2);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "res1", res1);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_symeig_out", res1);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> res1__storage_saved =
    res1_.has_storage() ? c10::optional<Storage>(res1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> res1__impl_saved;
  if (res1_.defined()) res1__impl_saved = res1_.getIntrusivePtr();
  c10::optional<Storage> res2__storage_saved =
    res2_.has_storage() ? c10::optional<Storage>(res2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> res2__impl_saved;
  if (res2_.defined()) res2__impl_saved = res2_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_symeig_out(res1_, res2_, self_, eigenvectors, upper);
  }
  #ifndef NDEBUG
  if (res1__storage_saved.has_value())
    AT_ASSERT(res1__storage_saved.value().is_alias_of(res1_.storage()));
  if (res1__impl_saved) AT_ASSERT(res1__impl_saved == res1_.getIntrusivePtr());
  if (res2__storage_saved.has_value())
    AT_ASSERT(res2__storage_saved.value().is_alias_of(res2_.storage()));
  if (res2__impl_saved) AT_ASSERT(res2__impl_saved == res2_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(res1);
  increment_version(res2);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( res1, res2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, res1);
    jit::tracer::addOutput(node, res2);
  }
  return std::forward_as_tuple(res1, res2);
}
Tensor VariableType::_th_tan(const Tensor & self) const {
  RECORD_FUNCTION("_th_tan", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_tan"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_tan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_tan(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
  RECORD_FUNCTION("_th_unfold", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_unfold"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_unfold");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_unfold(self_, dimension, size, step);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_th_uniform_(Tensor & self, double from, double to, Generator * generator) const {
  RECORD_FUNCTION("_th_uniform_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_uniform_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_uniform");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_uniform_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_uniform_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_uniform_(self_, from, to, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_th_var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
  RECORD_FUNCTION("_th_var_out", std::vector<c10::IValue>({result, self}), Function::peek_at_next_sequence_nr());
  auto& result_ = unpack(result, "result", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_th_var");
  }
  if (compute_requires_grad( result )) {
    throw_error_out_requires_grad("_th_var");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_var_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> result__storage_saved =
    result_.has_storage() ? c10::optional<Storage>(result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> result__impl_saved;
  if (result_.defined()) result__impl_saved = result_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_var_out(result_, self_, dim, unbiased, keepdim);
  }
  #ifndef NDEBUG
  if (result__storage_saved.has_value())
    AT_ASSERT(result__storage_saved.value().is_alias_of(result_.storage()));
  if (result__impl_saved) AT_ASSERT(result__impl_saved == result_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(result);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_th_view(const Tensor & self, IntArrayRef size) const {
  RECORD_FUNCTION("_th_view", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_view"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_th_view");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_th_view(self_, size);
  })();
  auto out = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::_th_zero_(Tensor & self) const {
  RECORD_FUNCTION("_th_zero_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_th_zero_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_th_zero");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_th_zero_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_th_zero_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_th_zero_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_thnn_adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
  RECORD_FUNCTION("_thnn_adaptive_avg_pool3d_backward", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_adaptive_avg_pool3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_adaptive_avg_pool3d_backward(grad_output_, self_);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size) const {
  RECORD_FUNCTION("_thnn_adaptive_avg_pool3d_forward_out", std::vector<c10::IValue>({output, self}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_adaptive_avg_pool3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_adaptive_avg_pool3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_adaptive_avg_pool3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_adaptive_avg_pool3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_adaptive_avg_pool3d_forward_out(output_, self_, output_size);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_avg_pool3d_forward(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
  RECORD_FUNCTION("_thnn_avg_pool3d_forward", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_avg_pool3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_avg_pool3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_avg_pool3d_forward(self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
  RECORD_FUNCTION("_thnn_col2im_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_col2im_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_col2im_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_col2im_backward(grad_output_, kernel_size, dilation, padding, stride);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_col2im_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
  RECORD_FUNCTION("_thnn_col2im_forward_out", std::vector<c10::IValue>({output, self}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_col2im_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_col2im_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_col2im_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_col2im_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_col2im_forward_out(output_, self_, output_size, kernel_size, dilation, padding, stride);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::_thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) const {
  RECORD_FUNCTION("_thnn_conv2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Function::peek_at_next_sequence_nr());
  auto grad_input_ = unpack_opt(grad_input, "grad_input", 0);
  auto grad_weight_ = unpack_opt(grad_weight, "grad_weight", 1);
  auto grad_bias_ = unpack_opt(grad_bias, "grad_bias", 2);
  auto& grad_output_ = unpack(grad_output, "grad_output", 3);
  auto& self_ = unpack(self, "self", 4);
  auto& weight_ = unpack(weight, "weight", 5);
  auto& finput_ = unpack(finput, "finput", 9);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 10);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("_thnn_conv2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("_thnn_conv2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_conv2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_weight__storage_saved =
    grad_weight_.has_storage() ? c10::optional<Storage>(grad_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_weight__impl_saved;
  if (grad_weight_.defined()) grad_weight__impl_saved = grad_weight_.getIntrusivePtr();
  c10::optional<Storage> grad_bias__storage_saved =
    grad_bias_.has_storage() ? c10::optional<Storage>(grad_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_bias__impl_saved;
  if (grad_bias_.defined()) grad_bias__impl_saved = grad_bias_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_conv2d_backward_out(grad_input_, grad_weight_, grad_bias_, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_weight__storage_saved.has_value())
    AT_ASSERT(grad_weight__storage_saved.value().is_alias_of(grad_weight_.storage()));
  if (grad_weight__impl_saved) AT_ASSERT(grad_weight__impl_saved == grad_weight_.getIntrusivePtr());
  if (grad_bias__storage_saved.has_value())
    AT_ASSERT(grad_bias__storage_saved.value().is_alias_of(grad_bias_.storage()));
  if (grad_bias__impl_saved) AT_ASSERT(grad_bias__impl_saved == grad_bias_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor VariableType::_thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
  RECORD_FUNCTION("_thnn_conv_depthwise2d_forward", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_depthwise2d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_depthwise2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_conv_depthwise2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
  RECORD_FUNCTION("_thnn_conv_dilated3d_forward", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_conv_dilated3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
  }
  Tensor output;
  Tensor columns;
  Tensor ones;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_conv_dilated3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_conv_dilated3d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation);
  })();
  std::tie(output, columns, ones) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output, columns, ones ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, columns);
    jit::tracer::addOutput(node, ones);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
Tensor VariableType::_thnn_elu_forward(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
  RECORD_FUNCTION("_thnn_elu_forward", std::vector<c10::IValue>({self, alpha, scale, input_scale}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_elu_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_elu_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_elu_forward(self_, alpha, scale, input_scale);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_elu_forward_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) const {
  RECORD_FUNCTION("_thnn_elu_forward_", std::vector<c10::IValue>({self, alpha, scale, input_scale}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_elu_forward_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_thnn_elu_forward");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_thnn_elu_forward_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_elu_forward_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_elu_forward_(self_, alpha, scale, input_scale);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_thnn_glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
  RECORD_FUNCTION("_thnn_glu_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_glu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_glu_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_glu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_glu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_glu_backward_out(grad_input_, grad_output_, self_, dim);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  RECORD_FUNCTION("_thnn_hardtanh_backward", std::vector<c10::IValue>({grad_output, self, min_val, max_val}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_hardtanh_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_hardtanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_hardtanh_backward(grad_output_, self_, min_val, max_val);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  RECORD_FUNCTION("_thnn_hardtanh_forward_out", std::vector<c10::IValue>({output, self, min_val, max_val}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_hardtanh_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_hardtanh_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_hardtanh_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_hardtanh_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_hardtanh_forward_out(output_, self_, min_val, max_val);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
  RECORD_FUNCTION("_thnn_im2col_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_im2col_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_im2col_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_im2col_backward(grad_output_, input_size, kernel_size, dilation, padding, stride);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_im2col_forward_out(Tensor & output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
  RECORD_FUNCTION("_thnn_im2col_forward_out", std::vector<c10::IValue>({output, self}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_im2col_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_im2col_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_im2col_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_im2col_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_im2col_forward_out(output_, self_, kernel_size, dilation, padding, stride);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_l1_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("_thnn_l1_loss_forward", std::vector<c10::IValue>({self, target}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_l1_loss_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_l1_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_l1_loss_forward(self_, target_, reduction);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
  RECORD_FUNCTION("_thnn_max_pool2d_with_indices_backward", std::vector<c10::IValue>({grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_max_pool2d_with_indices_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_pool2d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_max_pool2d_with_indices_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::_thnn_max_pool2d_with_indices_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
  RECORD_FUNCTION("_thnn_max_pool2d_with_indices_forward_out", std::vector<c10::IValue>({output, indices, self}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_max_pool2d_with_indices_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_max_pool2d_with_indices_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_pool2d_with_indices_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_max_pool2d_with_indices_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_max_pool2d_with_indices_forward_out(output_, indices_, self_, kernel_size, stride, padding, dilation, ceil_mode);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor & VariableType::_thnn_max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
  RECORD_FUNCTION("_thnn_max_pool3d_with_indices_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("_thnn_max_pool3d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_max_pool3d_with_indices_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_pool3d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_max_pool3d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_max_pool3d_with_indices_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) const {
  RECORD_FUNCTION("_thnn_max_unpool3d_forward", std::vector<c10::IValue>({self, indices}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_max_unpool3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_max_unpool3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_max_unpool3d_forward(self_, indices_, output_size, stride, padding);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  RECORD_FUNCTION("_thnn_multi_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, p, margin, weight}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 5);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_multi_margin_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_multi_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_multi_margin_loss_backward(grad_output_, self_, target_, p, margin, weight_, reduction);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  RECORD_FUNCTION("_thnn_multi_margin_loss_forward_out", std::vector<c10::IValue>({output, self, target, p, margin, weight}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight )) {
    throw_error_out_requires_grad("_thnn_multi_margin_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_multi_margin_loss_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_multi_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_multi_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_multi_margin_loss_forward_out(output_, self_, target_, p, margin, weight_, reduction);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor & VariableType::_thnn_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  RECORD_FUNCTION("_thnn_sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, output}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("_thnn_sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("_thnn_sigmoid_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_sigmoid_backward_out(grad_input_, grad_output_, output_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::_thnn_smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("_thnn_smooth_l1_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_smooth_l1_loss_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_smooth_l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_smooth_l1_loss_backward(grad_output_, self_, target_, reduction);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("_thnn_smooth_l1_loss_forward_out", std::vector<c10::IValue>({output, self, target}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("_thnn_smooth_l1_loss_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_smooth_l1_loss_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_smooth_l1_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_smooth_l1_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_smooth_l1_loss_forward_out(output_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_soft_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("_thnn_soft_margin_loss_forward", std::vector<c10::IValue>({self, target}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, target )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_soft_margin_loss_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_soft_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_soft_margin_loss_forward(self_, target_, reduction);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
  RECORD_FUNCTION("_thnn_softplus_forward", std::vector<c10::IValue>({self, beta, threshold}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_softplus_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_softplus_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_softplus_forward(self_, beta, threshold);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
  RECORD_FUNCTION("_thnn_tanh_backward", std::vector<c10::IValue>({grad_output, output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_tanh_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_tanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_tanh_backward(grad_output_, output_);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_tanh_forward_out(Tensor & output, const Tensor & self) const {
  RECORD_FUNCTION("_thnn_tanh_forward_out", std::vector<c10::IValue>({output, self}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_tanh_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_tanh_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_tanh_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_tanh_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_tanh_forward_out(output_, self_);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
  RECORD_FUNCTION("_thnn_upsample_bicubic2d_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_bicubic2d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_bicubic2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_upsample_bicubic2d_backward(grad_output_, output_size, input_size, align_corners);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_bicubic2d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
  RECORD_FUNCTION("_thnn_upsample_bicubic2d_forward_out", std::vector<c10::IValue>({output, self}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_upsample_bicubic2d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_upsample_bicubic2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_bicubic2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_bicubic2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_upsample_bicubic2d_forward_out(output_, self_, output_size, align_corners);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_nearest3d_forward(const Tensor & self, IntArrayRef output_size) const {
  RECORD_FUNCTION("_thnn_upsample_nearest3d_forward", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_nearest3d_forward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_nearest3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_upsample_nearest3d_forward(self_, output_size);
  })();
  auto output = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
Tensor VariableType::_thnn_upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
  RECORD_FUNCTION("_thnn_upsample_trilinear3d_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_thnn_upsample_trilinear3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_trilinear3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_thnn_upsample_trilinear3d_backward(grad_output_, output_size, input_size, align_corners);
  })();
  auto grad_input = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::_thnn_upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
  RECORD_FUNCTION("_thnn_upsample_trilinear3d_forward_out", std::vector<c10::IValue>({output, self}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_thnn_upsample_trilinear3d_forward");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("_thnn_upsample_trilinear3d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_upsample_trilinear3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_upsample_trilinear3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->_thnn_upsample_trilinear3d_forward_out(output_, self_, output_size, align_corners);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
std::tuple<Tensor,Tensor> VariableType::_unique(const Tensor & self, bool sorted, bool return_inverse) const {
  RECORD_FUNCTION("_unique", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UniqueBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UniqueBackward>(new UniqueBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_unique");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_unique(self_, sorted, return_inverse);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_unique2(const Tensor & self, bool sorted, bool return_inverse, bool return_counts) const {
  RECORD_FUNCTION("_unique2", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_unique2"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_unique2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->_unique2(self_, sorted, return_inverse, return_counts);
  })();
  std::tie(result0, result1, result2) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor> VariableType::_weight_norm_differentiable_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) const {
  RECORD_FUNCTION("_weight_norm_differentiable_backward", std::vector<c10::IValue>({grad_w, saved_v, saved_g, saved_norms}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm_differentiable_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_w", grad_w);
    jit::tracer::addInputs(node, "saved_v", saved_v);
    jit::tracer::addInputs(node, "saved_g", saved_g);
    jit::tracer::addInputs(node, "saved_norms", saved_norms);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::_weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) const {
  RECORD_FUNCTION("adaptive_avg_pool3d_backward", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AdaptiveAvgPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool3DBackwardBackward>(new AdaptiveAvgPool3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->adaptive_avg_pool3d_backward(grad_output_, self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) const {
  RECORD_FUNCTION("adaptive_max_pool1d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::adaptive_max_pool1d(self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> VariableType::adaptive_max_pool2d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) const {
  RECORD_FUNCTION("adaptive_max_pool2d_out", std::vector<c10::IValue>({out, indices, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("adaptive_max_pool2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("adaptive_max_pool2d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->adaptive_max_pool2d_out(out_, indices_, self_, output_size);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(out, indices);
}
Tensor VariableType::addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  RECORD_FUNCTION("addbmm", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<AddbmmBackward>(new AddbmmBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
    if (grad_fn->should_compute_output(1)) {
      grad_fn->batch2_ = SavedVariable(batch2, false);
    }
    grad_fn->alpha = alpha;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->batch1_ = SavedVariable(batch1, false);
    }
    grad_fn->beta = beta;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->addbmm(self_, batch1_, batch2_, beta, alpha);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
  RECORD_FUNCTION("addbmm_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  check_inplace(self);
  std::shared_ptr<AddbmmBackward> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    grad_fn = std::shared_ptr<AddbmmBackward>(new AddbmmBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, batch1, batch2 ));
    grad_fn->batch1_argsize_0 = batch1.size(0);
    grad_fn->batch1_argsize_1 = batch1.size(1);
    grad_fn->batch2_argsize_2 = batch2.size(2);
    if (grad_fn->should_compute_output(1)) {
      grad_fn->batch2_ = SavedVariable(batch2, false);
    }
    grad_fn->alpha = alpha;
    if (grad_fn->should_compute_output(2)) {
      grad_fn->batch1_ = SavedVariable(batch1, false);
    }
    grad_fn->beta = beta;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addbmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addbmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addbmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->addbmm_(self_, batch1_, batch2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
  RECORD_FUNCTION("addcdiv_out", std::vector<c10::IValue>({out, self, tensor1, tensor2, value}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& tensor1_ = unpack(tensor1, "tensor1", 2);
  auto& tensor2_ = unpack(tensor2, "tensor2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, tensor1, tensor2 )) {
    throw_error_out_requires_grad("addcdiv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addcdiv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addcdiv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcdiv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor1__storage_saved =
    tensor1_.has_storage() ? c10::optional<Storage>(tensor1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor1__impl_saved;
  if (tensor1_.defined()) tensor1__impl_saved = tensor1_.getIntrusivePtr();
  c10::optional<Storage> tensor2__storage_saved =
    tensor2_.has_storage() ? c10::optional<Storage>(tensor2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor2__impl_saved;
  if (tensor2_.defined()) tensor2__impl_saved = tensor2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->addcdiv_out(out_, self_, tensor1_, tensor2_, value);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor1__storage_saved.has_value())
    AT_ASSERT(tensor1__storage_saved.value().is_alias_of(tensor1_.storage()));
  if (tensor1__impl_saved) AT_ASSERT(tensor1__impl_saved == tensor1_.getIntrusivePtr());
  if (tensor2__storage_saved.has_value())
    AT_ASSERT(tensor2__storage_saved.value().is_alias_of(tensor2_.storage()));
  if (tensor2__impl_saved) AT_ASSERT(tensor2__impl_saved == tensor2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::addmv_out(Tensor & out, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
  RECORD_FUNCTION("addmv_out", std::vector<c10::IValue>({out, self, mat, vec, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat_ = unpack(mat, "mat", 2);
  auto& vec_ = unpack(vec, "vec", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat, vec )) {
    throw_error_out_requires_grad("addmv");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addmv");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat__storage_saved =
    mat_.has_storage() ? c10::optional<Storage>(mat_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat__impl_saved;
  if (mat_.defined()) mat__impl_saved = mat_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->addmv_out(out_, self_, mat_, vec_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat__storage_saved.has_value())
    AT_ASSERT(mat__storage_saved.value().is_alias_of(mat_.storage()));
  if (mat__impl_saved) AT_ASSERT(mat__impl_saved == mat_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::all(const Tensor & self, int64_t dim, bool keepdim) const {
  RECORD_FUNCTION("all", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AllBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AllBackward1>(new AllBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::all");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->all(self_, dim, keepdim);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::all(const Tensor & self) const {
  RECORD_FUNCTION("all", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AllBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AllBackward0>(new AllBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::all");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->all(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::any(const Tensor & self, int64_t dim, bool keepdim) const {
  RECORD_FUNCTION("any", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AnyBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AnyBackward1>(new AnyBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::any");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->any(self_, dim, keepdim);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::any(const Tensor & self) const {
  RECORD_FUNCTION("any", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AnyBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AnyBackward0>(new AnyBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::any");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->any(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::argmin(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) const {
  RECORD_FUNCTION("argmin", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argmin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::argmin(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
  RECORD_FUNCTION("as_strided", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsStridedBackward>(new AsStridedBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->storage_offset = storage_offset;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::as_strided");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "storage_offset", storage_offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->as_strided(self_, size, stride, storage_offset);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::as_strided_(Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
  RECORD_FUNCTION("as_strided_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AsStridedBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsStridedBackward>(new AsStridedBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_geometry = TensorGeometry(self);
    grad_fn->size = size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->storage_offset = storage_offset;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::as_strided");
    } else {
      op_name = jit::Symbol::fromQualString("aten::as_strided_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "storage_offset", storage_offset);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("as_strided_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->as_strided_(self_, size, stride, storage_offset);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) const {
  RECORD_FUNCTION("batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::batch_norm_elemt(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps) const {
  RECORD_FUNCTION("batch_norm_elemt", std::vector<c10::IValue>({input, weight, bias, mean, invstd}), Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack_opt(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  auto& mean_ = unpack(mean, "mean", 3);
  auto& invstd_ = unpack(invstd, "invstd", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, weight, bias, mean, invstd )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_elemt"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias, mean, invstd ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_elemt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->batch_norm_elemt(input_, weight_, bias_, mean_, invstd_, eps);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) const {
  RECORD_FUNCTION("bilinear", std::vector<c10::IValue>({input1, input2, weight, bias}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bilinear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::bilinear(input1, input2, weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::binary_cross_entropy_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) const {
  RECORD_FUNCTION("binary_cross_entropy_out", std::vector<c10::IValue>({out, self, target, weight}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto weight_ = unpack_opt(weight, "weight", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target, weight )) {
    throw_error_out_requires_grad("binary_cross_entropy");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("binary_cross_entropy");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("binary_cross_entropy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->binary_cross_entropy_out(out_, self_, target_, weight_, reduction);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::ceil(const Tensor & self) const {
  RECORD_FUNCTION("ceil", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CeilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CeilBackward>(new CeilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ceil");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->ceil(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::ceil_(Tensor & self) const {
  RECORD_FUNCTION("ceil_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CeilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CeilBackward>(new CeilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ceil");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ceil_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ceil_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->ceil_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::cholesky_solve_out(Tensor & out, const Tensor & self, const Tensor & input2, bool upper) const {
  RECORD_FUNCTION("cholesky_solve_out", std::vector<c10::IValue>({out, self, input2}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& input2_ = unpack(input2, "input2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, input2 )) {
    throw_error_out_requires_grad("cholesky_solve");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cholesky_solve");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "upper", upper);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_solve_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> input2__storage_saved =
    input2_.has_storage() ? c10::optional<Storage>(input2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input2__impl_saved;
  if (input2_.defined()) input2__impl_saved = input2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->cholesky_solve_out(out_, self_, input2_, upper);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (input2__storage_saved.has_value())
    AT_ASSERT(input2__storage_saved.value().is_alias_of(input2_.storage()));
  if (input2__impl_saved) AT_ASSERT(input2__impl_saved == input2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::clamp_min_out(Tensor & out, const Tensor & self, Scalar min) const {
  RECORD_FUNCTION("clamp_min_out", std::vector<c10::IValue>({out, self, min}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("clamp_min");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("clamp_min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_min_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->clamp_min_out(out_, self_, min);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::constant_pad_nd(const Tensor & self, IntArrayRef pad, Scalar value) const {
  RECORD_FUNCTION("constant_pad_nd", std::vector<c10::IValue>({self, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ConstantPadNdBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ConstantPadNdBackward>(new ConstantPadNdBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->pad = pad.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::constant_pad_nd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->constant_pad_nd(self_, pad, value);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) const {
  RECORD_FUNCTION("conv2d", std::vector<c10::IValue>({input, weight, bias}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv2d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor VariableType::conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) const {
  RECORD_FUNCTION("conv_transpose1d", std::vector<c10::IValue>({input, weight, bias}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor & VariableType::copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking) const {
  RECORD_FUNCTION("copy_sparse_to_sparse_", std::vector<c10::IValue>({self, src}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("copy_sparse_to_sparse_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::copy_sparse_to_sparse");
    } else {
      op_name = jit::Symbol::fromQualString("aten::copy_sparse_to_sparse_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copy_sparse_to_sparse_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->copy_sparse_to_sparse_(self_, src_, non_blocking);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::cosh_out(Tensor & out, const Tensor & self) const {
  RECORD_FUNCTION("cosh_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("cosh");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cosh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->cosh_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::cross_out(Tensor & out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) const {
  RECORD_FUNCTION("cross_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("cross");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("cross");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cross");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cross_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->cross_out(out_, self_, other_, dim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) const {
  RECORD_FUNCTION("ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ctc_loss(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) const {
  RECORD_FUNCTION("ctc_loss", std::vector<c10::IValue>({log_probs, targets, input_lengths, target_lengths}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) const {
  RECORD_FUNCTION("cudnn_affine_grid_generator_backward", std::vector<c10::IValue>({grad}), Function::peek_at_next_sequence_nr());
  auto& grad_ = unpack(grad, "grad", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_affine_grid_generator_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_affine_grid_generator_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad__storage_saved =
    grad_.has_storage() ? c10::optional<Storage>(grad_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad__impl_saved;
  if (grad_.defined()) grad__impl_saved = grad_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->cudnn_affine_grid_generator_backward(grad_, N, C, H, W);
  })();
  auto grad_theta = as_variable(tmp);
  #ifndef NDEBUG
  if (grad__storage_saved.has_value())
    AT_ASSERT(grad__storage_saved.value().is_alias_of(grad_.storage()));
  if (grad__impl_saved) AT_ASSERT(grad__impl_saved == grad_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_theta ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_theta);
  }
  return grad_theta;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
  RECORD_FUNCTION("cudnn_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<CudnnConvolutionBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<CudnnConvolutionBackwardBackward>(new CudnnConvolutionBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->cudnn_convolution_backward(self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  })();
  std::tie(result0, result1, result2) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cudnn_convolution_backward_bias(const Tensor & grad_output) const {
  RECORD_FUNCTION("cudnn_convolution_backward_bias", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_backward_bias"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward_bias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->cudnn_convolution_backward_bias(grad_output_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
  RECORD_FUNCTION("cudnn_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_backward_input"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self_size", self_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->cudnn_convolution_backward_input(self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
  RECORD_FUNCTION("cudnn_convolution_transpose", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<CudnnConvolutionTransposeBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<CudnnConvolutionTransposeBackward>(new CudnnConvolutionTransposeBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->cudnn_convolution_transpose(self_, weight_, bias_, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::diagflat(const Tensor & self, int64_t offset) const {
  RECORD_FUNCTION("diagflat", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diagflat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::diagflat(self, offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::div(const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("div", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<DivBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<DivBackward0>(new DivBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->div(self_, other_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::div(const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("div", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DivBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DivBackward1>(new DivBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->div(self_, other);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::div_(Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("div_", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<DivBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<DivBackward0>(new DivBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self.clone(), false);
    }
    grad_fn->other_ = SavedVariable(other, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::div");
    } else {
      op_name = jit::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->div_(self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::div_(Tensor & self, Scalar other) const {
  RECORD_FUNCTION("div_", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<DivBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DivBackward1>(new DivBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->other = other;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::div");
    } else {
      op_name = jit::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->div_(self_, other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor &,Tensor &> VariableType::eig_out(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) const {
  RECORD_FUNCTION("eig_out", std::vector<c10::IValue>({e, v, self}), Function::peek_at_next_sequence_nr());
  auto& e_ = unpack(e, "e", 0);
  auto& v_ = unpack(v, "v", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("eig");
  }
  if (compute_requires_grad( e, v )) {
    throw_error_out_requires_grad("eig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "e", e);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eig_out", e);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> e__storage_saved =
    e_.has_storage() ? c10::optional<Storage>(e_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> e__impl_saved;
  if (e_.defined()) e__impl_saved = e_.getIntrusivePtr();
  c10::optional<Storage> v__storage_saved =
    v_.has_storage() ? c10::optional<Storage>(v_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> v__impl_saved;
  if (v_.defined()) v__impl_saved = v_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->eig_out(e_, v_, self_, eigenvectors);
  }
  #ifndef NDEBUG
  if (e__storage_saved.has_value())
    AT_ASSERT(e__storage_saved.value().is_alias_of(e_.storage()));
  if (e__impl_saved) AT_ASSERT(e__impl_saved == e_.getIntrusivePtr());
  if (v__storage_saved.has_value())
    AT_ASSERT(v__storage_saved.value().is_alias_of(v_.storage()));
  if (v__impl_saved) AT_ASSERT(v__impl_saved == v_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(e);
  increment_version(v);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( e, v ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, e);
    jit::tracer::addOutput(node, v);
  }
  return std::forward_as_tuple(e, v);
}
Tensor VariableType::embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) const {
  RECORD_FUNCTION("embedding_backward", std::vector<c10::IValue>({grad, indices}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "sparse", sparse);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::empty_out(Tensor & out, IntArrayRef size) const {
  RECORD_FUNCTION("empty_out", std::vector<c10::IValue>({out}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("empty_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::empty_out(out, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::eq_out(Tensor & out, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("eq_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::eq_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::eq_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("eq_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::eq_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::exp(const Tensor & self) const {
  RECORD_FUNCTION("exp", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ExpBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ExpBackward>(new ExpBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::exp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->exp(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::exp_(Tensor & self) const {
  RECORD_FUNCTION("exp_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ExpBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ExpBackward>(new ExpBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::exp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::exp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->exp_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor VariableType::eye(int64_t n, const TensorOptions & options) const {
  RECORD_FUNCTION("eye", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::eye(n, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::eye(int64_t n, int64_t m, const TensorOptions & options) const {
  RECORD_FUNCTION("eye", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "m", m);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::eye(n, m, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool VariableType::fbgemm_is_cpu_supported() const {
  RECORD_FUNCTION("fbgemm_is_cpu_supported", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::fbgemm_is_cpu_supported();
  return result;
}
Tensor & VariableType::fill_(Tensor & self, Scalar value) const {
  RECORD_FUNCTION("fill_", std::vector<c10::IValue>({self, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FillBackward0>(new FillBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::full_like");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->fill_(self_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::fill_(Tensor & self, const Tensor & value) const {
  RECORD_FUNCTION("fill_", std::vector<c10::IValue>({self, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& value_ = unpack(value, "value", 1);
  check_inplace(self);
  std::shared_ptr<FillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<FillBackward1>(new FillBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::full_like");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> value__storage_saved =
    value_.has_storage() ? c10::optional<Storage>(value_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->fill_(self_, value_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (value__storage_saved.has_value())
    AT_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved) AT_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) const {
  RECORD_FUNCTION("fractional_max_pool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<FractionalMaxPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<FractionalMaxPool2DBackwardBackward>(new FractionalMaxPool2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->fractional_max_pool2d_backward(grad_output_, self_, kernel_size, output_size, indices_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::fractional_max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) const {
  RECORD_FUNCTION("fractional_max_pool3d", std::vector<c10::IValue>({self, random_samples}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  check_no_requires_grad(random_samples, "random_samples");
  std::shared_ptr<FractionalMaxPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FractionalMaxPool3DBackward>(new FractionalMaxPool3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->output_size = output_size.vec();
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> random_samples__storage_saved =
    random_samples_.has_storage() ? c10::optional<Storage>(random_samples_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> random_samples__impl_saved;
  if (random_samples_.defined()) random_samples__impl_saved = random_samples_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->fractional_max_pool3d(self_, kernel_size, output_size, random_samples_);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (random_samples__storage_saved.has_value())
    AT_ASSERT(random_samples__storage_saved.value().is_alias_of(random_samples_.storage()));
  if (random_samples__impl_saved) AT_ASSERT(random_samples__impl_saved == random_samples_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::fractional_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) const {
  RECORD_FUNCTION("fractional_max_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("fractional_max_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("fractional_max_pool3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->fractional_max_pool3d_backward_out(grad_input_, grad_output_, self_, kernel_size, output_size, indices_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::ge_out(Tensor & out, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("ge_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::ge_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::ge_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("ge_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::ge_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor> VariableType::geqrf(const Tensor & self) const {
  RECORD_FUNCTION("geqrf", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<GeqrfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GeqrfBackward>(new GeqrfBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor a;
  Tensor tau;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::geqrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->geqrf(self_);
  })();
  std::tie(a, tau) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( a, tau ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, a);
    jit::tracer::addOutput(node, tau);
  }
  return std::make_tuple(std::move(a), std::move(tau));
}
Tensor VariableType::ger(const Tensor & self, const Tensor & vec2) const {
  RECORD_FUNCTION("ger", std::vector<c10::IValue>({self, vec2}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& vec2_ = unpack(vec2, "vec2", 1);
  std::shared_ptr<GerBackward> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    grad_fn = std::shared_ptr<GerBackward>(new GerBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, vec2 ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->vec2_ = SavedVariable(vec2, false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ger");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->ger(self_, vec2_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::glu(const Tensor & self, int64_t dim) const {
  RECORD_FUNCTION("glu", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<GluBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<GluBackward>(new GluBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::glu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->glu(self_, dim);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
  RECORD_FUNCTION("glu_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("glu_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("glu_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::glu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->glu_backward_out(grad_input_, grad_output_, self_, dim);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
  RECORD_FUNCTION("grid_sampler", std::vector<c10::IValue>({input, grid}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::grid_sampler(input, grid, interpolation_mode, padding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode) const {
  RECORD_FUNCTION("grid_sampler_2d", std::vector<c10::IValue>({input, grid}), Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  std::shared_ptr<GridSampler2DBackward> grad_fn;
  if (compute_requires_grad( input, grid )) {
    grad_fn = std::shared_ptr<GridSampler2DBackward>(new GridSampler2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, grid ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grid_ = SavedVariable(grid, false);
    grad_fn->interpolation_mode = interpolation_mode;
    grad_fn->padding_mode = padding_mode;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->grid_sampler_2d(input_, grid_, interpolation_mode, padding_mode);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::gt_out(Tensor & out, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("gt_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::gt_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::gt_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("gt_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::gt_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
  RECORD_FUNCTION("hardtanh_backward", std::vector<c10::IValue>({grad_output, self, min_val, max_val}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<HardtanhBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<HardtanhBackwardBackward>(new HardtanhBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardtanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->hardtanh_backward(grad_output_, self_, min_val, max_val);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::histc_out(Tensor & out, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
  RECORD_FUNCTION("histc_out", std::vector<c10::IValue>({out, self, min, max}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("histc");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("histc");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("histc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->histc_out(out_, self_, bins, min, max);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::hspmm_out(Tensor & out, const Tensor & mat1, const Tensor & mat2) const {
  RECORD_FUNCTION("hspmm_out", std::vector<c10::IValue>({out, mat1, mat2}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mat1, mat2 )) {
    throw_error_out_requires_grad("hspmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("hspmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hspmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hspmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->hspmm_out(out_, mat1_, mat2_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  RECORD_FUNCTION("index_add", std::vector<c10::IValue>({self, index, source}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::index_add(self, dim, index, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
  RECORD_FUNCTION("index_add_", std::vector<c10::IValue>({self, index, source}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& source_ = unpack(source, "source", 3);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<IndexAddBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<IndexAddBackward>(new IndexAddBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->dim = dim;
    if (grad_fn->should_compute_output(1)) {
      grad_fn->index_ = SavedVariable(index, false);
    }
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->index_add_(self_, dim, index_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::inverse_out(Tensor & out, const Tensor & self) const {
  RECORD_FUNCTION("inverse_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("inverse");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("inverse");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("inverse_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->inverse_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::kl_div(const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("kl_div", std::vector<c10::IValue>({self, target}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  std::shared_ptr<KlDivBackward> grad_fn;
  if (compute_requires_grad( self, target )) {
    grad_fn = std::shared_ptr<KlDivBackward>(new KlDivBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kl_div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->kl_div(self_, target_, reduction);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
  RECORD_FUNCTION("kthvalue", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<KthvalueBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<KthvalueBackward>(new KthvalueBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->kthvalue(self_, k, dim, keepdim);
  })();
  std::tie(values, indices) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & VariableType::le_out(Tensor & out, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("le_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::le_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::le_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("le_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::le_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::leaky_relu_out(Tensor & out, const Tensor & self, Scalar negative_slope) const {
  RECORD_FUNCTION("leaky_relu_out", std::vector<c10::IValue>({out, self, negative_slope}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("leaky_relu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("leaky_relu");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->leaky_relu_out(out_, self_, negative_slope);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::lgamma(const Tensor & self) const {
  RECORD_FUNCTION("lgamma", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LgammaBackward>(new LgammaBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->lgamma(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lgamma_(Tensor & self) const {
  RECORD_FUNCTION("lgamma_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LgammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LgammaBackward>(new LgammaBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lgamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lgamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->lgamma_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::log10_out(Tensor & out, const Tensor & self) const {
  RECORD_FUNCTION("log10_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log10");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("log10");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log10_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->log10_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::log1p(const Tensor & self) const {
  RECORD_FUNCTION("log1p", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Log1PBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log1PBackward>(new Log1PBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log1p");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->log1p(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log1p_(Tensor & self) const {
  RECORD_FUNCTION("log1p_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Log1PBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log1PBackward>(new Log1PBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log1p");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log1p_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log1p_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->log1p_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) const {
  RECORD_FUNCTION("lstm", std::vector<c10::IValue>({input}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lstm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::lstm(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) const {
  RECORD_FUNCTION("lstm", std::vector<c10::IValue>({data, batch_sizes}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lstm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor & VariableType::lt_out(Tensor & out, const Tensor & self, Scalar other) const {
  RECORD_FUNCTION("lt_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::lt_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::lt_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("lt_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::lt_out(out, self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::lu_solve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
  RECORD_FUNCTION("lu_solve", std::vector<c10::IValue>({self, LU_data, LU_pivots}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& LU_data_ = unpack(LU_data, "LU_data", 1);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 2);
  check_no_requires_grad(LU_data, "LU_data");
  check_no_requires_grad(LU_pivots, "LU_pivots");
  std::shared_ptr<LuSolveBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LuSolveBackward>(new LuSolveBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lu_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "LU_data", LU_data);
    jit::tracer::addInputs(node, "LU_pivots", LU_pivots);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> LU_data__storage_saved =
    LU_data_.has_storage() ? c10::optional<Storage>(LU_data_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_data__impl_saved;
  if (LU_data_.defined()) LU_data__impl_saved = LU_data_.getIntrusivePtr();
  c10::optional<Storage> LU_pivots__storage_saved =
    LU_pivots_.has_storage() ? c10::optional<Storage>(LU_pivots_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_pivots__impl_saved;
  if (LU_pivots_.defined()) LU_pivots__impl_saved = LU_pivots_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->lu_solve(self_, LU_data_, LU_pivots_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (LU_data__storage_saved.has_value())
    AT_ASSERT(LU_data__storage_saved.value().is_alias_of(LU_data_.storage()));
  if (LU_data__impl_saved) AT_ASSERT(LU_data__impl_saved == LU_data_.getIntrusivePtr());
  if (LU_pivots__storage_saved.has_value())
    AT_ASSERT(LU_pivots__storage_saved.value().is_alias_of(LU_pivots_.storage()));
  if (LU_pivots__impl_saved) AT_ASSERT(LU_pivots__impl_saved == LU_pivots_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) const {
  RECORD_FUNCTION("margin_ranking_loss", std::vector<c10::IValue>({input1, input2, target}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::margin_ranking_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::margin_ranking_loss(input1, input2, target, margin, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::masked_select_out(Tensor & out, const Tensor & self, const Tensor & mask) const {
  RECORD_FUNCTION("masked_select_out", std::vector<c10::IValue>({out, self, mask}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mask_ = unpack(mask, "mask", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mask )) {
    throw_error_out_requires_grad("masked_select");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("masked_select");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_select_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->masked_select_out(out_, self_, mask_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::matrix_power(const Tensor & self, int64_t n) const {
  RECORD_FUNCTION("matrix_power", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matrix_power");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::matrix_power(self, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
  RECORD_FUNCTION("max_pool2d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
  RECORD_FUNCTION("max_pool2d_with_indices_backward", std::vector<c10::IValue>({grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  check_no_requires_grad(indices, "indices");
  std::shared_ptr<MaxPool2DWithIndicesBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<MaxPool2DWithIndicesBackwardBackward>(new MaxPool2DWithIndicesBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->max_pool2d_with_indices_backward(grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) const {
  RECORD_FUNCTION("max_pool3d_with_indices", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxPool3DWithIndicesBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxPool3DWithIndicesBackward>(new MaxPool3DWithIndicesBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->max_pool3d_with_indices(self_, kernel_size, stride, padding, dilation, ceil_mode);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) const {
  RECORD_FUNCTION("max_pool3d_with_indices_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool3d_with_indices_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool3d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->max_pool3d_with_indices_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
  RECORD_FUNCTION("min_out", std::vector<c10::IValue>({min, min_indices, self}), Function::peek_at_next_sequence_nr());
  auto& min_ = unpack(min, "min", 0);
  auto& min_indices_ = unpack(min_indices, "min_indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("min");
  }
  if (compute_requires_grad( min )) {
    throw_error_out_requires_grad("min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "min_indices", min_indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "min", min);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("min_out", min);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> min__storage_saved =
    min_.has_storage() ? c10::optional<Storage>(min_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> min__impl_saved;
  if (min_.defined()) min__impl_saved = min_.getIntrusivePtr();
  c10::optional<Storage> min_indices__storage_saved =
    min_indices_.has_storage() ? c10::optional<Storage>(min_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> min_indices__impl_saved;
  if (min_indices_.defined()) min_indices__impl_saved = min_indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->min_out(min_, min_indices_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (min__storage_saved.has_value())
    AT_ASSERT(min__storage_saved.value().is_alias_of(min_.storage()));
  if (min__impl_saved) AT_ASSERT(min__impl_saved == min_.getIntrusivePtr());
  if (min_indices__storage_saved.has_value())
    AT_ASSERT(min_indices__storage_saved.value().is_alias_of(min_indices_.storage()));
  if (min_indices__impl_saved) AT_ASSERT(min_indices__impl_saved == min_indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(min);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( min ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, min);
    jit::tracer::addOutput(node, min_indices);
  }
  return std::forward_as_tuple(min, min_indices);
}
Tensor & VariableType::min_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("min_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("min");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("min");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("min_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->min_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::miopen_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
  RECORD_FUNCTION("miopen_convolution_backward_weight", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_convolution_backward_weight"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->miopen_convolution_backward_weight(weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::miopen_depthwise_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) const {
  RECORD_FUNCTION("miopen_depthwise_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<MiopenDepthwiseConvolutionBackwardBackward> grad_fn;
  if (compute_requires_grad( self, grad_output, weight )) {
    grad_fn = std::shared_ptr<MiopenDepthwiseConvolutionBackwardBackward>(new MiopenDepthwiseConvolutionBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, grad_output, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_depthwise_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->miopen_depthwise_convolution_backward(self_, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  })();
  std::tie(result0, result1, result2) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::miopen_depthwise_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) const {
  RECORD_FUNCTION("miopen_depthwise_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, weight )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("miopen_depthwise_convolution_backward_input"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, weight ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_depthwise_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self_size", self_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->miopen_depthwise_convolution_backward_input(self_size, grad_output_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mkldnn_reshape(const Tensor & self, IntArrayRef shape) const {
  RECORD_FUNCTION("mkldnn_reshape", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_reshape"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_reshape");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->mkldnn_reshape(self_, shape);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mm(const Tensor & self, const Tensor & mat2) const {
  RECORD_FUNCTION("mm", std::vector<c10::IValue>({self, mat2}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  std::shared_ptr<MmBackward> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    grad_fn = std::shared_ptr<MmBackward>(new MmBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat2 ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->mat2_sizes = mat2.sizes().vec();
    grad_fn->mat2_ = SavedVariable(mat2, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->mm(self_, mat2_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::mse_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("mse_loss_out", std::vector<c10::IValue>({out, self, target}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("mse_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mse_loss");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mse_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mse_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->mse_loss_out(out_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::mul_out(Tensor & out, const Tensor & self, const Tensor & other) const {
  RECORD_FUNCTION("mul_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("mul");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mul");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->mul_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) const {
  RECORD_FUNCTION("multi_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, p, margin, weight}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multi_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::multilabel_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("multilabel_margin_loss_out", std::vector<c10::IValue>({out, self, target}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::multilabel_margin_loss_out(out, self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
  RECORD_FUNCTION("narrow_copy", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("narrow_copy"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::narrow_copy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->narrow_copy(self_, dim, start, length);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::neg_out(Tensor & out, const Tensor & self) const {
  RECORD_FUNCTION("neg_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("neg");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("neg");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("neg_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->neg_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::nonzero_out(Tensor & out, const Tensor & self) const {
  RECORD_FUNCTION("nonzero_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nonzero_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::nonzero_out(out, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
  RECORD_FUNCTION("norm_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("norm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("norm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->norm_out(out_, self_, p, dim, keepdim, dtype);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
  RECORD_FUNCTION("norm_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("norm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("norm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->norm_out(out_, self_, p, dim, keepdim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::normal_out(Tensor & out, const Tensor & mean, double std, Generator * generator) const {
  RECORD_FUNCTION("normal_out", std::vector<c10::IValue>({out, mean}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->normal_out(out_, mean_, std, generator);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::normal_out(Tensor & out, double mean, const Tensor & std, Generator * generator) const {
  RECORD_FUNCTION("normal_out", std::vector<c10::IValue>({out, std}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->normal_out(out_, mean, std_, generator);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::normal_out(Tensor & out, const Tensor & mean, const Tensor & std, Generator * generator) const {
  RECORD_FUNCTION("normal_out", std::vector<c10::IValue>({out, mean, std}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& std_ = unpack(std, "std", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( mean, std )) {
    throw_error_out_requires_grad("normal");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("normal");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> std__storage_saved =
    std_.has_storage() ? c10::optional<Storage>(std_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> std__impl_saved;
  if (std_.defined()) std__impl_saved = std_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->normal_out(out_, mean_, std_, generator);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (std__storage_saved.has_value())
    AT_ASSERT(std__storage_saved.value().is_alias_of(std_.storage()));
  if (std__impl_saved) AT_ASSERT(std__impl_saved == std_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::one_hot(const Tensor & self, int64_t num_classes) const {
  RECORD_FUNCTION("one_hot", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::one_hot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_classes", num_classes);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::one_hot(self, num_classes);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ones_like(const Tensor & self) const {
  RECORD_FUNCTION("ones_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ones_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ones_like(const Tensor & self, const TensorOptions & options) const {
  RECORD_FUNCTION("ones_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ones_like(self, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pdist(const Tensor & self, double p) const {
  RECORD_FUNCTION("pdist", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pdist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::pdist(self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
  RECORD_FUNCTION("prelu_backward", std::vector<c10::IValue>({grad_output, self, weight}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<PreluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<PreluBackwardBackward>(new PreluBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prelu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->prelu_backward(grad_output_, self_, weight_);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> VariableType::pstrf_out(Tensor & u, Tensor & pivot, const Tensor & self, bool upper, Scalar tol) const {
  RECORD_FUNCTION("pstrf_out", std::vector<c10::IValue>({u, pivot, self, tol}), Function::peek_at_next_sequence_nr());
  auto& u_ = unpack(u, "u", 0);
  auto& pivot_ = unpack(pivot, "pivot", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("pstrf");
  }
  if (compute_requires_grad( u, pivot )) {
    throw_error_out_requires_grad("pstrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pstrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "pivot", pivot);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "tol", tol);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "u", u);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pstrf_out", u);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> u__storage_saved =
    u_.has_storage() ? c10::optional<Storage>(u_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> u__impl_saved;
  if (u_.defined()) u__impl_saved = u_.getIntrusivePtr();
  c10::optional<Storage> pivot__storage_saved =
    pivot_.has_storage() ? c10::optional<Storage>(pivot_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> pivot__impl_saved;
  if (pivot_.defined()) pivot__impl_saved = pivot_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->pstrf_out(u_, pivot_, self_, upper, tol);
  }
  #ifndef NDEBUG
  if (u__storage_saved.has_value())
    AT_ASSERT(u__storage_saved.value().is_alias_of(u_.storage()));
  if (u__impl_saved) AT_ASSERT(u__impl_saved == u_.getIntrusivePtr());
  if (pivot__storage_saved.has_value())
    AT_ASSERT(pivot__storage_saved.value().is_alias_of(pivot_.storage()));
  if (pivot__impl_saved) AT_ASSERT(pivot__impl_saved == pivot_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(u);
  increment_version(pivot);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( u, pivot ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, u);
    jit::tracer::addOutput(node, pivot);
  }
  return std::forward_as_tuple(u, pivot);
}
Scalar VariableType::q_scale(const Tensor & self) const {
  RECORD_FUNCTION("q_scale", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->q_scale(self_);
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor VariableType::quantized_rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) const {
  RECORD_FUNCTION("quantized_rnn_tanh_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_rnn_tanh_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint_like(const Tensor & self, int64_t high) const {
  RECORD_FUNCTION("randint_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint_like(self, high);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint_like(const Tensor & self, int64_t low, int64_t high) const {
  RECORD_FUNCTION("randint_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint_like(self, low, high);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint_like(const Tensor & self, int64_t high, const TensorOptions & options) const {
  RECORD_FUNCTION("randint_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint_like(self, high, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint_like(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options) const {
  RECORD_FUNCTION("randint_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint_like(self, low, high, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::renorm_out(Tensor & out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
  RECORD_FUNCTION("renorm_out", std::vector<c10::IValue>({out, self, p, maxnorm}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("renorm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("renorm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("renorm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->renorm_out(out_, self_, p, dim, maxnorm);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
  RECORD_FUNCTION("replication_pad2d_backward", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReplicationPad2DBackwardBackward>(new ReplicationPad2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->replication_pad2d_backward(grad_output_, self_, padding);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::replication_pad3d(const Tensor & self, IntArrayRef padding) const {
  RECORD_FUNCTION("replication_pad3d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReplicationPad3DBackward>(new ReplicationPad3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->replication_pad3d(self_, padding);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) const {
  RECORD_FUNCTION("replication_pad3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->replication_pad3d_backward_out(grad_input_, grad_output_, self_, padding);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  RECORD_FUNCTION("s_copy_", std::vector<c10::IValue>({self, src}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("s_copy_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::s_copy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::s_copy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("s_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->s_copy_(self_, src_, non_blocking);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  RECORD_FUNCTION("scatter", std::vector<c10::IValue>({self, index, src}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::scatter(self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  RECORD_FUNCTION("scatter", std::vector<c10::IValue>({self, index, value}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::scatter(self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
  RECORD_FUNCTION("scatter_", std::vector<c10::IValue>({self, index, src}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<ScatterBackward0> grad_fn;
  if (compute_requires_grad( self, src )) {
    grad_fn = std::shared_ptr<ScatterBackward0>(new ScatterBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, src ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->scatter_(self_, dim, index_, src_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
  RECORD_FUNCTION("scatter_", std::vector<c10::IValue>({self, index, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  check_inplace(self);
  check_no_requires_grad(index, "index");
  std::shared_ptr<ScatterBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ScatterBackward1>(new ScatterBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
    grad_fn->index_ = SavedVariable(index, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->scatter_(self_, dim, index_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::sigmoid(const Tensor & self) const {
  RECORD_FUNCTION("sigmoid", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SigmoidBackward>(new SigmoidBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->sigmoid(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::sigmoid_(Tensor & self) const {
  RECORD_FUNCTION("sigmoid_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SigmoidBackward>(new SigmoidBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sigmoid");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sigmoid_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->sigmoid_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
  RECORD_FUNCTION("sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, output}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("sigmoid_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("sigmoid_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->sigmoid_backward_out(grad_input_, grad_output_, output_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::sign(const Tensor & self) const {
  RECORD_FUNCTION("sign", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SignBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SignBackward>(new SignBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sign");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->sign(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sign_(Tensor & self) const {
  RECORD_FUNCTION("sign_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SignBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SignBackward>(new SignBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sign");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sign_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->sign_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sinh_out(Tensor & out, const Tensor & self) const {
  RECORD_FUNCTION("sinh_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sinh");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sinh");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sinh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->sinh_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
int64_t VariableType::size(const Tensor & self, int64_t dim) const {
  auto result = TypeDefault::size(self, dim);
  return result;
}
Tensor VariableType::slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) const {
  RECORD_FUNCTION("slice", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SliceBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SliceBackward>(new SliceBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->start = start;
    grad_fn->end = end;
    grad_fn->step = step;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slice");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->slice(self_, dim, start, end, step);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::smm(const Tensor & self, const Tensor & mat2) const {
  RECORD_FUNCTION("smm", std::vector<c10::IValue>({self, mat2}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::smm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::smm(self, mat2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) const {
  RECORD_FUNCTION("smooth_l1_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SmoothL1LossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SmoothL1LossBackwardBackward>(new SmoothL1LossBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::smooth_l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->smooth_l1_loss_backward(grad_output_, self_, target_, reduction);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::softshrink_out(Tensor & out, const Tensor & self, Scalar lambd) const {
  RECORD_FUNCTION("softshrink_out", std::vector<c10::IValue>({out, self, lambd}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("softshrink");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("softshrink");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softshrink");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softshrink_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->softshrink_out(out_, self_, lambd);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor> VariableType::solve(const Tensor & self, const Tensor & A) const {
  RECORD_FUNCTION("solve", std::vector<c10::IValue>({self, A}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<SolveBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<SolveBackward>(new SolveBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->A_ = SavedVariable(A, false);
  }
  Tensor solution;
  Tensor LU;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->solve(self_, A_);
  })();
  std::tie(solution, LU) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( solution ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, solution);
    jit::tracer::addOutput(node, LU);
  }
  if (grad_fn) {
    grad_fn->solution_ = SavedVariable(solution, true);
  }
  return std::make_tuple(std::move(solution), std::move(LU));
}
int64_t VariableType::sparse_dim(const Tensor & self) const {
  RECORD_FUNCTION("sparse_dim", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->sparse_dim(self_);
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor & VariableType::sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
  RECORD_FUNCTION("sparse_resize_and_clear_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("sparse_resize_and_clear_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize_and_clear");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize_and_clear_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_resize_and_clear_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->sparse_resize_and_clear_(self_, size, sparse_dim, dense_dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::vector<Tensor> VariableType::split_with_sizes(const Tensor & self, IntArrayRef split_sizes, int64_t dim) const {
  RECORD_FUNCTION("split_with_sizes", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SplitWithSizesBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SplitWithSizesBackward>(new SplitWithSizesBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->split_sizes = split_sizes.vec();
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::split_with_sizes");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "split_sizes", split_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->split_with_sizes(self_, split_sizes, dim);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sqrt(const Tensor & self) const {
  RECORD_FUNCTION("sqrt", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqrtBackward>(new SqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->sqrt(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::sqrt_(Tensor & self) const {
  RECORD_FUNCTION("sqrt_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<SqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SqrtBackward>(new SqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sqrt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sqrt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sqrt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->sqrt_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::stack_out(Tensor & out, TensorList tensors, int64_t dim) const {
  RECORD_FUNCTION("stack_out", std::vector<c10::IValue>({out}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto tensors_ = unpack(tensors, "tensors", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( tensors )) {
    throw_error_out_requires_grad("stack");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("stack");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::stack");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("stack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (Tensor tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->stack_out(out_, tensors_, dim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::svd(const Tensor & self, bool some, bool compute_uv) const {
  RECORD_FUNCTION("svd", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<SvdBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SvdBackward>(new SvdBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->some = some;
    grad_fn->compute_uv = compute_uv;
  }
  Tensor U;
  Tensor S;
  Tensor V;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::svd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->svd(self_, some, compute_uv);
  })();
  std::tie(U, S, V) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( U, S, V ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, V);
  }
  if (grad_fn) {
    grad_fn->U_ = SavedVariable(U, true);
    grad_fn->S_ = SavedVariable(S, true);
    grad_fn->V_ = SavedVariable(V, true);
  }
  return std::make_tuple(std::move(U), std::move(S), std::move(V));
}
std::tuple<Tensor &,Tensor &> VariableType::symeig_out(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) const {
  RECORD_FUNCTION("symeig_out", std::vector<c10::IValue>({e, V, self}), Function::peek_at_next_sequence_nr());
  auto& e_ = unpack(e, "e", 0);
  auto& V_ = unpack(V, "V", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("symeig");
  }
  if (compute_requires_grad( e, V )) {
    throw_error_out_requires_grad("symeig");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "V", V);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "e", e);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("symeig_out", e);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> e__storage_saved =
    e_.has_storage() ? c10::optional<Storage>(e_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> e__impl_saved;
  if (e_.defined()) e__impl_saved = e_.getIntrusivePtr();
  c10::optional<Storage> V__storage_saved =
    V_.has_storage() ? c10::optional<Storage>(V_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> V__impl_saved;
  if (V_.defined()) V__impl_saved = V_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->symeig_out(e_, V_, self_, eigenvectors, upper);
  }
  #ifndef NDEBUG
  if (e__storage_saved.has_value())
    AT_ASSERT(e__storage_saved.value().is_alias_of(e_.storage()));
  if (e__impl_saved) AT_ASSERT(e__impl_saved == e_.getIntrusivePtr());
  if (V__storage_saved.has_value())
    AT_ASSERT(V__storage_saved.value().is_alias_of(V_.storage()));
  if (V__impl_saved) AT_ASSERT(V__impl_saved == V_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(e);
  increment_version(V);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( e, V ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, e);
    jit::tracer::addOutput(node, V);
  }
  return std::forward_as_tuple(e, V);
}
Tensor VariableType::tan(const Tensor & self) const {
  RECORD_FUNCTION("tan", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanBackward>(new TanBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->tan(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::tan_(Tensor & self) const {
  RECORD_FUNCTION("tan_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TanBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanBackward>(new TanBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::tan");
    } else {
      op_name = jit::Symbol::fromQualString("aten::tan_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->tan_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor VariableType::tanh_backward(const Tensor & grad_output, const Tensor & output) const {
  RECORD_FUNCTION("tanh_backward", std::vector<c10::IValue>({grad_output, output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  std::shared_ptr<TanhBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<TanhBackwardBackward>(new TanhBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->tanh_backward(grad_output_, output_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::thnn_col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
  RECORD_FUNCTION("thnn_col2im_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_col2im_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::thnn_col2im_backward(grad_output, kernel_size, dilation, padding, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
  RECORD_FUNCTION("thnn_conv2d", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) const {
  RECORD_FUNCTION("thnn_conv2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_bias", grad_bias);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
Tensor & VariableType::thnn_conv3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) const {
  RECORD_FUNCTION("thnn_conv3d_out", std::vector<c10::IValue>({out, self, weight, bias}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv3d_out(out, self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
  RECORD_FUNCTION("thnn_conv_depthwise2d_forward", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDepthwise2DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConvDepthwise2DBackward>(new ThnnConvDepthwise2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->thnn_conv_depthwise2d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) const {
  RECORD_FUNCTION("thnn_conv_dilated3d_forward", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 3);
  std::shared_ptr<ThnnConvDilated3DBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<ThnnConvDilated3DBackward>(new ThnnConvDilated3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor output;
  Tensor columns;
  Tensor ones;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_dilated3d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->thnn_conv_dilated3d_forward(self_, weight_, kernel_size, bias_, stride, padding, dilation);
  })();
  std::tie(output, columns, ones) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, columns);
    jit::tracer::addOutput(node, ones);
  }
  if (grad_fn) {
    grad_fn->columns_ = SavedVariable(columns, true);
    grad_fn->ones_ = SavedVariable(ones, true);
  }
  return std::make_tuple(std::move(output), std::move(columns), std::move(ones));
}
Tensor & VariableType::thnn_conv_transpose2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) const {
  RECORD_FUNCTION("thnn_conv_transpose2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_transpose2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_transpose2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv_transpose2d_out(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::thnn_im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) const {
  RECORD_FUNCTION("thnn_im2col_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_im2col_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::thnn_im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::threshold(const Tensor & self, Scalar threshold, Scalar value) const {
  RECORD_FUNCTION("threshold", std::vector<c10::IValue>({self, threshold, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ThresholdBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ThresholdBackward0>(new ThresholdBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->threshold = threshold;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::threshold");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->threshold(self_, threshold, value);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::threshold_(Tensor & self, Scalar threshold, Scalar value) const {
  RECORD_FUNCTION("threshold_", std::vector<c10::IValue>({self, threshold, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ThresholdBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ThresholdBackward1>(new ThresholdBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->threshold = threshold;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::threshold");
    } else {
      op_name = jit::Symbol::fromQualString("aten::threshold_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("threshold_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->threshold_(self_, threshold, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
std::tuple<Tensor &,Tensor &> VariableType::triangular_solve_out(Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
  RECORD_FUNCTION("triangular_solve_out", std::vector<c10::IValue>({X, M, self, A}), Function::peek_at_next_sequence_nr());
  auto& X_ = unpack(X, "X", 0);
  auto& M_ = unpack(M, "M", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("triangular_solve");
  }
  if (compute_requires_grad( X, M )) {
    throw_error_out_requires_grad("triangular_solve");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triangular_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "M", M);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "X", X);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("triangular_solve_out", X);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> X__storage_saved =
    X_.has_storage() ? c10::optional<Storage>(X_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> X__impl_saved;
  if (X_.defined()) X__impl_saved = X_.getIntrusivePtr();
  c10::optional<Storage> M__storage_saved =
    M_.has_storage() ? c10::optional<Storage>(M_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> M__impl_saved;
  if (M_.defined()) M__impl_saved = M_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->triangular_solve_out(X_, M_, self_, A_, upper, transpose, unitriangular);
  }
  #ifndef NDEBUG
  if (X__storage_saved.has_value())
    AT_ASSERT(X__storage_saved.value().is_alias_of(X_.storage()));
  if (X__impl_saved) AT_ASSERT(X__impl_saved == X_.getIntrusivePtr());
  if (M__storage_saved.has_value())
    AT_ASSERT(M__storage_saved.value().is_alias_of(M_.storage()));
  if (M__impl_saved) AT_ASSERT(M__impl_saved == M_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  increment_version(X);
  increment_version(M);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( X, M ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, X);
    jit::tracer::addOutput(node, M);
  }
  return std::forward_as_tuple(X, M);
}
Tensor VariableType::tril(const Tensor & self, int64_t diagonal) const {
  RECORD_FUNCTION("tril", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TrilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TrilBackward>(new TrilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tril");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->tril(self_, diagonal);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::tril_(Tensor & self, int64_t diagonal) const {
  RECORD_FUNCTION("tril_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TrilBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TrilBackward>(new TrilBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::tril");
    } else {
      op_name = jit::Symbol::fromQualString("aten::tril_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tril_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->tril_(self_, diagonal);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) const {
  RECORD_FUNCTION("tril_indices", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tril_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "row", row);
    jit::tracer::addInputs(node, "col", col);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->tril_indices(row, col, offset, options_);
  })();
  auto result = as_variable(tmp);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
  RECORD_FUNCTION("unfold", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnfoldBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnfoldBackward>(new UnfoldBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dimension = dimension;
    grad_fn->size = size;
    grad_fn->step = step;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unfold");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->unfold(self_, dimension, size, step);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::uniform_(Tensor & self, double from, double to, Generator * generator) const {
  RECORD_FUNCTION("uniform_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<UniformBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UniformBackward>(new UniformBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::uniform");
    } else {
      op_name = jit::Symbol::fromQualString("aten::uniform_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("uniform_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->uniform_(self_, from, to, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
  RECORD_FUNCTION("upsample_bicubic2d_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleBicubic2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackwardBackward>(new UpsampleBicubic2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bicubic2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->upsample_bicubic2d_backward(grad_output_, output_size, input_size, align_corners);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_bilinear2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners) const {
  RECORD_FUNCTION("upsample_bilinear2d_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_bilinear2d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_bilinear2d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bilinear2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->upsample_bilinear2d_out(out_, self_, output_size, align_corners);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::upsample_nearest1d_out(Tensor & out, const Tensor & self, IntArrayRef output_size) const {
  RECORD_FUNCTION("upsample_nearest1d_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("upsample_nearest1d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("upsample_nearest1d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->upsample_nearest1d_out(out_, self_, output_size);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) const {
  RECORD_FUNCTION("upsample_trilinear3d_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleTrilinear3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackwardBackward>(new UpsampleTrilinear3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_trilinear3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->upsample_trilinear3d_backward(grad_output_, output_size, input_size, align_corners);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::var_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) const {
  RECORD_FUNCTION("var_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("var");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("var");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("var_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->var_out(out_, self_, dim, unbiased, keepdim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::view(const Tensor & self, IntArrayRef size) const {
  RECORD_FUNCTION("view", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ViewBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ViewBackward>(new ViewBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::view");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return baseType->view(self_, size);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::zero_(Tensor & self) const {
  RECORD_FUNCTION("zero_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ZeroBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ZeroBackward>(new ZeroBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::zeros_like");
    } else {
      op_name = jit::Symbol::fromQualString("aten::zero_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("zero_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    baseType->zero_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::zeros(IntArrayRef size, const TensorOptions & options) const {
  RECORD_FUNCTION("zeros", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::zeros(size, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}

}} // namespace torch::autograd
