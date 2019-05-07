#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>

#include <THC/THCGeneral.h>
#include <THC/THCTensorRandom.h>
#include <THC/THCGenerator.hpp>
#include <THC/THCApply.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

/**
 * Note [Register spilling in curand call for CUDA < 10]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * For CUDA < 10, curandStatePhilox4_32_10_t engine achieves poor performance (60% SOL bandwidth)
 * when called to generate one random number at a time. This is because the line
 *            unsigned ret = (&state->output.x)[state->STATE++];
 * in
 *            QUALIFIERS unsigned int curand(curandStatePhilox4_32_10_t *state)
 * in curand_kernel.h dynamically indexes into state.output, preventing the compiler from ever
 * storing state.output in registers.
 *
 * CUDA 10 fixed this problem. However, for backwards compatibility, in the following kernels
 * we are using curand distributions that utilize curand4 call. curand4 call doesn't have the
 * register spilling problem.
 */
 
THCGenerator* THCRandom_getGenerator(THCState* state);

namespace {
// Increment should be at least the number of curand() random numbers used in
// each thread. It is the user's responsibility to make sure that the increment for philox is never
// smaller than the number of curand() calls. Increment value > the number of curand() calls
// won't harm but anything less would mean that you would be reusing random values from
// previous calls. 
// e.g. In many kernels below, we use distributions that utilize curand4 call in the kernel.
//      Hence, increment value should be at least 4 for those kernels.
std::pair<uint64_t, uint64_t> next_philox_seed(at::Generator* gen, uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
}

// utility function that calculates proper philox_offset
// for distributions utilizing TensorIterator. Note that,
// this function is just returning the number of elements
// per thread. That is because, for distributions using
// TensorIterator, we are using a grid-stride loop with each
// thread yielding one element per thread. Moreover, even though
// we use curand_uniform4, we only utilize the x value of the
// returned float4. Hence, since the number of curand() random
// is 1, the philox offset increment is the number of elements
// per thread.
uint64_t calc_philox_increment(uint64_t total_elements,
                               bool is_trivial_1d) {
  int block_size;
  if (is_trivial_1d) {
    block_size = launch_size_1d; 
  } else {
    block_size = launch_size_nd;
  }
  uint32_t grid_size = (total_elements + block_size - 1) / block_size;
  uint32_t blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid_size = std::min(
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
      grid_size);
  return ((total_elements - 1) / (block_size * grid_size) + 1);
}

template <typename scalar_t>
void poisson_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& lambda,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
      ret,
      lambda,
      [seeds] __device__(
          scalar_t & ret_val, const scalar_t& lambda) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        ret_val = static_cast<scalar_t>(curand_poisson(&state, lambda));
      });
}

template <typename scalar_t>
void gamma_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& alpha,
    std::pair<uint64_t, uint64_t> seeds) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
      ret,
      alpha,
      [seeds] __device__(
          scalar_t & ret_val, const scalar_t& alpha) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);

        auto uniform_lambda = [&state] __device__ () {
          return curand_uniform(&state);
        };
        BaseSampler<accscalar_t, decltype(uniform_lambda)> standard_uniform(uniform_lambda);

        auto normal_lambda = [&state] __device__ () {
          return curand_normal(&state);
        };
        BaseSampler<accscalar_t, decltype(normal_lambda)> standard_normal(normal_lambda);
        auto sample = sample_gamma<scalar_t, accscalar_t, decltype(uniform_lambda), decltype(normal_lambda)>(alpha, standard_uniform, standard_normal);
        auto min_value = std::numeric_limits<scalar_t>::min();
        ret_val = (min_value > sample) ? min_value : sample;
      });
}

template <typename scalar_t>
void gamma_grad_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& output) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      ret, self, output,
      [] __device__ (scalar_t& ret_val, const scalar_t& self_val, const scalar_t &output_val) {
        ret_val = standard_gamma_grad_one<scalar_t, accscalar_t>(self_val, output_val);
      });
}

template<typename scalar_t, typename prob_t>
void bernoulli_tensor_cuda_kernel(
    at::Tensor& ret, const at::Tensor& p,
    std::pair<uint64_t, uint64_t> seeds) {
  // The template argument `4` below indicates that we want to operate on four
  // element at each time. See NOTE [ CUDA_tensor_applyN helpers ] for details.
  at::cuda::CUDA_tensor_apply2<scalar_t, prob_t, 4>(
      ret, p,
      [seeds] __device__(
          int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
          const prob_t& p1, const prob_t& p2, const prob_t& p3, const prob_t& p4) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        // See Note [Register spilling in curand call for CUDA < 10]
        float4 rand = curand_uniform4(&state);
        switch (n) {
          case 4: {
            assert(0 <= p4 && p4 <= 1);
            v4 = static_cast<scalar_t>(rand.w <= p4);
            // fallthrough
          }
          case 3: {
            assert(0 <= p3 && p3 <= 1);
            v3 = static_cast<scalar_t>(rand.z <= p3);
            // fallthrough
          }
          case 2: {
            assert(0 <= p2 && p2 <= 1);
            v2 = static_cast<scalar_t>(rand.y <= p2);
            // fallthrough
          }
          case 1: {
            assert(0 <= p1 && p1 <= 1);
            v1 = static_cast<scalar_t>(rand.x <= p1);
          }
        }
      }
    );
}

template<typename scalar_t>
void bernoulli_scalar_cuda_kernel(
    at::Tensor& ret, double p_,
    std::pair<uint64_t, uint64_t> seeds) {
  float p = static_cast<float>(p_);
  // The template argument `4` below indicates that we want to operate on four
  // element at each time. See NOTE [ CUDA_tensor_applyN helpers ] for details.
  at::cuda::CUDA_tensor_apply1<scalar_t, 4>(
      ret, [seeds, p] __device__(
        int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
        curandStatePhilox4_32_10_t state;
        curand_init(
            seeds.first,
            blockIdx.x * blockDim.x + threadIdx.x,
            seeds.second,
            &state);
        // See Note [Register spilling in curand call for CUDA < 10]
        float4 rand = curand_uniform4(&state);
        switch (n) {
          case 4: {
            v4 = static_cast<scalar_t>(rand.w <= p);
            // fallthrough
          }
          case 3: {
            v3 = static_cast<scalar_t>(rand.z <= p);
            // fallthrough
          }
          case 2: {
            v2 = static_cast<scalar_t>(rand.y <= p);
            // fallthrough
          }
          case 1: {
            v1 = static_cast<scalar_t>(rand.x <= p);
          }
        }
      }
    );
}

template<typename scalar_t>
void dirichlet_scalar_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& gamma) {
  auto gamma_sum = gamma.sum(-1, true).expand(ret.sizes());
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(ret, gamma, gamma_sum,
  [] __device__(scalar_t &ret_val, const scalar_t &gamma, const scalar_t &gamma_sum) {
    ret_val = gamma / gamma_sum;
    auto min_value = std::numeric_limits<scalar_t>::min();
    auto max_value = 1 - std::numeric_limits<scalar_t>::epsilon();
    ret_val = (min_value > ret_val) ? min_value : ret_val;
    ret_val = (max_value < ret_val) ? max_value : ret_val;
  });
}

template <typename scalar_t>
void uniform_kernel_impl(
  at::TensorIterator& iter,
  scalar_t from_,
  scalar_t to,
  std::pair<uint64_t, uint64_t> seeds) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  at::native::gpu_nullary_kernel<true>(
    iter,
    [seeds, from_, to]GPU_LAMBDA(int linear_index) -> scalar_t {
      auto range = static_cast<accscalar_t>(to-from_);
      auto from = static_cast<accscalar_t>(from_);
    #ifdef __CUDA_ARCH__
      curandStatePhilox4_32_10_t state;
      curand_init(seeds.first,
                  blockIdx.x * blockDim.x + threadIdx.x,
                  seeds.second + linear_index / (blockDim.x*gridDim.x),
                  &state);
      accscalar_t rand;
      if (std::is_same<scalar_t, double>::value) {
        // See Note [Register spilling in curand call for CUDA < 10]
        rand = static_cast<accscalar_t>(curand_uniform2_double(&state).x);
      } else {
        // See Note [Register spilling in curand call for CUDA < 10]
        rand = static_cast<accscalar_t>(curand_uniform4(&state).x);
      }
      // reverse the bounds of curand4 from (0, 1] to [0, 1)
      // Note that this method is from legacy THCTensorRandom and is likely to give
      // you more 0-s, since, the probability of gettings 1-s is higher than 0-s and
      // by reversing the bounds, we are fliping the probabilities of 1-s and 0-s.
      auto reverse_bound_rand = rand == static_cast<accscalar_t>(1.0) ? static_cast<accscalar_t>(0.0) : rand;
      return static_cast<scalar_t>(reverse_bound_rand * range + from);
    #else
      return static_cast<scalar_t>(range + from);
    #endif
    });
}

} // namespace

namespace at { namespace native {
Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.scalar_type(), "poisson_cuda", [&] {
    poisson_cuda_kernel<scalar_t>(ret, lambda, next_philox_seed(gen, 20));
  });
  return ret;
}

Tensor _s_gamma_cuda(const Tensor& alpha, Generator* gen) {
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.scalar_type(), "gamma_cuda", [&] {
     gamma_cuda_kernel<scalar_t>(ret, alpha, next_philox_seed(gen, 10));
   });
  return ret;
}

Tensor _s_dirichlet_cuda(const Tensor& alpha, Generator* gen) {
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "dirichlet", [&] {
    Tensor gamma = at::empty(alpha.sizes(), alpha.options());
    gamma_cuda_kernel<scalar_t>(gamma, alpha, next_philox_seed(gen, 10));
    dirichlet_scalar_cuda_kernel<scalar_t>(ret, gamma);
  });
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "_standard_gamma_grad_cuda", [&] {
     gamma_grad_cuda_kernel<scalar_t>(ret, self, output);
   });
  return ret;
}

Tensor& bernoulli_tensor_cuda_(Tensor &self, const Tensor& p_, Generator* gen) {
  auto p = std::get<0>(expand_inplace(self, p_.to(kCUDA)));
  AT_DISPATCH_ALL_TYPES_AND(
    at::ScalarType::Half, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
      using self_t = scalar_t;
      auto seeds = next_philox_seed(gen, 10);
      AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, p.scalar_type(), "bernoulli_tensor_cuda_p_", [&] {
        using p_t = scalar_t;
        return bernoulli_tensor_cuda_kernel<self_t, p_t>(self, p, seeds);
      });
   });
  return self;
}

Tensor& bernoulli_scalar_cuda_(Tensor &self, double p, Generator* gen) {
  AT_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "bernoulli_scalar_cuda_", [&] {
    auto seeds = next_philox_seed(gen, 10);
    bernoulli_scalar_cuda_kernel<scalar_t>(self, p, seeds);
   });
  return self;
}

static void uniform_kernel_cuda(TensorIterator& iter, double from_, double to_, Generator* gen) {
  uint64_t counter_offset = calc_philox_increment(iter.numel(), iter.is_trivial_1d());
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "uniform_cuda", [&] {
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    AT_CHECK(from <= to,
      "uniform_ expects to return a [from, to) range, but found from=", from,
      " > to=", to);
    AT_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
          "uniform_ expects to-from ≤ std::numeric_limits<double>::max(), but found to=", to,
          " and from=", from, " which result in to-from to exceed the limit");
    uniform_kernel_impl<scalar_t>(iter, from, to, seeds);
   });
}

Tensor& uniform_cuda_(Tensor& self, double from, double to, Generator* gen) {
  auto iter = TensorIterator::nullary_op(self);
  uniform_kernel_cuda(*iter, from, to, gen);
  return self;
}


}} // namespace at::native
