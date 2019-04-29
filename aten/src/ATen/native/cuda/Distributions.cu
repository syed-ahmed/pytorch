#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/Array.h>
#include <ATen/AccumulateType.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>

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
 * Note [Register spilling in curand for CUDA < 10]
 */
THCGenerator* THCRandom_getGenerator(THCState* state);

namespace {
// increment should be at least the number of curand() random numbers used in
// each thread.
std::pair<uint64_t, uint64_t> next_philox_seed(at::Generator* gen, uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
}

// step value used in the CUDA_tensor_apply* for several kernels below
constexpr uint64_t UNROLL_FACTOR = 4;
// num_engine_calls value used in calc_philox_increment for
// several kernels below
constexpr uint64_t CURAND4_ENGINE_CALLS = 4;

// utility function that calculates proper philox_offset
// unroll factor is the step value in CUDA_tensor_apply*
// num_engine_calls correspond how many randoms are being
// requested to philox engine. For instance, curand_uniform4
// requests 4 uint32_t from the engine and hence, num_engine_calls is 4.
uint64_t calc_philox_increment(uint64_t total_elements,
                               uint64_t grid_size,
                               uint64_t block_size,
                               uint64_t unroll_factor,
                               uint64_t num_engine_calls) {
  return ((total_elements - 1)/(block_size * grid_size * unroll_factor)+1) * num_engine_calls;
}

// utility function that calculates grid size for functions utilizing CUDA_tensor_apply*
uint64_t calc_grid_size_cuda_apply_utils_n(uint64_t step, uint64_t total_elements) {
  // grid calculation from getApplyGrid() in CUDAApplyUtils.cuh
  uint64_t grid_size = (total_elements + (at::cuda::AT_APPLY_THREADS_PER_BLOCK * step) - 1) / (at::cuda::AT_APPLY_THREADS_PER_BLOCK * step);
  #if CUDA_VERSION < 9000
    if (!self.is_contiguous()) {
      uint64_t blocks_per_sm = AT_APPLY_BLOCKS_PER_SM;
      grid_size = std::min((unsigned int)(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm , grid_size);
    }   
  #endif
  return grid_size;
}

// utility function to concat two uint4 bits into at::cuda::Array<uint64_t, 4>
__device__ inline at::cuda::Array<uint64_t, 4> make_64_bits_from_32_bits(uint4 hi, uint4 lo) {
  at::cuda::Array<uint64_t, 4> ret;
  ret[0] = (static_cast<uint64_t>(hi.x) << 32) | lo.x;
  ret[1] = (static_cast<uint64_t>(hi.y) << 32) | lo.y;
  ret[2] = (static_cast<uint64_t>(hi.z) << 32) | lo.z;
  ret[3] = (static_cast<uint64_t>(hi.w) << 32) | lo.w;
  return ret;
};

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
          // Note that we are using curand_uniform here instead of curand_uniform4 since
          // this kernel does not care about perf.
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
        float4 rand = curand_uniform4(&state);
        switch (n) {
          case 4: {
            std::is_unsigned<prob_t>::value ? assert(p4 <= 1) : assert(0 <= p4 && p4 <= 1);
            v4 = static_cast<scalar_t>(rand.w <= p4);
            // fallthrough
          }
          case 3: {
            std::is_unsigned<prob_t>::value ? assert(p3 <= 1) : assert(0 <= p3 && p3 <= 1);
            v3 = static_cast<scalar_t>(rand.z <= p3);
            // fallthrough
          }
          case 2: {
            std::is_unsigned<prob_t>::value ? assert(p2 <= 1) : assert(0 <= p2 && p2 <= 1);
            v2 = static_cast<scalar_t>(rand.y <= p2);
            // fallthrough
          }
          case 1: {
            std::is_unsigned<prob_t>::value ? assert(p1 <= 1) : assert(0 <= p1 && p1 <= 1);
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

template<typename scalar_t>
void uniform_tensor_cuda_kernel(
    at::Tensor& ret,
    float from,
    float to,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply1<scalar_t, UNROLL_FACTOR>(
    ret, [seeds, from, to] __device__(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 rand = curand_uniform4(&state);

      // define lambda to reverse the bounds of curand_uniform4 from (0, 1] to [0, 1)
      auto reverse_bounds = [] __device__ (float value) {
        return value == 1.0f ? 0.0f : value;
      };
      switch (n) {
        case 4: {
          v4 = static_cast<scalar_t>(reverse_bounds(rand.w) * (to-from) + from);
          // fallthrough
        }
        case 3: {
          v3 = static_cast<scalar_t>(reverse_bounds(rand.z) * (to-from) + from);
          // fallthrough
        }
        case 2: {
          v2 = static_cast<scalar_t>(reverse_bounds(rand.y) * (to-from) + from);
          // fallthrough
        }
        case 1: {
          v1 = static_cast<scalar_t>(reverse_bounds(rand.x) * (to-from) + from);
        }
      }
    }
  );
}

// overloading normal_tensor_cuda_kernel for float mean, float std
template<typename scalar_t>
void normal_tensor_cuda_kernel(
    at::Tensor& ret,
    const float& mean,
    const float& std,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply1<scalar_t, UNROLL_FACTOR>(
    ret, [seeds, mean, std] __device__(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 randn = curand_normal4(&state);
      switch (n) {
        case 4: {
          v4 = static_cast<scalar_t>(randn.w * std + mean);
          // fallthrough
        }
        case 3: {
          v3 = static_cast<scalar_t>(randn.z * std + mean);
          // fallthrough
        }
        case 2: {
          v2 = static_cast<scalar_t>(randn.y * std + mean);
          // fallthrough
        }
        case 1: {
          v1 = static_cast<scalar_t>(randn.x * std + mean);
        }
      }
    }
  );
}

// overloading normal_tensor_cuda_kernel for Tensor mean, float std
template<typename scalar_t>
void normal_tensor_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& mean,
    const float& std,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, float, UNROLL_FACTOR>(
    ret, mean,
    [seeds, std] __device__(
        int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
        const float& m1, const float& m2, const float& m3, const float& m4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 randn = curand_uniform4(&state);
      switch (n) {
        case 4: {
          v4 = static_cast<scalar_t>(randn.w * std + m4);
          // fallthrough
        }
        case 3: {
          v3 = static_cast<scalar_t>(randn.z * std + m3);
          // fallthrough
        }
        case 2: {
          v2 = static_cast<scalar_t>(randn.y * std + m2);
          // fallthrough
        }
        case 1: {
          v1 = static_cast<scalar_t>(randn.x * std + m1);
        }
      }
    }
  );
}

// overloading normal_tensor_cuda_kernel for float mean, Tensor std
template<typename scalar_t>
void normal_tensor_cuda_kernel(
    at::Tensor& ret,
    const float& mean,
    const at::Tensor& std,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply2<scalar_t, float, UNROLL_FACTOR>(
    ret, std,
    [seeds, mean] __device__(
        int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
        const float& s1, const float& s2, const float& s3, const float& s4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 randn = curand_uniform4(&state);
      switch (n) {
        case 4: {
          assert(s4 > 0.0f);
          v4 = static_cast<scalar_t>(randn.w * s4 + mean);
          // fallthrough
        }
        case 3: {
          assert(s3 > 0.0f);
          v3 = static_cast<scalar_t>(randn.z * s3 + mean);
          // fallthrough
        }
        case 2: {
          assert(s2 > 0.0f);
          v2 = static_cast<scalar_t>(randn.y * s2 + mean);
          // fallthrough
        }
        case 1: {
          assert(s1 > 0.0f);
          v1 = static_cast<scalar_t>(randn.x * s1 + mean);
        } 
      }
    }
  );
}

// overloading normal_tensor_cuda_kernel for Tensor mean, Tensor std
template<typename scalar_t>
void normal_tensor_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& mean,
    const at::Tensor& std,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply3<scalar_t, float, float, UNROLL_FACTOR>(
    ret, mean, std,
    [seeds] __device__(
        int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4,
        const float& m1, const float& m2, const float& m3, const float& m4,
        const float& s1, const float& s2, const float& s3, const float& s4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 randn = curand_uniform4(&state);
      switch (n) {
        case 4: {
          assert(s4 > 0.0f);
          v4 = static_cast<scalar_t>(randn.w * s4 + m4);
          // fallthrough
        }
        case 3: {
          assert(s3 > 0.0f);
          v3 = static_cast<scalar_t>(randn.z * s3 + m3);
          // fallthrough
        }
        case 2: {
          assert(s2 > 0.0f);
          v2 = static_cast<scalar_t>(randn.y * s2 + m2);
          // fallthrough
        }
        case 1: {
          assert(s1 > 0.0f);
          v1 = static_cast<scalar_t>(randn.x * s1 + m1);
        }
      }
    }
  );
}

template<typename scalar_t>
void cauchy_tensor_cuda_kernel(
    at::Tensor& ret,
    float median,
    float sigma,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply1<scalar_t, UNROLL_FACTOR>(
    ret, [seeds, median, sigma] __device__(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 rand = curand_uniform4(&state);
      switch (n) {
        case 4: {
          v4 = static_cast<scalar_t>(median + sigma * ::tan(static_cast<float>(M_PI) * (rand.w-0.5f)));
          // fallthrough
        }
        case 3: {
          v3 = static_cast<scalar_t>(median + sigma * ::tan(static_cast<float>(M_PI) * (rand.z-0.5f)));
          // fallthrough
        }
        case 2: {
          v2 = static_cast<scalar_t>(median + sigma * ::tan(static_cast<float>(M_PI) * (rand.y-0.5f)));
          // fallthrough
        }
        case 1: {
          v1 = static_cast<scalar_t>(median + sigma * ::tan(static_cast<float>(M_PI) * (rand.x-0.5f)));
        }
      }
    }
  );
}

template<typename scalar_t>
void log_normal_tensor_cuda_kernel(
    at::Tensor& ret,
    float mean,
    float std,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply1<scalar_t, UNROLL_FACTOR>(
    ret, [seeds, mean, std] __device__(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 randn = curand_log_normal4(&state, mean, std);
      switch (n) {
        case 4: {
          v4 = static_cast<scalar_t>(randn.w);
          // fallthrough
        }
        case 3: {
          v3 = static_cast<scalar_t>(randn.z);
          // fallthrough
        }
        case 2: {
          v2 = static_cast<scalar_t>(randn.y);
          // fallthrough
        }
        case 1: {
          v1 = static_cast<scalar_t>(randn.x);
        }
      }
    }
  );
}

template<typename scalar_t>
void exponential_tensor_cuda_kernel(
    at::Tensor& ret,
    float lambda,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply1<scalar_t, UNROLL_FACTOR>(
    ret, [seeds, lambda] __device__(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 rand = curand_uniform4(&state);
      switch (n) {
        case 4: {
          v4 = static_cast<scalar_t>(-1.0f / lambda * ::log(rand.w));
          // fallthrough
        }
        case 3: {
          v3 = static_cast<scalar_t>(-1.0f / lambda * ::log(rand.z));
          // fallthrough
        }
        case 2: {
          v2 = static_cast<scalar_t>(-1.0f / lambda * ::log(rand.y));
          // fallthrough
        }
        case 1: {
          v1 = static_cast<scalar_t>(-1.0f / lambda * ::log(rand.x));
        }
      }
    }
  );
}

template<typename scalar_t>
void geometric_tensor_cuda_kernel(
    at::Tensor& ret,
    float p,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply1<scalar_t, UNROLL_FACTOR>(
    ret, [seeds, p] __device__(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      float4 rand = curand_uniform4(&state);
      switch (n) {
        case 4: {
          v4 = static_cast<scalar_t>(::ceil(::log(rand.w) / ::log(1.0f-p)));
          // fallthrough
        }
        case 3: {
          v3 = static_cast<scalar_t>(::ceil(::log(rand.z) / ::log(1.0f-p)));
          // fallthrough
        }
        case 2: {
          v2 = static_cast<scalar_t>(::ceil(::log(rand.y) / ::log(1.0f-p)));
          // fallthrough
        }
        case 1: {
          v1 = static_cast<scalar_t>(::ceil(::log(rand.x) / ::log(1.0f-p)));
        }
      }
    }
  );
}

template<typename scalar_t>
void random_tensor_cuda_kernel(
    at::Tensor& ret,
    uint64_t range,
    int64_t base,
    std::pair<uint64_t, uint64_t> seeds) {
  at::cuda::CUDA_tensor_apply1<scalar_t, UNROLL_FACTOR>(
    ret, [seeds, range, base] __device__(
      int n, scalar_t& v1, scalar_t& v2, scalar_t& v3, scalar_t& v4) {
      curandStatePhilox4_32_10_t state;
      curand_init(
          seeds.first,
          blockIdx.x * blockDim.x + threadIdx.x,
          seeds.second,
          &state);
      if (std::is_same<scalar_t, double>::value || std::is_same<scalar_t, int64_t>::value) {
        at::cuda::Array<uint64_t, 4> rand = make_64_bits_from_32_bits(curand4(&state), curand4(&state));
        switch (n) {
          case 4: {
            v4 = static_cast<scalar_t>(rand[3] % range + base);
            // fallthrough
          }
          case 3: {
            v3 = static_cast<scalar_t>(rand[2] % range + base);
            // fallthrough
          }
          case 2: {
            v2 = static_cast<scalar_t>(rand[1] % range + base);
            // fallthrough
          }
          case 1: {
            v1 = static_cast<scalar_t>(rand[0] % range + base);
          }
        }
      } else {
        auto rand = curand4(&state);
        auto range_ = static_cast<uint32_t>(range);
        auto base_ = static_cast<int32_t>(base);
        switch (n) {
          case 4: {
            v4 = static_cast<scalar_t>(rand.w % range_ + base_);
            // fallthrough
          }
          case 3: {
            v3 = static_cast<scalar_t>(rand.z % range_ + base_);
            // fallthrough
          }
          case 2: {
            v2 = static_cast<scalar_t>(rand.y % range_ + base_);
            // fallthrough
          }
          case 1: {
            v1 = static_cast<scalar_t>(rand.x % range_ + base_);
          }
        }
      }
    }
  );
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.scalar_type(), "dirichlet", [&] {
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

Tensor& uniform_tensor_cuda_(Tensor& self, double from_, double to_, Generator* gen) {
  // static_cast everything to float since we are using curand_uniform4
  float from = static_cast<float>(from_);
  float to = static_cast<float>(to_);
  AT_CHECK(from <= to,
           "uniform_ expects to return a [from, to) range, but found from=", from,
           " > to=", to);
  AT_CHECK((to - from) <= std::numeric_limits<float>::max(),
           "uniform_ expects to-from ≤ std::numeric_limits<float>::max(), but found to=", to,
           " and from=", from, " which result in to-from to exceed the limit");
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "uniform_tensor_cuda_", [&] {
    uniform_tensor_cuda_kernel<scalar_t>(self, from, to, seeds);
   });
  return self;
}

Tensor& cauchy_tensor_cuda_(Tensor& self, double median_, double sigma_, Generator* gen) {
  // static_cast everything to float since we are using curand_normal4
  float median = static_cast<float>(median_);
  float sigma = static_cast<float>(sigma_);
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "cauchy_tensor_cuda_", [&] {
    cauchy_tensor_cuda_kernel<scalar_t>(self, median, sigma, seeds);
   });
  return self;
}

Tensor& log_normal_tensor_cuda_(Tensor& self, double mean_, double std_, Generator* gen) {
  // static_cast everything to float since we are using curand_normal4
  float mean = static_cast<float>(mean_);
  float std = static_cast<float>(std_);
  AT_CHECK(std > 0.0f, "log_normal_ expects std > 0.0f, but found std=", std);
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "log_normal_tensor_cuda_", [&] {
    log_normal_tensor_cuda_kernel<scalar_t>(self, mean, std, seeds);
   });
  return self;
}

Tensor& exponential_tensor_cuda_(Tensor& self, double lambda_, Generator* gen) {
  // static_cast everything to float since we are using curand_normal4
  float lambda = static_cast<float>(lambda_);
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "exponential_tensor_cuda_", [&] {
    exponential_tensor_cuda_kernel<scalar_t>(self, lambda, seeds);
   });
  return self;
}

Tensor& geometric_tensor_cuda_(Tensor& self, double p_, Generator* gen) {
  // static_cast everything to float since we are using curand_normal4
  float p = static_cast<float>(p_);
  AT_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "geometric_tensor_cuda_", [&] {
    geometric_tensor_cuda_kernel<scalar_t>(self, p, seeds);
   });
  return self;
}

Tensor& random_tensor_cuda_(Tensor& self, Generator* gen) {
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset;
  if (self.scalar_type() == ScalarType::Long || self.scalar_type() == ScalarType::Double) {
    // when long or double type, we'll need 8 curand 32 bit values to make
    // 4 64-bit values. Hence, num_engine_calls = CURAND4_ENGINE_CALLS * 2 = 8
    counter_offset = calc_philox_increment(total_elements,
                                           grid_size,
                                           cuda::AT_APPLY_THREADS_PER_BLOCK,
                                           UNROLL_FACTOR,
                                           CURAND4_ENGINE_CALLS * 2);
  } else {
    counter_offset = calc_philox_increment(total_elements,
                                           grid_size,
                                           cuda::AT_APPLY_THREADS_PER_BLOCK,
                                           UNROLL_FACTOR,
                                           CURAND4_ENGINE_CALLS);
  }
  auto seeds = next_philox_seed(gen, counter_offset);
  int64_t base = static_cast<int64_t>(0);
  if (isFloatingType(self.scalar_type())) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "random_tensor_cuda_", [&] {
      uint64_t range = static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1);
      random_tensor_cuda_kernel<scalar_t>(self, range, base, seeds);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "random_tensor_cuda_", [&] {
      uint64_t range = static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1;
      random_tensor_cuda_kernel<scalar_t>(self, range, base, seeds);
    });
  }
  
  return self;
}

Tensor& clamped_random_tensor_cuda_(Tensor& self, int64_t from, int64_t to, Generator* gen) {
  AT_CHECK(from < to, "random_ expects 'to' to be greater than 'from', but got from=", from, " >= to=", to);
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset;
  if (self.scalar_type() == ScalarType::Long || self.scalar_type() == ScalarType::Double) {
    // when long or double type, we'll need 8 curand 32 bit values to make
    // 4 64-bit values. Hence, num_engine_calls = CURAND4_ENGINE_CALLS * 2 = 8
    counter_offset = calc_philox_increment(total_elements,
                                           grid_size,
                                           cuda::AT_APPLY_THREADS_PER_BLOCK,
                                           UNROLL_FACTOR,
                                           CURAND4_ENGINE_CALLS * 2);
  } else {
    counter_offset = calc_philox_increment(total_elements,
                                           grid_size,
                                           cuda::AT_APPLY_THREADS_PER_BLOCK,
                                           UNROLL_FACTOR,
                                           CURAND4_ENGINE_CALLS);
  }
  auto seeds = next_philox_seed(gen, counter_offset);
  uint64_t range = to - from;
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "clamped_random_tensor_cuda_", [&] {
    random_tensor_cuda_kernel<scalar_t>(self, range, from, seeds);
  });
  
  return self;
}

Tensor& capped_random_tensor_cuda_(Tensor& self, int64_t to, Generator* gen) {
  return clamped_random_tensor_cuda_(self, 0, to, gen);
}

Tensor& normal_tensor_cuda_(Tensor& self, double mean_, double std_, Generator* gen) {
  // static_cast everything to float since we are using curand_normal4
  float mean = static_cast<float>(mean_);
  float std = static_cast<float>(std_);
  AT_CHECK(std > 0.0f, "normal_ expects std > 0.0f, but found std=", std);
  uint64_t total_elements = self.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "normal_tensor_cuda_", [&] {
    normal_tensor_cuda_kernel<scalar_t>(self, mean, std, seeds);
   });
  return self;
}

Tensor& normal_out_cuda_(Tensor& output, const Tensor& mean_, double std_, Generator* gen) {
  auto mean = std::get<0>(expand_inplace(output, mean_.to(kCUDA)));
  // static_cast everything to float since we are using curand_normal4
  float std = static_cast<float>(std_);
  AT_CHECK(std > 0.0f, "normal_out expects std > 0.0f, but found std=", std);
  uint64_t total_elements = output.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(
    at::ScalarType::Half, output.scalar_type(), "normal_out_cuda_", [&] {
      normal_tensor_cuda_kernel<scalar_t>(output, mean, std, seeds);
   });
  return output;
}

Tensor& normal_out_cuda_(Tensor& output, double mean_, const Tensor& std_, Generator* gen) {
  auto std = std::get<0>(expand_inplace(output, std_.to(kCUDA)));
  // static_cast everything to float since we are using curand_normal4
  float mean = static_cast<float>(mean_);
  uint64_t total_elements = output.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(
    at::ScalarType::Half, output.scalar_type(), "normal_out_cuda_", [&] {
      normal_tensor_cuda_kernel<scalar_t>(output, mean, std, seeds);
   });
  return output;
}

Tensor& normal_out_cuda_(Tensor& output, const Tensor& mean_, const Tensor& std_, Generator* gen) {
  auto std = std::get<0>(expand_inplace(output, std_.to(kCUDA)));
  auto mean = std::get<0>(expand_inplace(output, mean_.to(kCUDA)));
  uint64_t total_elements = output.numel();
  uint64_t grid_size = calc_grid_size_cuda_apply_utils_n(UNROLL_FACTOR, total_elements);
  uint64_t counter_offset = calc_philox_increment(total_elements,
                                                  grid_size,
                                                  cuda::AT_APPLY_THREADS_PER_BLOCK,
                                                  UNROLL_FACTOR,
                                                  CURAND4_ENGINE_CALLS);
  auto seeds = next_philox_seed(gen, counter_offset);
  AT_DISPATCH_ALL_TYPES_AND(
    at::ScalarType::Half, output.scalar_type(), "normal_out_cuda_", [&] {
      normal_tensor_cuda_kernel<scalar_t>(output, mean, std, seeds);
   });
  return output; 
}

Tensor normal_cuda_(const Tensor& mean_, double std_, Generator* gen) {
  Tensor ret = at::empty(mean_.sizes(), mean_.options());
  normal_out_cuda_(ret, mean_, std_, gen);
  return ret;
}

Tensor normal_cuda_(double mean_, const Tensor& std_, Generator* gen) {
  Tensor ret = at::empty(std_.sizes(), std_.options());
  normal_out_cuda_(ret, mean_, std_, gen);
  return ret;
}

Tensor normal_cuda_(const Tensor& mean_, const Tensor& std_, Generator* gen) {
  AT_CHECK(mean_.sizes().equals(std_.sizes()),
           "normal expects 'mean' tensor to be the same size as 'std' tensor but got ",
           "mean tensor size=", mean_.sizes(), " and std tensor size=", std_.sizes());
  Tensor ret = at::empty(mean_.sizes(), mean_.options());
  normal_out_cuda_(ret, mean_, std_, gen);
  return ret;
}

}} // namespace at::native
