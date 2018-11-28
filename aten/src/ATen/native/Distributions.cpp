#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Config.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "c10/util/Exception.h"
#include "ATen/core/Generator.h"
#include "ATen/native/Distributions.h"
#include "ATen/native/DispatchStub.h"
#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <type_traits>
#include <functional>
#include <assert.h>
#include <cpuinfo.h>

#include "TH/THMath.h"

namespace {
/*
 * This section is a counterpart to Distributions.cu
 *
 */

// The function `sample_poisson`
// is adapted from Numpy's distributions.c implementation.
// It is MIT licensed, so here is the copyright:

/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


int64_t sample_poisson(double lambda, at::Generator* generator) {
  std::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
  auto& cpu_engine = generator->getCPUEngine();
  if (lambda >= 10) {
    // transformed rejection method, (Hoermann, 1993)
    int64_t k;
    double U, V, a, b, invalpha, vr, us;

    double slam = std::sqrt(lambda);
    double loglam = std::log(lambda);
    b = 0.931 + 2.53 * slam;
    a = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr = 0.9277 - 3.6224 / (b - 2);

    while (1) {
      U = standard_uniform(cpu_engine) - 0.5;
      V = standard_uniform(cpu_engine);
      us = 0.5 - std::fabs(U);
      k = (int64_t)std::floor((2 * a / us + b) * U + lambda + 0.43);
      if ((us >= 0.07) && (V <= vr)) {
        return k;
      }
      if ((k < 0) || ((us < 0.013) && (V > us))) {
        continue;
      }
      if ((std::log(V) + std::log(invalpha) - std::log(a / (us * us) + b)) <=
          (-lambda + k * loglam - std::lgamma((double)k + 1))) {
        return k;
      }
    }
  } else if (lambda == 0) {
    return 0;
  } else {
    int64_t X;
    double prod, U, enlam;

    enlam = std::exp(-lambda);
    X = 0;
    prod = 1.0;
    while (1) {
      U = standard_uniform(cpu_engine);
      prod *= U;
      if (prod > enlam) {
        X += 1;
      } else {
        return X;
      }
    }
  }
}

} // namespace

namespace at {
namespace native {

Tensor bernoulli(const Tensor& self, Generator* gen) {
  return at::empty_like(self).bernoulli_(self, gen);
}

Tensor bernoulli(const Tensor& self, double p, Generator* gen) {
  return at::empty_like(self).bernoulli_(p, gen);
}

Tensor& bernoulli_out(Tensor& result, const Tensor& self, Generator* gen) {
  // result.resize_as_(self) requires self to have same dtype as result, so we
  // use resize_ instead.
  // TODO: Fix resize_as_. See pytorch/pytorch#11665.
  return result.resize_(self.sizes()).bernoulli_(self, gen);
}

Tensor& bernoulli_tensor_cpu_(Tensor& self, const Tensor& p_, Generator* gen) {
  AT_DISPATCH_ALL_TYPES(self.type(), "bernoulli_tensor_cpu_self_", [&] {
    Generator* generator = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCPU));
    auto& cpu_engine = generator->getCPUEngine();
    using self_t = scalar_t;
    if (p_.scalar_type() == kDouble) {
      auto p = std::get<0>(expand_inplace(self, p_.to(kCPU)));
      CPU_tensor_apply2<self_t, double>(
        self, p, [&cpu_engine](self_t& ret_val, double& p_val) {
          std::bernoulli_distribution bernoulli(p_val);
          ret_val = static_cast<self_t>(bernoulli(cpu_engine));
        });
    } else {
      AT_DISPATCH_FLOATING_TYPES(p_.type(), "bernoulli_tensor_cpu_p_", [&] {
        auto p = std::get<0>(expand_inplace(self, p_.to(kCPU)));
        using p_t = scalar_t;
        CPU_tensor_apply2<self_t, p_t>(
          self, p, [&cpu_engine](self_t& ret_val, p_t& p_val) {
            std::bernoulli_distribution bernoulli(p_val);
            ret_val = static_cast<self_t>(bernoulli(cpu_engine));
          });
      });
    }
  });
  return self;
}

DEFINE_DISPATCH(bernoulli_mkl_stub);

Tensor& bernoulli_scalar_cpu_(Tensor& self, double p, Generator* gen) {
  AT_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
#if AT_MKL_ENABLED()
  if (cpuinfo_initialize() && cpuinfo_vendor_intel == cpuinfo_get_processor(0)->core->vendor) {
    bernoulli_mkl_stub(kCPU, self, p, gen);
    return self;
  }
#endif
  AT_DISPATCH_ALL_TYPES(self.type(), "bernoulli_scalar_cpu_", [&] {
    Generator* generator = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCPU));
    auto& cpu_engine = generator->getCPUEngine();
    CPU_tensor_apply1<scalar_t>(
        self, [&cpu_engine, p](scalar_t& ret_val) {
          std::bernoulli_distribution bernoulli(p);
          ret_val = static_cast<scalar_t>(bernoulli(cpu_engine));
        });
  });
  return self;
}


Tensor _standard_gamma_grad_cpu(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  AT_DISPATCH_FLOATING_TYPES(self.type(), "_standard_gamma_grad", [&] {
    CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(ret, self, output,
      [](scalar_t& ret_val, const scalar_t& self_val, const scalar_t &output_val) {
        ret_val = standard_gamma_grad_one<scalar_t, double>(self_val, output_val);
      }
    );
  });
  return ret;
}

/*
 * This section is a counterpart to Distributions.cu
 */

Tensor _s_poisson_cpu(const Tensor& lambda, Generator *gen) {
  Tensor ret = at::zeros(lambda.sizes(), lambda.options());
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "poisson", [&] {
    Generator* generator = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCPU));
    CPU_tensor_apply2<scalar_t, scalar_t>(ret, lambda,
      [generator](scalar_t& ret_val, const scalar_t& lambda){
        ret_val = static_cast<scalar_t>(sample_poisson(static_cast<double>(lambda), generator));
      }
    );
    });
  return ret;
}

Tensor _s_gamma_cpu(const Tensor& alpha, Generator *gen) {
  Tensor ret = at::zeros(alpha.sizes(), alpha.options());
  AT_DISPATCH_FLOATING_TYPES(ret.type(), "gamma", [&] {
    Generator* generator = detail::checkGeneratorWithDefault(gen, &detail::getDefaultGenerator(kCPU));
    auto& cpu_engine = generator->getCPUEngine();
    CPU_tensor_apply2<scalar_t, scalar_t>(ret, alpha,
      [&cpu_engine](scalar_t& ret_val, const scalar_t& alpha){
        BaseSampler<double> standard_uniform([&cpu_engine] () {
          std::uniform_real_distribution<double> standard_uniform(0.0, 1.0);
          return standard_uniform(cpu_engine);
        });
        BaseSampler<double> standard_normal([&cpu_engine] () {
          std::normal_distribution<double> normal{0.0, 1.0};
          return normal(cpu_engine);
        });
        auto sample = sample_gamma<scalar_t, double>(alpha, standard_uniform, standard_normal);
        ret_val = std::max(std::numeric_limits<scalar_t>::min(), (scalar_t) sample);
      }
    );
    });

  return ret;
}

}} // namespace at::native
