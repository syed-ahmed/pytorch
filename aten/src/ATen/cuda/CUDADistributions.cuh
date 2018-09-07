#pragma once

#include <math.h>
#include <stdint.h>
#include <cuda.h>

#include "ATen/core/Generator.h"

/*
* Distribution implementations adapted from THRandom
*/

/*
* Produces a uniform distribution in the range [0,1) of type double
*/
__device__  __inline__ double standard_uniform_distribution(at::Generator* _generator) {
  uint32_t random_val = _generator->getState()->engine();
  double result = static_cast<double>(random_val) * INV_UINT32_MAX_DOUBLE;
  if (result == 1) {
    return 0.0;
  } else {
    return result;
  }
}

/*
* Produces a normal distribution given philox random, mean and standard deviation
*/
__device__  __inline__ double normal_distribution(at::Generator* _generator) {
  // Box-Muller method
  if(!_generator->getState()->normal_is_valid) {
    double normal_x = standard_uniform_distribution(_generator);
    double normal_y = standard_uniform_distribution(_generator);
    double normal_rho = sqrt(-2.0 * log(1.0 - normal_y));
    _generator->setNormalDistValid(1);
    _generator->setNormalDistState(normal_x, normal_rho);
  } else {
    _generator->setNormalDistValid(0);
  }
  if(_generator->getState()->normal_is_valid) {
    return _generator->getState()->normal_rho*cos(2.0*M_PI*_generator->getState()->normal_x);
  } else {
    return _generator->getState()->normal_rho*sin(2.0*M_PI*_generator->getState()->normal_x);
  }
}

/*
* Produces a lognormal distribution given philox random, mean and standard deviation
*/
__device__  __inline__ double lognormal_distribution(at::Generator* _generator, double mean, double stdv) {
  return exp((normal_distribution(_generator) * stdv) + mean);
}

/*
* Produces a poisson distribution given philox random, and lambda
*/
__device__  __inline__ int64_t poisson_distribution(at::Generator* _generator, double lambda) {
  if (lambda >= 10) {
    // transformed rejection method, (Hoermann, 1993)
    int64_t k;
    double U, V, a, b, invalpha, vr, us;

    double slam = sqrt(lambda);
    double loglam = log(lambda);
    b = 0.931 + 2.53 * slam;
    a = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr = 0.9277 - 3.6224 / (b - 2);

    while (1) {
      U = standard_uniform_distribution(_generator) - 0.5;
      V = standard_uniform_distribution(_generator);
      us = 0.5 - fabs(U);
      k = (int64_t)floor((2 * a / us + b) * U + lambda + 0.43);
      if ((us >= 0.07) && (V <= vr)) {
        return k;
      }
      if ((k < 0) || ((us < 0.013) && (V > us))) {
        continue;
      }
      if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
          (-lambda + k * loglam - lgamma((double)k + 1))) {
        return k;
      }
    }
  } else if (lambda == 0) {
    return 0;
  } else {
    int64_t X;
    double prod, U, enlam;

    enlam = exp(-lambda);
    X = 0;
    prod = 1.0;
    while (1) {
      U = standard_uniform_distribution(_generator);
      prod *= U;
      if (prod > enlam) {
        X += 1;
      } else {
        return X;
      }
    }
  }
}