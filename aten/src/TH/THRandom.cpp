#include <TH/THGeneral.h>
#include "THRandom.h"

/* Creates (default seeded) new generator*/
static at::Generator* THGenerator_newUnseeded()
{
  at::Generator *self = &at::globalContext().createGenerator(at::kCPU);
  return self;
}

/* Creates new generator and makes sure it is randomly seeded*/
at::Generator* THGenerator_new()
{
  at::Generator *self = &at::globalContext().createGenerator(at::kCPU);
  self->seed();
  return self;
}

at::Generator* THGenerator_copy(at::Generator *self, at::Generator *from)
{
    THGeneratorState_copy(self->getState(), from->getState());
    return self;
}

void THGenerator_free(at::Generator *self)
{
  self->getState()->mutex.~mutex();
  THFree(self);
}

int THGeneratorState_isValid(GeneratorState *_gen_state)
{
  if (_gen_state != nullptr)
    return 1;

  return 0;
}

GeneratorState* THGeneratorState_copy(GeneratorState *self, GeneratorState *from)
{
  memcpy(self, from, sizeof(GeneratorState));
  return self;
}

uint64_t THRandom_seed(at::Generator *_generator)
{
  return _generator->seed();
}

void THRandom_manualSeed(at::Generator *_generator, uint64_t the_seed_)
{
  _generator->manualSeed(the_seed_);
}

uint64_t THRandom_initialSeed(at::Generator *_generator)
{
  return _generator->getStartingSeed();
}

uint32_t THRandom_random(at::Generator *_generator)
{
  return _generator->getState()->engine();
}

uint64_t THRandom_random64(at::Generator *_generator)
{
  uint64_t hi = _generator->getState()->engine();
  uint64_t lo = _generator->getState()->engine();
  return hi | lo;
}

/* generates a random number on [0,1)-double-interval */
static inline double uniform_double(at::Generator *_generator)
{
  std::uniform_real_distribution<double> uniform(0, 1);
  return uniform(_generator->getState()->engine);
}

/* generates a random number on [0,1)-double-interval */
static inline float uniform_float(at::Generator *_generator)
{
  std::uniform_real_distribution<float> uniform(0, 1);
  return uniform(_generator->getState()->engine);
}

double THRandom_uniform(at::Generator *_generator, double a, double b)
{
  return(uniform_double(_generator) * (b - a) + a);
}

float THRandom_uniformFloat(at::Generator *_generator, float a, float b)
{
  return(uniform_float(_generator) * (b - a) + a);
}

double THRandom_normal(at::Generator *_generator, double mean, double stdv)
{
  std::normal_distribution<double> normal{mean, stdv};
  return normal(_generator->getState()->engine);
}

double THRandom_exponential(at::Generator *_generator, double lambda)
{
  std::exponential_distribution<double> exponential(lambda);
  return exponential(_generator->getState()->engine);
}

double THRandom_cauchy(at::Generator *_generator, double median, double sigma)
{
  std::cauchy_distribution<double> cauchy(median, sigma);
  return cauchy(_generator->getState()->engine);
}

double THRandom_logNormal(at::Generator *_generator, double mean, double stdv)
{
  std::lognormal_distribution<double> logNormal(mean, stdv);
  return logNormal(_generator->getState()->engine);
}

int THRandom_geometric(at::Generator *_generator, double p)
{
  std::geometric_distribution<> geometric(p);
  return geometric(_generator->getState()->engine);
}

int THRandom_bernoulli(at::Generator *_generator, double p)
{
  std::bernoulli_distribution bernoulli(p);
  return bernoulli(_generator->getState()->engine);
}

int THRandom_bernoulliFloat(at::Generator *_generator, float p)
{
  std::bernoulli_distribution bernoulli(p);
  return bernoulli(_generator->getState()->engine);
}
