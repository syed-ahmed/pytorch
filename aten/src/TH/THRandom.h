#ifndef TH_RANDOM_INC
#define TH_RANDOM_INC

#include <TH/THGeneral.h>

#include "ATen/ATen.h"

struct GeneratorState;

/* Manipulate THGenerator objects */
TH_API at::Generator * THGenerator_new(void);
TH_API at::Generator * THGenerator_copy(at::Generator *self, at::Generator *from);
TH_API void THGenerator_free(at::Generator *gen);

/* Checks if given generator state is valid */
TH_API int THGeneratorState_isValid(GeneratorState *_gen_state);

/* Manipulate THGeneratorState objects */
TH_API GeneratorState * THGeneratorState_copy(GeneratorState *self, GeneratorState *from);

/* Initializes the random number generator from /dev/urandom (or on Windows
platforms with the current time (granularity: seconds)) and returns the seed. */
TH_API uint64_t THRandom_seed(at::Generator *_generator);

/* Initializes the random number generator with the given int64_t "the_seed_". */
TH_API void THRandom_manualSeed(at::Generator *_generator, uint64_t the_seed_);

/* Returns the starting seed used. */
TH_API uint64_t THRandom_initialSeed(at::Generator *_generator);

/* Generates a uniform 32 bits integer. */
TH_API uint32_t THRandom_random(at::Generator *_generator);

/* Generates a uniform 64 bits integer. */
TH_API uint64_t THRandom_random64(at::Generator *_generator);

/* Generates a uniform random double on [a, b). */
TH_API double THRandom_uniform(at::Generator *_generator, double a, double b);

/* Generates a uniform random float on [0,1). */
TH_API float THRandom_uniformFloat(at::Generator *_generator, float a, float b);

/** Generates a random number from a normal distribution.
    (With mean #mean# and standard deviation #stdv >= 0#).
*/
TH_API double THRandom_normal(at::Generator *_generator, double mean, double stdv);

/** Generates a random number from an exponential distribution.
    The density is $p(x) = lambda * exp(-lambda * x)$, where
    lambda is a positive number.
*/
TH_API double THRandom_exponential(at::Generator *_generator, double lambda);

/** Returns a random number from a Cauchy distribution.
    The Cauchy density is $p(x) = sigma/(pi*(sigma^2 + (x-median)^2))$
*/
TH_API double THRandom_cauchy(at::Generator *_generator, double median, double sigma);

/** Generates a random number from a log-normal distribution.
    (#mean > 0# is the mean of the log-normal distribution
    and #stdv# is its standard deviation).
*/
TH_API double THRandom_logNormal(at::Generator *_generator, double mean, double stdv);

/** Generates a random number from a geometric distribution.
    It returns an integer #i#, where $p(i) = (1-p) * p^(i-1)$.
    p must satisfy $0 < p < 1$.
*/
TH_API int THRandom_geometric(at::Generator *_generator, double p);

/* Returns true with double probability $p$ and false with probability $1-p$ (p > 0). */
TH_API int THRandom_bernoulli(at::Generator *_generator, double p);

/* Returns true with float probability $p$ and false with probability $1-p$ (p > 0). */
TH_API int THRandom_bernoulliFloat(at::Generator *_generator, float p);

#endif
