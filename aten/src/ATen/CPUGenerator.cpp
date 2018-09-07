#include "ATen/CPUGenerator.h"

namespace at {
namespace detail {

// Global generator state and constants
static std::once_flag cpu_device_flag;
static GeneratorState default_gen_state_cpu;
static std::unique_ptr<CPUGenerator> default_gen_cpu;
static std::unique_ptr<CPUGenerator> new_gen_cpu;

/* 
* Populates global values and creates a generator for CPU.
* Note: the generator on CPU is a 32 bit Mersenne Twister Engine.
* Warning: this function must only be called once!
*/
static void initGlobalGeneratorState(){
  std::lock_guard<std::mutex> lock(default_gen_state_cpu.mutex);
  default_gen_state_cpu.device = -1;
  default_gen_state_cpu.starting_seed = default_gen_state_cpu.engine.default_seed;
  default_gen_state_cpu.engine.seed(default_gen_state_cpu.engine.default_seed);
  default_gen_cpu.reset(new CPUGenerator(&default_gen_state_cpu));
}

/*
* Gets the default CPU generator state. Lazily creates on if
* there is none.
*/
Generator& CPUGenerator_getDefaultGenerator() {
  std::call_once(cpu_device_flag, initGlobalGeneratorState);
  return *(default_gen_cpu.get());
}

/*
* Creates a CPU generator instance. Note this is not the default generator
* that torch.random API uses. This function is meant for the
* torch.Generator API which lets the user create a Generator
* independent of the default generator.
*/
Generator& CPUGenerator_createGenerator() {
  GeneratorState new_gen_state;
  std::lock_guard<std::mutex> lock(new_gen_state.mutex);
  new_gen_state.device = -1;
  new_gen_state.starting_seed = new_gen_state.engine.default_seed;
  new_gen_state.engine.seed(new_gen_state.engine.default_seed);
  new_gen_cpu.reset(new CPUGenerator(&new_gen_state));
  return *(new_gen_cpu.get());
}

} // namespace detail

AT_HOST_DEVICE GeneratorState* CPUGenerator::getState() { 
  return this->state_; 
}
  
AT_HOST_DEVICE void CPUGenerator::setState(GeneratorState* state_in) {
  if(typeid(state_in->engine) != typeid(this->state_->engine)){
    AT_ERROR("Cannot set the generator engine with an engine of different type.");
  }else{
    this->state_ = state_in;
  }
}

/* 
* Randomly seeds the engine using the non-deterministic
* std::random_device generator
*
* Note that seed() function in the std library just
* initializes the engine with the default_seed. In our
* version of seed() function, we get a random number and 
* then set the seed of the engine to that number and then
* return the seed.
*/

AT_HOST_DEVICE uint32_t CPUGenerator::seed() {
  std::random_device rd;
  uint32_t seed_val;
  // some compilers may not work std::random_device
  // in those case use chrono
  if (rd.entropy() != 0) {
    seed_val = rd();
  }
  else {
    seed_val = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  this->state_->engine.seed(seed_val);
  this->state_->starting_seed = seed_val;
  return seed_val;
}

/* 
* Returns the current seed of the generator
*/
AT_HOST_DEVICE uint32_t CPUGenerator::getStartingSeed() {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  return this->state_->starting_seed;
}

/* 
* Manually seed the engine with the seed input
* Note that, in std library, this is not called manualSeed()
* but just seed(input_value)
*/
AT_HOST_DEVICE void CPUGenerator::manualSeed(uint32_t seed) {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  this->state_->engine.seed(seed);
  this->state_->starting_seed = seed;
}

AT_HOST_DEVICE uint32_t CPUGenerator::random() {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  return this->state_->engine();
}

AT_HOST_DEVICE uint64_t CPUGenerator::random64() {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  uint32_t hi = this->state_->engine();
  uint32_t lo = this->state_->engine();
  return hi | lo;
}

} // namespace at
