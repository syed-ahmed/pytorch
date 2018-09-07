#include "ATen/cuda/CUDAGenerator.h"

namespace at {
namespace detail {

// Global generator state and constants
static int64_t num_gpus = -1;
static std::once_flag num_gpu_init_flag;
static std::deque<std::once_flag> cuda_device_flags;
static std::vector<GeneratorState> default_gen_states_cuda;
static std::vector<std::unique_ptr<CUDAGenerator>> default_gens_cuda;
static std::unique_ptr<CUDAGenerator> new_gen_cuda;

/* 
* Populates the global default_gen_states_cuda vector
* Warning: this function must only be called once!
*/
static void initCUDAGenStatesVector(){
  num_gpus = at::cuda::getNumGPUs();
  cuda_device_flags.resize(num_gpus);

  // one default generator object per device
  // each object has their own state
  default_gen_states_cuda.resize(num_gpus);
  default_gens_cuda.resize(num_gpus);
}

/* 
* Populates global values and creates a generator for CUDA
* Note: the generator on CUDA is a 32 bit Philox Engine.
* Warning: this function must only be called once!
*/
static void initGlobalGeneratorState(int64_t device = -1){
  // Switches to the requested device so engines are properly associated
  // with it.
  at::DeviceGuard device_guard{(int)device};
  std::lock_guard<std::mutex> lock(default_gen_states_cuda[device].mutex);
  default_gen_states_cuda[device].device = device;
  default_gen_states_cuda[device].starting_seed = default_gen_states_cuda[device].engine.default_seed;
  default_gens_cuda[device].reset(new CUDAGenerator(&default_gen_states_cuda[device]));
}


/*
* Gets the default CUDA generator. Lazily creates on if
* there is none.
*/
Generator& CUDAGenerator_getDefaultGenerator(int64_t device = -1) {
  std::call_once(num_gpu_init_flag, initCUDAGenStatesVector);
  if (device == -1) device = at::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  std::call_once(cuda_device_flags[device], initGlobalGeneratorState, device);
  return *(default_gens_cuda[device].get());
}

/*
* Creates a CUDA generator instance. Note this is not the default generator
* that torch.cuda.random API uses. This function is meant for the
* torch.Generator API which lets the user create a Generator
* independent of the default generator.
*/
Generator& CUDAGenerator_createGenerator(int64_t device = -1) {
  GeneratorState new_gen_state;
  if (device == -1) device = at::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);

  at::DeviceGuard device_guard{(int)device};
  std::lock_guard<std::mutex> lock(new_gen_state.mutex);
  new_gen_state.device = device;
  new_gen_state.starting_seed = new_gen_state.engine.default_seed;
  new_gen_state.engine.seed(new_gen_state.engine.default_seed);
  new_gen_cuda.reset(new CUDAGenerator(&new_gen_state));
  return *(new_gen_cuda.get());
}

} // namespace detail

AT_HOST_DEVICE GeneratorState* CUDAGenerator::getState() { 
  return this->state_; 
}
  
AT_HOST_DEVICE void CUDAGenerator::setState(GeneratorState* state_in) {
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
* Note that the seed for CUDAGenerator just sets the 
* starting_seed value in GeneratorState
*/

AT_HOST_DEVICE uint32_t CUDAGenerator::seed() {
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
  this->state_->starting_seed = seed_val;
  return seed_val;
}

/* 
* Returns the starting seed of the generator
*/
AT_HOST_DEVICE uint32_t CUDAGenerator::getStartingSeed() {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  return this->state_->starting_seed;
}

/* 
* Manually set the stating_seed
*/
AT_HOST_DEVICE void CUDAGenerator::manualSeed(uint32_t seed) {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  this->state_->starting_seed = seed;
}

AT_HOST_DEVICE uint32_t CUDAGenerator::random() {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  return this->state_->engine();
}

AT_HOST_DEVICE uint64_t CUDAGenerator::random64() {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  uint32_t hi = this->state_->engine();
  uint32_t lo = this->state_->engine();
  return hi | lo;
}

// CUDA Generator specific methods
AT_HOST_DEVICE void CUDAGenerator::setNormalDistState(double x, double rho) {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  this->state_->normal_x = x;
  this->state_->normal_rho = rho;
}

AT_HOST_DEVICE void CUDAGenerator::setNormalDistValid(int valid_flag) {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  this->state_->normal_is_valid = valid_flag;
}

AT_HOST_DEVICE uint32_t CUDAGenerator::seedAll() {
  
}

AT_HOST_DEVICE void CUDAGenerator::manualSeedAll() {

}

AT_HOST_DEVICE std::pair<uint64_t, uint64_t> CUDAGenerator::next_philox_seed(uint64_t increment) {
  std::lock_guard<std::mutex> lock(this->state_->mutex);
  uint64_t offset = this->state_->philox_seed_offset.fetch_add(increment);
  return std::make_pair(this->state_->starting_seed, offset);
}

} // namespace at
