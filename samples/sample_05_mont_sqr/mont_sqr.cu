/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

/************************************************************************************************
 *  This benchmark generates 100,000+ random 1018-bit numbers and odd 1018-bit moduli
 *  (represented inside a 1024-bit CGBN environment), benchmarks repeated Montgomery modular
 *  squaring (cgbn_mont_sqr) in a loop per instance on the GPU using CUDA events, and verifies
 *  every final result on the CPU using GMP (mpz_powm with exponent 2^iterations).
 *
 *  Usage:
 *    ./mont_sqr [-n instances] [-i iterations] [-r runs] [-t tpi] [-s seed] [-a]
 ************************************************************************************************/

#define DEFAULT_INSTANCES  (100*1024)
#define DEFAULT_ITERATIONS 1024
#define DEFAULT_RUNS       10
#define DEFAULT_TPI        8
#define DEFAULT_SEED       1

#define ENV_BITS           1024
#define INPUT_BITS         1018

template<uint32_t tpi, uint32_t bits, uint32_t num_bits>
class mont_sqr_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance (8, 16, or 32 for 1024-bit)
  static const uint32_t BITS=bits;                 // CGBN environment size (must be divisible by 32)
  static const uint32_t NUM_BITS=num_bits;         // bit-width of generated random inputs & moduli
};

template<class params>
class mont_sqr_t {
  public:
  static_assert(params::BITS % 32 == 0, "BITS must be divisible by 32");
  static_assert(params::NUM_BITS >= 2 && params::NUM_BITS <= params::BITS, "NUM_BITS must be in [2, BITS]");

  static const uint32_t WORDS=params::BITS/32;

  // Four 1024-bit fields = 512 bytes per instance (128-byte aligned for coalesced loads/stores)
  typedef struct {
    cgbn_mem_t<params::BITS> x;
    cgbn_mem_t<params::BITS> modulus;
    cgbn_mem_t<params::BITS> mont_x;
    cgbn_mem_t<params::BITS> result;
  } instance_t;

  typedef cgbn_context_t<params::TPI, params> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t              bn_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;

  __device__ __forceinline__ mont_sqr_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance)
      : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
  }

  __device__ __forceinline__ void to_mont(bn_t &mont_x, const bn_t &x, const bn_t &modulus) {
    cgbn_bn2mont(_env, mont_x, x, modulus);
  }

  __device__ __forceinline__ void repeated_sqr(bn_t &result, const bn_t &mont_x, const bn_t &modulus, uint32_t iterations) {
    uint32_t np0=-cgbn_binary_inverse_ui32(_env, cgbn_get_ui32(_env, modulus));

    cgbn_set(_env, result, mont_x);
    #pragma nounroll
    for(uint32_t iter=0;iter<iterations;iter++) {
      cgbn_mont_sqr(_env, result, result, modulus, np0);
    }
  }

  __device__ __forceinline__ void from_mont(bn_t &result, const bn_t &mont_r, const bn_t &modulus) {
    uint32_t np0=-cgbn_binary_inverse_ui32(_env, cgbn_get_ui32(_env, modulus));

    cgbn_mont2bn(_env, result, mont_r, modulus, np0);
  }

  __host__ static instance_t *generate_instances(uint32_t count, unsigned long seed) {
    instance_t     *instances=(instance_t *)malloc(sizeof(instance_t)*count);
    gmp_randstate_t state;
    mpz_t           x, m;

    if(instances==NULL) {
      fprintf(stderr, "Failed to allocate %u instances\n", count);
      exit(1);
    }

    gmp_randinit_default(state);
    gmp_randseed_ui(state, seed);
    mpz_init(x);
    mpz_init(m);

    for(uint32_t index=0;index<count;index++) {
      mpz_urandomb(x, state, params::NUM_BITS);
      mpz_urandomb(m, state, params::NUM_BITS);

      // Ensure modulus is odd and >= 3
      mpz_setbit(m, 0);
      if(mpz_cmp_ui(m, 3)<0)
        mpz_set_ui(m, 3);

      // Ensure 0 <= x < modulus < 2^NUM_BITS
      int cmp=mpz_cmp(x, m);
      if(cmp>0) {
        mpz_swap(x, m);
        mpz_setbit(m, 0);
      }
      else if(cmp==0) {
        mpz_sub_ui(x, x, 1);
      }

      from_mpz(x, instances[index].x._limbs, WORDS);
      from_mpz(m, instances[index].modulus._limbs, WORDS);
      zero_words(instances[index].mont_x._limbs, WORDS);
      zero_words(instances[index].result._limbs, WORDS);
    }

    mpz_clear(x);
    mpz_clear(m);
    gmp_randclear(state);
    return instances;
  }

  __host__ static bool verify_results(instance_t *instances, uint32_t count, uint32_t iterations, double *cpu_seconds_out) {
    uint32_t num_threads=std::thread::hardware_concurrency();
    if(num_threads==0)
      num_threads=1;
    if(num_threads>count)
      num_threads=count;

    std::atomic<int32_t>     first_failure(-1);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    auto start_time=std::chrono::steady_clock::now();

    uint32_t chunk=(count+num_threads-1)/num_threads;
    for(uint32_t t=0;t<num_threads;t++) {
      uint32_t start=t*chunk;
      uint32_t end=(start+chunk<count) ? (start+chunk) : count;
      if(start>=end)
        break;

      workers.emplace_back([instances, start, end, iterations, &first_failure]() {
        mpz_t x, m, exp, computed, correct;

        mpz_init(x);
        mpz_init(m);
        mpz_init(exp);
        mpz_init(computed);
        mpz_init(correct);

        // Repeated squaring `iterations` times modulo m equals x^(2^iterations) mod m
        mpz_set_ui(exp, 0);
        mpz_setbit(exp, (mp_bitcnt_t)iterations);

        for(uint32_t index=start;index<end;index++) {
          if(first_failure.load(std::memory_order_relaxed)!=-1)
            break;

          to_mpz(x, instances[index].x._limbs, WORDS);
          to_mpz(m, instances[index].modulus._limbs, WORDS);
          to_mpz(computed, instances[index].result._limbs, WORDS);

          mpz_powm(correct, x, exp, m);
          if(mpz_cmp(correct, computed)!=0) {
            int32_t expected=-1;
            first_failure.compare_exchange_strong(expected, (int32_t)index);
            break;
          }
        }

        mpz_clear(x);
        mpz_clear(m);
        mpz_clear(exp);
        mpz_clear(computed);
        mpz_clear(correct);
      });
    }

    for(size_t t=0;t<workers.size();t++)
      workers[t].join();

    auto   end_time=std::chrono::steady_clock::now();
    double cpu_seconds=std::chrono::duration<double>(end_time-start_time).count();
    if(cpu_seconds_out!=NULL)
      *cpu_seconds_out=cpu_seconds;

    int32_t failed_index=first_failure.load();
    if(failed_index!=-1) {
      mpz_t x, m, exp, computed, correct;

      mpz_init(x);
      mpz_init(m);
      mpz_init(exp);
      mpz_init(computed);
      mpz_init(correct);

      mpz_set_ui(exp, 0);
      mpz_setbit(exp, (mp_bitcnt_t)iterations);

      to_mpz(x, instances[failed_index].x._limbs, WORDS);
      to_mpz(m, instances[failed_index].modulus._limbs, WORDS);
      to_mpz(computed, instances[failed_index].result._limbs, WORDS);
      mpz_powm(correct, x, exp, m);

      printf("GPU mont_sqr kernel failed on instance %d:\n", failed_index);
      gmp_printf("  x       : %#Zx\n", x);
      gmp_printf("  modulus : %#Zx\n", m);
      gmp_printf("  expected: %#Zx\n", correct);
      gmp_printf("  computed: %#Zx\n", computed);

      mpz_clear(x);
      mpz_clear(m);
      mpz_clear(exp);
      mpz_clear(computed);
      mpz_clear(correct);
      return false;
    }

    printf("All %u results match! (verified on %u CPU threads in %.3f s)\n", count, (uint32_t)workers.size(), cpu_seconds);
    return true;
  }
};

template<class params>
__global__ void kernel_to_mont(cgbn_error_report_t *report, typename mont_sqr_t<params>::instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  mont_sqr_t<params>                ms(cgbn_report_monitor, report, (int32_t)instance);
  typename mont_sqr_t<params>::bn_t x, m, mont_x;

  cgbn_load(ms._env, x, &(instances[instance].x));
  cgbn_load(ms._env, m, &(instances[instance].modulus));

  ms.to_mont(mont_x, x, m);

  cgbn_store(ms._env, &(instances[instance].mont_x), mont_x);
}

template<class params>
__global__ void kernel_mont_sqr(cgbn_error_report_t *report, typename mont_sqr_t<params>::instance_t *instances, uint32_t count, uint32_t iterations) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  mont_sqr_t<params>                ms(cgbn_report_monitor, report, (int32_t)instance);
  typename mont_sqr_t<params>::bn_t mont_x, m, r;

  cgbn_load(ms._env, mont_x, &(instances[instance].mont_x));
  cgbn_load(ms._env, m, &(instances[instance].modulus));

  ms.repeated_sqr(r, mont_x, m, iterations);

  cgbn_store(ms._env, &(instances[instance].result), r);
}

template<class params>
__global__ void kernel_from_mont(cgbn_error_report_t *report, typename mont_sqr_t<params>::instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  mont_sqr_t<params>                ms(cgbn_report_monitor, report, (int32_t)instance);
  typename mont_sqr_t<params>::bn_t mont_r, m, r;

  cgbn_load(ms._env, mont_r, &(instances[instance].result));
  cgbn_load(ms._env, m, &(instances[instance].modulus));

  ms.from_mont(r, mont_r, m);

  cgbn_store(ms._env, &(instances[instance].result), r);
}

template<class params>
bool run_benchmark(uint32_t instance_count, uint32_t iterations, uint32_t runs, unsigned long seed) {
  typedef typename mont_sqr_t<params>::instance_t instance_t;

  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  cudaEvent_t          start, stop;
  cudaDeviceProp       prop;
  uint32_t             TPB=(params::TPB==0) ? 128 : params::TPB;
  uint32_t             TPI=params::TPI, IPB=TPB/TPI;
  uint32_t             blocks=(instance_count+IPB-1)/IPB;
  double               total_sqrs=(double)instance_count*(double)iterations;

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("================================================================\n");
  printf("GPU             : %s\n", prop.name);
  printf("Instances       : %u (%u-bit random numbers in %u-bit CGBN env)\n", instance_count, params::NUM_BITS, params::BITS);
  printf("Launch config   : TPI=%u, TPB=%u (%u instances/block, %u blocks)\n", TPI, TPB, IPB, blocks);
  printf("Squarings/inst  : %u (%.2fM total cgbn_mont_sqr ops per launch)\n", iterations, total_sqrs/1e6);
  printf("Benchmark runs  : %u timed (+ 1 warmup)\n", runs);
  printf("================================================================\n");

  printf("Generating %u random %u-bit instances (seed=%lu) ...\n", instance_count, params::NUM_BITS, seed);
  instances=mont_sqr_t<params>::generate_instances(instance_count, seed);

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  // Convert inputs to Montgomery representation outside the timed squaring kernel
  kernel_to_mont<params><<<blocks, TPB>>>(report, gpuInstances, instance_count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  printf("Warming up GPU mont_sqr kernel ...\n");
  kernel_mont_sqr<params><<<blocks, TPB>>>(report, gpuInstances, instance_count, 5*iterations);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  printf("Benchmarking GPU cgbn_mont_sqr ...\n");
  double total_ms=0.0;
  float  min_ms=FLT_MAX;
  for(uint32_t run=0;run<runs;run++) {
    float elapsed_ms=0.0f;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_mont_sqr<params><<<blocks, TPB>>>(report, gpuInstances, instance_count, iterations);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CGBN_CHECK(report);

    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("  Run %u/%u: %.3f ms\n", run+1, runs, elapsed_ms);
    fflush(stdout);

    total_ms+=elapsed_ms;
    if(elapsed_ms<min_ms)
      min_ms=elapsed_ms;
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  double avg_ms=total_ms/(double)runs;
  double avg_sec=avg_ms/1000.0;
  double msqr_per_sec=(total_sqrs/avg_sec)/1e6;
  double best_msqr_per_sec=(total_sqrs/(min_ms/1000.0))/1e6;
  double ns_per_sqr=(avg_ms*1e6)/total_sqrs;

  printf("GPU Results (TPI=%u):\n", TPI);
  printf("  Avg kernel time : %.3f ms (best: %.3f ms)\n", avg_ms, min_ms);
  printf("  Avg throughput  : %.2f M mont_sqr/s (best: %.2f M mont_sqr/s)\n", msqr_per_sec, best_msqr_per_sec);
  printf("  Avg time/sqr    : %.3f ns\n", ns_per_sqr);

  // Convert results back from Montgomery representation to standard space
  kernel_from_mont<params><<<blocks, TPB>>>(report, gpuInstances, instance_count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));

  printf("Verifying the results on CPU ...\n");
  double cpu_sec=0.0;
  bool   ok=mont_sqr_t<params>::verify_results(instances, instance_count, iterations, &cpu_sec);
  if(ok && cpu_sec>0.0) {
    double cpu_msqr_per_sec=(total_sqrs/cpu_sec)/1e6;
    printf("  CPU throughput  : %.2f M sqr/s (GPU speedup: %.2fx)\n\n", cpu_msqr_per_sec, avg_sec>0.0 ? (cpu_sec/avg_sec) : 0.0);
  }

  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
  return ok;
}

static void print_usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  -n <instances>   Number of random %d-bit instances (default: %d)\n", INPUT_BITS, DEFAULT_INSTANCES);
  printf("  -i <iterations>  Montgomery squarings per instance (default: %d)\n", DEFAULT_ITERATIONS);
  printf("  -r <runs>        Number of timed GPU kernel runs (default: %d)\n", DEFAULT_RUNS);
  printf("  -t <tpi>         Threads per instance: 8, 16, or 32 (default: %d)\n", DEFAULT_TPI);
  printf("  -a               Benchmark and verify all supported TPIs (8, 16, 32)\n");
  printf("  -s <seed>        Random seed (default: %d)\n", DEFAULT_SEED);
  printf("  -h               Show this help message\n");
}

static bool parse_ulong_arg(const char *opt, const char *val_str, unsigned long *out) {
  if(val_str==NULL || *val_str=='\0' || *val_str=='-') {
    fprintf(stderr, "Error: invalid value for %s: '%s'\n", opt, val_str ? val_str : "");
    return false;
  }
  char *end=NULL;
  errno=0;
  unsigned long val=strtoul(val_str, &end, 10);
  if(errno!=0 || end==val_str || *end!='\0') {
    fprintf(stderr, "Error: invalid value for %s: '%s'\n", opt, val_str);
    return false;
  }
  *out=val;
  return true;
}

static bool parse_u32_arg(const char *opt, const char *val_str, uint32_t *out) {
  unsigned long val=0;
  if(!parse_ulong_arg(opt, val_str, &val) || val>UINT32_MAX) {
    if(val>UINT32_MAX)
      fprintf(stderr, "Error: value out of range for %s: '%s'\n", opt, val_str);
    return false;
  }
  *out=(uint32_t)val;
  return true;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IOLBF, 0);

  uint32_t      instances=DEFAULT_INSTANCES;
  uint32_t      iterations=DEFAULT_ITERATIONS;
  uint32_t      runs=DEFAULT_RUNS;
  uint32_t      tpi=DEFAULT_TPI;
  unsigned long seed=DEFAULT_SEED;
  bool          all_tpi=false;

  for(int arg=1;arg<argc;arg++) {
    if(strcmp(argv[arg], "-n")==0 && arg+1<argc) {
      if(!parse_u32_arg("-n", argv[++arg], &instances)) return 1;
    }
    else if(strcmp(argv[arg], "-i")==0 && arg+1<argc) {
      if(!parse_u32_arg("-i", argv[++arg], &iterations)) return 1;
    }
    else if(strcmp(argv[arg], "-r")==0 && arg+1<argc) {
      if(!parse_u32_arg("-r", argv[++arg], &runs)) return 1;
    }
    else if(strcmp(argv[arg], "-t")==0 && arg+1<argc) {
      if(!parse_u32_arg("-t", argv[++arg], &tpi)) return 1;
    }
    else if(strcmp(argv[arg], "-s")==0 && arg+1<argc) {
      if(!parse_ulong_arg("-s", argv[++arg], &seed)) return 1;
    }
    else if(strcmp(argv[arg], "-a")==0) {
      all_tpi=true;
    }
    else if(strcmp(argv[arg], "-h")==0 || strcmp(argv[arg], "--help")==0) {
      print_usage(argv[0]);
      return 0;
    }
    else {
      fprintf(stderr, "Unknown or incomplete option: %s\n", argv[arg]);
      print_usage(argv[0]);
      return 1;
    }
  }

  if(instances==0 || iterations==0 || runs==0) {
    fprintf(stderr, "Error: instances, iterations, and runs must be >= 1\n");
    return 1;
  }

  int device_count=0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if(device_count<=0) {
    fprintf(stderr, "Error: no CUDA-capable GPU device found.\n");
    return 1;
  }

  if(all_tpi) {
    bool ok=true;
    ok=run_benchmark<mont_sqr_params_t<8, ENV_BITS, INPUT_BITS>>(instances, iterations, runs, seed) && ok;
    ok=run_benchmark<mont_sqr_params_t<16, ENV_BITS, INPUT_BITS>>(instances, iterations, runs, seed) && ok;
    ok=run_benchmark<mont_sqr_params_t<32, ENV_BITS, INPUT_BITS>>(instances, iterations, runs, seed) && ok;
    return ok ? 0 : 1;
  }

  bool ok=false;
  if(tpi==8) {
    ok=run_benchmark<mont_sqr_params_t<8, ENV_BITS, INPUT_BITS>>(instances, iterations, runs, seed);
  }
  else if(tpi==16) {
    ok=run_benchmark<mont_sqr_params_t<16, ENV_BITS, INPUT_BITS>>(instances, iterations, runs, seed);
  }
  else if(tpi==32) {
    ok=run_benchmark<mont_sqr_params_t<32, ENV_BITS, INPUT_BITS>>(instances, iterations, runs, seed);
  }
  else {
    fprintf(stderr, "Unsupported TPI=%u for %d-bit CGBN (supported: 8, 16, 32)\n", tpi, ENV_BITS);
    return 1;
  }

  return ok ? 0 : 1;
}
