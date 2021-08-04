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

#include <cassert>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "chrono"

#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

// For this example, there are quite a few template parameters that are used to generate the actual code.
// In order to simplify passing many parameters, we use the same approach as the CGBN library, which is to
// create a container class with static constants and then pass the class.

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
//   WINDOW_BITS     - number of bits to use for the windowed exponentiation

template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class powm_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
  static const uint32_t WINDOW_BITS=window_bits;   // window size
};

template<class params>
class curve_t {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable

  // define the instance structure
  typedef struct {
    cgbn_mem_t<params::BITS> aX;
    cgbn_mem_t<params::BITS> aY;
    cgbn_mem_t<params::BITS> bX;
    cgbn_mem_t<params::BITS> bY;
    cgbn_mem_t<params::BITS> modulus;
    uint32_t d;

    uint32_t num_bits;
    char s_bits[100]; // TODO dynamically copy this
  } instance_t;

  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance; // which curve instance is this

  // Constructor
  __device__ __forceinline__ curve_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) :
      _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}


  /**
   * Simultaneously compute
   * pA = [2](pA)
   * pB = pA + pB
   *
   * everything (including d) in montgomery form
   */
  __device__ __forceinline__ void double_add(
          bn_t &aX, bn_t &aY,
          bn_t &bX, bn_t &bY,
          uint32_t d,
          const bn_t &modulus) {
    /**
     * compute S!(P) using repeated double and add
     * https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_doubling
     */

    bn_t C, D, A, B, CB, DA, AA, BB, temp, K, dK, w, v;

    // find np0 correctly
    uint32_t np0 = cgbn_bn2mont(_env, temp, aX, modulus);
    //printf("Hi %d => %d\n", _instance, np0);

    cgbn_add(_env, C, bY, bX);
    cgbn_sub(_env, D, bY, bX);

    // TODO remove this with do normalize or do mod or something.
    cgbn_add(_env, D, D, modulus);

    cgbn_add(_env, A, aY, aX);
    cgbn_sub(_env, B, aY, aX);

    // TODO remove this with do normalize or do mod or something.
    cgbn_add(_env, B, B, modulus);

/*
    cgbn_set(_env, aX, C);
    cgbn_set(_env, aY, D);
    cgbn_set(_env, bX, A);
    cgbn_set(_env, bY, B);
*/

    cgbn_mont_mul(_env, CB, C, B, modulus, np0);
    cgbn_mont_mul(_env, DA, D, A, modulus, np0);

    cgbn_mont_sqr(_env, AA, A, modulus, np0);
    cgbn_mont_sqr(_env, BB, B, modulus, np0);

/*
    cgbn_set(_env, aX, A);
    cgbn_set(_env, aY, B);
    cgbn_set(_env, bX, AA);
    cgbn_set(_env, bY, BB);
*/

    // Overwrite aX with result
    cgbn_mont_mul(_env, aX, AA, BB, modulus, np0);
    cgbn_sub(_env, K, AA, BB);
    // TODO remove this with do normalize or do mod or something.
    cgbn_add(_env, K, K, modulus);

    cgbn_mul_ui32(_env, dK, K, d);
    cgbn_add(_env, temp, BB, dK);

    // Overwrite aY with result: K(BB + dK)
    cgbn_mont_mul(_env, aY, K, temp, modulus, np0);

    cgbn_add(_env, w, DA, CB);
    cgbn_sub(_env, v, DA, CB);
    // TODO remove this with do normalize or do mod or something.
    cgbn_add(_env, v, v, modulus);

    // Overwrite bX
    cgbn_mont_sqr(_env, bX, w, modulus, np0);

    // Overwrite bY
    cgbn_mont_sqr(_env, temp, v, modulus, np0);
    cgbn_add(_env, bY, temp, temp);
  }

  __host__ static instance_t *generate_instances(uint32_t count) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

    mpz_t x;
    mpz_init(x);

    for(int index=0;index<count;index++) {
        instance_t &instance = instances[index];

        mpz_set_ui(x, 8);
        from_mpz(x, instance.aX._limbs, params::BITS/32);
        mpz_set_ui(x, 2);
        from_mpz(x, instance.aY._limbs, params::BITS/32);

        mpz_set_ui(x, 18);
        from_mpz(x, instance.bX._limbs, params::BITS/32);
        mpz_set_ui(x, 2960);
        from_mpz(x, instance.bY._limbs, params::BITS/32);

        mpz_set_ui(x, 2147483647);
        from_mpz(x, instance.modulus._limbs, params::BITS/32);

        instance.d = 23; // d_z in colab


        mpz_set_ui(x, 2520);
        instance.num_bits = mpz_sizeinbase(x, 2) - 1;
        assert( instance.num_bits <= 100 );
        for (int i = 0; i < instance.num_bits; i++) {
            instance.s_bits[i] = mpz_tstbit (x, instance.num_bits - 1 - i);
            //printf("%d => %d\n", i, instance.s_bits[i]);
        }

        instance.num_bits = 144343;
    }

    mpz_clear(x);

    return instances;
  }
};

// kernel implementation using cgbn
//
// Unfortunately, the kernel must be separate from the curve_t class

template<class params>
__global__ void kernel_double_add(cgbn_error_report_t *report, typename curve_t<params>::instance_t *instances, uint32_t count) {
  int32_t instance_i;
  //int32_t instance_j;

  // decode an instance_i number from the blockIdx and threadIdx
  instance_i=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  //instance_j=(blockIdx.x*blockDim.x + threadIdx.x)%params::TPI;
  if(instance_i >= count)
    return;

  curve_t<params>                 curve(cgbn_report_monitor, report, instance_i);
  typename curve_t<params>::bn_t  aX, aY, bX, bY, modulus;

  typename curve_t<params>::instance_t &instance = instances[instance_i];

  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(curve._env, aX, &(instance.aX));
  cgbn_load(curve._env, aY, &(instance.aY));
  cgbn_load(curve._env, bX, &(instance.bX));
  cgbn_load(curve._env, bY, &(instance.bY));
  cgbn_load(curve._env, modulus, &(instance.modulus));

  uint32_t d = instance.d;

  // Do the progressive queue thing.
  for (int b = 0; b < instance.num_bits; b++) {
    //if (instance_j == 0) printf("%d => %d\n", b, instance.s_bits[b]);
    if (instance.s_bits[b & 7] == 0) {
        curve.double_add(aX, aY, bX, bY, d, modulus);
    } else {
        curve.double_add(bX, bY, aX, aY, d, modulus);
    }
  }

  cgbn_store(curve._env, &(instance.aX), aX);
  cgbn_store(curve._env, &(instance.aY), aY);
  cgbn_store(curve._env, &(instance.bX), bX);
  cgbn_store(curve._env, &(instance.bY), bY);
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef typename curve_t<params>::instance_t instance_t;

  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block

  size_t gpu_count = (instance_count+IPB-1)/IPB;

  //printf("Genereating instances ...\n");
  instances = curve_t<params>::generate_instances(instance_count);

  //printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel<%ld> ...\n", gpu_count);

  auto start_t = std::chrono::high_resolution_clock::now();

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_double_add<params><<<gpu_count, TPB>>>(report, gpuInstances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  //printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));

  auto end_t = std::chrono::high_resolution_clock::now();
  double diff = std::chrono::duration<float>(end_t - start_t).count();
  printf("Testing %d candidates (%d BITS) for %d double_adds took %.4f = %.0f curves/second\n",
      instance_count, params::BITS, instances[0].num_bits, diff,
      instance_count / diff);


  mpz_t x, y;
  mpz_init(x);
  mpz_init(y);
  for(int index=0; index<instance_count; index++) {
    instance_t &instance = instances[index];

    to_mpz(x, instance.aX._limbs, params::BITS/32);
    to_mpz(y, instance.aY._limbs, params::BITS/32);
    //gmp_printf("pA: (%Zd, %Zd)\n", x, y);
    to_mpz(x, instance.bX._limbs, params::BITS/32);
    to_mpz(y, instance.bY._limbs, params::BITS/32);
    //gmp_printf("pB: (%Zd, %Zd)\n", x, y);
  }
  mpz_clear(x);
  mpz_clear(y);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  typedef powm_params_t<8, 1024, 5> params;

  run_test<params>(256);

  run_test<params>(16 * 63);
  run_test<params>(16 * 65);
  run_test<params>(16 * 100);
  run_test<params>(16 * 110);
  run_test<params>(1790);
  run_test<params>(1792);
  run_test<params>(1794);
  run_test<params>(2000);
  run_test<params>(1780 * 2);
}
