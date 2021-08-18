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

// See cgbn_error_t enum (cgbn.h:39)
#define cgbn_normalized_error ((cgbn_error_t) 14)

template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class ecm_params_t {
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
    char s_bits[64]; // TODO dynamically copy this
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
  __device__ __forceinline__ void double_add_v1(
          bn_t &aX, bn_t &aY,
          bn_t &bX, bn_t &bY,
          uint32_t d,
          uint32_t bit,
          const bn_t &modulus) {
    /**
     * compute S!(P) using repeated double and add
     * https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_doubling
     */

    bn_t C, D, A, B, CB, DA, AA, BB, temp, K, dK, w, v;

    // find np0 correctly
    uint32_t np0 = cgbn_bn2mont(_env, temp, aX, modulus);
    //printf("Hi v1 %d,%d => %d\n", _instance, bit, np0);

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
    return
// */

    cgbn_mont_mul(_env, CB, C, B, modulus, np0);
    cgbn_mont_mul(_env, DA, D, A, modulus, np0);

    cgbn_mont_sqr(_env, AA, A, modulus, np0);
    cgbn_mont_sqr(_env, BB, B, modulus, np0);

/*
    cgbn_set(_env, aX, A);
    cgbn_set(_env, aY, B);
    cgbn_set(_env, bX, AA);
    cgbn_set(_env, bY, BB);
// */

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


  // Verify 0 <= r < modulus
  __device__ __forceinline__ void assert_normalized(bn_t &r, const bn_t &modulus) {
    if (_context.check_errors()) {

        // Negative overflow
        if (cgbn_extract_bits_ui32(_env, r, params::BITS-1, 1)) {
            _context.report_error(cgbn_normalized_error);
        }
        // Positive overflow
        if (cgbn_compare(_env, r, modulus) >= 0) {
            _context.report_error(cgbn_normalized_error);
        }
    }
  }

  // Normalize after addition
  __device__ __forceinline__ void normalize_addition(bn_t &r, const bn_t &modulus) {

      if (cgbn_compare(_env, r, modulus) >= 0) {
          cgbn_sub(_env, r, r, modulus);
      }
  }

  // Normalize after subtraction
  __device__ __forceinline__ void normalize_subtraction(bn_t &r, const bn_t &modulus) {

      if (cgbn_extract_bits_ui32(_env, r, params::BITS-1, 1)) {
          cgbn_add(_env, r, r, modulus);
      }
  }

  // Normalize after multiplication by uint32
  //__device__ __forceinline__ void normalize_mul_ui32(env_t env, bn_t &r, const bn_t &modulus, uint32_t np0) {
  //    NO IDEA HOW TO IMPLEMENT.
  //    POSSIBLE IDEAS:
  //        implement cgbn_mont_reduce
  //        implement cgbn_mont_mul_ui32
  //            I don't think this is any different than cgbn_mul_ui32 with mont_reduce
  //            maybe needs extra carry info or something
  //
  //        use barrett (or mont) rem somehow
  //}

  __device__ __forceinline__ void double_add_v2(
          bn_t &q, bn_t &u,
          bn_t &w, bn_t &v,
          uint32_t d,
          uint32_t bit,
          const bn_t &modulus) {

    uint32_t thread_i = (blockIdx.x*blockDim.x + threadIdx.x)%params::TPI;

    // q = xA = aX
    // u = zA = aY
    // w = xB = bX
    // v = zB = bY

    //cgbn_set_ui32(_env, q, 0);
    //cgbn_set_ui32(_env, u, 0);
    //cgbn_set_ui32(_env, w, 0);
    //cgbn_set_ui32(_env, v, 0);

    // t2 is only needed once (BB + dK), see if it can be optimized around
    bn_t t, t2;
    // find np0 correctly
    uint32_t np0 = cgbn_bn2mont(_env, t, q, modulus);
    if (thread_i == 0) {
        printf("\tv2 %d,%d | np0 %d\n", _instance, bit, np0);
        printf("\t\tin\t(%d, %d),  (%d, %d)\n",
                cgbn_get_ui32(_env, q), cgbn_get_ui32(_env, u),
                cgbn_get_ui32(_env, w), cgbn_get_ui32(_env, v));
    }

    // Convert everything to mont
    cgbn_bn2mont(_env, q, q, modulus);
    cgbn_bn2mont(_env, u, u, modulus);
    cgbn_bn2mont(_env, w, w, modulus);
    cgbn_bn2mont(_env, v, v, modulus);
    { // TODO: move behind a flag
        assert_normalized(q, modulus);
        assert_normalized(u, modulus);
        assert_normalized(w, modulus);
        assert_normalized(v, modulus);
        if (thread_i == 0)
            printf("\t\t0\t(%d, %d),  (%d, %d)\n",
                    cgbn_get_ui32(_env, q), cgbn_get_ui32(_env, u),
                    cgbn_get_ui32(_env, w), cgbn_get_ui32(_env, v));
    }

    cgbn_add(_env, t, v, w); // t = (bY + bX)
    normalize_addition(t, modulus);
    cgbn_sub(_env, v, v, w); // v = (bY - bX)
    normalize_subtraction(v, modulus);
    cgbn_add(_env, w, u, q); // w = (aY + aX)
    normalize_addition(w, modulus);
    cgbn_sub(_env, u, u, q); // u = (aY - aX)
    normalize_subtraction(u, modulus);
    {
        assert_normalized(t, modulus);
        assert_normalized(v, modulus);
        assert_normalized(w, modulus);
        assert_normalized(u, modulus);
        if (thread_i == 0)
            printf("\t\t1\t(%d, %d),  (%d, %d)\n",
                    cgbn_get_ui32(_env, t), cgbn_get_ui32(_env, v),
                    cgbn_get_ui32(_env, w), cgbn_get_ui32(_env, u));
    }

    cgbn_mont_mul(_env, t, t, u, modulus, np0); // C*B
    cgbn_mont_mul(_env, v, v, w, modulus, np0); // D*A
    // TODO check if using temporary is faster?
    cgbn_mont_sqr(_env, w, w, modulus, np0);    // AA
    cgbn_mont_sqr(_env, u, u, modulus, np0);    // BB
    {
        assert_normalized(t, modulus);
        assert_normalized(v, modulus);
        assert_normalized(w, modulus);
        assert_normalized(u, modulus);
        if (thread_i == 0)
            printf("\t\t2\t(%d, %d),  (%d, %d)\n",
                    cgbn_get_ui32(_env, t), cgbn_get_ui32(_env, v),
                    cgbn_get_ui32(_env, w), cgbn_get_ui32(_env, u));
    }

    // q = aX is finalized
    cgbn_mont_mul(_env, q, u, w, modulus, np0); // AA*BB
        assert_normalized(q, modulus);
    cgbn_mont2bn(_env, q, q, modulus, np0);
        assert_normalized(q, modulus);

    cgbn_sub(_env, w, w, u); // K = AA-BB
    normalize_subtraction(w, modulus);

    // TODO use cgbn_mul_ui32 then normalize_mul_ui32
    //uint32_t carry = cgbn_mul_ui32(_env, t2, w, d); // dK
    //normalize_mul_ui32(_env, t2, modulus, np0);
    cgbn_set_ui32(_env, t2, d);  // d_z
    cgbn_bn2mont(_env, t2, t2, modulus); // TODO: pass d in montgomery form
    cgbn_mont_mul(_env, t2, w, t2, modulus, np0);  // dK
        assert_normalized(t2, modulus);


    cgbn_add(_env, u, u, t2); // BB + dK
    normalize_addition(u, modulus);
    {
        assert_normalized(w, modulus);
        assert_normalized(t2, modulus);
        assert_normalized(u, modulus);
        if (thread_i == 0)
            printf("\t\t3\tdecimal %d, d = %d | K = %d,  dK = %d,  BB + dk = %d\n",
                    cgbn_get_ui32(_env, q),
                    d,
                    cgbn_get_ui32(_env, w),
                    cgbn_get_ui32(_env, t2),
                    cgbn_get_ui32(_env, u));
    }

    // u = aY is finalized
    cgbn_mont_mul(_env, u, w, u, modulus, np0); // K(BB+dK)
        assert_normalized(u, modulus);
    cgbn_mont2bn(_env, u, u, modulus, np0);
        assert_normalized(u, modulus);

    cgbn_add(_env, w, v, t); // DA + CB
    normalize_addition(w, modulus);
    cgbn_sub(_env, v, v, t); // DA - CB
    normalize_subtraction(v, modulus);
    {
        assert_normalized(w, modulus);
        assert_normalized(v, modulus);
        if (thread_i == 0)
            printf("\t\t4\tdecimal %d | %d, %d\n",
                    cgbn_get_ui32(_env, u),
                    cgbn_get_ui32(_env, w),
                    cgbn_get_ui32(_env, v));
    }

    // w = bX is finalized
    cgbn_mont_sqr(_env, w, w, modulus, np0); // (DA+CB)^2 mod N
        assert_normalized(w, modulus);
    cgbn_mont2bn(_env, w, w, modulus, np0);
        assert_normalized(w, modulus);

    cgbn_mont_sqr(_env, v, v, modulus, np0); // (DA-CB)^2 mod N
        assert_normalized(v, modulus);

    // v = bY is finalized
    cgbn_add(_env, v, v, v); // double
    normalize_addition(v, modulus);
        assert_normalized(v, modulus);
    cgbn_mont2bn(_env, v, v, modulus, np0);
        assert_normalized(v, modulus);

    if (thread_i == 0)
        printf("\t\t5\tdecimal %d %d\n",
                cgbn_get_ui32(_env, w),
                cgbn_get_ui32(_env, v));
  }

  __host__ static instance_t *generate_instances(uint32_t count) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

    mpz_t x;
    mpz_init(x);

    for(int index=0;index<count;index++) {
        instance_t &instance = instances[index];

        // XXX: calc d_z from sigma
        // XXX: 2P_y depends on d which depends on bits!

        // N, P1_x, P1_y, 2P_x, 2P_y, "d_z", s
        char data[][100] = {
            // "2147483647", "2", "1", "9", "392", "6", "2"
            "2147483647", "2", "1", "9", "392", "6", "2520"
        };

        // N
        mpz_set_str(x, data[0], 10);
        from_mpz(x, instance.modulus._limbs, params::BITS/32);

        // P1 (X, Y)
        mpz_set_str(x, data[1], 10);
        from_mpz(x, instance.aX._limbs, params::BITS/32);
        mpz_set_str(x, data[2], 10);
        from_mpz(x, instance.aY._limbs, params::BITS/32);

        // 2P = P2 (X, Y)
        mpz_set_str(x, data[3], 10);
        from_mpz(x, instance.bX._limbs, params::BITS/32);
        mpz_set_str(x, data[4], 10);
        from_mpz(x, instance.bY._limbs, params::BITS/32);

        // d_z (not montgomery) (in colab) | d = (sigma / 2^32) mod N
        instance.d = atol(data[5]);

        // s
        mpz_set_str(x, data[6], 10);
        instance.num_bits = mpz_sizeinbase(x, 2) - 1;
        assert( instance.num_bits <= 100 );
        for (int i = 0; i < instance.num_bits; i++) {
            instance.s_bits[i] = mpz_tstbit (x, instance.num_bits - 1 - i);
            if (index == 0) {
             //   printf("%d => %d\n", i, instance.s_bits[i]);
            }
        }

        //instance.num_bits = 1;

        //instance.num_bits = 14434;
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
  // decode an instance_i number from the blockIdx and threadIdx
  int32_t instance_i=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  int32_t instance_j=(blockIdx.x*blockDim.x + threadIdx.x)%params::TPI;
  if(instance_i >= count)
    return;

  if (instance_j == -123) return;   // avoid unused warning

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

  /**
   * compute S!(P) using repeated double and add
   * https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_doubling
   */

  // TODO Do the progressive queue thing.
  for (int b = 0; b < instance.num_bits; b++) {
    //if (instance_j == 0) printf("%d => %d\n", b, instance.s_bits[b]);
    if (instance_j == 0) {
        printf("%d => %d\t|| (%d, %d),  (%d, %d)\n",
                b, instance.s_bits[b],
                cgbn_get_ui32(curve._env, aX), cgbn_get_ui32(curve._env, aY),
                cgbn_get_ui32(curve._env, bX), cgbn_get_ui32(curve._env, bY));
    }


    if (instance.s_bits[b & 63] == 0) { // TODO cleanup in real code
        curve.double_add_v2(aX, aY, bX, bY, d, b, modulus);
    } else {
        curve.double_add_v2(bX, bY, aX, aY, d, b, modulus);
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


  mpz_t x, y, n;
  mpz_init(x);
  mpz_init(y);
  mpz_init(n);
  for(int index=0; index<instance_count; index++) {
    if (index >= 1) break;
    instance_t &instance = instances[index];

    to_mpz(x, instance.aX._limbs, params::BITS/32);
    to_mpz(y, instance.aY._limbs, params::BITS/32);
    gmp_printf("pA: (%Zd, %Zd)\n", x, y);
    to_mpz(x, instance.bX._limbs, params::BITS/32);
    to_mpz(y, instance.bY._limbs, params::BITS/32);
    gmp_printf("pB: (%Zd, %Zd)\n", x, y);

    to_mpz(n, instance.modulus._limbs, params::BITS/32);
    to_mpz(x, instance.aX._limbs, params::BITS/32);
    to_mpz(y, instance.aY._limbs, params::BITS/32);

    mpz_invert(y, y, n);    // aY ^ (N-2) % N

    to_mpz(x, instance.aX._limbs, params::BITS/32);
    mpz_mul(x, x, y);         // aX * aY^-1
    mpz_mod(x, x, n);

    gmp_printf("X= %Zd\n", x);
  }
  mpz_clear(x);
  mpz_clear(y);
  mpz_clear(n);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  typedef ecm_params_t<8, 1024, 5> params;

  run_test<params>(1);
  /*
  // Warm up
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
  // */
}
