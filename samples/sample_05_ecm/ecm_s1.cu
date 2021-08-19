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
#define cgbn_positive_overflow ((cgbn_error_t) 15)
#define cgbn_negative_overflow ((cgbn_error_t) 16)

#define PRINT_DEBUG 0

// Seems to adds very small overhead (1-10%)
#define VERIFY_NORMALIZED 1
// Adds even less overhead (<1%)
#define CHECK_ERROR 1

// Can dramatically change compile time
#if 1
    #define FORCE_INLINE __forceinline__
#else
    #define FORCE_INLINE
#endif

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

// define the instance structure
typedef struct {
    char* n = NULL;

    // Number of curves to run
    uint32_t curves = 0;

    uint64_t B1 = 0;
    // Number of bits in S (based on B1)
    uint32_t num_bits;
    // Bits (malloc'ed in generate_instance)
    char    *s_bits;

    // output file for stage1 residual
    FILE    *file = stdout;

    // Sigma of first curve
    uint64_t sigma = 10;

} metadata_t;

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
  } instance_t;


  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance; // which curve instance is this

  // Constructor
  __device__ FORCE_INLINE curve_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) :
      _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}


  // Verify 0 <= r < modulus
  __device__ FORCE_INLINE void assert_normalized(bn_t &r, const bn_t &modulus) {
    //if (VERIFY_NORMALIZED && _context.check_errors()) {
    if (VERIFY_NORMALIZED && CHECK_ERROR) {

        // Negative overflow
        if (cgbn_extract_bits_ui32(_env, r, params::BITS-1, 1)) {
            _context.report_error(cgbn_negative_overflow);
        }
        // Positive overflow
        if (cgbn_compare(_env, r, modulus) >= 0) {
            _context.report_error(cgbn_positive_overflow);
        }
    }
  }

  // Normalize after addition
  __device__ FORCE_INLINE void normalize_addition(bn_t &r, const bn_t &modulus) {
      if (cgbn_compare(_env, r, modulus) >= 0) {
          cgbn_sub(_env, r, r, modulus);
      }
  }

  // Normalize after subtraction (handled instead by checking carry)
  /*
  __device__ FORCE_INLINE void normalize_subtraction(bn_t &r, const bn_t &modulus) {
      if (cgbn_extract_bits_ui32(_env, r, params::BITS-1, 1)) {
          cgbn_add(_env, r, r, modulus);
      }
  }
  */

  /**
   * Calculate (r * m) / 2^32 mod modulus
   *
   * This removes a factor of 2^32 which is not present in m.
   * Otherwise m (really d) needs to be passed as a bigint not a uint32
   */
  __device__ FORCE_INLINE void special_mult_ui32(bn_t &r, uint32_t m, const bn_t &modulus, uint32_t np0) {
    //uint32_t thread_i = (blockIdx.x*blockDim.x + threadIdx.x)%params::TPI;
    bn_t temp;

    uint32_t carry_t1 = cgbn_mul_ui32(_env, r, r, m);
    uint32_t t1_0 = cgbn_extract_bits_ui32(_env, r, 0, 32);
    uint32_t q = t1_0 * np0;
    uint32_t carry_t2 = cgbn_mul_ui32(_env, temp, modulus, q);

    cgbn_shift_right(_env, r, r, 32);
    cgbn_shift_right(_env, temp, temp, 32);
    // Add back overflow carry
    cgbn_insert_bits_ui32(_env, r, r, params::BITS-32, 32, carry_t1);
    cgbn_insert_bits_ui32(_env, temp, temp, params::BITS-32, 32, carry_t2);

    // This needs to be measured at block containing top bit of modulus
    int32_t carry_q = cgbn_add(_env, r, r, temp);
    carry_q += cgbn_add_ui32(_env, r, r, t1_0 != 0); // add 1
    while (carry_q != 0) {
        carry_q -= cgbn_sub(_env, r, r, modulus);
    }

    // 0 <= r, temp < modulus => r + temp + 1 < 2*modulus
    if (cgbn_compare(_env, r, modulus) >= 0) {
        cgbn_sub(_env, r, r, modulus);
    }
  }


  __device__ FORCE_INLINE void double_add_v2(
          bn_t &q, bn_t &u,
          bn_t &w, bn_t &v,
          uint32_t d,
          uint32_t bit_number,
          const bn_t &modulus,
          const uint32_t np0) {
    // q = xA = aX
    // u = zA = aY
    // w = xB = bX
    // v = zB = bY

    // Doesn't seem to be a large cost to using many extra variables
    bn_t t, CB, DA, AA, BB, K, dK;

    cgbn_add(_env, t, v, w); // t = (bY + bX)
    normalize_addition(t, modulus);
    if (cgbn_sub(_env, v, v, w)) // v = (bY - bX)
        cgbn_add(_env, v, v, modulus);


    cgbn_add(_env, w, u, q); // w = (aY + aX)
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, u, u, q)) // u = (aY - aX)
        cgbn_add(_env, u, u, modulus);
    if (VERIFY_NORMALIZED) {
        assert_normalized(t, modulus);
        assert_normalized(v, modulus);
        assert_normalized(w, modulus);
        assert_normalized(u, modulus);
    }

    cgbn_mont_mul(_env, CB, t, u, modulus, np0); // C*B
        normalize_addition(CB, modulus); // TODO: https://github.com/NVlabs/CGBN/issues/15
    cgbn_mont_mul(_env, DA, v, w, modulus, np0); // D*A
        normalize_addition(DA, modulus); // TODO: https://github.com/NVlabs/CGBN/issues/15

    cgbn_mont_sqr(_env, AA, w, modulus, np0);    // AA
    cgbn_mont_sqr(_env, BB, u, modulus, np0);    // BB
    normalize_addition(AA, modulus); // TODO: https://github.com/NVlabs/CGBN/issues/15
    normalize_addition(BB, modulus); // TODO: https://github.com/NVlabs/CGBN/issues/15
    if (VERIFY_NORMALIZED) {
        assert_normalized(CB, modulus);
        assert_normalized(DA, modulus);
        assert_normalized(AA, modulus);
        assert_normalized(BB, modulus);
    }

    // q = aX is finalized
    cgbn_mont_mul(_env, q, AA, BB, modulus, np0); // AA*BB
        normalize_addition(q, modulus); // TODO: https://github.com/NVlabs/CGBN/issues/15
        assert_normalized(q, modulus);

    if (cgbn_sub(_env, K, AA, BB)) // K = AA-BB
        cgbn_add(_env, K, K, modulus);

    // By definition of d = (sigma / 2^32) % MODN
    // K = k*R
    // dK = d*k*R = (K * R * sigma) >> 32
    cgbn_set(_env, dK, K);
    special_mult_ui32(dK, d, modulus, np0); // dK = K*d
        assert_normalized(dK, modulus);

    cgbn_add(_env, u, BB, dK); // BB + dK
    normalize_addition(u, modulus);
    if (VERIFY_NORMALIZED) {
        assert_normalized(K, modulus);
        assert_normalized(dK, modulus);
        assert_normalized(u, modulus);
    }

    // u = aY is finalized
    cgbn_mont_mul(_env, u, K, u, modulus, np0); // K(BB+dK)
        normalize_addition(u, modulus); // TODO: https://github.com/NVlabs/CGBN/issues/15
        assert_normalized(u, modulus);

    cgbn_add(_env, w, DA, CB); // DA + CB
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, v, DA, CB)) // DA - CB
        cgbn_add(_env, v, v, modulus);
    if (VERIFY_NORMALIZED) {
        assert_normalized(w, modulus);
        assert_normalized(v, modulus);
    }

    // w = bX is finalized
    cgbn_mont_sqr(_env, w, w, modulus, np0); // (DA+CB)^2 mod N
        normalize_addition(w, modulus); // TODO issue 15
        assert_normalized(w, modulus);

    cgbn_mont_sqr(_env, v, v, modulus, np0); // (DA-CB)^2 mod N
        normalize_addition(v, modulus); // TODO issue 15
        assert_normalized(v, modulus);

    // v = bY is finalized
    cgbn_shift_left(_env, v, v, 1); // double
    normalize_addition(v, modulus);
        assert_normalized(v, modulus);
  }

  __host__ static void compute_s_bits(mpz_t &s, int B1) {
      // Doesn't do even half of the smart things that compute_s does
      const int ACCUM_SIZE = 30;
      mpz_t prime, ppz, accum[ACCUM_SIZE];
      mpz_init(prime);
      mpz_init(ppz);
      for (int i = 0; i < ACCUM_SIZE; i++) {
          mpz_init_set_ui(accum[i], 1);
      }


      // Prime, prime power, max prime power;
      uint32_t p, pp, maxpp;
      // index
      int pi = 0;

      for (mpz_set_ui(prime, 2); (p = mpz_get_ui(prime)) <= B1; mpz_nextprime(prime, prime)) {
        maxpp = B1 / p;
        pp = p;
        while (pp <= maxpp) {
            pp *= p;
        }

        mpz_set_ui(ppz, pp);

        // Prefix product tree (TODO what is the name here)
        if ((pi & 1) == 0) {
            mpz_set(accum[0], ppz);
        } else {
            mpz_mul(accum[0], accum[0], ppz);
        }

        // printf("%d | %d | %d\n", pi, p, pp);

        int j = 0;
        while ((pi & (1 << j)) != 0) {
            if ((pi & (1 << j + 1)) == 0) {
                mpz_swap(accum[j+1], accum[j]);
            } else {
                mpz_mul(accum[j+1], accum[j+1], accum[j]);
            }
            mpz_set_ui(accum[j], 1);
            j++;
        }
        pi++;
      }

      // Multiply all accumulators
      mpz_set_ui(s, 1);
      for (int i = 0; i < ACCUM_SIZE; i++) {
        mpz_mul(s, s, accum[i]);
        mpz_clear(accum[i]);
      }
      mpz_clear(prime);
      mpz_clear(ppz);
  }

  __host__ static instance_t *generate_instances(metadata_t &run_data) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*run_data.curves);

    // P1_x, P1_y = (2,1)
    // 2P_x, 2P_y = (9, 64 * d + 8)

    mpz_t x, n;
    mpz_init(x);
    mpz_init(n);

    // B1 => s / s_bits
    assert( 2 <= run_data.B1 && run_data.B1 <= 11000000 );

    compute_s_bits(x, run_data.B1);
    uint32_t num_bits = mpz_sizeinbase(x, 2) - 1;
    //printf("B1=%lu S has %d bits", run_data.B1, num_bits);
    //if (num_bits < 200) {
    //    gmp_printf(": %Zd", x);
    //}
    //printf("\n");

    assert( num_bits <= 1442098 ); // B1 = 1e6, bits = 1.4e65
    run_data.num_bits = num_bits;
    // Use int* so that size can be stored in first element, could pass around extra size.
    run_data.s_bits = (char*) malloc(sizeof(char) * num_bits);

    for (int i = 0; i < num_bits; i++) {
        run_data.s_bits[i] = mpz_tstbit (x, num_bits - 1 - i);
        // print with verbose 3
        // if (PRINT_DEBUG)
        //    printf("S bit %d => %d\n", i, run_data.s_bits[i]);
    }

    // N
    mpz_set_str(n, run_data.n, 10);

    for(int index=0;index<run_data.curves;index++) {
        instance_t &instance = instances[index];

        // d = (sigma / 2^32) mod N BUT 2^32 handled by special_mul_ui32
        instance.d = run_data.sigma + index;

        // mod
        from_mpz(n, instance.modulus._limbs, params::BITS/32);

        // P1 (X, Y)
        mpz_set_ui(x, 2);
        from_mpz(x, instance.aX._limbs, params::BITS/32);
        mpz_set_ui(x, 1);
        from_mpz(x, instance.aY._limbs, params::BITS/32);

        // 2P = P2 (X, Y)
        // P2_y = 64 * d + 8
        mpz_set_ui(x, 9);
        from_mpz(x, instance.bX._limbs, params::BITS/32);

        // d = sigma * mod_inverse(2 ** 32, N)
        mpz_ui_pow_ui(x, 2, 32);
        mpz_invert(x, x, n);
        mpz_mul_ui(x, x, instance.d);
        // P2_y = 64 * d - 2;
        mpz_mul_ui(x, x, 64);
        mpz_add_ui(x, x, 8);
        mpz_mod(x, x, n);

        // if (PRINT_DEBUG)
        //    gmp_printf("%d => %Zd\n", instance.d, x);
        from_mpz(x, instance.bY._limbs, params::BITS/32);

    }

    mpz_clear(x);
    mpz_clear(n);

    return instances;
  }
};

// kernel implementation using cgbn
//
// Unfortunately, the kernel must be separate from the curve_t class

template<class params>
__global__ void kernel_double_add(
        cgbn_error_report_t *report,
        uint32_t num_bits,
        char* gpu_s_bits,
        typename curve_t<params>::instance_t *instances,
        uint32_t count) {
  // decode an instance_i number from the blockIdx and threadIdx
  int32_t instance_i=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  int32_t instance_j=(blockIdx.x*blockDim.x + threadIdx.x)%params::TPI;
  if(instance_i >= count)
    return;

  if (instance_j == -123) return;   // avoid unused warning

  cgbn_monitor_t monitor = CHECK_ERROR ? cgbn_report_monitor : cgbn_no_checks;

  curve_t<params> curve(monitor, report, instance_i);
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

  uint32_t np0;
  {
    // Convert everything to mont
    np0 = cgbn_bn2mont(curve._env, aX, aX, modulus);
    cgbn_bn2mont(curve._env, aY, aY, modulus);
    cgbn_bn2mont(curve._env, bX, bX, modulus);
    cgbn_bn2mont(curve._env, bY, bY, modulus);
    {
      curve.assert_normalized(aX, modulus);
      curve.assert_normalized(aY, modulus);
      curve.assert_normalized(bX, modulus);
      curve.assert_normalized(bY, modulus);
    }
  }

  // TODO Do the progressive queue thing.
  for (int b = 0; b < num_bits; b++) {
    if (PRINT_DEBUG && instance_j == 0) {
        printf("%d => %d\t|| (%u, %u),  (%u, %u)\n",
                b, gpu_s_bits[b],
                cgbn_get_ui32(curve._env, aX), cgbn_get_ui32(curve._env, aY),
                cgbn_get_ui32(curve._env, bX), cgbn_get_ui32(curve._env, bY));
    }
    if (gpu_s_bits[b] == 0) {
        curve.double_add_v2(aX, aY, bX, bY, d, b, modulus, np0);
    } else {
        curve.double_add_v2(bX, bY, aX, aY, d, b, modulus, np0);
    }
  }

  {
    // Convert everything back to bn
    cgbn_mont2bn(curve._env, aX, aX, modulus, np0);
    cgbn_mont2bn(curve._env, aY, aY, modulus, np0);
    cgbn_mont2bn(curve._env, bX, bX, modulus, np0);
    cgbn_mont2bn(curve._env, bY, bY, modulus, np0);
    {
      curve.assert_normalized(aX, modulus);
      curve.assert_normalized(aY, modulus);
      curve.assert_normalized(bX, modulus);
      curve.assert_normalized(bY, modulus);
    }
  }
  cgbn_store(curve._env, &(instance.aX), aX);
  cgbn_store(curve._env, &(instance.aY), aY);
  cgbn_store(curve._env, &(instance.bX), bX);
  cgbn_store(curve._env, &(instance.bY), bY);
}

template<class params>
void run_test(metadata_t &run_data) {
  typedef typename curve_t<params>::instance_t instance_t;

  char                *gpu_s_bits;
  instance_t          *instances, *gpu_instances;
  size_t               instance_size = sizeof(instance_t) * run_data.curves;
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI,
                       IPB=TPB/TPI;                                 // IPB is instances per block

  size_t gpu_block_count = (run_data.curves+IPB-1)/IPB;

  //printf("Generating instances ...\n");
  instances = curve_t<params>::generate_instances(run_data);
  assert(run_data.s_bits != NULL);

  //printf("Copying s_bits(%d) and instances(%d) to the GPU ...\n", run_data.num_bits, run_data.curves);
  CUDA_CHECK(cudaSetDevice(0));
  // Copy s_bits
  CUDA_CHECK(cudaMalloc((void **)&gpu_s_bits, sizeof(char) * run_data.num_bits));
  CUDA_CHECK(cudaMemcpy(gpu_s_bits, run_data.s_bits, sizeof(char) * (run_data.num_bits), cudaMemcpyHostToDevice));
  // Copy instances
  CUDA_CHECK(cudaMalloc((void **)&gpu_instances, instance_size));
  CUDA_CHECK(cudaMemcpy(gpu_instances, instances, instance_size, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel<%ld,%d> ...\n", gpu_block_count, TPB);
  auto start_t = std::chrono::high_resolution_clock::now();
  kernel_double_add<params><<<gpu_block_count, TPB>>>(
    report, run_data.num_bits, gpu_s_bits, gpu_instances, run_data.curves);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  if (report->_error) {
      printf("\n\nerror: %d\n", report->_error);
  }
  CGBN_CHECK(report);

  // Copy the instances back from gpuMemory
  //printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpu_instances, instance_size, cudaMemcpyDeviceToHost));

  auto end_t = std::chrono::high_resolution_clock::now();
  double diff = std::chrono::duration<float>(end_t - start_t).count();

  mpz_t x, y, n;
  mpz_init(x);
  mpz_init(y);
  mpz_init(n);

  // XXX: gmp-ecm returns results in reverse order
  for(int index=run_data.curves-1; index>=0; index--) {
    instance_t &instance = instances[index];

    if (PRINT_DEBUG && index == 0) {
        to_mpz(x, instance.aX._limbs, params::BITS/32);
        to_mpz(y, instance.aY._limbs, params::BITS/32);
        gmp_printf("pA: (%Zd, %Zd)\n", x, y);

        to_mpz(x, instance.bX._limbs, params::BITS/32);
        to_mpz(y, instance.bY._limbs, params::BITS/32);
        gmp_printf("pB: (%Zd, %Zd)\n", x, y);
    }

    to_mpz(n, instance.modulus._limbs, params::BITS/32);
    to_mpz(x, instance.aX._limbs, params::BITS/32);
    to_mpz(y, instance.aY._limbs, params::BITS/32);

    mpz_invert(y, y, n);    // aY ^ (N-2) % N

    to_mpz(x, instance.aX._limbs, params::BITS/32);
    mpz_mul(x, x, y);         // aX * aY^-1
    mpz_mod(x, x, n);

    gmp_fprintf(run_data.file, "METHOD=ECM; PARAM=3; SIGMA=%d; B1=%d; N=<OMITTED>; X=0x%Zx;\n", instance.d, run_data.B1, x);
  }
  mpz_clear(x);
  mpz_clear(y);
  mpz_clear(n);

  printf("Testing %d candidates (%d BITS) for %d double_adds took %.4f\n",
      run_data.curves, params::BITS, run_data.num_bits, diff);
  printf("Throughput: %.1f curves per second (on average %.2fms per Step 1)\n",
      run_data.curves / diff, 1000 * diff / run_data.curves);
  printf("\n");

  // clean up
  free(run_data.s_bits);
  free(instances);
  CUDA_CHECK(cudaFree(gpu_s_bits));
  CUDA_CHECK(cudaFree(gpu_instances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main(int argc, char** argv) {
  if (argc != 4) {
      printf("Usage: ecm_s1 SIGMA B1 N 2>results.txt\n");
      exit(1);
  }

  // TPI=8 is fastest, TPI=32 if only want to run a single curve
  typedef ecm_params_t<8, 1024 + 512, 5> params;

  metadata_t run_data;
  run_data.sigma = atol(argv[1]);
  run_data.B1 = atol(argv[2]);
  run_data.n = argv[3];
  run_data.file = stderr;

  //run_data.curves = 1;
  //run_test<params>(run_data);

  run_data.curves = 28*64;
  run_test<params>(run_data);

  /*
  // Try to find optimal curves / batch
  int tuning[] = {256, 16*63, 16*65, 16*100, 1790, 1792, 1794, 2000, 1780*2};
  for(int32_t curves : tuning) {
    run_data.curves = curves;
    run_test<params>(run_data);
  }
  // */
}
