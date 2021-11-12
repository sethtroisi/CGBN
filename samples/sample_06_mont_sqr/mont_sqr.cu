/***

Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

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

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 4
#define BITS (8*32)
#define INSTANCES 100000
#define OFFSET 0

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> res;
  cgbn_mem_t<BITS> res2;
} instance_t;

// support routine to generate random instances
instance_t *generate_instances(uint32_t count) {
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

  for(int index=0;index<OFFSET;index++) {
    random_words(instances[0].a._limbs, BITS/32);
    random_words(instances[0].a._limbs, BITS/32);
  }

  for(int index=0;index<count;index++) {
    // N
    random_words(instances[index].b._limbs, BITS/32);
    // Make sure b is odd
    instances[index].b._limbs[0] |= 1;
    // Make sure b is very large
    instances[index].b._limbs[BITS/32 - 1] |= 0x6DEAD000;

    // a
    random_words(instances[index].a._limbs, BITS/32);
    // Make sure a is smaller than N
    instances[index].a._limbs[BITS/32-1] &= instances[index].b._limbs[BITS/32-1];
    instances[index].a._limbs[BITS/32-2] &= instances[index].b._limbs[BITS/32-2];
    // TODO only do a even right now
    instances[index].a._limbs[0] &= 0xFFFFFFFD;
  }

  return instances;
}

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// the actual kernel
__global__ void kernel_test(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());
  env_t::cgbn_t  a, b, r;

  cgbn_load(bn_env, a, &(instances[instance].a));      // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b));      // load my instance's b value

  assert(cgbn_compare(bn_env, a, b) < 0);
  uint32_t np0 = cgbn_bn2mont(bn_env, a, a, b);
  //cgbn_store(bn_env, &(instances[instance].res), r);   // store r into res

  cgbn_mont_mul(bn_env, r, a, a, b, np0);
  cgbn_mont2bn(bn_env, r, r, b, np0);
  cgbn_store(bn_env, &(instances[instance].res), r);   // store squared number into res

  // reset r just in case
  cgbn_set_ui32(bn_env, r, 0);
  cgbn_mont_sqr(bn_env, r, a, b, np0);
  cgbn_mont2bn(bn_env, r, r, b, np0);
  cgbn_store(bn_env, &(instances[instance].res2), r);   // store mul number into  res2

  /*
  cgbn_mont_mul(bn_env, a, a, a, b, np0);
  cgbn_mont_mul(bn_env, a, a, a, b, np0);
  cgbn_mont_mul(bn_env, a, a, a, b, np0);
  cgbn_mont_mul(bn_env, r, a, a, b, np0);
  cgbn_mont2bn(bn_env, r, r, b, np0);
  cgbn_store(bn_env, &(instances[instance].res), r);   // store squared number into res
  */
}


__host__ static void verify_results(instance_t *instances, uint32_t instance_count) {
    mpz_t a, b, g, r, r2;
    mpz_init(a);
    mpz_init(b);
    mpz_init(g);
    mpz_init(r);
    mpz_init(r2);

    for(int index=0;index<instance_count;index++) {
      to_mpz(a, instances[index].a._limbs, BITS/32);
      to_mpz(b, instances[index].b._limbs, BITS/32);
      to_mpz(r, instances[index].res._limbs, BITS/32);
      to_mpz(r2, instances[index].res2._limbs, BITS/32);

      // mpz based mont_mul
      mpz_powm_ui(g, a, 2, b);

      if (mpz_cmp(g, r) != 0 || mpz_cmp(g, r2) != 0) {
          gmp_printf("\n%d\n", index);
          gmp_printf("N : %#Zx\n", b);
          gmp_printf("A : %#Zx\n", a);
          gmp_printf("G : %#Zx (gmp)\n", g);
          gmp_printf("R : %#Zx (mont_mul)\n", r);
          gmp_printf("R2: %#Zx (mont_sqr)\n", r2);
          assert(mpz_cmp(r, r2) == 0);
      }
    }
    printf("Verified %d mont_sqr(r, a, n) vs mont_mult(r, a, a, n)\n", instance_count);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(g);
    mpz_clear(r);
    mpz_clear(r2);
}

int main() {
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;

  instances=generate_instances(INSTANCES);

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*INSTANCES, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  // launch with 128 threads per block
  kernel_test<<<(INSTANCES+TPI-1)/TPI, 128>>>(report, gpuInstances, INSTANCES);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*INSTANCES, cudaMemcpyDeviceToHost));

  verify_results(instances, INSTANCES);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}
