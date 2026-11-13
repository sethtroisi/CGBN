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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

/************************************************************************************************
 *  Regression test for Montgomery multiplication (core_mont_wmad / imad / xmad):
 *
 *  1. Exercises cgbn_mont_mul and cgbn_mont_sqr across singleton (LIMBS = 1), even (LIMBS = 2, 4),
 *     odd (LIMBS = 3, 5), and padded (BITS < TPI * LIMBS * 32) configurations across TPI = 4, 8, 16,
 *     covering every compile-time branch of chain6 / chain7 in core_mont_wmad.cu.
 *  2. Tests deterministic edge-case vectors and random full-width instances across all in-place
 *     operand aliasing combinations:
 *       - out-of-place : cgbn_mont_mul(env, r, a, b, n, np0)
 *       - r == a       : cgbn_mont_mul(env, a, a, b, n, np0)
 *       - r == b       : cgbn_mont_mul(env, b, a, b, n, np0)
 *       - r == n       : cgbn_mont_mul(env, n, a, b, n, np0)   <-- fails upstream for LIMBS >= 2
 *       - mont_sqr r==n: cgbn_mont_sqr(env, n, a, n, np0)      <-- fails upstream for LIMBS >= 2
 *
 *  Why r == n fails in upstream core_mont_{wmad,imad,xmad}.cu (LIMBS >= 2):
 *    In core_t<env>::mont_mul, the epilogue writes the unreduced Montgomery sum (T mod 2^BITS)
 *    directly into r[0..LIMBS-1] and propagates carries into r BEFORE reading n[0..LIMBS-1] for
 *    the conditional subtraction of n (needed whenever T >= 2^BITS). When r aliases n, writing r
 *    clobbers the modulus n in registers prior to the conditional subtraction:
 *      - Deterministic vector 0 (N = 2^BITS - 1, A = N - 1, B = N - 2^PAD_BITS):
 *          Unreduced sum has top carry = 1 and low BITS limbs = 0 (expected result = 1).
 *          With r == n, n is overwritten with 0; t = n[0] - 1 underflows to 0xFFFFFFFF (~t = 0)
 *          on lane 0 word 0 while ~n[i] = 0xFFFFFFFF on all higher words, corrupting the output
 *          to 0xFFFFFFFF...FFFFFFFF00000000 (> N).
 *      - Deterministic vector 1 (N = 2^BITS - 1, A = N - 1, B = N - 2^(PAD_BITS + 1)):
 *          Unreduced sum has top carry = 1 and low BITS limbs = 1 (expected result = 2).
 *          With r == n, n is overwritten with 1, so the subtraction computes 1 - 1 = 0.
 ************************************************************************************************/

#ifndef RANDOM_INSTANCES
#define RANDOM_INSTANCES 2048
#endif

template<uint32_t tpi, uint32_t bits>
struct test_params_t {
  static const uint32_t TPB = 0;
  static const uint32_t MAX_ROTATION = 4;
  static const uint32_t SHM_LIMIT = 0;
  static const bool     CONSTANT_TIME = false;

  static const uint32_t TPI = tpi;
  static const uint32_t BITS = bits;
  static const uint32_t WORDS = bits / 32;
  static const uint32_t LIMBS = (WORDS + tpi - 1) / tpi;
  static const uint32_t UNPADDED_BITS = tpi * LIMBS * 32;
};

template<class params>
struct alias_instance_t {
  cgbn_mem_t<params::BITS> a;
  cgbn_mem_t<params::BITS> b;
  cgbn_mem_t<params::BITS> n;

  cgbn_mem_t<params::BITS> r_out;       // distinct r
  cgbn_mem_t<params::BITS> r_alias_a;   // r aliases a
  cgbn_mem_t<params::BITS> r_alias_b;   // r aliases b
  cgbn_mem_t<params::BITS> r_alias_n;   // r aliases n (mont_mul)
  cgbn_mem_t<params::BITS> sqr_out;     // distinct r (mont_sqr)
  cgbn_mem_t<params::BITS> sqr_alias_n; // r aliases n (mont_sqr)
};

template<class params>
__global__ void kernel_test_mont_aliasing(
    cgbn_error_report_t *report,
    alias_instance_t<params> *instances,
    uint32_t count) {
  int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (instance >= (int32_t)count)
    return;

  typedef cgbn_context_t<params::TPI, params> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t              bn_t;

  context_t context(cgbn_report_monitor, report, (uint32_t)instance);
  env_t     env(context);

  bn_t a, b, n, r, tmp;
  cgbn_load(env, a, &(instances[instance].a));
  cgbn_load(env, b, &(instances[instance].b));
  cgbn_load(env, n, &(instances[instance].n));

  uint32_t np0 = -cgbn_binary_inverse_ui32(env, cgbn_get_ui32(env, n));

  // 1. Out-of-place mont_mul: distinct destination r
  cgbn_mont_mul(env, r, a, b, n, np0);
  cgbn_store(env, &(instances[instance].r_out), r);

  // 2. In-place r == a
  cgbn_set(env, tmp, a);
  cgbn_mont_mul(env, tmp, tmp, b, n, np0);
  cgbn_store(env, &(instances[instance].r_alias_a), tmp);

  // 3. In-place r == b
  cgbn_set(env, tmp, b);
  cgbn_mont_mul(env, tmp, a, tmp, n, np0);
  cgbn_store(env, &(instances[instance].r_alias_b), tmp);

  // 4. In-place r == n (triggers aliasing bug when unreduced Montgomery sum >= 2^BITS)
  cgbn_set(env, tmp, n);
  cgbn_mont_mul(env, tmp, a, b, tmp, np0);
  cgbn_store(env, &(instances[instance].r_alias_n), tmp);

  // 5. Out-of-place vs in-place r == n for mont_sqr
  cgbn_mont_sqr(env, r, a, n, np0);
  cgbn_store(env, &(instances[instance].sqr_out), r);

  cgbn_set(env, tmp, n);
  cgbn_mont_sqr(env, tmp, a, tmp, np0);
  cgbn_store(env, &(instances[instance].sqr_alias_n), tmp);
}

static uint32_t host_np0(uint32_t n0) {
  // Newton-Raphson inverse mod 2^32: inv * n0 == 1 (mod 2^32), returns -inv mod 2^32
  uint32_t inv = n0;
  for (int i = 0; i < 4; i++)
    inv *= 2u - n0 * inv;
  return -inv;
}

static void cpu_cgbn_mont_mul_ref(
    uint32_t *out_words,
    const uint32_t *a_words,
    const uint32_t *b_words,
    const uint32_t *n_words,
    uint32_t words,
    uint32_t bits,
    uint32_t unpadded_bits) {
  mpz_t a, b, n, prod, add;
  mpz_inits(a, b, n, prod, add, NULL);

  to_mpz(a, const_cast<uint32_t *>(a_words), words);
  to_mpz(b, const_cast<uint32_t *>(b_words), words);
  to_mpz(n, const_cast<uint32_t *>(n_words), words);

  uint32_t np0 = host_np0(n_words[0]);
  uint32_t steps = unpadded_bits / 32;

  mpz_mul(prod, a, b);
  for (uint32_t step = 0; step < steps; step++) {
    uint32_t q = np0 * (uint32_t)mpz_get_ui(prod);
    mpz_mul_ui(add, n, q);
    mpz_add(prod, prod, add);
    mpz_fdiv_q_2exp(prod, prod, 32);
  }
  if (mpz_tstbit(prod, bits) == 1)
    mpz_sub(prod, prod, n);
  mpz_fdiv_r_2exp(prod, prod, bits);

  from_mpz(prod, out_words, words);
  mpz_clears(a, b, n, prod, add, NULL);
}

static void format_hex_summary(const uint32_t *w, uint32_t words, char *buf, size_t buf_len) {
  if (words <= 4) {
    size_t pos = snprintf(buf, buf_len, "0x");
    for (int i = (int)words - 1; i >= 0 && pos < buf_len; i--)
      pos += snprintf(buf + pos, buf_len - pos, "%08X", w[i]);
  } else {
    snprintf(buf, buf_len, "0x%08X_%08X...%08X_%08X",
             w[words - 1], w[words - 2], w[1], w[0]);
  }
}

template<class params>
static bool run_suite(const char *label, gmp_randstate_t rng) {
  const uint32_t words = params::WORDS;
  const uint32_t pad_bits = params::UNPADDED_BITS - params::BITS;
  const uint32_t total = 2 + RANDOM_INSTANCES;

  alias_instance_t<params> *h_inst =
      (alias_instance_t<params> *)calloc(total, sizeof(alias_instance_t<params>));
  if (h_inst == NULL) {
    fprintf(stderr, "Allocation failed\n");
    exit(1);
  }

  mpz_t a, b, n, pad_pow;
  mpz_inits(a, b, n, pad_pow, NULL);

  // Deterministic vector 0 (Case 1A):
  //   N = 2^BITS - 1, A = N - 1, B = N - 2^pad_bits.
  //   Produces unreduced Montgomery sum T = 2^BITS (carry = 1, low BITS = 0) => expected = 1.
  mpz_ui_pow_ui(n, 2, params::BITS);
  mpz_sub_ui(n, n, 1);
  mpz_sub_ui(a, n, 1);
  mpz_ui_pow_ui(pad_pow, 2, pad_bits);
  mpz_sub(b, n, pad_pow);
  from_mpz(a, h_inst[0].a._limbs, words);
  from_mpz(b, h_inst[0].b._limbs, words);
  from_mpz(n, h_inst[0].n._limbs, words);

  // Deterministic vector 1 (Case 1B):
  //   N = 2^BITS - 1, A = N - 1, B = N - 2^(pad_bits + 1).
  //   Produces unreduced Montgomery sum T = 2^BITS + 1 (carry = 1, low BITS = 1) => expected = 2.
  mpz_mul_2exp(pad_pow, pad_pow, 1);
  mpz_sub(b, n, pad_pow);
  from_mpz(a, h_inst[1].a._limbs, words);
  from_mpz(b, h_inst[1].b._limbs, words);
  from_mpz(n, h_inst[1].n._limbs, words);

  // Instances 2 .. total-1: Random full-width odd moduli N (MSB set so N > 2^(BITS-1)) and A, B < N
  for (uint32_t idx = 2; idx < total; idx++) {
    mpz_urandomb(n, rng, params::BITS);
    mpz_setbit(n, params::BITS - 1);
    mpz_setbit(n, 0);

    mpz_urandomm(a, rng, n);
    mpz_urandomm(b, rng, n);

    from_mpz(a, h_inst[idx].a._limbs, words);
    from_mpz(b, h_inst[idx].b._limbs, words);
    from_mpz(n, h_inst[idx].n._limbs, words);
  }
  mpz_clears(a, b, n, pad_pow, NULL);

  alias_instance_t<params> *d_inst = NULL;
  cgbn_error_report_t      *report = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_inst, sizeof(alias_instance_t<params>) * total));
  CUDA_CHECK(cudaMemcpy(d_inst, h_inst, sizeof(alias_instance_t<params>) * total, cudaMemcpyHostToDevice));
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  const uint32_t tpb = 128;
  const uint32_t ipb = tpb / params::TPI;
  kernel_test_mont_aliasing<params><<<(total + ipb - 1) / ipb, tpb>>>(report, d_inst, total);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  CUDA_CHECK(cudaMemcpy(h_inst, d_inst, sizeof(alias_instance_t<params>) * total, cudaMemcpyDeviceToHost));

  uint32_t expected_mul[params::WORDS];
  uint32_t expected_sqr[params::WORDS];
  uint32_t fail_out = 0, fail_a = 0, fail_b = 0, fail_n = 0, fail_sqr_out = 0, fail_sqr_n = 0;

  char exp_hex[64], got_hex[64];
  bool printed_sample_failure = false;

  for (uint32_t idx = 0; idx < total; idx++) {
    cpu_cgbn_mont_mul_ref(
        expected_mul,
        h_inst[idx].a._limbs,
        h_inst[idx].b._limbs,
        h_inst[idx].n._limbs,
        words,
        params::BITS,
        params::UNPADDED_BITS);
    cpu_cgbn_mont_mul_ref(
        expected_sqr,
        h_inst[idx].a._limbs,
        h_inst[idx].a._limbs,
        h_inst[idx].n._limbs,
        words,
        params::BITS,
        params::UNPADDED_BITS);

    if (compare_words(expected_mul, h_inst[idx].r_out._limbs, words) != 0)
      fail_out++;
    if (compare_words(expected_mul, h_inst[idx].r_alias_a._limbs, words) != 0)
      fail_a++;
    if (compare_words(expected_mul, h_inst[idx].r_alias_b._limbs, words) != 0)
      fail_b++;
    if (compare_words(expected_mul, h_inst[idx].r_alias_n._limbs, words) != 0) {
      fail_n++;
      if (idx < 2 || !printed_sample_failure) {
        format_hex_summary(expected_mul, words, exp_hex, sizeof(exp_hex));
        format_hex_summary(h_inst[idx].r_alias_n._limbs, words, got_hex, sizeof(got_hex));
        printf("    [FAIL %s] instance %u (%s) mont_mul(r==n):\n"
               "      expected: %s\n"
               "      got     : %s\n",
               label,
               idx,
               (idx == 0) ? "deterministic T == 2^BITS"
                          : (idx == 1) ? "deterministic T == 2^BITS + 1" : "random",
               exp_hex,
               got_hex);
        if (idx >= 2)
          printed_sample_failure = true;
      }
    }
    if (compare_words(expected_sqr, h_inst[idx].sqr_out._limbs, words) != 0)
      fail_sqr_out++;
    if (compare_words(expected_sqr, h_inst[idx].sqr_alias_n._limbs, words) != 0)
      fail_sqr_n++;
  }

  bool ok = (fail_out == 0 && fail_a == 0 && fail_b == 0 &&
             fail_n == 0 && fail_sqr_out == 0 && fail_sqr_n == 0);

  printf("  %-35s | out:%4u/%u  r==a:%4u/%u  r==b:%4u/%u  r==n:%4u/%u  sqr(r==n):%4u/%u  [%s]\n",
         label,
         fail_out, total,
         fail_a, total,
         fail_b, total,
         fail_n, total,
         fail_sqr_n, total,
         ok ? "PASS" : "FAIL");

  free(h_inst);
  CUDA_CHECK(cudaFree(d_inst));
  CUDA_CHECK(cgbn_error_report_free(report));
  return ok;
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  gmp_randstate_t rng;
  gmp_randinit_default(rng);
  gmp_randseed_ui(rng, 42);

  printf("Running CGBN Montgomery multiplication & in-place aliasing tests (%d random + 2 deterministic/suite)...\n",
         RANDOM_INSTANCES);
  printf("  %-35s | Failure counts (failed / total)\n", "Configuration");
  printf("  ------------------------------------+-----------------------------------------------------------------------------\n");

  bool all_ok = true;
  all_ok &= run_suite<test_params_t<8, 256>>  ("TPI=8,  BITS=256  (LIMBS=1, single)", rng);
  all_ok &= run_suite<test_params_t<4, 512>>  ("TPI=4,  BITS=512  (LIMBS=4, even)",   rng);
  all_ok &= run_suite<test_params_t<8, 512>>  ("TPI=8,  BITS=512  (LIMBS=2, even)",   rng);
  all_ok &= run_suite<test_params_t<8, 768>>  ("TPI=8,  BITS=768  (LIMBS=3, odd)",    rng);
  all_ok &= run_suite<test_params_t<8, 1024>> ("TPI=8,  BITS=1024 (LIMBS=4, even)",   rng);
  all_ok &= run_suite<test_params_t<8, 1280>> ("TPI=8,  BITS=1280 (LIMBS=5, odd)",    rng);
  all_ok &= run_suite<test_params_t<16, 1024>>("TPI=16, BITS=1024 (LIMBS=2, even)",   rng);
  all_ok &= run_suite<test_params_t<8, 704>>  ("TPI=8,  BITS=704  (LIMBS=3, padded)", rng);

  gmp_randclear(rng);

  if (!all_ok) {
    printf("\nRESULT: FAILED (Montgomery multiplication / aliasing mismatch detected)\n");
    return 1;
  }
  printf("\nRESULT: ALL TESTS PASSED\n");
  return 0;
}
