/***
 * Adversarial edge-case test for cgbn_mont_sqr.
 *
 * For every supported (TPI, BITS) configuration this builds a cross product of special
 * operands a (0, 1, 2^k, 2^k-1, all-ones, half-split boundary values, carry-ripple
 * patterns, a >= n, ...) and special odd moduli n (1, 3, 2^BITS-1, sparse, ...), plus
 * structured random values, and checks on the CPU with GMP that
 *
 *    cgbn_mont_sqr(r, a, n, np0)            (out of place)
 *    cgbn_mont_sqr(a, a, n, np0)            (in place)
 *    cgbn_mont_mul(r, a, a, n, np0)
 *
 * all equal the exact CGBN "almost Montgomery" result
 *    U = (a^2 + Q n) / R,  Q = -a^2 n^-1 mod R,  R = 2^(32*TPI*LIMBS),   r = (U >= 2^BITS) ? U - n : U.
 *
 * The instance count is chosen so that the final warp is only partially populated, which
 * exercises the non-full-warp (dynamic activemask) code paths.
 *
 *    ./mont_sqr_edge [-r random_instances] [-s seed] [-c TPIxBITS ...]
 ***/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"

template<uint32_t tpi, uint32_t bits>
class edge_params_t {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool     CONSTANT_TIME=false;
  static const uint32_t TPI=tpi;
  static const uint32_t BITS=bits;
};

template<class params>
struct edge_instance_t {
  cgbn_mem_t<params::BITS> a;
  cgbn_mem_t<params::BITS> n;
  cgbn_mem_t<params::BITS> r_sqr;
  cgbn_mem_t<params::BITS> r_alias;
  cgbn_mem_t<params::BITS> r_mul;
  cgbn_mem_t<params::BITS> a_red;      // a mod n (host)
  cgbn_mem_t<params::BITS> r_b2m;      // cgbn_bn2mont(a_red)
  cgbn_mem_t<params::BITS> r_back;     // cgbn_mont2bn(r_b2m)
};

template<class params>
__global__ void kernel_edge(cgbn_error_report_t *report, edge_instance_t<params> *instances, uint32_t count) {
  typedef cgbn_context_t<params::TPI, params>    context_t;
  typedef cgbn_env_t<context_t, params::BITS>    env_t;

  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  context_t      ctx(cgbn_report_monitor, report, instance);
  env_t          env(ctx);
  typename env_t::cgbn_t a, n, r1, r2;

  cgbn_load(env, a, &(instances[instance].a));
  cgbn_load(env, n, &(instances[instance].n));
  uint32_t np0=-cgbn_binary_inverse_ui32(env, cgbn_get_ui32(env, n));

  cgbn_mont_sqr(env, r1, a, n, np0);
  cgbn_mont_mul(env, r2, a, a, n, np0);
  cgbn_mont_sqr(env, a, a, n, np0);

  cgbn_store(env, &(instances[instance].r_sqr), r1);
  cgbn_store(env, &(instances[instance].r_mul), r2);
  cgbn_store(env, &(instances[instance].r_alias), a);

  // bn2mont / mont2bn round trip (division-free ladder when LIMBS > TPI)
  cgbn_load(env, a, &(instances[instance].a_red));
  uint32_t np0_b=cgbn_bn2mont(env, r1, a, n);
  cgbn_store(env, &(instances[instance].r_b2m), r1);
  cgbn_mont2bn(env, r2, r1, n, np0_b);
  cgbn_store(env, &(instances[instance].r_back), r2);
}

static void to_mpz(mpz_t r, const uint32_t *x, uint32_t words) {
  mpz_import(r, words, -1, sizeof(uint32_t), 0, 0, x);
}

static void from_mpz(const mpz_t s, uint32_t *x, uint32_t words) {
  size_t count=0;
  memset(x, 0, sizeof(uint32_t)*words);
  if(mpz_sizeinbase(s, 2)>words*32) {
    fprintf(stderr, "from_mpz: value too large\n");
    exit(1);
  }
  mpz_export(x, &count, -1, sizeof(uint32_t), 0, 0, s);
}

struct mpz_list_t {
  std::vector<mpz_t *> v;
  void add(const mpz_t x) {
    mpz_t *p=(mpz_t *)malloc(sizeof(mpz_t));
    mpz_init_set(*p, x);
    v.push_back(p);
  }
  ~mpz_list_t() {
    for(size_t i=0;i<v.size();i++) { mpz_clear(*v[i]); free(v[i]); }
  }
};

// words filled from a repeating pattern
static void pattern(mpz_t r, uint32_t words, const uint32_t *pat, uint32_t plen) {
  std::vector<uint32_t> x(words);
  for(uint32_t i=0;i<words;i++)
    x[i]=pat[i%plen];
  to_mpz(r, x.data(), words);
}

// random value where each word is random / 0 / 0xFFFFFFFF / single-bit with some probability
static void structured_random(mpz_t r, gmp_randstate_t state, uint32_t words) {
  std::vector<uint32_t> x(words);
  uint32_t mode=gmp_urandomm_ui(state, 4);
  for(uint32_t i=0;i<words;i++) {
    uint32_t w=(uint32_t)gmp_urandomb_ui(state, 32);
    uint32_t sel=gmp_urandomm_ui(state, 8);
    if(mode==1) {
      if(sel<3) w=0xFFFFFFFFu; else if(sel<5) w=0u;
    }
    else if(mode==2) {
      if(sel<6) w=0xFFFFFFFFu;
    }
    else if(mode==3) {
      if(sel<6) w=0u; else if(sel==6) w=1u<<(w&31);
    }
    x[i]=w;
  }
  to_mpz(r, x.data(), words);
}

template<class params>
static bool run_config(uint32_t random_count, unsigned long seed) {
  typedef edge_instance_t<params> instance_t;
  const uint32_t BITS=params::BITS, TPI=params::TPI, WORDS=BITS/32;
  const uint32_t LIMBS=(WORDS+TPI-1)/TPI, RBITS=32*TPI*LIMBS;   // Montgomery radix is 2^RBITS (> 2^BITS when padded)
  mpz_list_t      as, ns;
  mpz_t           t, R, RM, one;
  gmp_randstate_t state;

  mpz_init(t); mpz_init(R); mpz_init_set_ui(one, 1);
  mpz_mul_2exp(R, one, BITS);
  mpz_init(RM);
  mpz_mul_2exp(RM, one, RBITS);
  gmp_randinit_default(state);
  gmp_randseed_ui(state, seed);

  // ---- special a values ----
  mpz_set_ui(t, 0); as.add(t);
  mpz_set_ui(t, 1); as.add(t);
  mpz_set_ui(t, 2); as.add(t);
  mpz_set_ui(t, 0xFFFFFFFFul); as.add(t);
  mpz_sub_ui(t, R, 1); as.add(t);
  mpz_sub_ui(t, R, 2); as.add(t);
  {
    uint32_t ks[]={31, 32, 33, 63, 64, 65, BITS/8, BITS/4-1, BITS/4, BITS/4+1, BITS/2-33, BITS/2-32, BITS/2-1, BITS/2, BITS/2+1,
                   BITS/2+31, BITS/2+32, 3*BITS/4-1, 3*BITS/4, 3*BITS/4+1, BITS-65, BITS-64, BITS-33, BITS-32, BITS-2, BITS-1};
    for(uint32_t i=0;i<sizeof(ks)/sizeof(ks[0]);i++) {
      mpz_mul_2exp(t, one, ks[i]); as.add(t);                 // 2^k
      mpz_sub_ui(t, t, 1); as.add(t);                         // 2^k - 1
      mpz_add_ui(t, t, 2); as.add(t);                         // 2^k + 1
      mpz_mul_2exp(t, one, ks[i]); mpz_sub(t, R, t); as.add(t); // R - 2^k
    }
  }
  {
    uint32_t p0[]={0xFFFFFFFFu, 0u}, p1[]={0u, 0xFFFFFFFFu}, p2[]={0x80000000u}, p3[]={0x7FFFFFFFu}, p4[]={1u},
             p5[]={0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u}, p6[]={0u, 0u, 0xFFFFFFFFu, 0xFFFFFFFFu}, p7[]={0xAAAAAAAAu}, p8[]={0x55555555u},
             p9[]={0xFFFFFFFEu}, p10[]={0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u, 0u},
             p11[]={0u, 0u, 0u, 0u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}, p12[]={0x80000000u, 0u}, p13[]={0u, 0x80000000u};
    pattern(t, WORDS, p0, 2); as.add(t);   pattern(t, WORDS, p1, 2); as.add(t);
    pattern(t, WORDS, p2, 1); as.add(t);   pattern(t, WORDS, p3, 1); as.add(t);
    pattern(t, WORDS, p4, 1); as.add(t);   pattern(t, WORDS, p5, 4); as.add(t);
    pattern(t, WORDS, p6, 4); as.add(t);   pattern(t, WORDS, p7, 1); as.add(t);
    pattern(t, WORDS, p8, 1); as.add(t);   pattern(t, WORDS, p9, 1); as.add(t);
    pattern(t, WORDS, p10, 8); as.add(t);  pattern(t, WORDS, p11, 8); as.add(t);
    pattern(t, WORDS, p12, 2); as.add(t);  pattern(t, WORDS, p13, 2); as.add(t);
  }
  // low half / high half only
  mpz_mul_2exp(t, one, BITS/2); mpz_sub_ui(t, t, 1); mpz_mul_2exp(t, t, BITS/2); as.add(t);
  mpz_mul_2exp(t, one, BITS/2-1); mpz_sub_ui(t, t, 1); mpz_mul_2exp(t, t, BITS/2); as.add(t);
  for(uint32_t i=0;i<8;i++) {
    mpz_urandomb(t, state, BITS); as.add(t);
    mpz_urandomb(t, state, BITS/2); as.add(t);
    mpz_urandomb(t, state, BITS/2); mpz_mul_2exp(t, t, BITS/2); as.add(t);
    structured_random(t, state, WORDS); as.add(t);
  }

  // ---- special odd n values ----
  mpz_set_ui(t, 1); ns.add(t);
  mpz_set_ui(t, 3); ns.add(t);
  mpz_set_ui(t, 0xFFFFFFFFul); ns.add(t);
  mpz_sub_ui(t, R, 1); ns.add(t);
  mpz_sub_ui(t, R, 3); ns.add(t);
  mpz_mul_2exp(t, one, BITS-1); mpz_add_ui(t, t, 1); ns.add(t);
  mpz_mul_2exp(t, one, BITS-1); mpz_sub_ui(t, t, 1); ns.add(t);
  mpz_mul_2exp(t, one, BITS/2); mpz_add_ui(t, t, 1); ns.add(t);
  mpz_mul_2exp(t, one, BITS/2); mpz_sub_ui(t, t, 1); ns.add(t);
  mpz_mul_2exp(t, one, BITS-6); mpz_sub_ui(t, t, 1); ns.add(t);
  mpz_mul_2exp(t, one, BITS/2); mpz_sub_ui(t, t, 1); mpz_mul_2exp(t, t, BITS/2); mpz_add_ui(t, t, 1); ns.add(t);
  {
    uint32_t p0[]={1u}, p1[]={0x80000001u}, p2[]={0xFFFFFFFFu, 0u}, p3[]={0x55555555u}, p4[]={0x7FFFFFFFu};
    pattern(t, WORDS, p0, 1); ns.add(t);   pattern(t, WORDS, p1, 1); ns.add(t);
    pattern(t, WORDS, p2, 2); ns.add(t);   pattern(t, WORDS, p3, 1); ns.add(t);
    pattern(t, WORDS, p4, 1); ns.add(t);
  }
  for(uint32_t i=0;i<6;i++) {
    mpz_urandomb(t, state, BITS); mpz_setbit(t, 0); ns.add(t);
    mpz_urandomb(t, state, BITS-6); mpz_setbit(t, 0); ns.add(t);
    structured_random(t, state, WORDS); mpz_setbit(t, 0); ns.add(t);
  }

  // ---- build instances ----
  std::vector<instance_t> inst;
  instance_t              cur;
  for(size_t ni=0;ni<ns.v.size();ni++) {
    from_mpz(*ns.v[ni], cur.n._limbs, WORDS);
    for(size_t ai=0;ai<as.v.size();ai++) {
      from_mpz(*as.v[ai], cur.a._limbs, WORDS);
      inst.push_back(cur);
    }
    // values relative to n
    for(int32_t delta=-2;delta<=2;delta++) {
      for(uint32_t mult=1;mult<=2;mult++) {
        mpz_mul_ui(t, *ns.v[ni], mult);
        if(delta<0) {
          if(mpz_cmp_ui(t, (unsigned long)(-delta))<0) continue;
          mpz_sub_ui(t, t, (unsigned long)(-delta));
        }
        else
          mpz_add_ui(t, t, (unsigned long)delta);
        if(mpz_cmp(t, R)>=0) continue;
        from_mpz(t, cur.a._limbs, WORDS);
        inst.push_back(cur);
      }
    }
    mpz_sub(t, R, *ns.v[ni]); from_mpz(t, cur.a._limbs, WORDS); inst.push_back(cur);
  }
  for(uint32_t i=0;i<random_count || (inst.size()%8)!=1;i++) {
    structured_random(t, state, WORDS); from_mpz(t, cur.a._limbs, WORDS);
    structured_random(t, state, WORDS); mpz_setbit(t, 0); from_mpz(t, cur.n._limbs, WORDS);
    inst.push_back(cur);
  }

  for(size_t i=0;i<inst.size();i++) {
    mpz_t ta, tn;
    mpz_init(ta); mpz_init(tn);
    to_mpz(ta, inst[i].a._limbs, WORDS);
    to_mpz(tn, inst[i].n._limbs, WORDS);
    mpz_mod(ta, ta, tn);
    from_mpz(ta, inst[i].a_red._limbs, WORDS);
    mpz_clear(ta); mpz_clear(tn);
  }

  uint32_t count=(uint32_t)inst.size();
  uint32_t TPB=128, IPB=TPB/TPI, blocks=(count+IPB-1)/IPB;

  instance_t          *gpu;
  cgbn_error_report_t *report;
  CUDA_CHECK(cudaMalloc((void **)&gpu, sizeof(instance_t)*count));
  CUDA_CHECK(cudaMemcpy(gpu, inst.data(), sizeof(instance_t)*count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cgbn_error_report_alloc(&report));
  kernel_edge<params><<<blocks, TPB>>>(report, gpu, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  std::vector<instance_t> out(count);
  CUDA_CHECK(cudaMemcpy(out.data(), gpu, sizeof(instance_t)*count, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(gpu));
  CUDA_CHECK(cgbn_error_report_free(report));

  // ---- verify ----
  mpz_t a, n, q, u, g;
  mpz_init(a); mpz_init(n); mpz_init(q); mpz_init(u); mpz_init(g);
  uint32_t fail_sqr=0, fail_alias=0, fail_mul=0, fail_b2m=0, fail_back=0, sub_cases=0;
  for(uint32_t i=0;i<count;i++) {
    to_mpz(a, inst[i].a._limbs, WORDS);
    to_mpz(n, inst[i].n._limbs, WORDS);
    mpz_mul(u, a, a);
    mpz_invert(q, n, RM);
    mpz_mul(q, q, u);
    mpz_neg(q, q);
    mpz_mod(q, q, RM);
    mpz_addmul(u, q, n);
    mpz_tdiv_q_2exp(u, u, RBITS);
    if(mpz_cmp(u, R)>=0) {
      mpz_sub(u, u, n);
      sub_cases++;
    }
    to_mpz(g, out[i].r_sqr._limbs, WORDS);
    if(mpz_cmp(g, u)!=0) {
      if(fail_sqr<3) gmp_printf("  mont_sqr MISMATCH instance %u\n    a=%Zx\n    n=%Zx\n    got=%Zx\n    exp=%Zx\n", i, a, n, g, u);
      fail_sqr++;
    }
    to_mpz(g, out[i].r_alias._limbs, WORDS);
    if(mpz_cmp(g, u)!=0) fail_alias++;
    {
      // bn2mont must return the canonical a_red * R mod n; mont2bn may return n instead of 0 (CGBN: 0 <= res <= n)
      mpz_t ar, e;
      mpz_init(ar); mpz_init(e);
      to_mpz(ar, inst[i].a_red._limbs, WORDS);
      mpz_mul_2exp(e, ar, RBITS);
      mpz_mod(e, e, n);
      to_mpz(g, out[i].r_b2m._limbs, WORDS);
      if(mpz_cmp(g, e)!=0) {
        if(fail_b2m<3) gmp_printf("  bn2mont MISMATCH instance %u\n    a=%Zx\n    n=%Zx\n    got=%Zx\n    exp=%Zx\n", i, ar, n, g, e);
        fail_b2m++;
      }
      to_mpz(g, out[i].r_back._limbs, WORDS);
      if(mpz_cmp(g, ar)!=0 && !(mpz_sgn(ar)==0 && mpz_cmp(g, n)==0)) fail_back++;
      mpz_clear(ar); mpz_clear(e);
    }
    to_mpz(g, out[i].r_mul._limbs, WORDS);
    if(mpz_cmp(g, u)!=0) {
      if(fail_mul<3) gmp_printf("  mont_mul MISMATCH instance %u\n    a=%Zx\n    n=%Zx\n    got=%Zx\n    exp=%Zx\n", i, a, n, g, u);
      fail_mul++;
    }
  }
  bool ok=(fail_sqr==0 && fail_alias==0 && fail_mul==0 && fail_b2m==0 && fail_back==0);
  printf("EDGE TPI=%u BITS=%u: %u instances (%u with final -n), mont_sqr fails=%u, in-place fails=%u, mont_mul fails=%u, bn2mont fails=%u, mont2bn fails=%u : %s\n",
         TPI, BITS, count, sub_cases, fail_sqr, fail_alias, fail_mul, fail_b2m, fail_back, ok ? "PASS" : "FAIL");
  fflush(stdout);

  mpz_clear(a); mpz_clear(n); mpz_clear(q); mpz_clear(u); mpz_clear(g);
  mpz_clear(t); mpz_clear(R); mpz_clear(RM); mpz_clear(one);
  gmp_randclear(state);
  return ok;
}

int main(int argc, char **argv) {
  uint32_t      random_count=20000;
  unsigned long seed=1;
  std::vector<const char *> configs;

  for(int arg=1;arg<argc;arg++) {
    if(strcmp(argv[arg], "-r")==0 && arg+1<argc) random_count=(uint32_t)strtoul(argv[++arg], NULL, 10);
    else if(strcmp(argv[arg], "-s")==0 && arg+1<argc) seed=strtoul(argv[++arg], NULL, 10);
    else if(strcmp(argv[arg], "-c")==0 && arg+1<argc) configs.push_back(argv[++arg]);
    else { fprintf(stderr, "usage: %s [-r random_instances] [-s seed] [-c TPIxBITS ...]\n", argv[0]); return 1; }
  }

  bool ok=true, any=false;
  #define EDGE_CONFIG(T, B) \
    { bool sel=configs.empty(); \
      for(size_t i=0;i<configs.size();i++) if(strcmp(configs[i], #T "x" #B)==0) sel=true; \
      if(sel) { any=true; ok=run_config<edge_params_t<T, B> >(random_count, seed) && ok; } }
  EDGE_CONFIG(4, 256)
  EDGE_CONFIG(4, 512)
#ifndef EDGE_NO_WIDE          // wide-limb (LIMBS > TPI) needs the division free cgbn_bn2mont
  EDGE_CONFIG(4, 1024)
  EDGE_CONFIG(4, 2048)
  EDGE_CONFIG(8, 3072)
  EDGE_CONFIG(8, 4096)
#endif
  EDGE_CONFIG(4, 96)          // padded / odd LIMBS configurations (generic paths)
  EDGE_CONFIG(4, 160)
  EDGE_CONFIG(8, 768)
  EDGE_CONFIG(8, 1280)
  EDGE_CONFIG(16, 1568)
  EDGE_CONFIG(8, 512)
  EDGE_CONFIG(8, 1024)
  EDGE_CONFIG(16, 1024)
  EDGE_CONFIG(8, 1536)
  EDGE_CONFIG(8, 2048)
  EDGE_CONFIG(16, 2048)
  EDGE_CONFIG(16, 3072)
  EDGE_CONFIG(16, 4096)
  EDGE_CONFIG(32, 4096)
  EDGE_CONFIG(32, 2048)
  if(!any) { fprintf(stderr, "no matching configuration\n"); return 1; }
  printf("%s\n", ok ? "ALL EDGE TESTS PASSED" : "EDGE TESTS FAILED");
  return ok ? 0 : 1;
}
