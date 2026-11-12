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

#ifdef CGBN_DEBUG_MONT_SQR
#include <assert.h>
#endif

namespace cgbn {

__device__ __forceinline__ uint32_t wmad_shfl_idx8(const uint32_t mask, const uint32_t val, const uint32_t src_lane) {
  uint32_t res;
  asm ("shfl.sync.idx.b32 %0, %1, %2, 0x181f, %3;" : "=r"(res) : "r"(val), "r"(src_lane), "r"(mask));
  return res;
}

template<uint32_t C_IMM>
__device__ __forceinline__ uint32_t wmad_shfl_up8_zero(const uint32_t mask, const uint32_t val) {
  uint32_t res;
  asm ("{\n\t"
       ".reg .pred p;\n\t"
       "shfl.sync.up.b32 %0|p, %1, 1, %2, %3;\n\t"
       "selp.b32 %0, %0, 0, p;\n\t"
       "}" : "=r"(res) : "r"(val), "n"(C_IMM), "r"(mask));
  return res;
}

__device__ __forceinline__ uint32_t wmad_shfl_xor8(const uint32_t mask, const uint32_t val, const uint32_t lane_mask) {
  uint32_t res;
  asm ("shfl.sync.bfly.b32 %0, %1, %2, 0x181f, %3;" : "=r"(res) : "r"(val), "r"(lane_mask), "r"(mask));
  return res;
}

__device__ __forceinline__ void wmad_mul_2x2(
    uint32_t &p0, uint32_t &p1, uint32_t &p2, uint32_t &p3,
    const uint32_t x0, const uint32_t x1, const uint32_t y0, const uint32_t y1) {
  uint64_t w00=mad_wide(x0, y0, 0);
  uint64_t w11=mad_wide(x1, y1, 0);
  uint64_t m=mad_wide(x0, y1, 0);
  uint32_t m0=ulow(m), m1=uhigh(m), mc;
  m0=madlo_cc(x1, y0, m0);
  m1=madhic_cc(x1, y0, m1);
  mc=addc(0, 0);
  p0=ulow(w00);
  p1=add_cc(uhigh(w00), m0);
  p2=addc_cc(ulow(w11), m1);
  p3=addc(uhigh(w11), mc);
}

template<int32_t THREAD>
__device__ __forceinline__ void wmad_cios_pair_tpi8_l4(
    const uint32_t shfl_mask,
    const uint32_t group_thread,
    const uint32_t next_lane,
    uint32_t ra[6],
    uint32_t ru[5],
    uint32_t &c,
    const uint32_t v[4],
    const uint32_t n[4],
    const uint32_t np0) {
  #pragma unroll
  for(int32_t word=0;word<8;word+=2) {
    uint32_t t0, t1, q;
    if(word<4) {
      t0=wmad_shfl_idx8(shfl_mask, v[word], THREAD);
      t1=wmad_shfl_idx8(shfl_mask, v[word+1], THREAD);
    }
    else {
      t0=wmad_shfl_idx8(shfl_mask, v[word-4], THREAD+1);
      t1=wmad_shfl_idx8(shfl_mask, v[word-3], THREAD+1);
    }

    if(THREAD==0 && word==0) {
      uint64_t p1=mad_wide(v[1], t0, 0);
      uint64_t p3=mad_wide(v[3], t0, 0);
      uint64_t p0=mad_wide(v[0], t0, 0);
      uint64_t p2=mad_wide(v[2], t0, 0);
      ru[0]=ulow(p1);
      ru[1]=uhigh(p1);
      ru[2]=ulow(p3);
      ru[3]=uhigh(p3);
      ra[0]=ulow(p0);
      ra[1]=uhigh(p0);
      ra[2]=ulow(p2);
      ra[3]=uhigh(p2);
      ra[4]=0;
    }
    else {
      chain_t<5> chain1;
      ra[0]=chain1.add(ra[0], c);
      ru[0]=chain1.madlo(v[1], t0, ru[0]);
      ru[1]=chain1.madhi(v[1], t0, ru[1]);
      ru[2]=chain1.madlo(v[3], t0, ru[2]);
      ru[3]=chain1.madhi(v[3], t0, ru[3]);

      chain_t<5> chain2;
      ra[0]=chain2.madlo(v[0], t0, ra[0]);
      ra[1]=chain2.madhi(v[0], t0, ra[1]);
      ra[2]=chain2.madlo(v[2], t0, ra[2]);
      ra[3]=chain2.madhi(v[2], t0, ra[3]);
      ra[4]=chain2.add(ra[4], 0);
    }

    chain_t<6> chain3;
    q=wmad_shfl_idx8(shfl_mask, ra[0], 0)*np0;
    ra[0]=chain3.madlo(n[0], q, ra[0]);
    ra[1]=chain3.madhi(n[0], q, ra[1]);
    ra[2]=chain3.madlo(n[2], q, ra[2]);
    ra[3]=chain3.madhi(n[2], q, ra[3]);
    t0=wmad_shfl_idx8(shfl_mask, ra[0], next_lane);
    if(THREAD==0 && word==0)
      ra[4]=chain3.add(t0, 0);
    else
      ra[4]=chain3.add(ra[4], t0);
    ra[5]=chain3.add(0, 0);
#ifdef CGBN_DEBUG_MONT_SQR
    if(group_thread==0) assert(ra[0]==0u);
#endif

    chain_t<5> chain4;
    ru[0]=chain4.madlo(n[1], q, ru[0]);
    ru[1]=chain4.madhi(n[1], q, ru[1]);
    ru[2]=chain4.madlo(n[3], q, ru[2]);
    ru[3]=chain4.madhi(n[3], q, ru[3]);
    ru[4]=chain4.add(0, 0);

    chain_t<5> chain5;
    ru[0]=chain5.madlo(v[0], t1, ru[0]);
    ru[1]=chain5.madhi(v[0], t1, ru[1]);
    ru[2]=chain5.madlo(v[2], t1, ru[2]);
    ru[3]=chain5.madhi(v[2], t1, ru[3]);
    ru[4]=chain5.add(ru[4], 0);

    chain_t<4> chain6;
    ra[2]=chain6.madlo(v[1], t1, ra[2]);
    ra[3]=chain6.madhi(v[1], t1, ra[3]);
    ra[4]=chain6.madlo(v[3], t1, ra[4]);
    ra[5]=chain6.madhi(v[3], t1, ra[5]);

    chain_t<6> chain7;
    ru[0]=chain7.add(ru[0], ra[1]);
    q=wmad_shfl_idx8(shfl_mask, ru[0], 0)*np0;
    ra[0]=chain7.madlo(n[1], q, ra[2]);
    ra[1]=chain7.madhi(n[1], q, ra[3]);
    ra[2]=chain7.madlo(n[3], q, ra[4]);
    ra[3]=chain7.madhi(n[3], q, ra[5]);
    ra[4]=chain7.add(0, 0);

    uint32_t t1_low;
    chain_t<6> chain8;
    t1_low=chain8.madlo(n[0], q, ru[0]);
    c=chain8.madhi(n[0], q, ru[1]);
    ru[0]=chain8.madlo(n[2], q, ru[2]);
    ru[1]=chain8.madhi(n[2], q, ru[3]);
    t1=wmad_shfl_idx8(shfl_mask, t1_low, next_lane);
    ru[2]=chain8.add(ru[4], t1);
    ru[3]=chain8.add(0, 0);
#ifdef CGBN_DEBUG_MONT_SQR
    if(group_thread==0) assert(t1_low==0u);
#endif
  }
}

__device__ __forceinline__ void wmad_reduce_zero_tpi8_l4(
    const uint32_t shfl_mask,
    const uint32_t group_thread,
    const uint32_t next_lane,
    uint32_t ra[6],
    uint32_t ru[5],
    uint32_t &c,
    const uint32_t n[4],
    const uint32_t np0) {
  #pragma unroll
  for(int32_t step=0;step<8;step++) {
    uint32_t t0, t1, t1_low, q;

    chain_t<5> chain1;
    ra[0]=chain1.add(ra[0], c);
    q=wmad_shfl_idx8(shfl_mask, ra[0], 0)*np0;
    ru[0]=chain1.madlo(n[1], q, ru[0]);
    ru[1]=chain1.madhi(n[1], q, ru[1]);
    ru[2]=chain1.madlo(n[3], q, ru[2]);
    ru[3]=chain1.madhi(n[3], q, ru[3]);

    chain_t<6> chain2;
    uint32_t ra0_low=chain2.madlo(n[0], q, ra[0]);
    ra[1]=chain2.madhi(n[0], q, ra[1]);
    ra[2]=chain2.madlo(n[2], q, ra[2]);
    ra[3]=chain2.madhi(n[2], q, ra[3]);
    t0=wmad_shfl_idx8(shfl_mask, ra0_low, next_lane);
    if(step==0)
      ra[4]=chain2.add(ra[4], t0);
    else
      ra[4]=chain2.add(t0, 0);
    ra[5]=chain2.add(0, 0);
#ifdef CGBN_DEBUG_MONT_SQR
    if(group_thread==0) assert(ra0_low==0u);
#endif

    chain_t<5> chain3;
    ru[0]=chain3.add(ru[0], ra[1]);
    q=wmad_shfl_idx8(shfl_mask, ru[0], 0)*np0;
    ra[0]=chain3.madlo(n[1], q, ra[2]);
    ra[1]=chain3.madhi(n[1], q, ra[3]);
    ra[2]=chain3.madlo(n[3], q, ra[4]);
    ra[3]=chain3.madhi(n[3], q, ra[5]);

    chain_t<6> chain4;
    t1_low=chain4.madlo(n[0], q, ru[0]);
    c=chain4.madhi(n[0], q, ru[1]);
    ru[0]=chain4.madlo(n[2], q, ru[2]);
    ru[1]=chain4.madhi(n[2], q, ru[3]);
    t1=wmad_shfl_idx8(shfl_mask, t1_low, next_lane);
    ru[2]=chain4.add(t1, 0);
    ru[3]=chain4.add(0, 0);
#ifdef CGBN_DEBUG_MONT_SQR
    if(group_thread==0) assert(t1_low==0u);
#endif
  }
}

template<class env>
__device__ __forceinline__ void wmad_fast_prop_tpi8_l4(
    const uint32_t shfl_mask,
    const uint32_t carry,
    uint32_t x[4]) {
  uint32_t tid=threadIdx.x;
  asm volatile ("" : "+r"(tid));
  uint32_t gt=tid & 7u;
  uint32_t wt=tid & 31u;
  uint32_t sync=0xFFu<<(wt & 0x18u);
  uint32_t lane=(gt==0u) ? 0u : (1u<<wt);
  uint32_t land=mpland<4>(x);
  uint32_t land_nz=(gt==0u) ? 0u : land;
  uint32_t g, p;
  asm volatile ("{\n\t"
                ".reg .pred pg, pp;\n\t"
                "setp.ne.u32 pg, %2, 0;\n\t"
                "setp.eq.u32 pp, %3, 0xffffffff;\n\t"
                "vote.sync.ballot.b32 %0, pg, %5;\n\t"
                "vote.sync.ballot.b32 %1, pp, %5;\n\t"
                "and.b32 %0, %0, %4;\n\t"
                "and.b32 %1, %1, %4;\n\t"
                "}" : "=r"(g), "=r"(p) : "r"(carry), "r"(land_nz), "r"(sync), "r"(shfl_mask));
  uint32_t sum=g+g+p;
  uint32_t c_in=((lane & (p ^ sum))!=0u) ? 1u : 0u;
  x[0]=add_cc(x[0], c_in);
  x[1]=addc_cc(x[1], 0);
  x[2]=addc_cc(x[2], 0);
  x[3]=addc(x[3], 0);
}

template<class env>
__device__ __forceinline__ void wmad_mont_sqr_tpi8_l4(
    uint32_t r[4],
    const uint32_t a[4],
    const uint32_t n[4],
    const uint32_t np0) {
  uint32_t shfl_mask;
  asm ("activemask.b32 %0;" : "=r"(shfl_mask));
  uint32_t group_thread=threadIdx.x & 7u;
  uint32_t g4=group_thread<<2;
  uint32_t g_bit=1u<<group_thread;


  // Pack all 8 Step-3 3-bit lane LUT indices, d_src, sh_amt, and prev_lane into two
  // loop-invariant 32-bit registers (lut_o, lut_e) while allowing all 6 lane predicates
  // to hoist freely (0 LOP3/ISETP inside loops). Because SHFL.IDX with c=0x181f masks
  // src_lane & 7 in hardware, packed nibbles require only a single right shift (and 0
  // shifts for bit [2:0]), hitting REG=48 (83.3% SM occupancy = 40 warps/SM on sm_89)
  // and 632 dynamic fast-loop instructions.
  uint32_t lut_o=((0x32540002u>>g4) & 7u)
                | (((0x43657537u>>g4) & 7u)<<4)
                | (((0x26431110u>>g4) & 7u)<<8)
                | (((0x57766421u>>g4) & 7u)<<12)
                | ((4u+(group_thread>>1))<<16)
                | ((group_thread>>2)<<20);
  uint32_t lut_e=((0x25430001u>>g4) & 7u)
                | (((0x47656427u>>g4) & 7u)<<4)
                | (((0x11321132u>>g4) & 7u)<<8)
                | (((0x53765376u>>g4) & 7u)<<12)
                | (((group_thread+7u) & 7u)<<16);
  asm ("" : "+r"(lut_o), "+r"(lut_e));
  uint32_t lo=lut_o, le=lut_e;
  asm volatile ("" : "+r"(lo), "+r"(le));


#ifdef CGBN_DEBUG_MONT_SQR
  uint32_t n0_dbg=wmad_shfl_idx8(shfl_mask, n[0], 0);
  assert((n0_dbg & 1u)==1u);
  assert(n0_dbg*np0==0xFFFFFFFFu);
  assert((wmad_shfl_idx8(shfl_mask, a[3], 7)>>30)==0u);
  assert((wmad_shfl_idx8(shfl_mask, n[3], 7)>>28)==0u);
#endif

  uint32_t d_src=lo>>16;
  uint32_t s0=wmad_shfl_idx8(shfl_mask, a[0], d_src);
  uint32_t s2=wmad_shfl_idx8(shfl_mask, a[2], d_src);
  uint32_t s1=wmad_shfl_idx8(shfl_mask, a[1], d_src);
  uint32_t s3=wmad_shfl_idx8(shfl_mask, a[3], d_src);
  bool     odd_lane=(group_thread & 1u)!=0u;
  uint32_t d0=odd_lane ? s2 : s0;
  uint32_t d1=odd_lane ? s3 : s1;

  uint32_t sh_in=wmad_shfl_up8_zero<0x1c00>(shfl_mask, a[3]);
  uint32_t sh_amt=lo>>20;

  uint32_t v[4];
  v[0]=uleft_wrap(sh_in, a[0], sh_amt);
  v[1]=uleft_wrap(a[0], a[1], sh_amt);
  v[2]=uleft_wrap(a[1], a[2], sh_amt);
  v[3]=uleft_wrap(a[2], a[3], sh_amt);

  uint32_t ra[6]={0, 0, 0, 0, 0, 0};
  uint32_t ru[5]={0, 0, 0, 0, 0};
  uint32_t c=0;
  uint32_t next_lane=group_thread+1u;

  wmad_cios_pair_tpi8_l4<0>(shfl_mask, group_thread, next_lane, ra, ru, c, v, n, np0);
  wmad_cios_pair_tpi8_l4<2>(shfl_mask, group_thread, next_lane, ra, ru, c, v, n, np0);

  wmad_reduce_zero_tpi8_l4(shfl_mask, group_thread, next_lane, ra, ru, c, n, np0);

  uint32_t r1;
  chain_t<5> chain5;
  ra[0]=chain5.add(ra[0], c);
  ra[1]=chain5.add(ra[1], ru[0]);
  ra[2]=chain5.add(ra[2], ru[1]);
  ra[3]=chain5.add(ra[3], ru[2]);
  r1=chain5.add(ru[3], 0);

  asm volatile ("" : "+r"(d0), "+r"(d1) : "r"(r1));

  uint32_t E0, E1, E2, E3, c_E;
  {
    uint32_t i0=lo;
    uint32_t j0=lo>>4;
    uint32_t O0, O1, O2, O3;
    {
      uint32_t xi0=wmad_shfl_idx8(shfl_mask, d0, i0);
      uint32_t xi1=wmad_shfl_idx8(shfl_mask, d1, i0);
      uint32_t xj0=wmad_shfl_idx8(shfl_mask, d0, j0);
      uint32_t xj1=wmad_shfl_idx8(shfl_mask, d1, j0);
      wmad_mul_2x2(O0, O1, O2, O3, xi0, xi1, xj0, xj1);
    }

    uint32_t i1=lo>>8;
    uint32_t j1=lo>>12;
    uint32_t P0, P1, P2, P3;
    {
      uint32_t xi0=wmad_shfl_idx8(shfl_mask, d0, i1);
      uint32_t xi1=wmad_shfl_idx8(shfl_mask, d1, i1);
      uint32_t xj0=wmad_shfl_idx8(shfl_mask, d0, j1);
      uint32_t xj1=wmad_shfl_idx8(shfl_mask, d1, j1);
      wmad_mul_2x2(P0, P1, P2, P3, xi0, xi1, xj0, xj1);
    }

    bool accum_p=(g_bit & 0xBEu)!=0u;
    uint32_t c_O;
    {
      uint32_t a0=accum_p ? P0 : 0u;
      uint32_t a1=accum_p ? P1 : 0u;
      uint32_t a2=accum_p ? P2 : 0u;
      uint32_t a3=accum_p ? P3 : 0u;
      O0=add_cc(O0, a0);
      O1=addc_cc(O1, a1);
      O2=addc_cc(O2, a2);
      O3=addc_cc(O3, a3);
      c_O=addc(0, 0);
    }

    uint32_t x0=wmad_shfl_xor8(shfl_mask, O0, 4);
    uint32_t x1=wmad_shfl_xor8(shfl_mask, O1, 4);
    uint32_t x2=wmad_shfl_xor8(shfl_mask, O2, 4);
    uint32_t x3=wmad_shfl_xor8(shfl_mask, O3, 4);
    uint32_t xc=wmad_shfl_xor8(shfl_mask, c_O, 4);

    O0=accum_p ? O0 : P0;
    O1=accum_p ? O1 : P1;
    O2=accum_p ? O2 : P2;
    O3=accum_p ? O3 : P3;

    {
      bool recv_o=(g_bit & 0x1Cu)!=0u;
      uint32_t r0_m=recv_o ? x0 : 0u;
      uint32_t r1_m=recv_o ? x1 : 0u;
      uint32_t r2_m=recv_o ? x2 : 0u;
      uint32_t r3_m=recv_o ? x3 : 0u;
      uint32_t rc_m=recv_o ? xc : 0u;
      O0=add_cc(O0, r0_m);
      O1=addc_cc(O1, r1_m);
      O2=addc_cc(O2, r2_m);
      O3=addc_cc(O3, r3_m);
      c_O=addc(c_O, rc_m);
    }

    {
      E0=wmad_shfl_up8_zero<0x1800>(shfl_mask, O2);
      uint32_t e1_in=wmad_shfl_up8_zero<0x1800>(shfl_mask, O3);
      uint32_t upc_m=wmad_shfl_up8_zero<0x1800>(shfl_mask, c_O);

      bool     not_g7=group_thread!=7u;
      uint32_t o0_m=not_g7 ? O0 : 0u;
      uint32_t o1_m=not_g7 ? O1 : 0u;

      uint64_t intra=mad_wide(d0, d1, 0);
      E2=add_cc(o0_m, upc_m);
      E3=addc_cc(o1_m, 0u);
      c_E=addc(0u, 0u);

      E1=add_cc(e1_in, ulow(intra));
      E2=addc_cc(E2, uhigh(intra));
      E3=addc_cc(E3, 0u);
      c_E=addc(c_E, 0u);
    }

    uint32_t ie=le;
    uint32_t je=le>>4;
    {
      uint32_t xi0=wmad_shfl_idx8(shfl_mask, d0, ie);
      uint32_t xi1=wmad_shfl_idx8(shfl_mask, d1, ie);
      uint32_t xj0=wmad_shfl_idx8(shfl_mask, d0, je);
      uint32_t xj1=wmad_shfl_idx8(shfl_mask, d1, je);
      wmad_mul_2x2(P0, P1, P2, P3, xi0, xi1, xj0, xj1);
    }

    uint32_t im=le>>8;
    uint32_t jm=le>>12;
    {
      uint32_t xi0=wmad_shfl_idx8(shfl_mask, d0, im);
      uint32_t xi1=wmad_shfl_idx8(shfl_mask, d1, im);
      uint32_t dj0=wmad_shfl_idx8(shfl_mask, d0, jm);
      uint32_t dj1=wmad_shfl_idx8(shfl_mask, d1, jm);
      bool     owns_mid=(g_bit & 0x3Cu)!=0u;
      uint32_t bj=owns_mid ? dj1 : dj0;

      uint64_t m0=mad_wide(xi0, bj, 0);
      uint32_t h0=ulow(m0);
      uint64_t m1=mad_wide(xi1, bj, (uint64_t)uhigh(m0));
      uint32_t h1=ulow(m1);
      uint32_t h2=uhigh(m1);

      bool     keep_e=(g_bit & 0x7Eu)!=0u;
      uint32_t kp0=keep_e ? P0 : 0u;
      uint32_t kp1=keep_e ? P1 : 0u;
      uint32_t kp2=keep_e ? P2 : 0u;
      uint32_t kp3=keep_e ? P3 : 0u;
      E0=add_cc(E0, kp0);
      E1=addc_cc(E1, kp1);
      E2=addc_cc(E2, kp2);
      E3=addc_cc(E3, kp3);
      c_E=addc(c_E, 0);

      uint32_t sp0=keep_e ? 0u : P0;
      uint32_t sp1=keep_e ? 0u : P1;
      uint32_t sp2=keep_e ? 0u : P2;
      uint32_t sp3=keep_e ? 0u : P3;
      uint32_t S0=add_cc(h0, sp0);
      uint32_t S1=addc_cc(h1, sp1);
      uint32_t S2=addc_cc(h2, sp2);
      uint32_t S3=addc_cc(0u, sp3);
      uint32_t Sc=addc(0u, 0u);

      uint32_t rx0=wmad_shfl_xor8(shfl_mask, S0, 4);
      uint32_t rx1=wmad_shfl_xor8(shfl_mask, S1, 4);
      uint32_t rx2=wmad_shfl_xor8(shfl_mask, S2, 4);
      uint32_t rx3=wmad_shfl_xor8(shfl_mask, S3, 4);
      uint32_t rxc=wmad_shfl_xor8(shfl_mask, Sc, 4);

      rx1=add_cc(rx1, h0);
      rx2=addc_cc(rx2, h1);
      rx3=addc_cc(rx3, h2);
      rxc=addc(rxc, 0u);

      uint32_t mm0=owns_mid ? rx0 : 0u;
      uint32_t mm1=owns_mid ? rx1 : 0u;
      uint32_t mm2=owns_mid ? rx2 : 0u;
      uint32_t mm3=owns_mid ? rx3 : 0u;
      uint32_t mmc=owns_mid ? rxc : 0u;
      E0=add_cc(E0, mm0);
      E1=addc_cc(E1, mm1);
      E2=addc_cc(E2, mm2);
      E3=addc_cc(E3, mm3);
      c_E=addc(c_E, mmc);
    }

    c_E=(c_E<<1) | (E3>>31);
    E3=uleft_wrap(E2, E3, 1);
    E2=uleft_wrap(E1, E2, 1);
    E1=uleft_wrap(E0, E1, 1);
    E0=E0<<1;

    E0=madlo_cc(d0, d0, E0);
    E1=madhic_cc(d0, d0, E1);
    E2=madloc_cc(d1, d1, E2);
    E3=madhic_cc(d1, d1, E3);
    c_E=addc(c_E, 0);
  }

  ra[0]=add_cc(ra[0], E0);
  ra[1]=addc_cc(ra[1], E1);
  ra[2]=addc_cc(ra[2], E2);
  ra[3]=addc_cc(ra[3], E3);
  r1=addc(r1, c_E);

#ifdef CGBN_DEBUG_MONT_SQR
  if(group_thread==7u) assert(r1==0u);
#endif

  uint32_t t_up=wmad_shfl_idx8(shfl_mask, r1, le>>16);

  ra[0]=add_cc(ra[0], t_up);
  ra[1]=addc_cc(ra[1], 0);
  ra[2]=addc_cc(ra[2], 0);
  ra[3]=addc_cc(ra[3], 0);
  c=addc(0, 0);

#ifdef CGBN_DEBUG_MONT_SQR
  if(group_thread==7u) assert(c==0u);
  assert(c<=1u);
#endif

  if(__builtin_expect(__any_sync(shfl_mask, c!=0u), 0))
    wmad_fast_prop_tpi8_l4<env>(shfl_mask, c, ra);

#ifdef CGBN_DEBUG_MONT_SQR
  {
    uint32_t ref_r[4];
    core_t<env>::mont_mul(ref_r, a, a, n, np0);
    for(int32_t index=0;index<4;index++)
      assert(ra[index]==ref_r[index]);
  }
#endif

  r[0]=ra[0];
  r[1]=ra[1];
  r[2]=ra[2];
  r[3]=ra[3];
}

template<class env>
__device__ __forceinline__ void core_t<env>::mont_sqr(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  if(TPI==8 && LIMBS==4 && PADDING==0) {
    wmad_mont_sqr_tpi8_l4<env>(r, a, n, np0);
    return;
  }
  mont_mul(r, a, a, n, np0);
}

#if 1
template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t t, t0, t1, q, r1, ra[LIMBS+2], ru[LIMBS+1], c=0;
    
  #pragma unroll
  for(int32_t index=0;index<=LIMBS;index++) {
    ra[index]=0;
    ru[index]=0;
  }
  ra[LIMBS+1]=0;
  
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread+=2) {
    #pragma unroll
    for(int32_t word=0;word<2*LIMBS;word+=2) {
      if(word<LIMBS) 
        t0=__shfl_sync(sync, b[word], thread, TPI);
      else
        t0=__shfl_sync(sync, b[word-LIMBS], thread+1, TPI);
      if(word+1<LIMBS)
        t1=__shfl_sync(sync, b[word+1], thread, TPI);
      else
        t1=__shfl_sync(sync, b[word+1-LIMBS], thread+1, TPI);
    
      /* FIRST HALF */
      
      chain_t<> chain1;                               // unaligned: T0 * A_odd
      ra[0]=chain1.add(ra[0], c);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain1.madlo(a[index+1], t0, ru[index]);
        ru[index+1]=chain1.madhi(a[index+1], t0, ru[index+1]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain1.add(ru[LIMBS-1], 0);
        
      chain_t<> chain2;                               // aligned:   T0 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain2.madlo(a[index], t0, ra[index]);
        ra[index+1]=chain2.madhi(a[index], t0, ra[index+1]);
      }
      if(LIMBS%2==0)
        ra[LIMBS]=chain2.add(ra[LIMBS], 0);
      
      chain_t<> chain3;                               // aligned:   Q0 * N_even
      q=__shfl_sync(sync, ra[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain3.madlo(n[index], q, ra[index]);
        ra[index+1]=chain3.madhi(n[index], q, ra[index+1]);
      }
      ra[LIMBS+LIMBS%2]=chain3.add(ra[LIMBS+LIMBS%2], 0);
              
      chain_t<> chain4;                               // unaligned: Q0 * N_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain4.madlo(n[index+1], q, ru[index]);
        ru[index+1]=chain4.madhi(n[index+1], q, ru[index+1]);
      }
      ru[LIMBS-LIMBS%2]=chain4.add(ru[LIMBS-LIMBS%2], 0);

      /* SECOND HALF */

      t0=ra[0];

      chain_t<> chain5;                               // unaigned: T1 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ru[index]=chain5.madlo(a[index], t1, ru[index]);
        ru[index+1]=chain5.madhi(a[index], t1, ru[index+1]);
      }
      ru[LIMBS-LIMBS%2]=chain5.add(ru[LIMBS-LIMBS%2], 0);
      
      chain_t<> chain6;                               // aligned:   T1 * A_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ra[index+2]=chain6.madlo(a[index+1], t1, ra[index+2]);
        ra[index+3]=chain6.madhi(a[index+1], t1, ra[index+3]);
      }
      if(LIMBS%2==1)
        ra[LIMBS+1]=chain3.add(ra[LIMBS+1], 0);

      chain_t<> chain7;                               // aligned:   Q1 * N_odd
      ru[0]=chain7.add(ru[0], ra[1]);
      q=__shfl_sync(sync, ru[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<(int32_t)LIMBS-3;index+=2) {
        ra[index]=chain7.madlo(n[index+1], q, ra[index+2]);
        ra[index+1]=chain7.madhi(n[index+1], q, ra[index+3]);
      }
      ra[LIMBS-2-LIMBS%2]=chain3.madlo(n[LIMBS-1-LIMBS%2], q, ra[LIMBS-LIMBS%2]);
      ra[LIMBS-1-LIMBS%2]=chain3.madhi(n[LIMBS-1-LIMBS%2], q, ra[LIMBS+1-LIMBS%2]);
      if(LIMBS%2==1) {
        ra[LIMBS-1]=chain3.add(ra[LIMBS+1], 0);
        ra[LIMBS]=0;
      }
      else 
        ra[LIMBS-LIMBS%2]=chain3.add(0, 0);

      chain_t<> chain8;                               // unaigned: Q1 * N_even
      t1=chain8.madlo(n[0], q, ru[0]);
      c=chain8.madhi(n[0], q, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain8.madlo(n[index+2], q, ru[index+2]);
        ru[index+1]=chain8.madhi(n[index+2], q, ru[index+3]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain8.add(0, 0);
      else {
        ru[LIMBS-2]=chain8.add(ru[LIMBS], 0);
        ru[LIMBS-1]=0;
      }
      ru[LIMBS]=0;
            
      t0=__shfl_sync(sync, t0, threadIdx.x+1, TPI);
      t1=__shfl_sync(sync, t1, threadIdx.x+1, TPI);
        
      ra[LIMBS-2]=add_cc(ra[LIMBS-2], t0);
      ra[LIMBS-1]=addc_cc(ra[LIMBS-1], t1);
      ra[LIMBS]=addc(ra[LIMBS], 0);
      ra[LIMBS+1]=0;
    }
  }
  
  chain_t<LIMBS+1> chain9;
  r[0]=chain9.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++) 
    r[index]=chain9.add(ra[index], ru[index-1]);
  r1=chain9.add(ra[LIMBS], ru[LIMBS-1]);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain10;
  r[0]=chain10.add(r[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain10.add(r[index], 0);
  c=chain10.add(r1, 0);

  c=-fast_propagate_add(c, r);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain11;
  r[0]=chain11.add(r[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain11.add(r[index], ~n[index] & c);
  c=chain11.add(0, 0);
  fast_propagate_add(c, r);
  clear_padding(r);
}
#endif

template<class env>
__device__ __forceinline__ void core_t<env>::mont_reduce_wide(uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t n[LIMBS], const uint32_t np0, const bool zero) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t t0, t1, q, ra[LIMBS+2], ru[LIMBS+1], c=0, top;

  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    ra[index]=lo[index];
    ru[index]=0;
  }
  ra[LIMBS]=0;
  ru[LIMBS]=0;
  ra[LIMBS+1]=0;
  
  c=0;
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread+=2) {
    #pragma unroll
    for(int32_t l=0;l<2*LIMBS;l+=2) {
      chain_t<> chain1;                               // unaligned: Q0 * N_odd
      ra[0]=chain1.add(ra[0], c);
      q=__shfl_sync(sync, ra[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain1.madlo(n[index+1], q, ru[index]);
        ru[index+1]=chain1.madhi(n[index+1], q, ru[index+1]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain1.add(0, 0);
 
      chain_t<> chain2;                               // aligned:   Q0 * N_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain2.madlo(n[index], q, ra[index]);
        ra[index+1]=chain2.madhi(n[index], q, ra[index+1]);
      }
      if(LIMBS%2==0)
        ra[LIMBS]=chain2.add(ra[LIMBS], 0);      

      t0=__shfl_sync(sync, ra[0], threadIdx.x+1, TPI);
      if(!zero) {
        if(l<LIMBS)
          top=__shfl_sync(sync, hi[l], thread, TPI);
        else
          top=__shfl_sync(sync, hi[l-LIMBS], thread+1, TPI);
        t0=(group_thread==TPI-1) ? top : t0;
      }        

      chain_t<> chain3;                               // aligned:   Q1 * N_odd
      ru[0]=chain3.add(ru[0], ra[1]);
      q=__shfl_sync(sync, ru[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<(int32_t)LIMBS-3;index+=2) {
        ra[index]=chain3.madlo(n[index+1], q, ra[index+2]);
        ra[index+1]=chain3.madhi(n[index+1], q, ra[index+3]);
      }
      ra[LIMBS-2-LIMBS%2]=chain3.madlo(n[LIMBS-1-LIMBS%2], q, ra[LIMBS-LIMBS%2]);
      ra[LIMBS-1-LIMBS%2]=chain3.madhi(n[LIMBS-1-LIMBS%2], q, ra[LIMBS+1-LIMBS%2]);
      if(LIMBS%2==1)
        ra[LIMBS-1]=chain3.add(0, 0);

      chain_t<> chain4;                               // unaligned: Q1 * N_even
      t1=chain4.madlo(n[0], q, ru[0]);
      c=chain4.madhi(n[0], q, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain4.madlo(n[index+2], q, ru[index+2]);
        ru[index+1]=chain4.madhi(n[index+2], q, ru[index+3]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=0;
      else 
        ru[LIMBS-2]=chain4.add(0, 0);
      ru[LIMBS-1+LIMBS%2]=0;

      t1=__shfl_sync(sync, t1, threadIdx.x+1, TPI);
      if(!zero) {
        if(l+1<LIMBS)
          top=__shfl_sync(sync, hi[l+1], thread, TPI);
        else
          top=__shfl_sync(sync, hi[l+1-LIMBS], thread+1, TPI);
        t1=(group_thread==TPI-1) ? top : t1;
      }
      
      ra[LIMBS-2]=add_cc(ra[LIMBS-2], t0);
      ra[LIMBS-1]=addc_cc(ra[LIMBS-1], t1);
      ra[LIMBS]=addc(0, 0);
    }
  }
  
  chain_t<> chain5;
  ra[0]=chain5.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chain5.add(ra[index], ru[index-1]);
  c=chain5.add(ra[LIMBS], 0);
  
  c=fast_propagate_add(c, ra);
  
  if(!zero && c!=0) {
    t0=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

    chain_t<LIMBS+1> chain3;
    ra[0]=chain3.add(ra[0], ~t0);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      ra[index]=chain3.add(ra[index], ~n[index]);
    c=chain3.add(0, 0);
    fast_propagate_add(c, ra);
    clear_padding(ra);
  }

  mpset<LIMBS>(r, ra);
}

#if 0
template<uint32_t LIMBS>
__device__ __forceinline__ void fwmont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t t, t0, t1, q, r1, ra[LIMBS+2], ru[LIMBS+1], c=0;
    
  #pragma unroll
  for(int32_t index=0;index<=LIMBS;index++) {
    ra[index]=0;
    ru[index]=0;
  }
  ra[LIMBS+1]=0;
  
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread+=2) {
    #pragma unroll
    for(int32_t word=0;word<2*LIMBS;word+=2) {
      if(word<LIMBS) 
        t0=__shfl_sync(sync, b[word], thread, TPI);
      else
        t0=__shfl_sync(sync, b[word-LIMBS], thread+1, TPI);
      if(word+1<LIMBS)
        t1=__shfl_sync(sync, b[word+1], thread, TPI);
      else
        t1=__shfl_sync(sync, b[word+1-LIMBS], thread+1, TPI);
    
      /* FIRST HALF */
      
      chain_t<> chain1;                               // aligned:   T0 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain1.madlo(a[index], t0, ra[index]);
        ra[index+1]=chain1.madhi(a[index], t0, ra[index+1]);
      }
      ra[LIMBS+LIMBS%2]=chain1.add(ra[LIMBS+LIMBS%2], 0);

      chain_t<> chain2;                               // unaligned: T0 * A_odd
      ra[0]=chain2.add(ra[0], c);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain2.madlo(a[index+1], t0, ru[index]);
        ru[index+1]=chain2.madhi(a[index+1], t0, ru[index+1]);
      }
      ru[LIMBS-LIMBS%2]=chain2.add(ru[LIMBS-LIMBS%2], 0);
        
      q=__shfl_sync(sync, ra[0], 0, TPI)*np0;

      chain_t<> chain3;                               // aligned:   Q0 * N_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain3.madlo(n[index], q, ra[index]);
        ra[index+1]=chain3.madhi(n[index], q, ra[index+1]);
      }
      t=__shfl_sync(sync, ra[0], threadIdx.x+1, TPI);
      if(LIMBS%2==0)
        ra[LIMBS]=chain3.add(ra[LIMBS], t);
      ra[LIMBS+1]=chain3.add(ra[LIMBS+1], 0);
        
      chain_t<> chain4;                               // unaligned: Q0 * N_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain4.madlo(n[index+1], q, ru[index]);
        ru[index+1]=chain4.madhi(n[index+1], q, ru[index+1]);
      }
      if(LIMBS%2==1) 
        ru[LIMBS-1]=chain4.add(ru[LIMBS-1], t);
      ru[LIMBS]=chain4.add(ru[LIMBS], 0);
            
      /* SECOND HALF */

      chain_t<> chain5;                               // unaigned: T1 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ru[index]=chain5.madlo(a[index], t1, ru[index]);
        ru[index+1]=chain5.madhi(a[index], t1, ru[index+1]);
      }
      if(LIMBS%2==0)
        ru[LIMBS]=chain5.add(ru[LIMBS], 0);
      
      chain_t<> chain6;                               // aligned:   T1 * A_odd
      ru[0]=chain6.add(ru[0], ra[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ra[index+2]=chain6.madlo(a[index+1], t1, ra[index+2]);
        ra[index+3]=chain6.madhi(a[index+1], t1, ra[index+3]);
      }
      if(LIMBS%2==1)
        ra[LIMBS+1]=chain6.add(ra[LIMBS+1], 0);
      
      q=__shfl_sync(sync, ru[0], 0, TPI)*np0;

      chain_t<> chain7;                               // unaigned: Q1 * N_even
      t=chain7.madlo(n[0], q, ru[0]);
      c=chain7.madhi(n[0], q, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain7.madlo(n[index+2], q, ru[index+2]);
        ru[index+1]=chain7.madhi(n[index+2], q, ru[index+3]);
      }
      t=__shfl_sync(sync, t, threadIdx.x+1, TPI);
      if(LIMBS%2==0)
        ru[LIMBS-2]=chain7.add(ru[LIMBS], t);
      ru[LIMBS-1]=chain7.add(0, 0);
      ru[LIMBS]=0;
            
      chain_t<> chain8;                               // aligned:   Q1 * N_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ra[index]=chain8.madlo(n[index+1], q, ra[index+2]);
        ra[index+1]=chain8.madhi(n[index+1], q, ra[index+3]);
      }
      if(LIMBS%2==1)
        ra[LIMBS-1]=chain8.add(ra[LIMBS+1], t);
      ra[LIMBS]=chain8.add(0, 0);
      ra[LIMBS+1]=0;
    }
  }
  
  chain_t<LIMBS+1> chain9;
  r[0]=chain9.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++) 
    r[index]=chain9.add(ra[index], ru[index-1]);
  r1=chain9.add(ra[LIMBS], ru[LIMBS-1]);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain10;
  r[0]=chain10.add(r[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain10.add(r[index], 0);
  c=chain10.add(r1, 0);

  c=-fast_propagate_add(c, r);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain11;
  r[0]=chain11.add(r[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain11.add(r[index], ~n[index] & c);
  c=chain11.add(0, 0);
  fast_propagate_add(c, r);
  clear_padding(r);
}
#endif

} /* namespace cgbn */
