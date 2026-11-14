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

namespace cgbn {

/****************************************************************************************************
 * One radix 2^64 step of word serial Montgomery multiplication, for even LIMBS and no padding:
 *
 *    (ra, ru, c) = ((ra, ru, c) + (t1:t0) * a + (q1:q0) * n) / 2^64
 *
 * The running value is kept in the same redundant form as in mont_mul below:  ra holds the aligned
 * words (ra[i] at word i), ru the unaligned words (ru[i] at word i+1) and c is a carry into ra[0].
 * q0 and q1 are chosen so that the two low words (on thread 0) become zero.  Each thread passes its
 * two low words down to the thread below; the top thread shifts in zeros.
 *
 * With multiply==false the (t1:t0) * a terms are skipped, which gives a Montgomery reduction step.
 * In that case a, t0 and t1 are not used.
 ****************************************************************************************************/
template<class env> template<bool multiply>
__device__ __forceinline__ void core_t<env>::mont_step(uint32_t ra[LIMBS+2], uint32_t ru[LIMBS+1], uint32_t &c, const uint32_t a[LIMBS], const uint32_t t0, const uint32_t t1, const uint32_t n[LIMBS], const uint32_t np0, const uint32_t sync) {
  uint32_t q, t;

  /* FIRST HALF */

  chain_t<> chain1;                                   // unaligned: T0 * A_odd
  ra[0]=chain1.add(ra[0], c);
  if(multiply) {
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index+=2) {
      ru[index]=chain1.madlo(a[index+1], t0, ru[index]);
      ru[index+1]=chain1.madhi(a[index+1], t0, ru[index+1]);
    }

    chain_t<> chain2;                                 // aligned:   T0 * A_even
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index+=2) {
      ra[index]=chain2.madlo(a[index], t0, ra[index]);
      ra[index+1]=chain2.madhi(a[index], t0, ra[index+1]);
    }
    ra[LIMBS]=chain2.add(ra[LIMBS], 0);
  }

  q=__shfl_sync(sync, ra[0], 0, TPI)*np0;

  // when not multiplying, this chain continues chain1, to pick up the carry from ra[0]+c
  chain_t<CGBN_INF_CHAIN, !multiply> chain3;          // unaligned: Q0 * N_odd
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index+=2) {
    ru[index]=chain3.madlo(n[index+1], q, ru[index]);
    ru[index+1]=chain3.madhi(n[index+1], q, ru[index+1]);
  }
  if(multiply)
    ru[LIMBS]=chain3.add(0, 0);                       // a reduction step can not carry out here

  chain_t<> chain4;                                   // aligned:   Q0 * N_even
  ra[0]=chain4.madlo(n[0], q, ra[0]);
  ra[1]=chain4.madhi(n[0], q, ra[1]);
  t=__shfl_sync(sync, ra[0], threadIdx.x+1, TPI);     // ra[0] is now zero on thread 0
  #pragma unroll
  for(int32_t index=2;index<LIMBS;index+=2) {
    ra[index]=chain4.madlo(n[index], q, ra[index]);
    ra[index+1]=chain4.madhi(n[index], q, ra[index+1]);
  }
  ra[LIMBS]=chain4.add(ra[LIMBS], t);
  ra[LIMBS+1]=chain4.add(0, 0);

  /* SECOND HALF */

  if(multiply) {
    chain_t<> chain5;                                 // unaligned: T1 * A_even
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index+=2) {
      ru[index]=chain5.madlo(a[index], t1, ru[index]);
      ru[index+1]=chain5.madhi(a[index], t1, ru[index+1]);
    }
    ru[LIMBS]=chain5.add(ru[LIMBS], 0);

    chain_t<> chain6;                                 // aligned:   T1 * A_odd
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index+=2) {
      ra[index+2]=chain6.madlo(a[index+1], t1, ra[index+2]);
      ra[index+3]=chain6.madhi(a[index+1], t1, ra[index+3]);
    }
  }

  chain_t<> chain7;                                   // aligned:   Q1 * N_odd
  ru[0]=chain7.add(ru[0], ra[1]);
  q=__shfl_sync(sync, ru[0], 0, TPI)*np0;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index+=2) {
    ra[index]=chain7.madlo(n[index+1], q, ra[index+2]);
    ra[index+1]=chain7.madhi(n[index+1], q, ra[index+3]);
  }
  if(multiply)
    ra[LIMBS]=chain7.add(0, 0);
  else
    ra[LIMBS]=0;                                      // a reduction step can not carry out here

  chain_t<> chain8;                                   // unaligned: Q1 * N_even
  t=chain8.madlo(n[0], q, ru[0]);
  c=chain8.madhi(n[0], q, ru[1]);
  t=__shfl_sync(sync, t, threadIdx.x+1, TPI);         // t was zero on thread 0
  #pragma unroll
  for(int32_t index=0;index<LIMBS-2;index+=2) {
    ru[index]=chain8.madlo(n[index+2], q, ru[index+2]);
    ru[index+1]=chain8.madhi(n[index+2], q, ru[index+3]);
  }
  ru[LIMBS-2]=chain8.add(ru[LIMBS], t);
  ru[LIMBS-1]=chain8.add(0, 0);
  ru[LIMBS]=0;
}

/****************************************************************************************************
 * Final step of a Montgomery multiplication.  On each thread, ra + r1 * 2^(32*LIMBS) holds the
 * thread's share of the (unresolved) result U, with r1 small and U < 2R.  Resolves the carries
 * between the threads and subtracts n if U >= R (almost Montgomery reduction).
 ****************************************************************************************************/
template<class env>
__device__ __forceinline__ void core_t<env>::mont_normalize(uint32_t r[LIMBS], uint32_t ra[LIMBS], uint32_t r1, const uint32_t n[LIMBS], const uint32_t sync) {
  const uint32_t TOP_THREADS=(0xFFFFFFFF/TPI_ONES)<<TPI-1;      // the top thread of each instance in the warp
  uint32_t       group_thread=threadIdx.x & TPI-1;
  uint32_t       t, c, vote;

  t=__shfl_up_sync(sync, r1, 1, TPI);

  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain1;
  ra[0]=chain1.add(ra[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chain1.add(ra[index], 0);
  c=chain1.add(r1, 0);

  // c is set if a carry leaves the thread or, on the top thread, if U >= R.  Unless the modulus uses all
  // of the bits, both are rare, so take one ballot across the warp and usually skip the rest.
  vote=__ballot_sync(sync, c!=0);
  if(vote!=0) {
    // A carry between threads is very rare.  Without one, only top threads have voted, and the
    // instance needs the subtraction if its top thread did.
    if((vote & ~TOP_THREADS)!=0)
      c=fast_propagate_add(c, ra);
    else
      c=(vote & instance_sync_mask())!=0;
    c=-c;

    // compute -n
    t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

    chain_t<LIMBS+1> chain2;
    ra[0]=chain2.add(ra[0], ~t & c);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      ra[index]=chain2.add(ra[index], ~n[index] & c);
    c=chain2.add(0, 0);
    fast_propagate_add(c, ra);
  }
  mpset<LIMBS>(r, ra);
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

  if(MONT_UNROLLED) {
    // even LIMBS, no padding, up to 4096 bits:  fully unrolled, with warp wide syncs
    sync=warp_sync_mask();
    #pragma unroll
    for(int32_t thread=0;thread<TPI;thread++) {
      #pragma unroll
      for(int32_t word=0;word<LIMBS;word+=2) {
        t0=__shfl_sync(sync, b[word], thread, TPI);
        t1=__shfl_sync(sync, b[word+1], thread, TPI);
        mont_step<true>(ra, ru, c, a, t0, t1, n, np0, sync);
      }
    }

    chain_t<LIMBS+1> chain0;
    ra[0]=chain0.add(ra[0], c);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      ra[index]=chain0.add(ra[index], ru[index-1]);
    r1=chain0.add(ra[LIMBS], ru[LIMBS-1]);

    mont_normalize(r, ra, r1, n, sync);
    return;
  }

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
        ra[LIMBS+1]=chain6.add(ra[LIMBS+1], 0);

      chain_t<> chain7;                               // aligned:   Q1 * N_odd
      ru[0]=chain7.add(ru[0], ra[1]);
      q=__shfl_sync(sync, ru[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<(int32_t)LIMBS-3;index+=2) {
        ra[index]=chain7.madlo(n[index+1], q, ra[index+2]);
        ra[index+1]=chain7.madhi(n[index+1], q, ra[index+3]);
      }
      ra[LIMBS-2-LIMBS%2]=chain7.madlo(n[LIMBS-1-LIMBS%2], q, ra[LIMBS-LIMBS%2]);
      ra[LIMBS-1-LIMBS%2]=chain7.madhi(n[LIMBS-1-LIMBS%2], q, ra[LIMBS+1-LIMBS%2]);
      if(LIMBS%2==1) {
        ra[LIMBS-1]=chain7.add(ra[LIMBS+1], 0);
        ra[LIMBS]=0;
      }
      else 
        ra[LIMBS-LIMBS%2]=chain7.add(0, 0);

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
  ra[0]=chain9.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++) 
    ra[index]=chain9.add(ra[index], ru[index-1]);
  r1=chain9.add(ra[LIMBS], ru[LIMBS-1]);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain10;
  ra[0]=chain10.add(ra[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chain10.add(ra[index], 0);
  c=chain10.add(r1, 0);

  c=-fast_propagate_add(c, ra);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain11;
  ra[0]=chain11.add(ra[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chain11.add(ra[index], ~n[index] & c);
  c=chain11.add(0, 0);
  fast_propagate_add(c, ra);
  clear_padding(ra);
  mpset<LIMBS>(r, ra);
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
  ra[0]=chain9.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++) 
    ra[index]=chain9.add(ra[index], ru[index-1]);
  r1=chain9.add(ra[LIMBS], ru[LIMBS-1]);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain10;
  ra[0]=chain10.add(ra[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chain10.add(ra[index], 0);
  c=chain10.add(r1, 0);

  c=-fast_propagate_add(c, ra);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain11;
  ra[0]=chain11.add(ra[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chain11.add(ra[index], ~n[index] & c);
  c=chain11.add(0, 0);
  fast_propagate_add(c, ra);
  clear_padding(ra);
  mpset<LIMBS>(r, ra);
}
#endif

} /* namespace cgbn */
