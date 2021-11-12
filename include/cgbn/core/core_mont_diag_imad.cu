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

#include <assert.h>

namespace cgbn {

template<class env>
__device__ __forceinline__ void core_t<env>::mont_sqr(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  //return mont_mul(r, a, a, n, np0);

  //assert(LIMBS > 1);
  // TODO validate this works for n == 0, 1, 2, 3
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t x[LIMBS+1];
  uint32_t t, q, c;
  uint32_t x1=0;

  mpzero<LIMBS+1>(x); // Partial row sum

  // OHHH NO, the zero term doing inv_two means carry into the other threads!
  // Can potentially handle in each thread group?

  // If a is odd, summation starts as modular_inverse(2, N)
  // a[0] * a[0] >> (1) needs inv_two
  // Alternatively this can be handled by adding r_inv at the end
  t = __shfl_sync(sync, a[0], 0, TPI);
  if (t & 1) {
    // Set x = 2^-1 mod N = N / 2 + 1

    // Bit from above (or zero if top thread)
    q = __shfl_down_sync(sync, n[0] & 1, 1, TPI);
    q = q * (group_thread != (TPI-1));

    // Add 1 if bottom group_thread
    chain_t<LIMBS+1> chain1;
    x[0] = chain1.add((n[0] >> 1) | (n[1] << 31), group_thread == 0);

    #pragma unroll
    for(int32_t index=1;index<LIMBS-1;index++) {
      x[index] = chain1.add(n[index] >> 1, n[index+1] << 31);
    }
    if (LIMBS >= 2) {
      // TODO ACCOUNT FOR PAD LIMB ???
      x[LIMBS-1] = chain1.add(n[LIMBS-1] >> 1, q << 31);
    }
    x[LIMBS] = chain1.add(x[LIMBS], 0);
  }

  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread++) {
    // Maybe should break this into three groups (pre, thread=thread_group  post)

    for(int word=0;word<LIMBS;word++) {
      t=__shfl_sync(sync, a[word], thread, TPI);

      // Handle adding the shifted down square_low bit from thread above us when word == 0
      if (word == 0) {
        chain_t<2> chain1;
        x[LIMBS-1] = chain1.add(x[LIMBS-1], ((group_thread + 1 == thread) && (t & 1)) << 31);
        x[LIMBS] = chain1.add(x[LIMBS], 0);
      }

      if (thread == group_thread) {
        // First handle diagonal term, then proceed like normal

        // TODO should be LIMBS+1-word, should be using INF CHAIN to avoid check?
        chain_t<LIMBS+1> chain1;

        // t = a[word] because we are on thread == group_thread
        // Add a[word] * a[word] >> 1
        // group_thread == 0 and word == 0 handled with terrible inv_two block
        if ((t & 1) && (word > 0)) {
            x[word-1] = chain1.add(x[word-1], 1u << 31);
        }
        uint32_t square_high = __umulhi(t, t);
        uint32_t square_low = t * t;
		    square_low = (square_low >> 1) | (square_high << 31); // "|" is the same as "+" here
        square_high >>= 1;

        x[word] = chain1.add(square_low, x[word]);
        // low 32 bits of a[index] * t
        #pragma unroll
        for(int32_t index=word+1;index<LIMBS;index++) {
          x[index]=chain1.madlo(a[index], t, x[index]);
        }
        x[LIMBS]=chain1.add(x[LIMBS], 0);

        // TODO should be LIMBS+1-word
        chain_t<LIMBS+1> chain2;
        x[word+1] = chain2.add(square_high, x[word+1]);
        // High 32 bits of a[index] * t
        #pragma unroll
        for(int32_t index=word+1;index<LIMBS;index++) {
          x[index+1]=chain2.madhi(a[index], t, x[index+1]);
        }
        x1=chain2.add(0, 0);
      } else if (thread > group_thread) {
        // Multiply this threads section of a * a[thread * TPI + word]

        // Low 32 bits of a[index] * t
        chain_t<LIMBS+1> chain1;
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++) {
          x[index]=chain1.madlo(a[index], t, x[index]);
        }
        x[LIMBS]=chain1.add(x[LIMBS], 0);

        // High 32 bits of a[index] * t
        chain_t<LIMBS+1> chain2;
        #pragma unroll
        for(int32_t index=0;index<LIMBS;index++) {
          x[index+1]=chain2.madhi(a[index], t, x[index+1]);
        }
        x1=chain2.add(0, 0);
      }

      // s += (s * n' mod r) * n) / r
      // ^ this is all XXX bits of r, instead we do 32 bits at a time.
      // in fast_squaring.py I do summation = (summation + q * n) >> 32

      q=__shfl_sync(sync, x[0], 0, TPI)*np0;
      //if (group_thread==0) printf("%u,%u | %#x\n", thread, word, q);

      // If inv_two is moved to "+ r_inv" at the end, then this can be
      // skipped when thread < group_thread (as x will be 0)
      chain_t<LIMBS+2> chain3;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++) {
        x[index]=chain3.madlo(n[index], q, x[index]);
      }

      // Getting the bottom chunk from above us and continue the reduce
      t=__shfl_sync(sync, x[0], threadIdx.x+1, TPI);
      x[LIMBS]=chain3.add(x[LIMBS], t);
      x1=chain3.add(x1, 0);

      chain_t<LIMBS+1> chain4;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index++)
        x[index]=chain4.madhi(n[index], q, x[index+1]);
      x[LIMBS]=chain4.add(x1, 0);
      x1=0;
    }
  }

  /* Consider handling inv_two here by adding r_inv
   * 1. r_inv is not calculated yet.
   * 2. would avoid need for reduce in group_thread < thread
   */

  { // Resolve lazy carry limb
    // x[LIMB]...x[0] <= 0x00000002 0xFFFFFFFD
    t=__shfl_up_sync(sync, x[LIMBS], 1, TPI);
    assert(t <= 2);

    // all but most significant thread clears x[LIMB]
    x[LIMBS]=x[LIMBS]*(group_thread==TPI-1);
    // least significant limb doesn't have carry in.
    t=t*(group_thread>0);

    chain_t<LIMBS+1> chain1;
    x[0]=chain1.add(x[0], t);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      x[index]=chain1.add(x[index], 0);
    c=chain1.add(x[LIMBS], 0);
    // Handled by [c]arry from here on.
    x[LIMBS] = 0;

    /*
    if (group_thread==0) printf("FIRST\n");
    for (int32_t s=TPI-1; s>=0; s--)
        if (group_thread == s) {
            printf("%d,%d | c: %u \n", group_thread, LIMBS, c);
            for(int32_t index=LIMBS-1;index>=0;index--)
              printf("%d,%d | %#x\n", group_thread, index, x[index]);
        }
    // */

    // compute and add -n if carry out from resolve lazy carry
    c=-fast_propagate_add(c, x);

    //if (group_thread==0) printf("c: %u\n", c);
    {
      t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

      chain_t<LIMBS+1> chain2;
      x[0]=chain2.add(x[0], ~t & c);
      #pragma unroll
      for(int32_t index=1;index<LIMBS;index++)
        x[index]=chain2.add(x[index], ~n[index] & c);
      c=chain2.add(0, 0);
      c=-fast_propagate_add(c, x);
      // Shouldn't have x - n >= R
      assert(c == 0);
    }
  }

  // x = 2*x
  {
    chain_t<LIMBS+2> chain1;
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++) {
      r[index]=chain1.add(x[index], x[index]);
    }
    c=chain1.add(0, 0);

    // compute and add -n if carry out from doubling
    /*
    if (group_thread==0) printf("DOUBLED\n");
    for (int32_t s=TPI-1; s>=0; s--)
        if (group_thread == s) {
            printf("%d,%d | c: %u \n", group_thread, LIMBS, c);
            for(int32_t index=LIMBS-1;index>=0;index--)
              printf("%d,%d | %#x\n", group_thread, index, r[index]);
        }
    */

    // Propogate carry up, and determine if carry out
    c=-fast_propagate_add(c, r);
    //if (group_thread==0) printf("c: %u\n\n", c);

    q=0;  // count of reducotions
    while (c) {
      q++;
      t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
      chain_t<LIMBS+1> chain2;
      r[0]=chain2.add(r[0], ~t & c);
      #pragma unroll
      for(int32_t index=1;index<LIMBS;index++)
        r[index]=chain2.add(r[index], ~n[index] & c);
      c=chain2.add(0, 0);
      c=-fast_propagate_add(c, r);
    }
    // Sometimes 2!
    if (group_thread==0 && q>2) printf("reduces: %u\n", q);

    // In the case that R > x > n, then 2*x - n can STILL be greater than R
    // Need either R > n > x, then n > 2*x - n

  }
  clear_padding(r);
}

} /* namespace cgbn */
