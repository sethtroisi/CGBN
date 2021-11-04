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

template<class env>
__device__ __forceinline__ void core_t<env>::mont_sqr(uint32_t &r, const uint32_t a, const uint32_t n, const uint32_t np0) {
  return mont_mul(r, a, a, n, np0);
}

/* single limb per thread version */
template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t n, const uint32_t np0) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t r0=0, r1=0, r2, t, c;

  // ME = threadIdx.x % TPI (via threadIdx.x & (TPI-1))
  // Seth: each thread has a[ME], b[ME], n[ME] (modulo), and the universal uint32 np0

  // Seth: See http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf page 49 (3 in pdf)
  // for imad, xmad, wmad (which are respectively i32, i16, i8)

  // Seth: Throughput numbers in
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-instruction-throughput
  // MY 1080ti is probably __CUDA_ARCH__ 610 => XMAD

  #pragma unroll
  for(int32_t thread=0;thread<TPI;thread++) {
    // Seth: At step [thread], everyone sets t = b[thread]] (broadcast from thread [thread])
    t=__shfl_sync(sync, b, thread, TPI);

    // Seth: multiply t = a[ME] * b[i], storing result back in a r0 (low), r1 (high), r2 (high-high = carry)
    r0=madlo_cc(a, t, r0);
    r1=madhic_cc(a, t, r1);
    r2=addc(0, 0);

    // Seth: Every thread gets running bottom sum
    t=__shfl_sync(sync, r0, 0, TPI)*np0;

    // r{2,1,0} += modulo * bottom limb * np0
    // modulo * np0 = -1 (mod limb)
    // so this subtracts the bottom limb (MAYBE this handles reducing r2?)
    //
    // See http://eprints.utar.edu.my/2494/1/CS-2017-1401837-1.pdf Page 29
    // s = a * b
    // s += (s * n' mod r) * n) / r
    //   s is stored in r{2,1,0}
    //   (s * n') mod r
    //          being computed via ripple adder here, (s*n') is determined solely by the bottom limp?
    //          (s * n') % r is stored in s
    //   s [r{2,1,0}] += (s * n' mod r) [t] * n
    // return s >= n ? s - n : s
    r0=madlo_cc(n, t, r0);
    r1=madhic_cc(n, t, r1);
    r2=addc(r2, 0);
    // TODO is there some crazy way of moving r2=addc(r2,0) after the sync and reorgarinzing the r1 = addc(r2,0)
    // so that we only do one final r1=addc(carry1, carry2)

    // shift right by 32 bits (top thread gets zero)
    // TODO should this be __shfl_down_sync (or does that not wrap and hence doesn't work)
    r0=__shfl_sync(sync, r0, threadIdx.x+1, TPI);
    r0=add_cc(r0, r1);
    r1=addc(r2, 0);
  }

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);

  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  r0=add_cc(r0, t);
  c=addc(r1, 0);

  c=fast_propagate_add(c, r0);

  // compute -n
  t=n-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
  t=~t & -c;

  r0=add_cc(r0, t);
  c=addc(0, 0);
  fast_propagate_add(c, r0);

  r=r0;
}

template<class env>
__device__ __forceinline__ void core_t<env>::mont_reduce_wide(uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t n, const uint32_t np0, const bool zero) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t r0=lo, r1=0, t, top;

  #pragma unroll
  for(int32_t thread=0;thread<TPI;thread++) {
    t=__shfl_sync(sync, r0, 0, TPI)*np0;
    r0=madlo_cc(n, t, r0);
    r1=madhic_cc(n, t, r1);

    // shift right by 32 bits (top thread gets zero)
    r0=__shfl_sync(sync, r0, threadIdx.x+1, TPI);
    if(!zero) {
      top=__shfl_sync(sync, hi, thread, TPI);
      r0=(group_thread==TPI-1) ? top : r0;
    }

    // add it in
    r0=add_cc(r0, r1);
    r1=addc(0, 0);
  }

  r1=fast_propagate_add(r1, r0);

  if(!zero && r1!=0) {
    // compute -n
    t=n-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
    t=~t & -r1;

    r0=add_cc(r0, t);
    r1=addc(0, 0);
    fast_propagate_add(r1, r0);
  }
  r=r0;
}

} /* namespace cgbn */

