import math
import random

def to_mont(a, r, n):
    return a * r % n

def from_mont(a, r_inv, n):
    return a * r_inv % n

def mont_sqr_test(a, r, r_inv, n):
    return a * a * r_inv % n

def mont_sqr_redc(a, r, n_inv, n):
    t = a * a
    # Division here is "easy"
    u = (t + ((t * n_inv) % r) * n ) // r
    if u >= n:
        return u - n
    return u

def chunk(t, min_length=0):
    BLOCK = 2**32 - 1
    # How many 32 bit chunks in r
    bits = n.bin_length()
    blocks = (bits-1)/32 + 1
    # low 32, 33-64, 65-96, ...
    A = [(a >> (32*i)) & BLOCK for i in range(min(blocks, min_length)]
    return blocks, A

def mont_sqr_block1(a, r, r_inv, n):
    blocks, Ai = chunk(a)

    summation = 0
    for i in range(0, blocks):
        for j in range(0, blocks):
            summation += Ai[i] * Ai[j] << (32 * (i+j))

    return summation * r_inv % n

def mont_sqr_block_cios(a, n, r, block_inv):
    blocks, Ai = chunk(a)

    summation = 0
    for i in range(0, blocks):
        for j in range(0, blocks):
            summation += Ai[i] * Ai[j] << (32 * j)

        # one round of reduce
        #   m = summation[0] * n'[0] mod 2^32
        #   summation[j-1] = summation[j] + m * n[j]
        m = -(summation * block_inv) % (2 ** 32)
        summation = (summation + m * n) >> 32

    # Final reduction by block_inv
    if summation < 0:
        summation += n

    if summation >= n:
        summation -= n

    assert 0 <= summation < n, summation
    return summation

def mont_sqr_block_cios_fast(a, n, r, block_inv):
    blocks, Ai = chunk(a)

    summation = 0
    for i in range(0, blocks):
        # Is there some clever way to add this to summation without doubling
        # Or to handle the double carry from?
        temp = 0
        for j in range(i+1, blocks):
            temp += Ai[i] * Ai[j] << (32 * j)

        square_term = Ai[i] * Ai[i] << (32 * i)
        summation += 2 * temp + square_term

        # one round of reduce
        #   m = summation[0] * n'[0] mod 2^32
        #   summation[j-1] = summation[j] + m * n[j]
        m = -(summation * block_inv) % (2 ** 32)
        summation = (summation + m * n) >> 32

    if summation < 0:
        summation += n

    # from 2 * temp
    if summation > n:
        summation -= n

    assert 0 <= summation < n
    return summation

def mont_sqr_block_cios_fast_inplace(a, n, r, block_inv):
    blocks, Ai = chunk(a)

    summation = 0
    if a & 1:
        # inv_two for i = 0
        summation = N // 2 + 1
        assert summation * 2 % N == 1
    #print(hex(summation & (2**64-1)))

    for i in range(0, blocks):
        #summation, low_bit = divmod(summation, 2)

        square_term = Ai[i] * Ai[i] << (32 * i)
        # This could ripple carry the whole length of summation
        summation += (square_term >> 1)
        # Can maybe handle that by also passing array of (limb = 0xFFFF)
        #   Carry is only now reduced to computing a 1032/32bit carry = 32bit carry maybe
        #

        #print(i, "\t", hex(Ai[i]), "\tsquare", hex((Ai[i] * Ai[i] >> 33)), hex((Ai[i] * Ai[i] >> 1) & BLOCK))

        # Is there some clever way to add this to summation without doubling
        # Or to handle the double carry from?
        for j in range(i+1, blocks):
            summation += Ai[i] * Ai[j] << (32 * j)

        # one round of reduce
        #   m = summation[i] * n'[0] mod 2^32
        #   summation[j-1] = summation[j] + m * n[j]
        m = -(summation * block_inv) % (2 ** 32)
        print(f"{i}\t0\t: {summation >> 32 & BLOCK:08x} {summation & BLOCK:08x}\t{m:#x}")
        summation = (summation + m * n) >> 32
        #print("\t0\t", hex(summation >> 32 & BLOCK), hex(summation & BLOCK))
        #print()

    print("---")
    print(summation)
    print(hex(summation))
    for c in reversed(list(map(hex, chunk(summation)[1]))): print("\t", c)

    if summation >= R:
        summation -= n

    print("after sub")
    print(summation)
    print(hex(summation))


    summation *= 2
    # TODO could also handle inv_two logic here
    #if a & 1:
    #    summation += R_inv

    print("after double")
    print(summation)
    print(hex(summation))

    # from 2 * temp
    if summation > n:
        summation -= n

    print("after 2nd sub")
    print(summation)
    print(hex(summation))
    for c in reversed(list(map(hex, chunk(summation)[1]))): print("\t", c)

    #if summation > n:
    #    summation -= n

    #assert 0 <= summation < n, (summation, summation / n)
    return summation


def mont_sqr_block_cios_fast_two_stage(a, n, r, block_inv, n_inv):
    blocks, Ai = chunk(a)

    summation = 0
    for i in range(0, blocks):
        #summation, low_bit = divmod(summation, 2)

        # Is there some clever way to add this to summation without doubling
        # Or to handle the double carry from?
        for j in range(0, i):
            summation += Ai[i] * Ai[j] << (32 * j)

        # one round of reduce
        #   m = summation[0] * n'[0] mod 2^32
        #   summation[j-1] = summation[j] + m * n[j]
        #m = -(summation * block_inv) % (2 ** 32)
        summation_low = summation & 0xFFFFFFFF
        assert block_inv <= 0xFFFFFFFF
        m = -((summation_low * block_inv) & 0xFFFFFFFF)

        summation = (summation + m * n) >> 32

    if summation < 0:
        summation += n
    if summation >= n:
        summation -= n

    # Double non-diagonal part of summary
    summation <<= 1  # summation *= 2

    if summation >= n:
        summation -= n
    assert 0 <= summation < n

    # Trace of limbs
    square_sum = 0
    for i in range(0, blocks):
        square_sum += Ai[i] * Ai[i] << (32 * (i + i))

    t = square_sum
    # TODO(2025): This feels like one round of REDC reduce?
    u1 = (t + ((t * n_inv) % r) * n ) // r
    if u1 >= n:
        u1 -= n
    assert 0 <= u1 < n

    m = (u1 & 0xFFFFFFFF) * block_inv

    summation += u1
    if summation >= n:
        summation -= n

    #print("summation  :", summation)
    #print("       % n :", summation % n)

    return summation


def mont_mul_2026_gpu_code(a, n, r, block_inv, n_inv):
    """This is my attempt to model what the GPU code is doing in each step"""
    blocks, Ai = chunk(a)
    Bi = Ai

    np0 = block_inv
    assert 0 <= np0 < 0xFFFFFFFF
    assert (np0 * n) & 0xFFFFFFFF == 1

    TPI = 8
    LIMBS = (blocks + TPI-1) / TPI
    print(blocks, TPI, LIMBS)

    sync_state = [None for i in range(TPI)
    for thread in range(TPI):
        for word in range(LIMBS)
            # This loop is implicit in GPU code.
            for gpu_thread in range(TPI):
                t = Bi[thread*LIMBS + word]
                x = [0 for i in range(LIMBS+1)]
                x1 = 0 # x[LIMBS+1]
                carry = 0
                for index in range(LIMBS)
                    multiply_lo = a[index] * t + x[index] + carry
                    carry = multiply_lo >> 32
                    assert 0 <= carry <= 1
                    x[index] += (multiply_lo & 0xFFFFFFFF)

                x[LIMBS] += carry

                carry = 0
                for index in range(LIMBS):
                    multiply_hi = ((a[index] * t) >> 32) + x[index] + carry
                    carry = multiply_hi >> 32
                    assert 0 <= carry <= 1
                    x[index+1] += (multiply_hi & 0xFFFFFFFF)
                x1 = carry

                # all syncronize on q = x[0]
                sync_state[gpu_thread] = (x, x1)

            for gpu_thread in range(TPI)
                q = sync_state[0][0] * np0
                x, x1 = sync_state[gpu_thread]

                carry = 0
                for index in range(LIMBS):
                    multiply_lo = n[index] * q + carry
                    carry = multiply_lo > 0xFFFFFFFF
                    x[index] += (multiply_lo & 0xFFFFFFFF)

                # sync on t = x[0] a second time
                sync_state[gpu_thread] = (x, x1, q, carry)

            for gpu_thread in range(TPI)
                t = sync_state[0][0] * np0
                x, x1, q, carry = sync_state[gpu_thread]
                partial = x[LIMBS] + t
                carry = partial >> 32
                assert 0 <= carry <= 1
                x[LIMBS] = partial & 0xFFFFFFFF
                x1 = x1 + carry

                carry = 0
                for index in range(LIMBS):
                    multiply_hi = ((n[index] * q) >> 32) + x[index+1] + carry
                    carry = multiply_hi >> 32
                    assert 0 <= carry <= 1
                    x[index] += (multiply_hi & 0xFFFFFFFF)
                x[LIMBS] = x1 + carry

                sync_state[gpu_thread] = (x, x1)

    # Have to resolve lazy carry in two stages
    for gpu_thread in range(TPI)
        # Get carry from lower thread
        t = sync_state[max(gpu_thread-1, 0)][0][LIMBS]
        x, x1 = sync_state[gpu_thread]

        # Clear the spilled carry limb (except top thread)
        x[LIMBS] *= (gpu_threada == TPI-1)

        # Bottom threads doesn't need carry in
        t = t * (group_thread > 0)

        # Carry up our limbs
        add = x[0] + t
        carry = add > 0xFFFFFFFF
        r[0] = add & 0xFFFFFFFF
        for index in range(1, LIMBS):
            add = x[index] + carry
            carry = add > 0xFFFFFFFF
            r[index] = add & 0xFFFFFFFF
        c = x[LIMBS] + carry
        x[LIMBS] = 0

        # Fast propagate add is magic which sucks that I have to implement it here
        sync_state[gpu_thread] = (x, r)



def  mont_sqr_block_cios_2026(a, n, r, block_inv, n_inv):
    blocks, Ai = chunk(a)

    # Ai shifted over 1, reduced mod n
    t = (a << 1)
    if t > r:
        t -= n
    blocks2, Ai_shifted = chunk(t)
    if blocks2 < blocks:
        Ai_shifted.extend([0] * (blocks2 - blocks))

    summation = 0
    for i in range(0, blocks):
        for j in range(0, i):
            summation += Ai[i] * Ai_shifted[j] << (32 * j)

        summation += Ai[i] * Ai[i] << (32 * i)

        # one round of reduce
        #   m = summation[0] * n'[0] mod 2^32
        #   summation[j-1] = summation[j] + m * n[j]
        #m = -(summation * block_inv) % (2 ** 32)
        summation_low = summation & 0xFFFFFFFF
        assert block_inv <= 0xFFFFFFFF
        m = -((summation_low * block_inv) & 0xFFFFFFFF)

        summation = (summation + m * n) >> 32
        print(i, summation.bit_length(), summation)

    if summation < 0:
        summation += n
    if summation >= n:
        summation -= n

    assert 0 <= summation < n

    return summation



random.seed(1)


N = 0xfffadcafd43b06fb42ec7cb0585c4e2ad78fa438d36b7796b9ea4adcc67e2a97823e5f01
assert N % 2 == 1

# TODO(2025) What is this?
R = 2 ** (9*32)
R_inv = pow(R, -1, N)
#print ("R_inv: ", R_inv)
assert R > N

BLOCK = 0xFFFFFFFF
N_block_inv = pow(N, -1, 2 ** 32) # Same as (N_inv & BLOCK)
#print ("np0: ", N_block_inv)

assert R * R_inv % N == 1
assert (N * N_block_inv) % (2 ** 32) == 1

N_inv = pow(-N, -1, R)
assert R_inv * R - N * N_inv == 1

if True:
    As = [0xedd01ca6843b0429deaac33a40fb26fa78febb43f902481a7fdba45c89ec1b188d3c8f54]
else:
    As = [random.randrange(0, N) for i in range(100)]

for A in As:
    RESULT = A * A % N

    A_mont = to_mont(A, R, N)
    RESULT_mont = to_mont(RESULT, R, N)

    print()
    T1 = mont_sqr_test(A_mont, R, R_inv, N)
    T2 = mont_sqr_redc(A_mont, R, N_inv, N)
    T3 = mont_sqr_block1(A_mont, R, R_inv, N)
    print()
#    T4 = mont_sqr_block_cios(A_mont, N, R, N_block_inv)
#    T5 = mont_sqr_block_cios_fast(A_mont, N, R, N_block_inv)
    T6 = mont_sqr_block_cios_fast_inplace(A_mont, N, R, N_block_inv)
    T7 = mont_sqr_block_cios_fast_two_stage(A_mont, N, R, N_block_inv, N_inv)
    T8 = mont_sqr_block_cios_2026(A_mont, N, R, N_block_inv, N_inv)

    print()
    print(hex(RESULT_mont), "=")
    print(" ", hex(from_mont(RESULT_mont, R_inv, N)))
    print("-" * len(hex(RESULT_mont)))
    print(hex(T1))
#    print(hex(T2))
#    print(hex(T3))
#    print(hex(T4))
#    print(hex(T5))
    print(hex(T6))
    print(hex(T7))
    assert T1 == T2 and T1 == T3
#    assert T1 == T4
#    assert T1 == T5
#    assert T1 == T6
    assert T1 == T7

