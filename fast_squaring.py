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

def chunk_a(a, r):
    BLOCK = 2**32 - 1
    # How many 32 bit chunks in r
    bits = len(bin(r)) - 3
    blocks, test = divmod(bits, 32)
    assert test == 0, bits

    # low 32, 33-64, 65-96, ...
    A = [(a >> (32*i)) & BLOCK for i in range(blocks)]
    return blocks, A

def mont_sqr_block1(a, r, r_inv, n):
    blocks, Ai = chunk_a(a, r)

    summation = 0
    for i in range(0, blocks):
        for j in range(0, blocks):
            summation += Ai[i] * Ai[j] << (32 * (i+j))

    return summation * r_inv % n

def mont_sqr_block_cios(a, n, r, block_inv):
    blocks, Ai = chunk_a(a, r)

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
    blocks, Ai = chunk_a(a, r)

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
    blocks, Ai = chunk_a(a, r)

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
    for c in reversed(list(map(hex, chunk_a(summation, r)[1]))): print("\t", c)

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
    for c in reversed(list(map(hex, chunk_a(summation, r)[1]))): print("\t", c)

    #if summation > n:
    #    summation -= n

    #assert 0 <= summation < n, (summation, summation / n)
    return summation


def mont_sqr_block_cios_fast_two_stage(a, n, r, block_inv, n_inv):
    blocks, Ai = chunk_a(a, r)

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
        m = -(summation * block_inv) % (2 ** 32)
        summation = (summation + m * n) >> 32

    if summation > n:
        summation -= n
    summation *= 2
    if summation >= n:
        summation -= n
    assert 0 <= summation < n

    square_sum = 0
    for i in range(0, blocks):
        square_sum += Ai[i] * Ai[i] << (32 * (i + i))

    t = square_sum
    u = (t + ((t * n_inv) % r) * n ) // r
    if u >= n:
        u -= n

    summation += u
    if summation >= n:
        summation -= n

    #print("summation  :", summation)
    #print("       % n :", summation % n)

    return summation

random.seed(1)


N = 0xfffadcafd43b06fb42ec7cb0585c4e2ad78fa438d36b7796b9ea4adcc67e2a97823e5f01
assert N % 2 == 1

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

for _ in range(1):
    A = random.randrange(0, N)
    # TODO hardcoded for now
    A = 0xedd01ca6843b0429deaac33a40fb26fa78febb43f902481a7fdba45c89ec1b188d3c8f54
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
#    T7 = mont_sqr_block_cios_fast_two_stage(A_mont, N, R, N_block_inv, N_inv)

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
#    print(hex(T7))
    assert T1 == T2 and T1 == T3
#    assert T1 == T4
#    assert T1 == T5
#    assert T1 == T6
#    assert T1 == T7

