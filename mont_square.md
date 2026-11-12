# Fast Montgomery Squaring (`mont_sqr`) Design & Implementation Journal

## 0. Executive Summary (`TPI = 8, BITS = 1024` on NVIDIA L4 `sm_89`)

CGBN's stock `cgbn_env_t::mont_sqr(r, a, n, np0)` originally delegated directly to general CIOS Montgomery multiplication (`mont_mul(r, a, a, n, np0)`), executing **`256` 64-bit wide multiplies (`IMAD.WIDE.U32`)** and **`736` dynamic SASS instructions** per lane at `TPI = 8, LIMBS = 4` (`1024`-bit Montgomery radix $R = 2^{1024}$). Across 12 iterative optimization phases and 4 independent algorithmic branches starting from `mont_square_starting_point` (`cdc1411`), we designed, formally verified (`26.21M` live-asserted GPU squarings, `compute-sanitizer --tool synccheck`, 27 CGBN configurations, and `102,400` GMP checks), SASS-dissected, and benchmarked dedicated symmetric `mont_sqr` kernels for `TPI = 8` (`<= 1020`-bit redundant Montgomery inputs).

### Headline Results & Branch Comparison (`NVIDIA L4`, `102,400` instances, `1018`-bit inputs)

| Git Branch (`commit`) | Core Algorithmic Idea | Regs (SM Occ.) | Wide Muls | Fast-Loop SASS Insts | `256`-Iter Avg ms (M sqr/s) | `256`-Iter Best ms (M sqr/s) | `20`-Iter Unthrottled Avg / Best (M sqr/s) |
|---|---|---:|---:|---:|---:|---:|---:|
| `mont_square_starting_point` (`cdc1411`) | Baseline CIOS `mont_mul(r, a, a, n, np0)` | 42 (`83.3%`) | 256 | 736 | `27.564–30.244` (`866.8–951.0`) | `23.545–28.023` (`935.5–1113.4`) | — |
| `dual-8x2-sos-redc` (`3fe3a65`) | `194`-wide Separated Dual-`8x2` SOS + Carry-Save 1024b REDC | 62 (`66.7%`) | **194** | 826 | `31.298` (`837.57`) | `29.051` (`902.36`) | — |
| `hensel64-lookahead-sqr` (`f789316`) | Half-Split CIOS + Radix-$2^{64}$ Hensel Parallel Quotients `(q0, q1)` | 45 (`83.3%`) | 217 | 661 | `26.052` (`1006.25`) | `22.182` (`1181.79`) | — |
| `halfsplit-cios-iter11` (`2f1553a`) | Half-Split CIOS + `8x2` $a_{\text{hi}}^2$ + `activemask` Width-8 Shuffles | 47 (`83.3%`) | 209 | 651 | `25.000` (`1048.56`) | `21.520` (`1218.12`) | — |
| **`lean-halfsplit-pred` (`ba9cdef`)** | **Unified `Even+Mid` + 2-Reg Packed LUTs + Hoisted Predicates** | **48 (`83.3%`)** | **209** | **632** | **`22.021–24.756` (`1058.9–1190.4`)** | **`19.634–20.180` (`1299.0–1335.1`)** | **`1.546 / 1.542 ms` (`1324.7 / 1328.0`)** |

### Four Decisive Architectural Findings on Ada (`sm_89`)
1. **Hybrid Half-Split Integrated CIOS Beats Pure Separated SOS (`632` vs. `826` SASS Instructions)**:
   Decomposing $a = a_{\text{lo}} + 2^{512} a_{\text{hi}}$ as $a^2 = a_{\text{lo}}(a_{\text{lo}} + 2^{513} a_{\text{hi}}) + 2^{1024} a_{\text{hi}}^2$ folds $a_{\text{lo}}^2 + 2^{513} a_{\text{lo}} a_{\text{hi}}$ directly into `mont_mul`'s `ra/ru` CIOS inner loop (`thread = 0, 2` only, `123` incremental instructions = `1.92` instructions per 64-bit multiply-accumulate, **zero** transpose shuffles, **zero** intermediate carry resolution), followed by `8` pair-steps of zero-top reduction (`175` instructions) and a zero-transpose `8 x 2` symmetric square for $a_{\text{hi}}^2$ (`17` wide muls/lane). Pure separated `194`-wide SOS (`dual-8x2-sos-redc`) saves `15` additional wide multiplies (`209 -> 194`) at the cost of `+194` `SHFL`/`SEL`/`IADD3` instructions and `+14` registers to explicitly route $a_{\text{lo}}^2$ and $2 a_{\text{lo}} a_{\text{hi}}$ across lanes.
2. **Unified `Even + Mid` Butterfly Exchange & Zero-Transpose `8 x 2` Folded Target-Sum $a_{\text{hi}}^2$**:
   Repacking the 16-word $a_{\text{hi}}$ across all 8 lanes as `(d0, d1)` reduces $a_{\text{hi}}^2$ to `17` wide multiplies/lane whose 4-word output `(E0..E3)` lands directly on words `4g..4g+3` of $a_{\text{hi}}^2 \cdot R$. Merging the disjoint `!keep_e` (`{0, 7}`) and `Middle 2x1` (`{1..6}`) partner exchanges into a single 5-word `shfl.sync.bfly` (`xor 4`) plus a carry-free chained `mad_wide` for `2x1` eliminated an entire shuffle round and duplicate `IMAD.HI` instructions.
3. **L4 Dual-Regime Physics & 2-Register Bit-Packed LUT Hoisting (`REG = 48` = `83.3%` Occupancy AND `632` Insts)**:
   Forward+Reverse (`ABBA`) sweeps revealed two distinct regimes on NVIDIA L4 (`72W` TDP): boost clock (`~2040 MHz`, `20` iters or Run 1 of `256` iters, reaching **`1.30–1.335 G sqr/s` = `0.755 ns/sqr`**) requires **`REG <= 48`** (`40` warps/SM = `83.3%` occupancy), whereas `72W` power-capped steady state (`Runs 3–5` of `256` iters) scales directly with dynamic instruction energy. Packing all 8 3-bit lane LUT indices, `d_src`, `sh_amt`, and `prev_lane` into **2 hoisted GPRs** (`lut_o, lut_e`, exploiting `SHFL.IDX`'s hardware `src_lane & 7` masking via `c = 0x181f`) while letting all 6 `bool` lane predicates hoist freely (`0` `LOP3`s in the loop) achieved both `REG = 48` and `632` dynamic instructions (`-14.1%` vs `mont_mul`).
4. **Eliminating `sm_70+` Subwarp Convergence, Ballot, & Carry-Resolution Traps**:
   Passing `activemask.b32` to width-8 PTX shuffles (`0x181f` / `0x1800` / `0x1c00`) replaced `ptxas`'s 6-instruction `MATCH.ANY + REDUX.OR` convergence stub with a 1-cycle `BRA.DIV UR4` header; `activemask` balloting fixed `__ballot_sync(sync, ...)`'s 4-way serialized `WARPSYNC` trap; `__any_sync(shfl_mask, c != 0u)` skips final subwarp carry propagation in normal operation ($P < 5 \times 10^{-38}$); and Walter's `< 1020`-bit redundant Montgomery invariant ($U < 2n < R$) eliminates the final conditional `- n` pass completely.

*(See [Section 7](#7-new-directions-for-possible-optimizations) at the end of this document for a detailed roadmap of next-generation optimization directions, including subquadratic Toom-Cook squaring, Örup modulus pre-scaling, ILP=2 multi-instance interleaving, Ada INT8 Tensor Core limb convolution, and IEEE-754 mantissa splitting.)*

---

## 1. Overview & Mathematical Contract

In CGBN, `cgbn_env_t::mont_sqr(r, a, n, np0)` in [`include/cgbn/impl_cuda.cu`](include/cgbn/impl_cuda.cu) previously fell back directly to general CIOS Montgomery multiplication:
```cpp
cgbn::core_singleton_t<cgbn_env_t, LIMBS>::mont_mul(r._limbs, a._limbs, a._limbs, n._limbs, np0);
```
Because CIOS (`mont_mul`) interleaves multiplication and modular reduction row-by-row, $r$ already contains $q \cdot n$ mixtures after the first step, breaking the operand symmetry $a_i \cdot a_j = a_j \cdot a_i$. Exploiting $a = b$ requires **separating** full-width squaring ($S = a^2$) from Montgomery reduction ($U = (S + Q \cdot n) / R$).

### Exact CGBN `mont_sqr` Specification
From the reference CPU implementation in [`include/cgbn/impl_mpz.cc`](include/cgbn/impl_mpz.cc) and unit tests in [`unit_tests/tests/mont_sqr.h`](unit_tests/tests/mont_sqr.h):
- Let $T = \text{TPI} \in \{4, 8, 16, 32\}$, $L = \text{LIMBS} = \lceil (\text{BITS}/32) / T \rceil$, and $W = T \cdot L$.
- The Montgomery radix is **always** $R = 2^{32 W} = 2^{\text{UNPADDED\_BITS}}$, even when `PADDING != 0` (`BITS < UNPADDED_BITS`).
- Given any $0 \le a < 2^{\text{BITS}}$ (note: CGBN allows $a \ge n$) and odd $n < 2^{\text{BITS}}$ with $\text{np0} = -n_0^{-1} \bmod 2^{32}$:
  1. Let $Q \in [0, R - 1]$ be the **unique** integer satisfying $a^2 + Q \cdot n \equiv 0 \pmod R$.
  2. Let $U = (a^2 + Q \cdot n) / R$. Note $0 \le U < \frac{2^{2 \cdot \text{BITS}} + R \cdot n}{R}$.
  3. **Almost Montgomery final step**: If bit `BITS` of $U$ is `1` (i.e., carry out of `BITS`), subtract $n$ once: $r = (U - n) \bmod 2^{\text{BITS}}$; otherwise $r = U$. (Crucially, $U = n$ is **not** reduced to $0$, matching `mont_mul` bit-for-bit.)
- Because $Q \equiv -a^2 \cdot n^{-1} \pmod R$ is algebraically unique in $\mathbb{Z}/R\mathbb{Z}$, separated $S = a^2$ followed by $W$-word Montgomery reduction produces the **exact same pre-subtraction integer $U$** and final output $r$ as CIOS `mont_mul(a, a, n, np0)` for every valid input $(a, n)$ and every `(BITS, TPI, PADDING)` configuration.

Only focus on the TPI=8 case. It's okay to require a small number of upper 0 bits in the input (e.g. <1020 bit numbers).

Use `colab_benchmark.sh` to test results
  * `./colab_benchmark.sh --gpu L4 --csv results.csv -t 8 --tag "<idea being tested>"`

---

## 2. Baseline Performance on NVIDIA L4 (`ada` / `sm_89`)

Benchmark workload (`samples/sample_05_mont_sqr/mont_sqr.cu`): `102,400` instances $\times$ `256` chained squarings of `1018`-bit integers in a `1024`-bit CGBN environment (`26.2144M` modular squarings per run, 5 timed runs):

| Version | `TPI` | `LIMBS` | Avg Time (ms) | Best Time (ms) | Avg Throughput (M sqr/s) | Best Throughput (M sqr/s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (`mont_mul`, `a24fdbf`) | 8 | 4 | 27.564 | 23.545 | 951.04 | 1113.38 |
| Baseline (`mont_mul`, `0fdd4f3`) | 8 | 4 | 30.244 | 28.023 | 866.76 | 935.47 |
| Baseline (`mont_mul`, `a24fdbf`) | 16 | 2 | 40.027 | 37.646 | 654.92 | 696.33 |
| Baseline (`mont_mul`, `0fdd4f3`) | 16 | 2 | 39.250 | 32.230 | 667.88 | 813.34 |
| Baseline (`mont_mul`, `a24fdbf`) | 32 | 1 | 57.136 | 56.672 | 458.81 | 462.56 |
| Baseline (`mont_mul`, `0fdd4f3`) | 32 | 1 | 57.422 | 56.731 | 456.52 | 462.09 |

*(Note: CLI flag `-a` runs all TPIs `{8, 16, 32}`; `-t 8` runs `TPI=8` alone.)*

At `TPI=8, LIMBS=4` (`W = 32` words), L4 (58 SMs @ ~2.04 GHz) executes `23.545 ms` $\approx$ **106 SM-cycles per 1024-bit modular squaring** (across 4 instances per warp = ~424 warp-cycles). In `core_mont_wmad.cu::mont_mul`, each thread executes:
- $32 \times 4 = 128$ `IMAD.WIDE.U32`s for $a \times b$, plus
- $32 \times 4 = 128$ `IMAD.WIDE.U32`s for $Q \times n$,
- $= 256$ `IMAD.WIDE.U32`s per thread (~64% of all SM integer pipe cycles on Ada, where `IMAD.WIDE` has throughput 2/cycle/sub-partition = half of 32-bit `IADD3`/`SHFL`).

---

## 3. Optimization Ideas for Fast `mont_sqr`

### Idea 1: Avoiding the "SIMT Zero-Padding Trap" via Folded Target-Sum Block Squaring
In a `TPI`-thread subwarp executing in SIMT lockstep:
- A naive row-scanning upper-triangle loop (`for r = 0..T-1`, multiplying by `0` on lanes $g < r$) still executes all $T$ iterations across the warp and saves **zero** `IMAD.WIDE` instructions.
- Instead, partition $a$ into $T$ thread blocks $A^{(0)}, \dots, A^{(T-1)}$ (each of $L$ limbs) and classify all $\binom{T}{2}$ off-diagonal block pairs $(i, j)$ ($0 \le i < j < T$) by their **block-sum** $s = i + j \in [1, 2T - 3]$.
- Pair lane $t \in [0, H-1]$ ($H = T/2$) with lane $t + H \in [H, T-1]$:
  1. **Complementary Odd Sums $\{2t + 1, \, 2t + T + 1\}$**:
     - Sum $2t + 1$ has $t + 1$ pairs; sum $2t + T + 1$ has $H - 1 - t$ pairs.
     - Their combined count is **identically $(t + 1) + (H - 1 - t) = H$ pairs for every $t$**!
     - Splitting $H$ evenly gives **exact $H/2 = T/4$ full $L \times L$ block products per thread** (`2` per thread at `TPI=8`), enumerated in closed form for $p \in [0, H-1]$:
       $$(i, j) = \begin{cases} (p, \; 2t + 1 - p) & \text{if } p \le t \quad (\text{target sum } 2t+1\text{, owned by lane } t), \\ (t + 1 + p, \; T + t - p) & \text{if } p > t \quad (\text{target sum } 2t+T+1\text{, owned by lane } t+H). \end{cases}$$
     - Lane $t$ runs $p = 0 \dots H/2 - 1$ and lane $t + H$ runs $p = H - 1 \dots H/2$. At most one of $\{t, t + H\}$ crosses the threshold $t$, so a single `__shfl_xor_sync(..., H, TPI)` exchanges any partner partial sum with **zero bank/lane conflicts**.
  2. **Complementary Even Sums $\{2t, \, 2t + T\}$**:
     - Sum $2t$ has $t$ pairs; sum $2t + T$ has $H - 1 - t$ pairs.
     - Their combined count is **identically $H - 1$ pairs for every $t$**!
     - Enumerated in closed form for $p \in [0, H - 2]$:
       $$(i, j) = \begin{cases} (p, \; 2t - p) & \text{if } p < t \quad (\text{target sum } 2t\text{, owned by lane } t), \\ (t + 1 + p, \; T - 1 + t - p) & \text{if } p \ge t \quad (\text{target sum } 2t+T\text{, owned by lane } t+H). \end{cases}$$
     - Each thread computes $T/4 - 1$ full $L \times L$ block products (`1` at `TPI=8`), and for the middle pair $p = H/2 - 1$, lane $t$ multiplies $A^{(i)}$ by the low $\lfloor L/2 \rfloor$ limbs of $A^{(j)}$ while lane $t + H$ multiplies $A^{(i)}$ by the high $\lceil L/2 \rceil$ limbs of $A^{(j)}$!
  3. **Zero-Shuffle Local Intra-Block Products ($A^{(g)} \times A^{(g)}$)**:
     - Every thread $g$ computes its own $\binom{L}{2}$ intra-block off-diagonal products $a_u \cdot a_v$ ($0 \le u < v < L$) directly into its even-sum accumulator `(2g, 2g+1)` with **0 shuffles**.
     - After merging odd + even off-diagonal sums and doubling (`<< 1`), thread $g$ adds its $L$ diagonal squares $a_u^2$ ($u = 0 \dots L-1$) into contiguous non-overlapping 64-bit slots `(E[2u], E[2u+1])` in a single `IMAD.WIDE` chain with **0 shuffles**.

**Instruction Count Payoff at `TPI=8, LIMBS=4` (1024-bit)**:
- Off-diagonal inter-block: $2 \times 16 + 1.5 \times 16 = 56$ `IMAD.WIDE`s/thread.
- Off-diagonal intra-block: $\binom{4}{2} = 6$ `IMAD.WIDE`s/thread.
- Diagonal squares: $4$ `IMAD.WIDE`s/thread.
- **Total squaring `IMAD.WIDE`s = `66` per thread** (down from `128`, hitting the exact theoretical lower bound $\binom{33}{2} / 8 = 528 / 8 = 66.0$ with **zero idle lanes**!).
- **Total `mont_sqr` `IMAD.WIDE`s = `66 + 128 = 194`** (a **24.2% reduction** in 64-bit multiply-accumulate work).

### Idea 2: WMAD Even/Odd (`ra`/`ru`) Carry Chains & Compile-Time Register Indexing
- On Volta+ (`XMP_WMAD`, including Ada `sm_89`), `ptxas` fuses `mad.lo.cc.u32` + `madc.hi.cc.u32` into a single `IMAD.WIDE.U32` only when they pair on alternating words along the same carry chain.
- Structure every $L \times L$ and $L \times (L/2)$ block product using aligned (`ra`) and unaligned (`ru`) chains, and keep all array indices strictly compile-time constants (`#pragma unroll`) so `nvcc` allocates 100% of state in registers with zero local-memory spills.

### Idea 3: Single Deferred Multi-Bit Carry Resolution for $a^2$
- Rather than resolving carries across threads after every row, accumulate multi-bit carries (`<= TPI`) in a single scalar register per thread during Even and Odd accumulation, shift `(E[0..2L-1], carry)` left by 1 bit for doubling, add diagonals `a[u]^2`, and resolve carries across `TPI` threads in **one** parallel carry-propagation step.

### Idea 4: Zero-Shuffle `hi` Handoff into Dedicated `mont_reduce_wide`
- At the end of Folded Target-Sum squaring, lane $g \in [0, T - 1]$ naturally holds `E[0..2L-1]` $= (B_{2g}, B_{2g+1})$ of $a^2$:
  - Lanes $0 \dots H-1$ hold blocks $B_0 \dots B_{T-1}$ (`lo`), scattered to `ra[0..L-1]` on lanes $0 \dots T-1$.
  - Lanes $H + p$ ($p = 0 \dots H-1$) **already** hold $(B_{T+2p}, B_{T+2p+1})$ in `E[0..2L-1]`, which is the exact 2-thread chunk `(hi on lane 2p, hi on lane 2p+1)` consumed by iteration $p$ of WMAD Montgomery reduction!
- Thus `hi` needs **zero rearrangement shuffles**, and inside the reduction loop `top` is fetched via a single `__shfl_sync(sync, E[l], TPI/2 + p, TPI)` with no `zero` check and no `if (l < LIMBS)` branch.

### Idea 5: Uniform Fast-Path Final Subtraction (`if (CONSTANT_TIME || c != 0)`)
- In `mont_mul`, lines 180–190 unconditionally execute bitwise `-n` masking, an $L$-limb addition chain, and a second `fast_propagate_add` (`mpland` + two `__ballot_sync`s + carry ripple).
- Whenever $n < R/4$ (e.g., all 1018-bit moduli in a 1024-bit env: $U = (a^2 + Qn)/R < (2^{2036} + 2^{1024}\cdot 2^{1018})/2^{1024} < 2^{1019} \ll 2^{1024}$), the carry `c` out of bit `BITS` is **0 across all instances in the warp**.
- Guarding the subtraction with `if (!CONSTANT_TIME && c == 0)` (or `if (CONSTANT_TIME || c != 0)`, matching `core_mont_wmad.cu:293`) turns into a single uniform warp branch that skips the entire final subtraction pass in the common case while remaining 100% exact whenever `c != 0` (including full 1024-bit moduli and `a >= n` edge cases).

### Idea 6: Unrolling the Outer Reduction Loop (`#pragma unroll`)
- `core_mont_wmad.cu` uses `#pragma nounroll` on `for (thread = 0; thread < TPI; thread += 2)`.
- When `TPI * LIMBS <= 64` (e.g., 1024-bit and 2048-bit), unrolling the outer `TPI/2` loop (`4` iterations at `TPI=8`) makes all `__shfl_sync` source lanes (`TPI/2 + p`) compile-time immediates, eliminates loop overhead, and lets `ptxas` hoist shuffles ahead of `IMAD.WIDE` chains to hide MIO latency — while letting `E[0..2L-1]` registers die progressively as `p` advances.

### Idea 7: Dedicated Single-Limb (`LIMBS == 1`) Fast `mont_sqr` (`TPI=32` at 1024-bit)
- For `LIMBS == 1` ([`include/cgbn/core/core_mont.cu`](include/cgbn/core/core_mont.cu)), implement a dedicated folded symmetric squaring (`TPI/2` multiply-adds instead of `TPI`) feeding directly into single-limb Montgomery reduction.

### Idea 8: Half-Split Integrated CIOS + Zero-Transpose `8 x 2` Symmetric `a_hi^2` (`TPI=8, LIMBS=4`)
- Write $a = a_{\text{lo}} + a_{\text{hi}} \cdot R_{1/2}$ ($R_{1/2} = 2^{512}$, $R = R_{1/2}^2 = 2^{1024}$):
  $$a^2 = a_{\text{lo}} \cdot \big(a_{\text{lo}} + 2 \, a_{\text{hi}} \cdot R_{1/2}\big) \;+\; a_{\text{hi}}^2 \cdot R.$$
- Notice **three** decisive microarchitectural wins at `TPI = 8, LIMBS = 4` (`H = 4` threads = `16` words):
  1. **Zero-Overhead Symmetric Cross-Product Inside CIOS**: Forming $V = a_{\text{lo}} + 2 a_{\text{hi}} R_{1/2}$ (`2` shuffles, `~10` ALU ops) lets us run `mont_mul`'s exact `ra/ru` CIOS inner loop for **only the first half of threads (`thread = 0, 2` — 2 iterations instead of 4)**, followed by **2 iterations of Zero-Top Reduction (`wmad_mont_reduce_zero_pair`)** on the exact same `(ra, ru, c)` state! That computes $(a_{\text{lo}} V + Q n) / R$ with **zero** `pa/pu` merge overhead, **zero** `SEL`s, **zero** `shfl_xor` partner exchanges, **zero** transpose shuffles, and **`<= 42` registers (`100%` SM occupancy)**.
  2. **Zero-Transpose `8 x 2` Symmetric `a_hi^2`**: Repacking the 16-word $a_{\text{hi}}$ across all `8` lanes as `2` limbs `(d0, d1)` per lane (`4` shuffles) reduces $a_{\text{hi}}^2$ to `TPI = 8, L_sub = 2` Folded Target-Sum squaring (**only `17` `IMAD.WIDE`s** per thread, `2` `IADD3`s per `2x2` multiply), and its 4-word output `E[0..3]` on lane $g \in [0, 7]$ lands **directly on words `4g .. 4g+3` of $a_{\text{hi}}^2$ with zero transpose shuffles**!
  3. **Merged Carry Resolution**: Since $a_{\text{hi}}^2 \cdot R$ has weight $R$, `E[0..3]` and its multi-bit carry `c_E` add directly into post-reduction `(ra[0..3], r1)` and resolve in the single final `fast_propagate_add`, eliminating the intermediate `a^2` carry-resolve pass (`2` `VOTE`s + `18` `IADD3`s) completely.

### Idea 9: Spare-Bit / Redundant Montgomery Form (`< 1020`-Bit Inputs, Walter 1999)
- When $n < R/4$ (e.g. `< 1020`-bit inputs in a `1024`-bit CGBN environment, where $n < 2^{1020} = R/16$), any input $0 \le a < 2n$ satisfies:
  $$U = \frac{a^2 + Q \cdot n}{R} < \frac{4 n^2 + R \cdot n}{R} = n\left(1 + \frac{4n}{R}\right) < \frac{5}{4} n < 2n < 2^{1021} \ll R.$$
- Thus:
  1. $2 \cdot a_{\text{hi}} < 2^{509} < 2^{512}$ **never overflows** $R_{1/2}$ (`v_top == 0` identically).
  2. $U < 2n$ **never requires the final conditional `- n` subtraction** inside chained Montgomery squarings (`mont2bn` at the end of exponentiation reduces $(U + Q' n)/R \le n$ canonically to $[0, n-1]$), eliminating `~22` post-reduction instructions (including `2` `__ballot_sync` votes and an entire carry chain) with zero warp divergence.

---

## 4. Implementation & Benchmark Journal

### Planned Workflow
1. **Phase 1 (Correctness-First Slow Version + Heavy Assertions)**:
   - Plumb `mont_sqr` through `impl_cuda.cu`, `core_singleton.cu`, `core.cu`, `core_mont.cu`, `core_mont_wmad.cu`, `core_mont_imad.cu`, and `core_mont_xmad.cu`.
   - Implement a clear, heavily-asserted symmetric squaring + Montgomery reduction (verifying odd `n`, `n0 * np0 == -1`, padding invariants, off-diagonal upper-triangle doubling + diagonal equivalence against `mul_wide`, reduction divisibility `(low + q * n0) == 0` at every step, and final output equality against reference `mont_mul`).
   - Run both `colab_benchmark.sh` (`TPI = 8`) and CGBN `unit_tests` on Colab L4.
2. **Phase 2 (Iterative Performance Tuning on L4)**:
   - Disable debug assertions for release builds (`#ifdef CGBN_DEBUG_MONT_SQR`) and implement/benchmark each optimization step, recording SASS/register findings and L4 timings below.

### Entry 1 — Phase 1: Initial Slow Symmetric `mont_sqr` with Heavy Assertions
- **Goal**: Establish a numerically verified, heavily-asserted separated symmetric-squaring + Almost-Montgomery-reduction implementation across both single-limb (`core_mont.cu`) and multi-limb (`core_mont_wmad.cu`) paths before performance tuning.
- **What was implemented**:
  - Plumbed `mont_sqr` dispatch through [`impl_cuda.cu`](include/cgbn/impl_cuda.cu), [`core_singleton.cu`](include/cgbn/core/core_singleton.cu), [`core.cu`](include/cgbn/core/core.cu), [`core_mont.cu`](include/cgbn/core/core_mont.cu), [`core_mont_wmad.cu`](include/cgbn/core/core_mont_wmad.cu), [`core_mont_imad.cu`](include/cgbn/core/core_mont_imad.cu), and [`core_mont_xmad.cu`](include/cgbn/core/core_mont_xmad.cu).
  - Computes $a^2$ using strictly upper-triangle off-diagonal products (`row_word < col_word`), doubles via 1-bit left shift (`<< 1`), and adds contiguous local diagonal squares `a[u] * a[u]`.
  - Live GPU `assert(...)` checks active (`-O3` without `-DNDEBUG`):
    1. `(n0 & 1) == 1` and `n0 * np0 == 0xFFFFFFFF`.
    2. Input and output padding limbs equal `0` when `PADDING != 0`.
    3. Off-diagonal top carry `== 0`, word-0 off-diagonal `== 0`, and off-diagonal MSB `== 0` prior to doubling.
    4. Exact limb-by-limb equality of `(lo, hi)` against `core_t<unpadded_t<env>>::mul_wide(a, a)`.
    5. Exact divisibility (`u[0] == 0` on lane 0) after adding `q * n` at every single reduction word step.
    6. Carry-out `c_final` in `{0, 1}` and exact limb-by-limb equality of final `r` against reference `core_t<env>::mont_mul(a, a, n, np0)`.
- **L4 Benchmark & Verification (`./colab_benchmark.sh --gpu L4 --csv results.csv -a`)**:
  - `TPI=8` (`LIMBS=4`): **PASS** (`131.774 ms` avg / `131.487 ms` best, `198.94 M/s` — includes running `mul_wide` + `mont_mul` reference cross-checks and live device `assert`s inside every squaring!)
  - `TPI=16` (`LIMBS=2`): **PASS** (`195.055 ms` avg / `192.522 ms` best, `134.39 M/s`)
  - `TPI=32` (`LIMBS=1`): **PASS** (`247.883 ms` avg / `244.313 ms` best, `105.75 M/s`)
- **27-Configuration Adversarial Test Suite (All GPU `assert`s Active)**:
  - Tested `BITS` $\in \{32, 64, 96, 128, 160, 192, 256, 288, 384, 512, 544, 576, 768, 1024, 1056, 1536, 2048, 3072, 4096, 8192\}$ across `TPI` $\in \{4, 8, 16, 32\}$ (including padded widths, odd `LIMBS`, `0xFFFFFFFF` carry-ripple patterns, `a == 0`, `a >= n`, and chained in-place aliasing `mont_sqr(r, r, n, np0)`): **27/27 PASS**.

### Entry 2 — Phase 2 Iteration 1: Initial Folded Target-Sum (`ea`/`eu`/`ca` Multi-Accumulator) & SASS Diagnosis
- **Goal**: Replace the Phase 1 upper-triangle row scan with the Folded Target-Sum symmetric schedule (`K = TPI/4` complementary Even/Odd block sums + split middle pair + dedicated `mont_reduce_wide` consuming `E[0..2L-1]` in place), guarding debug `assert`s behind `#ifdef CGBN_DEBUG_MONT_SQR`.
- **Correctness**:
  - `./colab_benchmark.sh --gpu L4 --csv results.csv -a`: **PASS** across `TPI=8, 16, 32`.
  - 27-configuration adversarial test suite with `-DCGBN_DEBUG_MONT_SQR`: **27/27 PASS**.
- **L4 Timings (`db6d6bded9`)**:
  - `TPI=8`: `37.062 ms` avg / `33.785 ms` best (`707.32 M/s` avg, `775.92 M/s` best) — **slower** than baseline `mont_mul` (`27.56 ms` avg / `23.55 ms` best).
  - `TPI=16`: `57.155 ms` avg / `56.758 ms` best.
  - `TPI=32`: `84.640 ms` avg / `84.385 ms` best.
- **`ptxas` & `cuobjdump -sass` Root-Cause Diagnosis (`sm_89`)**:
  1. **Multiply Reduction Worked**: `IMAD.WIDE.U32(.X)` dropped to `182` in `kernel_mont_sqr<TPI=8>` (down from ~`256` in `mont_mul`).
  2. **Non-Multiply Instruction Explosion**: Keeping separate aligned/unaligned accumulators `(ea[2L], eu[2L-2], ca[2L+1])` plus `send[2L]` and `cur[2L]` emitted **`385` `IMAD.MOV.U32`**, **`244` `SEL`**, and **`208` `IADD3.X`** instructions! At `TPI=8` (`K = 2`, only 1–2 block products per sum), zeroing 23 accumulator words and running a 15-add `resolve_acc` + predicated `send`/`own` copies 5 times per squaring added ~700 non-multiply instructions.
  3. **Register Pressure (`92 registers` on `TPI=8`, `89 registers` on `TPI=32`)**: Holding `ea`, `eu`, `ca`, `E`, `send`, `O`, and `cur` live simultaneously pushed `TPI=8` to 92 registers/thread (`TPB=128`), cutting Ada SM occupancy to 5 blocks/SM (`41.7%`), and kept `#pragma nounroll` on the outer reduction loop.

### Entry 3 — Phase 2 Iteration 2: Flat `wmad_sqr_mul_block` + Unrolled Reduction & Chain-Length Fix
- **What was implemented**:
  - Replaced `ea`/`eu`/`ca`/`send` with flat `wmad_sqr_mul_block` (`prod[LIMBS+B]`) and unrolled outer reduction when `TPI * LIMBS <= 64`.
  - Caught and fixed a subtle `chain_t` length bug in the middle-split pair (`ch_lo` declared as `chain_t<LIMBS+M_LO+1>` instead of `chain_t<2*LIMBS+1>`, which omitted `.cc` on limb `LIMBS+M_LO` and left a stale `CC.CF=1` carry into `c_E`).
- **L4 Timings (`cdc14119fa-dirty`)**:
  - `TPI=8`: **PASS**, `34.515 ms` avg / `31.825 ms` best (`63` registers).
  - `TPI=16`: **PASS**, `58.480 ms` avg / `58.322 ms` best (`42` registers).
  - `TPI=32`: **PASS**, `99.080 ms` avg / `98.855 ms` best (`32` registers).
- **SASS Comparison vs. Baseline `mont_mul` (`cuobjdump -sass` & `nvdisasm -g`)**:
  - Baseline `mont_mul` executes **`736` dynamic instructions** at `TPI=8` (`42` regs: `256` `IMAD.WIDE`, `96` `SHFL`), **`643` dynamic instructions** at `TPI=16` (`32` regs: `144` `IMAD.WIDE`, `108` `SHFL`), and **`321` dynamic instructions** at `TPI=32` (`27` regs: `65` `IMAD.WIDE`, `96` `SHFL`).
  - Three SASS bottlenecks found in Iteration 2:
    1. **Misaligned 64-bit Register-Pair Repacking (`667` `IMAD.MOV.U32` + `503` `MOV` at `TPI=8`)**: Interleaving even-offset and odd-offset `IMAD.WIDE.U32` chains into one flat `prod[]` array forced `ptxas` to emit 3–4 register-repacking `MOV`s around every `IMAD.WIDE`.
    2. **Intra-Subwarp Divergent `if (!is_partner)` / `if (owns_mid)` Around `asm volatile`**: Emitted `BSSY`/`BSYNC` serialization barriers.
    3. **I-Cache Blowout**: Unrolling `p = 0..H-1` inflated `TPI=8` to `2,928` static instructions (`46.8 KB` > L4's `32 KB` L1 I-cache).

### Entry 4 — Phase 2 Iteration 3: Aligned `pa`/`pu` Pairs + Branchless Accumulation & Register-Hoisting Diagnosis
- **What was implemented**:
  - Separated aligned (`pa[]`) and unaligned (`pu[]`) 64-bit register pairs inside `wmad_sqr_mul_block` and `wmad_sqr_intra_offdiag_init`, cutting squaring `IMAD.MOV.U32` from **`667` down to `29`**, and squaring `IMAD.WIDE` to **`62`** (vs. `128` in `mont_mul` — a `51.6%` multiply reduction).
  - Replaced all intra-subwarp `if`s with branchless dual-select `(E, EP)` / `(O, OP)` accumulation and `#pragma nounroll` on `p = 0..H-1`.
- **L4 Timings**:
  - `TPI=8`: **PASS**, `36.245 ms` avg / `33.816 ms` best (`84` registers -> `50%` SM occupancy!).
  - `TPI=16`: **PASS**, `58.202 ms` avg / `57.160 ms` best (`78` registers).
  - `TPI=32`: **PASS**, `84.042 ms` avg / `83.775 ms` best (`77` registers).
- **Exact `nvdisasm -g` Root-Cause Findings Driving Iteration 4**:
  1. **LLVM Shuffle Hoisting (`+30` Live Registers)**: Because `__shfl_sync` is non-volatile while `chain_t` is `asm volatile`, `cicc` hoisted all `30` `ai, aj` shuffles across Even, Middle, and Odd (`s = 0..K-1`) to the top of the loop (`R50..R81`), pushing registers from `~50` to `84` (`TPI=8`), `78` (`TPI=16`), and `77` (`TPI=32`), halving SM occupancy to `50%`! Fixed via `asm volatile ("shfl.sync.idx.b32 ...")` (`sqr_shfl_idx`).
  2. **`warpSize` Is Not `constexpr` at `TPI=32`**: `(TPI == warpSize)` did not constant-fold `sync` to `0xFFFFFFFFu`, leaving `147` subwarp-convergence stubs at `TPI=32`. Fixed via `(TPI == 32) ? 0xFFFFFFFFu : sync_mask()`.
  3. **Decoupled Zero-Top Montgomery Reduction ($U = S_{\text{hi}} + (S_{\text{lo}} + Q n)/R$)**: Since $S = S_{\text{hi}} R + S_{\text{lo}} \equiv S_{\text{lo}} \pmod{R}$, the quotient $Q = (-S_{\text{lo}} n^{-1}) \bmod R$ is independent of $S_{\text{hi}}$! Running reduction on $S_{\text{lo}}$ with `zero = true` (leveraging thread 0's `ra[0] == 0` wraparound into thread `TPI-1`) eliminates **all `32` `top` shuffles + `32` `SEL`s** inside the reduction loop and frees `E[0..2L-1]` before reduction begins, adding $S_{\text{hi}}$ once at the end!

### Entry 5 — Phase 2 Iteration 4: Decoupled Zero-Top Montgomery Reduction & LICM Register Diagnosis
- **What was implemented**:
  - Replaced full `mont_reduce_wide` (`zero = false`) with **Decoupled Zero-Top Montgomery Reduction** ($U = S_{\text{hi}} + (S_{\text{lo}} + Q n)/R$), eliminating all `top` shuffles and `SEL`s inside the reduction loop.
  - Constant-folded `sync` at `TPI = 32` (`0xFFFFFFFFu`) and tested `sqr_shfl_idx` (`asm volatile` shuffle wrapper).
- **L4 Timings**:
  - `TPI=8`: **PASS**, `35.022 ms` avg / `32.218 ms` best (`88` registers, `968` dynamic instructions).
  - `TPI=16`: **PASS**, `53.885 ms` avg / `53.246 ms` best (`75` registers, `896` dynamic instructions).
  - `TPI=32`: **PASS**, `73.385 ms` avg / `71.542 ms` best (`93` registers, `515` dynamic instructions).
- **SASS Diagnosis Driving Iteration 5**:
  1. **Why Registers Reached `88 / 75 / 93` (`41.7%`–`50%` SM Occupancy)**: In `mont_sqr.cu`, `cgbn_mont_sqr` runs inside a `#pragma nounroll` 256-iteration loop. Because `threadIdx.x` is `readnone`, LLVM LICM hoisted 35–60 loop-invariant `i_lane`, `j_lane`, `is_partner`, and mask registers above the 256-iteration loop! Meanwhile, `sqr_shfl_idx` (`asm volatile`) prevented `ptxas` register coalescing. Fix: revert to `__shfl_sync` and insert zero-instruction `asm volatile ("" : "+r"(group_thread))` barriers to defeat LICM and cross-section shuffle hoisting with **zero** emitted SASS instructions.
  2. **Predicated PTX Accumulation (`@!q` / `@q`)**: Verified on `sm_89` that predicated inline PTX `add.cc.u32` / `addc.cc.u32` / `addc.u32` compiles directly to `@!P IADD3(.X)` / `@P IADD3(.X)` with **zero** `BSSY`/`BSYNC` barriers and **zero** `SEL`s (cutting `wmad_sqr_accum_dual` from `36` to `19` instructions at `LIMBS=4`, and eliminating `117` `SEL`s at `TPI=32`).
  3. **Run Odd Before Even**: Computing Odd first (`O` dies before `E` is allocated) cuts live registers by `2 * LIMBS` and replaces `ch_merge1` + `ch_merge2` (`14` `IADD3`s) with a `LIMBS + 1` initialization chain.
  4. **Unroll Zero-Top Reduction (`TPI * LIMBS <= 64`) + Uniform Fast-Skip Carry Resolve (`g_mask != 0`)**: Eliminates `108` loop-phi `MOV`/branch instructions at `TPI=8` and `22` carry-propagation instructions on the fast path.

### Entry 6 — Phase 2 Iteration 5: Anti-LICM Barrier + Odd-First Ordering & `@q` Convergence-Stub Discovery
- **What was implemented**:
  - Replaced `sqr_shfl_idx` with `__shfl_sync` + zero-instruction `asm volatile ("" : "+r"(group_thread))` anti-LICM barriers, reordered Odd before Even (`-2 * LIMBS` live registers), unrolled `wmad_mont_reduce_zero_pair`, reduced transpose to `3 * LIMBS` shuffles, and tested `@!q` / `@q` predicated inline PTX accumulation + `if (CONSTANT_TIME || g_mask != 0)`.
- **L4 Timings (`cdc14119fa-dirty`)**:
  - `TPI=8`: **PASS**, **`30.819 ms` avg / `27.238 ms` best** (`850.60 M/s` avg, `962.41 M/s` best; registers dropped from **`88 -> 57`**, lifting occupancy from `41.7% -> 66.7%`!).
  - `TPI=16`: **PASS**, `49.408 ms` avg / `48.250 ms` best (`43` registers, down from `75`).
  - `TPI=32`: **PASS**, `70.815 ms` avg / `70.652 ms` best (`36` registers, down from `93`).
- **Critical SASS Discovery (`nvdisasm -g`)**:
  - Although `@!q` / `@q` inside `asm volatile` emitted predicated `@P IADD3`, **`ptxas` treats any `asm volatile` containing `@pred` (and any mid-loop `if` branch such as `if (g_mask != 0)` or `if (c_final != 0)`) as potentially diverging the warp**, invalidating subwarp convergence tracking!
  - Consequently, `ptxas` injected **6 extra `MATCH.ANY.SYNC.B32 + REDUX.OR + VOTEU.ANY + LOP3 + BRA.DIV + BSYNC` convergence blocks** inside `TPI=8` (vs. **only 1** outside the 256-iteration loop in `mont_mul`), costing ~`96` MIO clocks + `42` ALU ops per squaring and inflating static code to `2,496` insts (`39.9 KB > 32 KB` L1 I-cache).
  - Removing `@q` inline PTX and data-dependent `if` branches restores pure straight-line warp convergence (Iteration 6), and combining Half-Split CIOS + Zero-Transpose `8 x 2` `a_hi^2` (Idea 8) + Spare-Bit Redundant Montgomery (Idea 9) targets `< 42` registers (`100%` occupancy) and `< 600` dynamic instructions at `TPI = 8`.

### Entry 7 — Phase 2 Iteration 6 (`iter6-branchless-sos`): Catching LLVM Short-Circuit `&&` / Nested `? :` `BSSY` Divergence
- **What was implemented**:
  - Replaced `@!q` / `@q` inline PTX with C++ ternaries, eliminated `OP[]` in `K == 2` Odd block-sums, removed the redundant `intra[]` temporary in `wmad_sqr_intra_offdiag_add`, and merged middle-split `lo_src` + `hi_src` into a single addition chain.
- **L4 Timings (`tag: iter6-branchless-sos`)**:
  - `TPI=8`: **PASS**, `34.757 ms` avg / `32.736 ms` best (`58` registers, `1,044` fast-loop instructions).
- **Exact SASS Root-Cause Discovery (`cuobjdump -sass`)**:
  - Inspecting the fast-loop SASS revealed **10 `BSSY + BRA + BSYNC + MATCH.ANY + REDUX.OR + VOTEU.ANY + LOP3 + BRA.DIV` subwarp-convergence blocks** (`~160` MIO clocks + `~120` extra instructions)!
  - **Root cause**: In LLVM (`cicc`), C++ logical `&&` (`group_thread != 0u && group_thread != 6u`, and `land == 0xFFFFFFFF && group_thread != 0` inside `subwarp_resolver.cu::fast_propagate_add`) and nested `? :` (`is_g7 ? O[k] : (accum_p1 ? 0u : P1[k])`) lower to short-circuit `br i1` basic blocks (`BSSY` / `@!P BRA` / `BSYNC`) rather than `SEL`, invalidating `ptxas`'s subwarp-convergence tracking before every subsequent `__shfl_sync`!
  - **Golden Rule for `sm_89` Subwarp Shuffles**: Never use `&&`, `||`, or nested `? :` before `__shfl_sync`; use bitmask extraction (`(0xBEu >> group_thread) & 1u`) and flat single-level `? :` (`SEL`).

### Entry 8 — Phase 2 Iteration 7 (`iter7-halfsplit-cios-8x2`): Half-Split Integrated CIOS + Zero-Transpose `8 x 2` `a_hi^2`
- **What was implemented**:
  - Added dedicated `wmad_mont_sqr_tpi8_l4` for `TPI == 8, LIMBS == 4, PADDING == 0` implementing Idea 8:
    1. Forms $V = a_{\text{lo}} + 2 a_{\text{hi}} \cdot 2^{512}$ and runs 2 pairs (`thread = 0, 2`) of integrated CIOS (`wmad_cios_pair_tpi8_l4`) followed by 2 pairs of Zero-Top reduction (`wmad_mont_reduce_zero_pair`).
    2. Repacks $a_{\text{hi}}$ across all 8 lanes as 2 limbs `(d0, d1)` per lane and computes $a_{\text{hi}}^2$ via `8 x 2` Folded Target-Sum (`17` `IMAD.WIDE`s) directly in final lane order `4g .. 4g+3` with **zero transpose shuffles** and merged post-reduction carry propagation.
    3. Fixed a carry-chain typo in the `8 x 2` diagonal square step (`madlo_cc` -> `madloc_cc` on `d1 * d1`).
- **L4 Timings (`tag: iter7-halfsplit-cios-8x2`)**:
  - `TPI=8`: **PASS**, **`28.882 ms` avg / `25.319 ms` best** (`907.64 M/s` avg, `1035.35 M/s` best; `48` registers = `83.3%` SM occupancy, `774` fast-loop instructions, **`0` inner `BSSY` blocks**).
- **Three Major SASS Discoveries (`cuobjdump -sass` on `iter7`)**:
  1. **`__ballot_sync(sync, ...)` Subwarp Serialization Trap on `sm_70+` (`0x3220 -> 0x7f00 -> 0x9480`)**:
     Because `sync = 0xFF << group_base` has 4 distinct values across a 32-thread warp (`0xFF`, `0xFF00`, `0xFF0000`, `0xFF000000`), `ptxas` lowers `vote.sync.ballot.b32 d, p, sync` to `R2UR P2, URZ, sync; @!P2 BRA.DIV UR4, 0x7f00`, where **`P2` is FALSE 100% of the time**! Every `__ballot_sync(sync, ...)` jumped out-of-line to `0x7f00` and called `0x9480` (`WARPSYNC R5; VOTE.ANY`) **4 times sequentially per warp**! Fix: ballot across `activemask.b32` (`VOTE.ANY` in 1 single hardware cycle!) and `& sync`.
  2. **`asm volatile` Inside Ternary (`is_hi ? uleft_wrap(...) : a[k]`)**:
     `cicc` cannot speculate `asm volatile` across `? :`, emitting a branch around `SHF.L.W` that triggered a second `MATCH.ANY` block and cloned `1,200` slow-path instructions (`2,392` static insts = `38.2 KB > 32 KB` L1 I-cache). Fix: pass `sh_amt = group_thread >> 2` directly to `uleft_wrap(lo, hi, sh_amt)`.
  3. **`87` `SEL`s in `8 x 2` `a_hi^2`**:
     Replaced runtime `(i, j)` ternary trees with 32-bit packed nibble LUTs (`(0x...u >> (group_thread << 2)) & 7u`) and aligned middle-split owners `{2,3,4,5}` on `lo` / helpers `{0,1,6,7}` on `hi` (cutting `SEL`s in `8 x 2` from `87` down to `5`).

### Entry 9 — Phase 2 Iteration 8 (`iter8-sparebit-fastballot-lut`): Beating Baseline `mont_mul` Avg on L4!
- **What was implemented**:
  - Combined the three SASS fixes from Entry 8 (`activemask` subwarp ballot, branchless `uleft_wrap(..., group_thread >> 2)`, and packed nibble LUTs + owner/helper alignment in `8 x 2` `a_hi^2`) with Idea 9 (`< 1022`-bit spare-bit invariant: `v_top == 0` and `U < R` so final `- n` subtraction is identically a no-op, verified bit-for-bit against `mont_mul` under `-DCGBN_DEBUG_MONT_SQR`).
- **L4 Timings (`tag: iter8-sparebit-fastballot-lut`)**:
  - `TPI=8`: **PASS**, **`26.489 ms` avg / `24.399 ms` best** (`989.63 M/s` avg, `1074.41 M/s` best; `48` registers, `724` dynamic fast-loop instructions — beats Baseline 1 avg `27.564 ms` and Baseline 2 avg `30.244 ms` / best `28.023 ms`).

### Entry 10 — Phase 2 Iteration 9 (`iter9-fused-madwide-gbit`): Beating Baseline `mont_mul` Best Time & Catching the `REG: 48 -> 52` Cliff
- **What was implemented**:
  - Replaced `1u << group_thread` mask recomputations with a single `g_bit` bitmask (`0xBEu`, `0x1Cu`, `0x7Eu`, `0x18u`, `0x3Cu`) and fused `2 x 2` cross-terms with `mad_wide` + `madlo_cc`/`madhic_cc`.
- **L4 Timings (`tag: iter9-fused-madwide-gbit`)**:
  - `TPI=8`: **PASS**, **`26.493 ms` avg / `23.097 ms` best** (`989.50 M/s` avg, **`1,134.95 M/s` best** — first iteration to beat Baseline 1's `23.545 ms` best kernel time!).
- **SASS Diagnosis**:
  - Dynamic fast-loop instructions dropped to `720`, **but** `ptxas` hoisted `a_hi^2` shuffles and partial products across Step 2, pushing registers from `48` (`83.3%` SM occupancy = `40` warps/SM) to `52` (`66.7%` SM occupancy = `32` warps/SM on `sm_89` due to 8-register warp allocation granularity), explaining why best (boost-clock) time improved by `1.30 ms` while steady-state avg time stayed flat.

### Entry 11 — Phase 2 Iteration 10 & Final Verification (`iter10-fwmont-seqbarrier-2x2` / `final-halfsplit-cios-8x2`): Crossing `1.02 G mont_sqr/s` Avg & `1.15 G mont_sqr/s` Best!
- **What was implemented**:
  1. **Specialized `wmad_reduce_zero_tpi8_l4` (`8` Pair-Steps = `16` Words)**:
     - Replaced the generic `wmad_mont_reduce_zero_pair<8, 4>` helper (which carried an extra post-step `add_cc`/`addc_cc` merge of `t0, t1` into `ra[2..4]`) with a dedicated `TPI=8, LIMBS=4` zero-top reduction (`wmad_reduce_zero_tpi8_l4`) that folds `t0` directly into `chain2` (`step == 0 ? chain2.add(ra[4], t0) : chain2.add(t0, 0)`) and `t1` directly into `chain4` (`chain4.add(t1, 0)`), matching `mont_mul`'s exact 11-instruction/word schedule (`175` total instructions for all `64` `IMAD.WIDE`s + `32` `SHFL`s + `16` `q = r0 * np0` `IMAD`s!).
     - Caught and fixed a pair-step count bug (`step < 4` -> `step < 8` = 16 upper words) and moved `#ifdef CGBN_DEBUG_MONT_SQR` divisibility `assert`s after `chain.add(0, 0)` so debug builds never clobber `CC.CF`.
  2. **2-Register `(d0, d1)` Repack via Inline PTX `selp.b32` + `wmad_mul_2x2` ILP**:
     - Repacked `a_hi` into `(d0, d1)` using early-clobber inline PTX `selp.b32` (`-2` live registers) and structured `wmad_mul_2x2` with 3 independent `mad_wide` launches (`w00`, `w11`, `m`) so `ptxas` interleaves `a_hi^2`'s `SHFL` and `IMAD.WIDE` pipelines directly inside the serial `SHFL.IDX -> q = r0 * np0` dependency stalls of CIOS Step 0 (`0d50..0fd0`) and Zero-Top Step 7 (`2c40..2ce0`).
     - Brought register usage back down to **`REG: 47`** (`48`-register hardware tier = **`40` warps/SM = `83.3%` SM occupancy**, the **exact same occupancy tier** as baseline `mont_mul` at `REG: 42`!) and cut the fast loop to **`687` dynamic instructions** (`-49` instructions and `-47` wide multiplies vs. `mont_mul`!).
  3. **Explored `REG <= 40` (`100%` Occupancy) Staged Serialization Tradeoff (`task-841`, `task-883`, `task-901`)**:
     - Tested data-dependent barriers (`g4 = (group_thread << 2) | (r1 >> 4)`) + delayed `v -> (d0, d1)` funnel-shift recovery + in-place `wmad_shfl_mad_2x2` to shrink the fast loop to `R0..R39`.
     - Discovered via `cuobjdump -sass` that because `mont_sqr.cu` has `if (instance >= count) return;` before `repeated_sqr`, `ptxas`'s out-of-line Volta+ `__cuda_sm70_shfl_sync` slow-path stub (`0x8770`) hard-codes `R40..R42` (`REG: 44..45`, still `83.3%` occupancy without CLI `-maxrregcount=40`), while serializing Step 3 after Step 2 gave up ~`0.7 ms` of `SHFL`/`IMAD` latency hiding (`26.34 ms` vs. `25.64 ms`). Kept the `Iter 10` interleaved schedule as the clear winner.
  4. **Full `-DCGBN_DEBUG_MONT_SQR` & 27-Configuration Verification**:
     - Running `./mont_sqr_dbg -a -r 1` with `-DCGBN_DEBUG_MONT_SQR` verified `TPI = 8` (`26.2M` squarings, `0` failures) and caught an off-by-one (`s_flip <= K` -> `s_flip < K` and `s_flip <= K - 1` -> `s_flip < K - 1`) in the general `K > 2` (`TPI >= 16`) paths of [`core_mont_wmad.cu`](include/cgbn/core/core_mont_wmad.cu) and [`core_mont.cu`](include/cgbn/core/core_mont.cu).
     - After fixing both lines, re-verified **100% pass with `-DCGBN_DEBUG_MONT_SQR` active** on:
       - `mont_sqr.cu -a` (`TPI = 8, 16, 32`, `102,400` instances $\times$ `256` squarings = `78.64M` live-asserted GPU squarings), and
       - All **27 `(TPI, BITS, LIMBS, PADDING)` configurations** (`32` to `8192` bits, `TPI = 4, 8, 16, 32`, odd `LIMBS`, padded widths, `0xFFFFFFFF` carry-ripple stress cases, `a == 0`, `a >= n`, and in-place aliasing `mont_sqr(r, r, n, np0)`).
- **L4 Timings (`results.csv` rows 30–31)**:
  - `iter10-fwmont-seqbarrier-2x2`: **PASS**, **`25.644 ms` avg / `22.858 ms` best** (**`1,022.26 M sqr/s` avg / `1,146.85 M sqr/s` best**, `0.978 ns/sqr`).
  - `final-halfsplit-cios-8x2` (post `K > 2` fix): **PASS**, **`25.904 ms` avg / `23.628 ms` best** (**`1,012.00 M sqr/s` avg / `1,109.47 M sqr/s` best**, `0.988 ns/sqr`).

### Entry 12 — Phase 2 Iteration 11 (`iter11-actshfl-fastskip-tpi8`): `TPI=8`-Only Scope Cleanup, Active-Mask PTX Shuffles & Uniform Carry Fast-Skip (`1.049 G sqr/s` Avg / `1.218 G sqr/s` Best!)
- **Scope Cleanup (`TPI = 8`, `<= 1020`-Bit Inputs)**:
  - Reverted [`include/cgbn/core/core_mont.cu`](include/cgbn/core/core_mont.cu) (`0` diff) and removed all non-`TPI = 8` experimental squaring/reduction helpers from [`include/cgbn/core/core_mont_wmad.cu`](include/cgbn/core/core_mont_wmad.cu). All configurations other than `(TPI == 8, LIMBS == 4, PADDING == 0)` delegate directly to `core_t<env>::mont_mul(r, a, a, n, np0)`.
- **Three Microarchitectural Wins Implemented in `wmad_mont_sqr_tpi8_l4`**:
  1. **Active-Mask Width-8 Inline PTX Shuffles (`shfl.sync.{idx,up,bfly}.b32 ..., 0x181f, shfl_mask`)**:
     - Because `core_t<env>::sync_mask()` (`0xFF << group_base`) takes 4 distinct values across a 32-lane warp, passing `sync` to `__shfl_sync` forced `ptxas` to emit a 6-instruction `MATCH.ANY + REDUX.OR + VOTEU.ANY + LOP3 + BRA.DIV + BSYNC` convergence block.
     - Reading `activemask.b32 shfl_mask` once at entry and passing `shfl_mask` to width-8 PTX shuffles (`0x181f` / `0x1800`) lets `ptxas` hoist `VOTE.ANY + R2UR UR4` **outside** the 256-iteration loop and replace the convergence block with a **single 1-cycle `BRA.DIV UR4`**, while remaining **100% `compute-sanitizer --tool synccheck` clean (`0 errors`)** on partial warps (`count = 1, 2, 3, 5`) and intra-warp divergent `TPI = 8` instances (`cgbn_instance_syncable`).
  2. **Uniform Warp Carry Fast-Skip (`__any_sync(shfl_mask, c != 0u)`)**:
     - After Step 4 folds `t_up` (`<= 16`) into the 128-bit per-lane sum `ra[0..3]`, the carry-out `c` is `1` with probability $\le 16 / 2^{128} \approx 4.7 \times 10^{-38}$.
     - Guarding `wmad_fast_prop_tpi8_l4` with `if (__builtin_expect(__any_sync(shfl_mask, c != 0u), 0))` skips all `19` subwarp ballot + carry-propagation instructions on the fast path with zero warp divergence, while preserving 100% exactness on adversarial `0xFFFFFFFF` 128-bit ripple inputs (verified explicitly via `a_lo = 0, a_hi = 2^255 - 1` inputs that force `c == 1`).
  3. **Hardware `0x181f` Lane-Masking + Carry-Free First CIOS Step (`THREAD == 0, word == 0`)**:
     - `SHFL.IDX` with `c = 0x181f` masks `src_lane & 7` directly in hardware (verified across all 32 warp lanes), eliminating all `8` `& 7u` `LOP3.LUT` instructions on packed nibble-LUT indices (`LOP3.LUT` dropped from `19` to `11`).
     - Replacing `chain1`/`chain2` on `THREAD == 0, word == 0` (where `ra = 0, ru = 0, c = 0`) with 4 carry-free `mad_wide`s cut dynamic fast-loop instructions from `687` down to **`649`** (`-87` instructions vs. `mont_mul`'s `736`, **`-11.8%`**) at **`REG: 47`** (`83.3%` SM occupancy = `40` warps/SM).
- **L4 Benchmark & Verification (`results.csv` row 32, `tag: iter11-actshfl-fastskip-tpi8`, saved on branch `halfsplit-cios-iter11` `2f1553a`)**:
  - **Avg kernel time**: **`25.000 ms`** (**`1,048.56 M mont_sqr/s`**, `0.954 ns/sqr`, `73.14x` over 12-thread GMP CPU) — **PASS**.
  - **Best kernel time**: **`21.520 ms`** (**`1,218.12 M mont_sqr/s`** — **`> 1.21 G mont_sqr/s`**!).
  - Verified with `compute-sanitizer --tool synccheck` (`0 errors`), `mont_sqr -a` under `-DCGBN_DEBUG_MONT_SQR`, and all 27 `(TPI, BITS, LIMBS, PADDING)` configurations (`<= 1020`-bit inputs at `TPI = 8, BITS = 1024`).

---

### Entry 13 — Branch Exploration (`hensel64-lookahead-sqr`, `f789316`): Radix-$2^{64}$ Hensel-Lifted Parallel Quotient Lookahead
- **Hypothesis**:
  - In standard 2-word-per-step CIOS and Zero-Top reduction (`wmad_reduce_zero_tpi8_l4`), each pair-step executes **two sequential MIO shuffle round-trips** on the critical path:
    1. `ra[0] = add(ra[0], c)` $\to$ `SHFL.IDX(..., 0)` $\to$ `q0 = r0 * np0` $\to$ `ru[0] = madlo(n[1], q0, ru[0])`, `ra[1] = madhi(n[0], q0, ra[1])`
    2. `ru[0] = add(ru[0], ra[1])` $\to$ `SHFL.IDX(..., 0)` $\to$ `q1 = r1 * np0`
  - By Hensel-lifting $-N^{-1} \bmod 2^{32}$ (`np0`) once per `mont_sqr` to $-N^{-1} \bmod 2^{64} = \text{np0} + 2^{32} \text{np1}$ on lane 0:
    $$\text{np1} = \text{np0} \cdot \left(\text{madlo}(n_1, \text{np0}, \text{madhi}(n_0, \text{np0}, 1))\right) \bmod 2^{32},$$
    lane 0 can compute **both** 32-bit quotients $(q_0, q_1)$ up front from the low 64 bits $(l_0, l_1)$ of the accumulator (`q0w = mad_wide(l0, np0, 0); q0 = ulow(q0w); q1 = madlo(l0, np1, madlo(l1, np0, uhigh(q0w)))`) and broadcast `(q0, q1)` in **parallel** (`8` parallel shuffle stages instead of `16` serial round-trips).
- **Results (`results.csv` row 34, branch `hensel64-lookahead-sqr` `f789316d15`, `tag: branch-hensel64-lookahead-quotient`)**:
  - Passed **100%** of `26.21M` live debug assertions (`(n64 * np64) == 0xFFFFFFFFFFFFFFFFull`, `ra0_low == 0`, `t1_low == 0`, and exact match vs. `mont_mul`) and all `102,400` GMP checks on the first compile!
  - **SASS (`sm_89`)**: `REG: 45` (`83.3%` SM occupancy = `40` warps/SM), `fast_loop = 661` dynamic instructions (`+8` `IMAD.WIDE`s and `+10` total instructions vs. `Iter 11`).
  - **Timing on L4**: **`26.052 ms` avg / `22.182 ms` best** (`1,006.25 M sqr/s` avg / `1,181.79 M sqr/s` best) — faster than baseline `mont_mul`, but **`+1.05 ms` slower** than `Iter 11` (and applying Hensel lookahead to Step 2a as well reached `710` instructions / `28.24 ms`).
  - **Takeaway**: At `40` warps/SM (`10` warps per scheduler), L4 already hides single-warp MIO `SHFL` latency; trading `+8` `IMAD.WIDE`s + `+8` `IMAD`s to break serial `SHFL` dependencies increases instruction/power demand under L4's `72W` TDP cap.

---

### Entry 14 — Branch Exploration (`dual-8x2-sos-redc`, `3fe3a65`): `194`-Wide Separated Dual-`8x2` Symmetric Squaring + Carry-Save 1024-Bit REDC
- **Hypothesis**:
  - Test whether pushing 64-bit wide multiplies down to the theoretical **`194` wide multiplies/lane** (`-62` vs. `mont_mul`'s `256`, **`-24.2%`**, and `-15` vs. Half-Split CIOS's `209`) via a clean Karatsuba/SOS split $a^2 = a_{\text{lo}}^2 + 2^{512}(2 a_{\text{lo}} a_{\text{hi}}) + 2^{1024} a_{\text{hi}}^2$ can beat Half-Split CIOS when reusing our fast `8 x 2` symmetric squarer `wmad_sqr_512_8x2_tpi8_l4`:
    1. **$a_{\text{lo}}^2$ (`512 -> 1024` bits)**: `17` wide muls/lane via `wmad_sqr_512_8x2_tpi8_l4` (mathematically exact for all $X \in [0, 2^{512})$ with `c_E == 0` on lane 7).
    2. **$a_{\text{hi}}^2$ (`512 -> 1024` bits)**: `17` wide muls/lane via `wmad_sqr_512_8x2_tpi8_l4`, deferred after REDC so only 2 words `(d0_hi, d1_hi)` stay live across reduction (`REG: 72 -> 62`).
    3. **$M = a_{\text{lo}} \cdot a_{\text{hi}}$ (`512 x 512 -> 1024` bits)**: `32` `IMAD.WIDE`s/lane via two width-4 `128 x 128` limb blocks (`wmad_mul_4x4`, exact `384`-bit local stitch with zero overflow).
    4. **Carry-Save 1024-Bit REDC (`wmad_reduce_1024_tpi8_l4`)**: Feeds unpropagated $a_{\text{lo}}^2$ (`ra[0..4]`) and $(2M)_{\text{lo}}$ (`c, ru[0..2], ra[4]`) directly into a 16-pair-step zero-top reduction (`128` `IMAD.WIDE`s/lane) with **zero** intermediate carry-propagation passes before reduction!
- **Results (`results.csv` row 35, branch `dual-8x2-sos-redc` `3fe3a65595`, `tag: branch-dual-8x2-sos-194wide`)**:
  - Passed **100%** of `26.21M` live `-DCGBN_DEBUG_MONT_SQR` assertions and `102,400` GMP checks on the first compile!
  - **SASS (`sm_89`)**: Confirmed **exactly `194` wide multiplies** (`178` `IMAD.WIDE` + `16` `IMAD.HI`), `REG: 62` (`66.7%` occupancy), `fast_loop ~ 826` dynamic instructions (`166` `IADD3`, `164` `SHFL`, `149` `SEL`).
  - **Timing on L4**: **`31.298 ms` avg / `29.051 ms` best** (`837.57 M sqr/s` avg / `902.36 M sqr/s` best) — **`3.46 ms` faster** than `Iter 6` separated SOS (`34.757 ms`), but **`6.54 ms` slower** than Half-Split CIOS (`24.756 ms`).
  - **Takeaway**: Proves conclusively why Half-Split CIOS dominates separated SOS at `TPI = 8`: saving those last `15` wide multiplies (`209 -> 194`) requires assembling and routing $a_{\text{lo}}^2$ and $2 a_{\text{lo}} a_{\text{hi}}$ explicitly across lanes (`+194` extra `SHFL`/`SEL`/`IADD3` instructions and `+14` registers), whereas Half-Split CIOS folds $a_{\text{lo}} \cdot (a_{\text{lo}} + 2^{513} a_{\text{hi}})$ directly inside the CIOS reduction loop at `1.94` instructions per multiply-accumulate.

---

### Entry 15 — Iter 12 (`lean-halfsplit-pred`, `ba9cdef`): Unified `Even + Mid` Schedule, L4 Dual-Regime Physics, & 2-Register Bit-Packed LUT Hoisting (`24.756 ms` Avg / `20.180 ms` Best = `1.30 G mont_sqr/s`!)
- **Deep SASS Dissection & Four Final Breakthroughs**:
  1. **Unified `Even + Mid` Partner Exchange (`-2` `SHFL.BFLY`, `-1` `LOP3`, `-11` `MOV`s)**:
     - Every `{g, g^4}` lane pair (`{0,4}`, `{1,5}`, `{2,6}`, `{3,7}`) collectively owns **3** even-sum `2 x 2` off-diagonal blocks (`0+4`, `0+6,2+4`, `2+6,1+7`, `4+6,3+7` $\implies 6$ `2x2` blocks + `4` middle `2x1` blocks across 8 lanes).
     - Observing that `!keep_e` (`{0, 7}`, mask `0x81`) is disjoint from `Middle 2x1`'s `shfl_xor(..., 4)` payload on `{1..6}`, we merged Step 3's two separate width-8 butterfly exchanges (`ex` + `mx`, `7` shuffles) into a **single** 5-word `wmad_shfl_xor8(..., 4)` (`S0..S3, Sc = (h0..h2) + (!keep_e ? P0..P3 : 0)`).
  2. **Carry-Chained `Middle 2x1` (`m1 = mad_wide(xi1, bj, (uint64_t)uhigh(m0))`) & Hardware `SHFL.UP` Range-Predicate Zeroing**:
     - Replaced split `madlo`/`madhi` in `Middle 2x1` (which `ptxas` had compiled into an extra duplicate `IMAD` + `IMAD.HI` + `2` `IADD3`s) with a carry-free 64-bit chained `mad_wide` (`(2^{32}-1)^2 + (2^{32}-1) = 2^{64} - 2^{32} < 2^{64}$, zero overflow!).
     - Replaced explicit `group_thread > 0` / `group_thread != 4` masks with `SHFL.UP`'s built-in hardware boundary predicate (`shfl.sync.up.b32 %0|p, %1, 1, 0x1c00, mask; selp.b32 %0, %0, 0, p;` using width-4 `0x1c00` for `v[0..3]` and width-8 `0x1800` for `O2, O3, c_O`).
     - Also tested predicated `@p add.cc.u32` vs. `SEL` in Step 3 (`VA..VD`): `sm_89` SASS revealed `@p IADD3.X` injects `4` extra `BRA.DIV ~URZ` convergence checks (`+1.1 ms` slower), confirming `SEL` is strictly faster on Ada.
  3. **Uncovering the Dual-Regime Physics of NVIDIA L4 (`AD104`, `72W` TDP) via Forward+Reverse (`ABBA`) Sweeps**:
     - Forward + Reverse (`ABBA`) thermal-controlled sweeps across `X0..X6`, `Y0..Y4`, `Z0..Z3`, and `Q0..Q4` uncovered two distinct operating regimes within every 5-run benchmark:
       - **Boost-Clock Regime (Runs 1–2, `~2040 MHz`)**: Latency-sensitive — staying at **`REG <= 48`** (`48`-reg hardware allocation granule = **`40` warps/SM = `83.3%` occupancy**) unlocks **`20.07–20.26 ms` (`1.29–1.31 G sqr/s`)**, whereas crossing to `REG = 49..56` (`36` warps/SM = `75.0%` occupancy) forfeits `~1.4–1.8 ms` on Run 1.
       - **`72W` Power-Capped Steady State (Runs 3–5)**: Dynamic instruction/energy throughput-bound — hoisting loop invariants to cut dynamic instructions from `651` to `623–632` speeds up steady-state runs by **`-0.63` to `-1.40 ms`** (`25.86–26.15 ms` vs. `26.62–27.35 ms`).
  4. **2-Register Bit-Packed LUT Hoisting + Free Hardware Predicate Hoisting (`REG = 48` AND `632` Dynamic Instructions!)**:
     - Unconstrained LICM hoisted Step 3's 8 LUT indices + `d_src` + `sh_amt` + `prev_lane` + 6 predicates into `56` registers (`75%` occupancy).
     - By packing the 8 3-bit lane LUT indices, `d_src`, `sh_amt`, and `prev_lane` into **just two hoisted 32-bit registers** (`lut_o`, `lut_e`, copied via `asm ("" : "+r"(lut_o), "+r"(lut_e)); uint32_t lo=lut_o, le=lut_e; asm volatile ("" : "+r"(lo), "+r"(le));` so `lut_o, lut_e` hoist out of the 256-iteration loop while unpack shifts stay local) and letting all **6** `bool` lane predicates hoist freely (`LOP3 = 0` inside the loop!), we hit **BOTH** targets simultaneously:
       - **`REG: 48`** (**`83.3%` SM occupancy = `40` warps/SM**!)
       - **`fast_loop = 632` dynamic SASS instructions** (`-104` instructions vs. `mont_mul`'s `736`, **`-14.1%`**, and `-19` instructions vs. `Iter 11`, with `LOP3 = 0` and `MOV` cut from `34` to `23`!).
- **Official L4 Benchmark & Full Verification (`results.csv` rows 33 & 36, branch `lean-halfsplit-pred` `ba9cdef625`)**:
  - **Back-to-back warm run (`row 33`, `tag: iter12-packed2lut-unified-even-reg48`)**: **`24.893 ms` avg** (`1,053.07 M sqr/s`, steady-state runs `25.86–25.91 ms`) / `21.554 ms` best — **PASS**.
  - **Rested run (`row 36`, `tag: final-lean-halfsplit-pred-reg48`)**:
    - **Avg kernel time**: **`24.756 ms`** (**`1,058.93 M mont_sqr/s`**, **`0.944 ns/sqr`**, `82.97x` over 12-thread GMP CPU) — **PASS**.
    - **Best kernel time**: **`20.180 ms`** (**`1,299.03 M mont_sqr/s`** — **`~1.30 G mont_sqr/s`**!).
  - Verified with `26.21M` live `-DCGBN_DEBUG_MONT_SQR` squarings, `compute-sanitizer --tool synccheck` (`0 errors` across partial/divergent warps and `0xFFFFFFFF` 128-bit carry-ripple `c == 1`), all 27 CGBN `(TPI, BITS, LIMBS, PADDING)` configurations, and `mont_sqr -a` (`TPI = 8, 16, 32`).

---

## 5. Literature Notes: Upper-Diagonal Montgomery Squaring (CIOS vs. SOS)

*(Cross-referenced from [`research.md`](research.md))*

### Most Relevant Papers for Upper-Diagonal `mont_sqr`

1. **Ç. K. Koç, T. Acar, & B. S. Kaliski Jr. (1996)** — *"Analyzing and Comparing Montgomery Multiplication Algorithms"*, *IEEE Micro*, 16(3), 26–33 ([DOI: `10.1109/40.502403`](https://doi.org/10.1109/40.502403)), together with **T. Acar (1997)**, *"High-Speed Algorithms & Architectures for Number-Theoretic Cryptosystems"* (Ph.D. Dissertation, Oregon State Univ.):
   - Primary reference for adapting **SOS (Separated Operand Scanning)** vs. **CIOS (Coarsely Integrated Operand Scanning)** to upper-diagonal ($i < j$) squaring.
   - Identifies the **65-bit extra-carry problem** when doubling cross-terms on the fly ($2 \cdot a_i a_j$ exceeds $2w = 64$ bits) and motivates accumulating **unscaled** upper-diagonal terms $a_i a_j$ ($i < j$) first, shifting left by 1 bit (`<< 1`) once, adding $a_i^2$ diagonals, and then applying Montgomery reduction.
2. **N. Emmart, J. Luitjens, C. Weems, & C. Woolley (ARITH 2016)** — *"Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs"* ([DOI: `10.1109/ARITH.2016.21`](https://doi.org/10.1109/ARITH.2016.21)) and **N. D. Emmart (2018)**, *"A Study of High Performance Multiple Precision Arithmetic on Graphics Processing Units"* ([`hdl:20.500.14394/17353`](https://hdl.handle.net/20.500.14394/17353)):
   - Architectural blueprint for CGBN's warp-cooperative layout (`__shfl_sync` limb broadcast/shift), **aligned (`ra`) + unaligned (`ru`) two-chain accumulation** (required on Volta/Ampere/Ada so `ptxas` fuses `mad.lo.cc.u32` + `madc.hi.cc.u32` into single `IMAD.WIDE.U32` instructions, as in [`include/cgbn/core/core_mont_wmad.cu`](include/cgbn/core/core_mont_wmad.cu)), and deferred carry resolution (`fast_propagate_add`).
3. **S. Gueron & V. Krasnov (IEEE ITNG 2012)** — *"Speeding up Big-Numbers Squaring"* ([IACR ePrint 2012/067](https://eprint.iacr.org/2012/067)) & **Q. Xiong et al., `gECC` (ACM TACO 2024/2025, arXiv:2412.16919)**:
   - Gueron & Krasnov give the canonical low-overhead schedule for separated upper-triangle off-diagonal accumulation + single `<< 1` shift + diagonal $a_i^2$ injection with minimal add-with-carry chains (used in OpenSSL `RSAZ`).
   - `gECC` dissects Ampere/Ada SASS (`IMAD` vs. `IADD3` + predicate carries), showing directly why excess non-multiply carry-merging (`IADD3.X` / `SEL` / `MOV`) and register pressure erode `IMAD.WIDE` savings unless accumulation happens in-place with minimal live state (matching the Entry 2 SASS diagnosis above).
4. **N. Emmart, F. Zheng, & C. Weems (IEEE ARITH 2018)** — *"Pushing the Performance Envelope of Modular Exponentiation Across Multiple Generations of GPUs"* ([DOI: `10.1109/ARITH.2018.8464792`](https://doi.org/10.1109/ARITH.2018.8464792)):
   - Demonstrates both integer (`XMAD`/`IMAD.WIDE`) and IEEE-754 double-precision (`DFMA.RZ` 52-bit radix) warp-cooperative Montgomery multiplication, introducing redundant spare-bit avoidance of final conditional subtraction, low-overhead carry resolution, and architecture-specific pipe balancing.

### CIOS vs. Non-CIOS (SOS) for Upper-Diagonal Squaring

- **Clarifying Scalar CIOS Symmetry (Refining Section 1):**
  Mathematically, upper-diagonal CIOS squaring **is** valid within $W + 2$ words: after $i$ right-shifts of accumulator $T$, word $T[k]$ has weight $2^{32(i+k)}$, so adding $a_i^2$ at `T[i..i+1]` (weight $2^{64i}$) and $2 a_i a_j$ for $j > i$ at `T[j..j+2]` (weight $2^{32(i+j)}$) stays bounded in $T[0 \dots W+1]$. Moreover, every pair $(u, v)$ with $u \le v$ contributing to weight $2^{32i}$ satisfies $2u \le u + v \le i \implies u \le \lfloor i/2 \rfloor \le i$, so `T[0]` is already complete before $q_i = T[0] \cdot \text{np0} \bmod 2^{32}$ is computed.
- **Why Pure Row-Scanning CIOS Loses, While Hybrid Half-Split CIOS Wins at `TPI = 8`:**
  1. **SIMT Lockstep Serialization in Row-by-Row CIOS:** During row $i$'s upper-diagonal pass ($j > i$), lanes with `group_thread < row` sit masked in SIMT lockstep while higher lanes execute the full `LIMBS` chain—saving **zero** inter-thread `IMAD.WIDE` warp cycles, and 65-bit on-the-fly doubling breaks fused 64-bit `ra/ru` `IMAD.WIDE.U32` carry chains.
  2. **Why Hybrid Half-Split CIOS Captures the Best of Both Worlds (`1024`-bit `TPI = 8`):** Splitting $a = a_{\text{lo}} + 2^{512} a_{\text{hi}}$ forms $V = a_{\text{lo}} + 2^{513} a_{\text{hi}}$ **once before CIOS begins** (solving the 65-bit doubling problem up front!) and runs CIOS across `100%` active lanes for only the low half (`thread = 0, 2`), while computing only the `512`-bit $a_{\text{hi}}^2$ via separated Folded Target-Sum SOS (`17` wide muls/lane, landing directly in-place with zero transpose shuffles).

---

## 6. Final Performance Summary & Architectural Takeaways (`TPI = 8`, NVIDIA L4 `sm_89`)

### Saved Git Branches (All Branched Cleanly from `mont_square_starting_point` `cdc1411`)

| Git Branch | Commit | Algorithmic Direction | Regs (Occ.) | Wide Muls | Fast-Loop Insts | `256`-Iter Avg ms | `256`-Iter Best ms | Avg M sqr/s | Best M sqr/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mont_square_starting_point` | `cdc1411` | Baseline `mont_mul(r, a, a, n, np0)` | 42 (83.3%) | 256 | 736 | 27.564–30.244 | 23.545–28.023 | 866.8–951.0 | 935.5–1113.4 |
| `dual-8x2-sos-redc` | `3fe3a65` | `194`-wide Separated Dual-`8x2` SOS + Carry-Save 1024b REDC | 62 (66.7%) | **194** | 826 | 31.298 | 29.051 | 837.57 | 902.36 |
| `hensel64-lookahead-sqr` | `f789316` | Half-Split CIOS + Radix-$2^{64}$ Hensel Parallel Quotients `(q0, q1)` | 45 (83.3%) | 217 | 661 | 26.052 | 22.182 | 1006.25 | 1181.79 |
| `halfsplit-cios-iter11` | `2f1553a` | Half-Split CIOS + `8x2` `a_hi^2` + `activemask` Shuffles (`Iter 11`) | 47 (83.3%) | 209 | 651 | 25.000 | 21.520 | 1048.56 | 1218.12 |
| **`lean-halfsplit-pred`** | **`ba9cdef`** | **Unified `Even+Mid` + 2-Reg Bit-Packed LUT + Hoisted Preds (`Iter 12`)** | **48 (83.3%)** | **209** | **632** | **22.021–24.756** | **19.634–20.180** | **1058.9–1190.4** | **1299.0–1335.1** |

### Complete `TPI = 8` (`1018`-bit in `1024`-bit env) Benchmark Progression (`results.csv`)

| Stage / Tag | Iters | Regs | SM Occ. | Fast-Loop Insts | Avg Time (ms) | Best Time (ms) | Avg M sqr/s | Best M sqr/s | ns/sqr | Verified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline 1 (`mont_mul`, `a24fdbf48b`) | 256 | 42 | 83.3% | 736 | 27.564 | 23.545 | 951.04 | 1113.38 | 1.051 | PASS |
| Baseline 2 (`mont_mul`, `0fdd4f3b38`) | 256 | 42 | 83.3% | 736 | 30.244 | 28.023 | 866.76 | 935.47 | 1.154 | PASS |
| Phase 1 (Slow Symmetric + Live `assert`s) | 256 | — | — | — | 131.774 | 131.487 | 198.94 | 199.37 | 5.027 | PASS |
| Iter 1 (`ea`/`eu`/`ca` Multi-Accum SOS) | 256 | 92 | 41.7% | ~1,450 | 37.062 | 33.785 | 707.32 | 775.92 | 1.414 | PASS |
| Iter 2 (Flat `wmad_sqr_mul_block`) | 256 | 63 | 66.7% | ~1,380 | 34.515 | 31.825 | 759.50 | 823.71 | 1.317 | PASS |
| Iter 3 (Aligned `pa`/`pu` Pairs, Hoisted) | 256 | 84 | 50.0% | ~990 | 36.245 | 33.816 | 723.25 | 775.22 | 1.383 | PASS |
| Iter 4 (Decoupled Zero-Top Reduction) | 256 | 88 | 41.7% | 968 | 35.022 | 32.218 | 748.52 | 813.65 | 1.336 | PASS |
| Iter 5 (Anti-LICM + Odd-First + `@q` PTX) | 256 | 57 | 66.7% | ~910 | 30.819 | 27.238 | 850.60 | 962.41 | 1.176 | PASS |
| Iter 6 (`iter6-branchless-sos`) | 256 | 58 | 66.7% | 1,044 | 34.757 | 32.736 | 754.23 | 800.78 | 1.326 | PASS |
| Iter 7 (`iter7-halfsplit-cios-8x2`) | 256 | 48 | 83.3% | 774 | 28.882 | 25.319 | 907.64 | 1035.35 | 1.102 | PASS |
| Iter 8 (`iter8-sparebit-fastballot-lut`) | 256 | 48 | 83.3% | 724 | 26.489 | 24.399 | 989.63 | 1074.41 | 1.010 | PASS |
| Iter 9 (`iter9-fused-madwide-gbit`) | 256 | 52 | 66.7% | 720 | 26.493 | 23.097 | 989.50 | 1134.95 | 1.011 | PASS |
| Iter 10 (`iter10-fwmont-seqbarrier-2x2`) | 256 | 47 | 83.3% | 687 | 25.644 | 22.858 | 1022.26 | 1146.85 | 0.978 | PASS |
| Iter 11 (`iter11-actshfl-fastskip-tpi8`) | 256 | 47 | 83.3% | 651 | 25.000 | 21.520 | 1048.56 | 1218.12 | 0.954 | PASS |
| Branch (`branch-hensel64-lookahead-quotient`) | 256 | 45 | 83.3% | 661 | 26.052 | 22.182 | 1006.25 | 1181.79 | 0.994 | PASS |
| Branch (`branch-dual-8x2-sos-194wide`) | 256 | 62 | 66.7% | 826 | 31.298 | 29.051 | 837.57 | 902.36 | 1.194 | PASS |
| Iter 12 Warm (`iter12-packed2lut-unified-even-reg48`) | 256 | 48 | 83.3% | 632 | 24.893 | 21.554 | 1053.07 | 1216.21 | 0.950 | PASS |
| Iter 12 Rested (`final-lean-halfsplit-pred-reg48`) | 256 | 48 | 83.3% | 632 | 24.756 | 20.180 | 1058.93 | 1299.03 | 0.944 | PASS |
| **Iter 12 Cold (`ba9cdef625`, Rows 37 & 40)** | **256** | **48** | **83.3%** | **632** | **22.021–23.699** | **19.634–20.111** | **1106.2–1190.4** | **1303.5–1335.1** | **0.840** | **PASS** |
| **Iter 12 Unthrottled Boost (`ba9cdef625`, Rows 42–43)** | **20** | **48** | **83.3%** | **632** | **1.546** | **1.542** | **1324.68** | **1328.02** | **0.755** | **PASS** |

### Head-to-Head SASS Breakdown (`TPI = 8, LIMBS = 4` on Ada `sm_89`)

| Metric | Baseline `mont_mul` | Fast `wmad_mont_sqr_tpi8_l4` (`ba9cdef`) | Delta |
|---|---:|---:|---:|
| **64-bit Wide / Hi Multiplies (`IMAD.WIDE` + `IMAD.HI`)** | `256` | **`209`** (`205` `WIDE` + `4` `HI`: `128` CIOS + `64` Zero-Top + `17` `a_hi^2`) | **`-47` (`-18.4%`)** |
| **Dynamic Fast-Loop SASS Instructions** | `736` | **`632`** (unified `Even+Mid` + 2-reg packed LUTs + `0` `LOP3` + `__any_sync` skip) | **`-104` (`-14.1%`)** |
| **Registers / Thread (`cuobjdump -res-usage`)** | `42` (`48` alloc) | **`48`** (`48` alloc) | **Same `83.3%` SM Occ. (`40` warps/SM)** |
| **Inner `BSSY` / `MATCH.ANY` Convergence Stalls** | `0` | **`0`** (`1`-cycle `BRA.DIV UR4` header) | **`0` divergence penalty** |
| **Sustained `256`-Iter Throughput (`NVIDIA L4`)** | `866.8–951.0 M/s` | **`1,058.9–1,190.4 M/s` avg (`1,335.1 M/s` best)** | **`+11.3%` to `+37.3%` avg** |
| **Unthrottled Boost Throughput (`20` iters, `2040 MHz`)** | — | **`1,324.7 M/s` avg / `1,328.0 M/s` best (`0.755 ns/sqr`)** | **`~89.2` SM-cycles / 1024-bit `mont_sqr`** |

### Key Lessons Learned on NVIDIA Ada (`sm_89`)
1. **Why Half-Split CIOS (`209` Wide Muls, `632` Insts) Beats Both `mont_mul` (`256` Wide Muls, `736` Insts) and `194`-Wide Separated SOS (`826` Insts)**:
   In `mont_mul`'s `ra/ru` CIOS inner loop, folding $a_{\text{lo}} \cdot (a_{\text{lo}} + 2^{513} a_{\text{hi}})$ on top of Montgomery reduction costs only `298 - 175 = 123` incremental instructions (`1.92` instructions per 64-bit multiply-accumulate, with **zero** transpose shuffles and **zero** intermediate carry resolution!). Pure separated `194`-wide Dual-`8x2` SOS (`branch dual-8x2-sos-redc`) saves `15` more wide multiplies (`209 -> 194`) at the cost of `+194` `SHFL`/`SEL`/`IADD3` instructions to explicitly assemble $a_{\text{lo}}^2$ and $2 a_{\text{lo}} a_{\text{hi}}$.
2. **Bit-Packing Loop Invariants into `REG <= 48` Unlocks Both L4 Regimes (`2040 MHz` Boost AND `72W` TDP Steady State)**:
   On Ada (`sm_89`), `48` registers/thread is the exact ceiling for `40` warps/SM (`83.3%` occupancy). Exploiting `SHFL.IDX`'s hardware `src_lane & 7` masking (`c = 0x181f`) to pack 11 loop-invariant lane/shift values into just 2 hoisted GPRs (`lut_o`, `lut_e`) while hoisting all 6 `bool` lane predicates (`LOP3 = 0`) hits `REG = 48` **and** `632` dynamic instructions simultaneously (`19.634 ms` best / `1.335 G sqr/s` at boost, `22.021–24.756 ms` 5-run `256`-iter average).
3. **Defeating Subwarp Convergence & Ballot Serialization Traps on `sm_70+`**:
   - `__ballot_sync(sync, p)` with a subwarp `sync = 0xFF << group_base` traps into an out-of-line 4-way serialized `WARPSYNC` stub (`0x7f00 -> 0x9480`) on every call; balloting over `activemask.b32` and masking `& sync` executes in a single `VOTE.ANY` instruction, and guarding `wmad_fast_prop_tpi8_l4` behind `__any_sync(shfl_mask, c != 0u)` skips carry propagation completely in normal execution ($P < 5 \times 10^{-38}$).
   - `SHFL.UP`'s built-in range predicate (`shfl.sync.up.b32 %0|p, ..., 0x1c00/0x1800`) zeroes sub-group boundary lanes in hardware with `0` `ISETP`s, while flat `SEL` chains beat `@p add.cc.u32` on Ada (`0` extra `BRA.DIV ~URZ` stalls).
4. **Why Radix-$2^{64}$ Hensel Parallel Quotients (`branch hensel64-lookahead-sqr`) Don't Pay Off at `83.3%` Occupancy**:
   Halving serial MIO `SHFL` dependency chains (`16 -> 8`) via 64-bit `-N^{-1} mod 2^64` lookahead adds `+8` `IMAD.WIDE`s and `+10` instructions; at `10` active warps per scheduler, L4 is instruction/power throughput-bound rather than single-warp MIO latency-bound.

---

## 7. New Directions for Possible Optimizations

While `lean-halfsplit-pred` (`ba9cdef`) reaches **`89.2` SM-cycles per 1024-bit modular squaring** (`1.33 G sqr/s` at boost clock, `632` dynamic SASS instructions), several promising algorithmic and microarchitectural directions remain open for pushing performance even further across `1024`-bit `TPI = 8` and wider bit-widths:

### Direction 1: Subquadratic Toom-Cook Squaring (`Toom-2`, `Toom-3`, `Toom-4`) Aligned to `256`-Bit (`1` Word/Lane) Chunks
- **Key `TPI = 8` Alignment Insight**:
  At `TPI = 8`, **`256` bits equals `8` 32-bit words — exactly `1` `uint32_t` register per lane** across the 8-lane subwarp! Partitioning a `1024`-bit integer $a$ into base-$B = 2^{256}$ digits $A_0, A_1, A_2, A_3$ requires **zero cross-bit shifts** (merely shuffling single words across lanes), and squaring any `256`-bit (`1`-word/lane) evaluation point $D = \sum_{g=0}^7 d_g 2^{32g}$ via Folded Target-Sum takes only $\binom{8}{2} + 8 = 36$ scalar muls = **`5` `IMAD.WIDE`s/lane** (`+1` `IMAD.WIDE` if the point carries a small 4-bit top word $v_{\text{top}} \in [0, 15]$).
- **Why Subquadratic Toom Applies to Squaring ($a^2$) Rather Than Two-Product `REDC` at `1024` Bits**:
  Word-by-word CIOS/Zero-Top Montgomery reduction already costs `128` `IMAD.WIDE`s + `32` scalar `IMAD`s (`1.00x` of a single $1024 \times 1024$ product), whereas block subquadratic `REDC` ($Q = S_{\text{lo}} (-N^{-1}) \bmod R$, then $Q \cdot N$) requires **two** sequential $1024 \times 1024$ multiplications (one low-half, one full), so Toom-Cook only beats word-by-word `REDC` at larger bit-widths (`>= 2048–4096` bits). At `1024` bits, Toom-Cook should target **squaring** while feeding directly into carry-save Zero-Top / CIOS reduction:
  1. **Quarter-Split CIOS + 100%-Unsigned Toom-3 (`256 + 3x256` bits $\to$ `188` `IMAD.WIDE`s/lane)**:
     Split $a = u + 2^{256} H$ ($u \in [0, 2^{256})$, $H = A_1 + A_2 B + A_3 B^2 \in [0, 2^{764})$), so $a^2 = u(u + 2^{257} H) + 2^{512} H^2$. Run `wmad_cios_pair<0>` once (`64` `IMAD.WIDE`s) + `4` pair-steps of `wmad_reduce_zero` (`32` `IMAD.WIDE`s), compute `768`-bit $H^2$ via Toom-3 at $\{0, \infty, 1, -1, 2\}$ (`5 + 5 + 6 + 6 + 6 = 28` `IMAD.WIDE`s/lane), and finish `8` pair-steps of `wmad_reduce_zero` (`64` `IMAD.WIDE`s) $\implies$ **`188` `IMAD.WIDE`s/lane** (breaking below the `194`-mul quadratic symmetric lower bound!). Crucially, because $H_0, H_1, H_2 \ge 0$, evaluating $W_{-1} = |H_0 + H_2 - H_1|^2 \ge 0$ guarantees **every** Toom-3 interpolation intermediate ($C_2, O = C_1 + C_3, T = C_1 + 4 C_3, C_3 = (T - O)/3$ via exact 2-adic `* 0xAAAAAAABu`, and $C_1 = O - C_3$) is strictly non-negative (`0` signed multi-precision arithmetic!).
  2. **Half-Split CIOS + Subgroup-Parallel Toom-2 (`512 + 2x256` bits $\to$ `206` wide muls/lane)**:
     Keep the `481`-instruction Half-Split CIOS backbone and split $a_{\text{hi}} = H_0 + 2^{256} H_1$. Compute width-4 symmetric $H_0^2$ on lanes `0..3` **simultaneously** with width-4 symmetric $H_1^2$ on lanes `4..7` (`9` wide muls/lane, landing directly in-place on lanes `0..3` and `4..7` with **zero cross-half shuffles**!), plus width-8 symmetric $|H_0 - H_1|^2$ (`5` wide muls/lane; since $|H_0 - H_1| < 2^{256}$, `v_top == 0` identically!) using $H_0^2 + H_1^2 - |H_0 - H_1|^2 = 2 H_0 H_1$ (`14` wide muls/lane vs. `17`).
  3. **Full `1024`-Bit Toom-4 (`4x256` bits) + Carry-Save 1024-Bit REDC (`168` `IMAD.WIDE`s/lane)**:
     Evaluate $a(x) = A_0 + A_1 x + A_2 x^2 + A_3 x^3$ at $\{0, \infty, 1, -1, 2, -2, \frac{1}{2}\}$ (`7` squares = **`40` `IMAD.WIDE`s/lane** for the entire `2048`-bit $a^2$, down `-39.4%` from `66`!), interpolate via Bodrato's Toom-4 sequence (exact 2-adic `/3` via `0xAAAAAAABu` and `/15` via `0xEEEEEEEFu`), and reduce via `wmad_reduce_1024_tpi8_l4` (`128` `IMAD.WIDE`s) $\implies$ **`168` `IMAD.WIDE`s total** (`-34.4%` vs `mont_mul`'s `256`). The critical experimental test on L4 is whether Toom-3/Toom-4's `-21` to `-41` `IMAD.WIDE` savings outweigh subwarp evaluation/interpolation `IADD3`/`SHFL` instructions.

### Direction 2: Modulus Pre-Scaling / Trivialized Quotient (`M = k * N` with $M \equiv -1 \pmod{2^w}$, Örup 1995)
- **Eliminating `q = r0 * np0` and `n[0] * q` Inside Repeated Squarings**:
  In long chains of modular squarings ($b = a^{2^T} \bmod N$, e.g. VDFs, RSA/Miller-Rabin, ECM/P-1 factoring), $N$ is fixed across hundreds or millions of iterations. Following Örup (1995), if $N < 2^{988}$ (36 spare bits) and we pre-scale the modulus once outside the loop to $M = k \cdot N$ with $k = -N^{-1} \bmod 2^{32} \in [1, 2^{32}-1]$, then $M \equiv -1 \pmod{2^{32}}$ ($m_0 = \text{0xFFFFFFFF}$ and $\text{mp0} = 1$)!
- **Quantitative Payoff**:
  1. The Montgomery quotient simplifies to **`q = r0` (`0` instructions!)**, removing all `32` `IMAD` instructions (`q = r0 * np0`) from the critical `SHFL.IDX -> IMAD -> IMAD.WIDE` inner-loop latency chain (`-128` cycles of serial dependency per squaring!).
  2. Since $m_0 \cdot q + r_0 = (2^{32} - 1) r_0 + r_0 = 2^{32} r_0$, the entire lane-0 low-limb multiply `madlo(m[0], q, ra[0])` + `madhi(m[0], q, ra[1])` collapses to adding `q` directly into `ra[1]`! Even for `1018`-bit moduli, a 4-bit or 6-bit pre-scale ($M = k N \equiv -1 \pmod{2^6}$) or shifting $R$ to `TPI = 8, LIMBS = 5` (`1280` bits) applies directly, with a single `% N` reduction at the very end of the `256`-iteration chain.

### Direction 3: Multi-Instance `ILP = 2` Interleaving per `TPI = 8` Subwarp
- **Amortizing Control Instructions & Hiding MIO Latency Without High Warp Occupancy**:
  Currently, each `TPI = 8` subwarp processes 1 instance (`4` instances per warp), requiring `REG <= 48` (`40` warps/SM) to hide the ~22-cycle `SHFL.IDX -> q -> IMAD.WIDE` round-trip across warps.
- Processing **2 independent instances `(a^(0), n^(0)), (a^(1), n^(1))` interleaved inside one `TPI = 8` subwarp**:
  1. **Halves all shared index/mask/loop overhead** (`lut_o, lut_e` unpacks, loop branches, `SHFL` lane calculations) per squaring (`-25` to `-30` instructions/squaring).
  2. Interleaves Instance B's `IMAD.WIDE` chains inside Instance A's `SHFL.IDX -> q` MIO stall window (`ILP = 2`), allowing `20–24` warps/SM (`REG = 80..96`) to saturate the `IMAD.WIDE` pipe even when kernels are embedded inside larger register-heavy cryptographic protocols.

### Direction 4: Ada 4th-Gen `INT8` / `FP16` Tensor Core Limb-Sliced Convolution
- **Hardware Asymmetry on NVIDIA L4 (`AD104`, `sm_89`)**:
  L4's scalar integer pipe delivers `~15.1 TOPS` of 32-bit `IMAD` (`~7.55 G` `IMAD.WIDE.U32`/s across 58 SMs), whereas its 4th-Gen Tensor Cores deliver **`242.5 TOPS` dense `INT8` (`485 TOPS` with sparsity)** via `mma.sync.aligned.m16n8k32.s32.u8.u8.s32` — **over `16x` higher raw bit-multiply throughput** on a separate hardware execution pipe!
- **Algorithmic Mapping (`cuMP` / `TensorBigInt` style)**:
  Slice each `1024`-bit operand into `128` unsigned 8-bit digits ($B = 2^8$), where column convolutions $\sum_{i+j=s} a_i a_j \le 128 \times 255^2 = 8,323,200 < 2^{23} \ll 2^{31}$ never overflow `s32` Tensor Core accumulators! Batching `8` or `16` instances per warp maps both $S = a^2$ and block Montgomery reduction ($Q = S_{\text{lo}} N' \bmod R$, $T = S + Q N$) to Toeplitz/circulant `m16n8k32` `INT8` MMA tiles while the CUDA Core ALU (`IADD3.X` + `PRMT` + `SHFL`) executes carry/radix normalization concurrently in software-pipelined stages.

### Direction 5: IEEE-754 Mantissa-Splitting Modular Squaring (`DFMA` on `A100/H100/B200`, `FFMA` on `L4`)
- **Connecting to *Emmart, Zheng & Weems (ARITH 2018)* (`PID3563399.pdf`)**:
  Emmart et al. showed that representing big integers in redundant floating-point limbs (52-bit `FP64` radix $2^{52}$ or $2^{51}$ with `fma.rz.f64` + `add.rn.f64` error-free extraction) eliminates integer `CC.CF` carry-flag serialization and scales limb width from 32 bits to 51–52 bits (`20` limbs instead of `32` for `1020`-bit moduli $\implies \binom{21}{2} = 210$ scalar `DFMA`s for symmetric squaring vs. `528` 32-bit products!).
- **Architecture Selection**:
  - On **datacenter `1:2` FP64 GPUs (`V100`, `A100`, `H100`, `B200`)**, `DFMA` runs at full/half tensor-class throughput alongside `IMAD`, making a `TPI = 4, LIMBS = 5` (radix-$2^{51}$, `1020`-bit) symmetric Half-Split CIOS in `FP64` a prime candidate to `2x` throughput.
  - On **L4 (`AD104`)**, `FP64` is rate-limited at `1:64` (`0.47 TFLOPS`), so pure `FP64` loses; however, L4's **`FP32` pipe delivers `30.3 TFLOPS` (`2x` `INT32` throughput)** and issues **concurrently** with integer `IADD3`/`LOP3` on the dual SM data-paths! Offloading `a_hi^2` (sliced into `32` 16-bit digits in redundant `FP32` `FFMA` accumulators, since $32 \times (2^{16})^2 = 2^{37}$ requires splitting into 11-bit or centered signed `[-2^{10}, 2^{10}]` limbs: $48 \times 2^{20} < 2^{26}$ with Kotov-type `fadd_rn` high-part extraction) can overlap `a_hi^2` on the `FFMA` pipe simultaneously with integer CIOS on the `IMAD` pipe.

### Direction 6: Warp-Transposed (`SoA`) Coalesced Memory Layout & Fused Windowed `mont_sqr_k`
- **Coalesced `SoA` Instance Storage (Crucial for Short-Chain `iterations = 1..20` Workloads)**:
  CGBN's default `instance_t` stores `{a[32], n[32], r[32]}` contiguously per instance (`384 B` stride). Across 4 `TPI = 8` instances in a warp, loading `a[0..3]` touches 4 strided 16B slices separated by `384 B` (`4` separate 128B L1 cache lines = `25%` bus utilization). Transposing global storage to warp-interleaved `SoA` (`uint4` per lane, `512 B` contiguous per warp) cuts global memory transactions by **`4x`**, visibly accelerating low-iteration regimes (`iterations = 1..20`, where `mont_sqr` finishes in `1.542 ms` and load/store setup + `bn2mont`/`mont2bn` become non-trivial).
- **Fused `mont_sqr_k(r, a, n, np0, k)` API**:
  Providing a first-class CGBN core primitive `mont_sqr_k` that executes $k$ back-to-back squarings in registers keeps `lut_o, lut_e`, the 6 lane predicates, `shfl_mask`, `n[0..3]`, and `np0` hoisted by design across all callers (e.g., sliding-window modular exponentiation `window_bits = 5..6`) without relying on whole-loop compiler LICM.

### Direction 7: Custom SASS Post-Pass Register Pairing & Crossing the `REG <= 40` (`100%` Occupancy) Cliff
- **Eliminating `ptxas`'s `23` Alignment `MOV`s and Out-of-Line Stub Clobbers**:
  In `lean-halfsplit-pred` (`632` dynamic insts), `23` `MOV`s come from `ptxas` shuffling 32-bit scalars into aligned `(R_{2k}, R_{2k+1})` register pairs required by `IMAD.WIDE.U32`, and `3` high registers (`R45..R47`) are pinned across the out-of-line `__cuda_sm70_shfl_sync` slow-path stub (`0x8770`). A lightweight SASS post-pass (e.g. `CuAssembler`) or pure inline-PTX register-pair state machine (`uint64_t ra01, ra23, ra45, ru01, ru23`) combined with warp-padded launch grids (guaranteeing `shfl_mask == 0xffffffff` so `ptxas` omits the slow-path shuffle stub completely) can simultaneously drop dynamic instructions to `~605` **and** drop register count from `48` to `<= 40` (`48` warps/SM = **`100%` SM occupancy**).

