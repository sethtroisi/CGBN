# Montgomery benchmark and test tools (not for upstream)

This branch sits on top of the `code_review/*` branches and only adds what was used to measure and
check them.  Nothing here is needed by the library.

| Path | Purpose |
|---|---|
| `samples/sample_05_mont_sqr` | Benchmark: chained `cgbn_mont_sqr` / `cgbn_mont_mul` on 102,400 random instances, verified with GMP.  `-m sqr|mul`, `-b bits`, `-t tpi`, `-f` (full width inputs), `-i iterations`.  Build with `-DBENCH_BIG` for 1536 / 3072 / 4096 bits and `-t 4` above 512 bits. |
| `samples/sample_06_mont_sqr_edge` | Edge test: `mont_sqr` (out of place and in place), `mont_mul(a, a)`, `bn2mont` and the `mont2bn` round trip against GMP, on special and structured operands with a partial final warp, for 21 (TPI, bits) configurations (padded, odd LIMBS and LIMBS>TPI included).  `-DEDGE_NO_WIDE` leaves out LIMBS>TPI, for branches below `code_review/wide-limbs`. |
| `tools/mont_bench/bench.sh` | Upload a worktree to a Colab session, build the benchmark once, run several configurations, append to a CSV. |
| `tools/mont_bench/edge.sh`, `unit.sh` | Build and run the edge test / CGBN's `unit_tests` on the Colab session. |
| `tools/mont_bench/sass.sh`, `live.py` | Register counts, SASS opcode histograms and a live register profile of the main loop. |
| `tools/mont_bench/sched_check.py` | Brute force check of the `mont_sqr_high` schedule for 4, 8, 16 and 32 threads. |

The scripts expect the `colab` CLI and a running GPU session named `cgbn-bench`.  The CLI sometimes
answers `HTTP 429: Already-active SSH session` for a while after a run; the scripts retry.

Timing comparisons need to be interleaved in one session (A B B A): single runs differ by about 1%, and
sessions by several percent.

## Results (NVIDIA L4, sm_89, 102,400 instances, 10 timed runs each, GMP verified)

One session, order master / review / peak / peak / review / master, mean of the two runs.  "review" is
`code_review/wide-limbs`, "peak" is the experimental branch `generic-halfsplit-sqr`, times are ms for
1024 iterations (512 at 1536 / 2048 bits, 256 at 3072 / 4096 bits).

| Operation | Bits | TPI | master | review | peak | review vs peak | master / review |
|---|---:|---:|---:|---:|---:|---:|---:|
| mont_mul | 512 | 4 | 33.812 | 27.844 | 27.358 | +1.8% | 1.214x |
| mont_mul | 1024 | 4 | - | 93.008 | 93.269 | -0.3% | |
| mont_mul | 1024 (full width) | 4 | - | 94.600 | 94.894 | -0.3% | |
| mont_mul | 1024 | 8 | 126.496 | 109.626 | 109.888 | -0.2% | 1.154x |
| mont_mul | 1024 | 16 | 173.924 | 145.706 | | | 1.194x |
| mont_mul | 2048 | 4 | - | 172.544 | 173.245 | -0.4% | |
| mont_mul | 2048 | 8 | 199.289 | 186.572 | 186.576 | 0.0% | 1.068x |
| mont_mul | 2048 | 16 | 244.888 | 222.557 | | | 1.100x |
| mont_mul | 4096 | 8 | - | 350.247 | 349.796 | +0.1% | |
| mont_mul | 4096 | 16 | | 375.980 | 375.989 | 0.0% | |
| mont_sqr | 1024 | 4 | - | 91.232 | 91.928 | -0.8% | |
| mont_sqr | 1024 (full width) | 4 | - | 94.156 | 94.621 | -0.5% | |
| mont_sqr | 1024 | 8 | 127.451 | 110.994 | 108.573 | +2.2% | 1.148x |
| mont_sqr | 1536 | 4 | - | 93.614 | 94.138 | -0.6% | |
| mont_sqr | 2048 | 4 | - | 158.956 | 160.933 | -1.2% | |
| mont_sqr | 2048 (full width) | 4 | - | 161.722 | 163.716 | -1.2% | |
| mont_sqr | 2048 | 8 | 200.632 | 176.476 | 174.969 | +0.9% | 1.137x |
| mont_sqr | 3072 | 8 | - | 179.798 | 181.633 | -1.0% | |
| mont_sqr | 4096 | 8 | - | 310.430 | 316.748 | -2.0% | |
| mont_sqr | 4096 (full width) | 8 | - | 313.833 | 319.818 | -1.9% | |

"-" means the configuration does not compile on master (LIMBS>TPI); blank means not measured.

Against the fastest configuration master can run at each size, the fastest review configuration is
1.36x (mont_mul) and 1.40x (mont_sqr) at 1024 bits, and 1.16x and 1.26x at 2048 bits.

Known gaps to the experimental code: `mont_sqr` at 1024 bits with TPI=8 uses `mont_mul(a, a)` (the
hand built 8 thread schedule was not carried over), and `mont_mul` at 512 bits with TPI=4 is about 2%
slower, with no confirmed cause.

## Checks run on each branch

| Branch | Edge test | CGBN `unit_tests` |
|---|---|---|
| `code_review/resolver-warp-sync` | pass (`-DEDGE_NO_WIDE`) | 880 / 880 |
| `code_review/mont-mul` | pass (`-DEDGE_NO_WIDE`) | 880 / 880 |
| `code_review/mont-sqr` | pass (`-DEDGE_NO_WIDE`) | 1040 / 1040 |
| `code_review/wide-limbs` | pass | 1040 / 1040 |
