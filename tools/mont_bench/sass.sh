#!/usr/bin/env bash
# usage: sass.sh <worktree> "<nvcc extra flags>"  -> per-kernel register count + SASS opcode histogram for kernel_mont_sqr / kernel_mont_mul
set -euo pipefail
WT="$1"; EXTRA="${2:-}"
export PATH=$HOME/.venv/bin:$PATH
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
( cd "$WT" && { git ls-files -z; git ls-files -z --others --exclude-standard; } | tar --null -T - -czf $TMP/cgbn.tar.gz )
colab upload -s cgbn-bench $TMP/cgbn.tar.gz content/cgbn_s.tar.gz >/dev/null
colab ssh -s cgbn-bench -- bash -s -- "$EXTRA" <<'REMOTE'
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
rm -rf /content/cgbn_s && mkdir -p /content/cgbn_s && tar -xzf /content/cgbn_s.tar.gz -C /content/cgbn_s
cd /content/cgbn_s/samples/sample_05_mont_sqr
nvcc -O3 -std=c++14 -Xcompiler -pthread $1 -I../../include -arch=sm_89 mont_sqr.cu -o mont_sqr -lgmp -lpthread 2>&1 | grep -a "error" | head || true
cuobjdump -res-usage mont_sqr 2>/dev/null > /content/res.txt || true
cuobjdump -sass mont_sqr 2>/dev/null > /content/sass.txt
python3 - <<'PY'
import re,collections
reg={}
L=open("/content/res.txt",errors="replace").read().split("\n")
for i,l in enumerate(L):
    k=re.search(r"kernel_mont_(sqr|mul)I19mont_bench_params_tILj(\d+)ELj(\d+)",l)
    if k and i+1<len(L):
        m=re.search(r"REG:(\d+)",L[i+1])
        if m: reg["%s t%s b%s"%k.groups()]=m.group(1)
F={}; cur=None
for line in open("/content/sass.txt",errors="replace"):
    m=re.search(r"Function : (\S+)",line)
    if m:
        cur=None
        k=re.search(r"kernel_mont_(sqr|mul)I19mont_bench_params_tILj(\d+)ELj(\d+)",m.group(1))
        if k: cur="%s t%s b%s"%k.groups(); F[cur]=[]
        continue
    if cur is not None:
        m=re.search(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?P\d\s+)?([A-Z0-9_.]+)(.*?);",line)
        if m: F[cur].append((int(m.group(1),16),m.group(2),m.group(3)))
def cls(op):
    if op.startswith("IMAD.WIDE") or op.startswith("IMAD.HI"): return "WIDE"
    if op.startswith("IMAD.MOV"): return "MOV"
    b=op.split(".")[0]
    return "IMAD32" if b=="IMAD" else b
keys=["TOTAL","WIDE","IMAD32","IADD3","SHFL","SEL","MOV","SHF","LOP3","ISETP","VOTE","BRA"]
for f in sorted(F):
    ins=F[f]; loops=[]
    for a,op,rest in ins:
        if op.startswith("BRA"):
            t=re.search(r"0x([0-9a-f]+)\s*$",rest.strip())
            if t and int(t.group(1),16)<a: loops.append((int(t.group(1),16),a))
    best=None
    for lo,hi in loops:
        inner=[(l2,h2) for l2,h2 in loops if lo<=l2 and h2<=hi and (l2,h2)!=(lo,hi)]
        H=collections.Counter()
        for a,op,rest in ins:
            if lo<=a<=hi and not any(l2<=a<=h2 for l2,h2 in inner):
                H[cls(op)]+=1; H["TOTAL"]+=1
        if best is None or H["WIDE"]>best[0]["WIDE"]: best=(H,lo,hi)
    if best:
        H=best[0]
        print(f, "REG=%s static=%d loop:"%(reg.get(f,"?"),len(ins)), " ".join("%s=%d"%(k,H[k]) for k in keys))
PY
REMOTE
