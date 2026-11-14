import re,sys,collections
want=sys.argv[1]  # e.g. "sqr 8 2048"
kind,t,b=want.split()
ins=[];cur=False
for line in open("/content/sass.txt",errors="replace"):
    m=re.search(r"Function : (\S+)",line)
    if m:
        cur=("kernel_mont_%sI19mont_bench_params_tILj%sELj%sE"%(kind,t,b)) in m.group(1); continue
    if cur:
        m=re.search(r"/\*([0-9a-f]{4,})\*/\s+(@!?P\d\s+)?([A-Z0-9_.]+)\s*(.*?);",line)
        if m: ins.append((int(m.group(1),16),m.group(3),m.group(4),bool(m.group(2))))
# main loop = backward branch span with most IMAD.WIDE
best=None
for a,op,rest,_ in ins:
    if op.startswith("BRA"):
        tm=re.search(r"0x([0-9a-f]+)\s*$",rest.strip())
        if tm and int(tm.group(1),16)<a:
            lo=int(tm.group(1),16); n=sum(1 for x in ins if lo<=x[0]<=a and x[1].startswith("IMAD.WIDE"))
            if best is None or n>best[0]: best=(n,lo,a)
_,lo,hi=best
body=[x for x in ins if lo<=x[0]<=hi]
NODEST=("ISETP","BRA","STG","VOTE","BSSY","BSYNC","NOP","RET","EXIT","R2UR","WARPSYNC","CALL","BMSK")
defs=[];uses=[]
for a,op,rest,pred in body:
    regs=[int(x) for x in re.findall(r"\bR(\d+)\b",rest)]
    d=[];u=regs
    if regs and not op.startswith(NODEST):
        first=re.match(r"\s*R(\d+)",rest)
        if first:
            r0=int(first.group(1)); d=[r0]; u=regs[1:]
            if op.startswith("IMAD.WIDE"): d=[r0,r0+1]
            # 64-bit addend of IMAD.WIDE: Rn implies Rn+1
            if op.startswith("IMAD.WIDE") and len(u)>=3: u=u+[u[-1]+1]
            if pred: u=u+d   # predicated write keeps the old value live
    defs.append(set(d));uses.append(set(u))
N=len(body)
# backward liveness, treating loop as cyclic (2 passes)
live=set();L=[None]*N
for _ in range(2):
    for i in range(N-1,-1,-1):
        live=(live-defs[i])|uses[i]
        L[i]=set(live)
cnt=[len(x) for x in L]
print("%s: loop insts=%d maxlive=%d at idx %d (addr %#x) maxreg=%d"%(want,N,max(cnt),cnt.index(max(cnt)),body[cnt.index(max(cnt))][0],max(max(d) for d in defs if d)))
W=max(1,N//40)
for i in range(0,N,W):
    seg=body[i:i+W]; h=collections.Counter()
    for a,op,rest,_ in seg:
        k="W" if op.startswith("IMAD.WIDE") or op.startswith("IMAD.HI") else "S" if op.startswith("SHFL") else "A" if op.startswith("IADD") else "L" if op.startswith("SEL") else "M" if "MOV" in op else "B" if op.startswith("BRA") else "o"
        h[k]+=1
    print("%5d %#07x live max=%2d min=%2d  %s"%(i,seg[0][0],max(cnt[i:i+W]),min(cnt[i:i+W])," ".join("%s%d"%(k,h[k]) for k in "WSALMBo" if h[k])))
alldefs=set().union(*defs)
inv=[len(x-alldefs) for x in L]
print("loop-invariant live regs (never written in loop): max=%d"%max(inv), sorted(L[cnt.index(max(cnt))]-alldefs))
# at coarse points, show for live non-invariant regs how long ago they were defined
lastdef={}
age_at={}
for i in range(N):
    for r in defs[i]: lastdef[r]=i
    if i%W==0:
        ages=sorted(i-lastdef.get(r,-1) for r in L[i] if r in alldefs and r in lastdef)
        b=collections.Counter("<50" if a<50 else "<200" if a<200 else "<600" if a<600 else ">=600" for a in ages)
        print("%5d live=%2d inv=%2d ages: %s"%(i,cnt[i],inv[i],dict(b)))
