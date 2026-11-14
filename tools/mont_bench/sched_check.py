# brute-force validation of the generic folded target-sum schedule (odd/even/mid) + own-prefix property
import random
def check(T, M):
    H=T//2; K=H//2; ML=M//2
    B=1<<(32*M)
    D=[random.getrandbits(32*M) for _ in range(T)]
    own_o=[0]*T; oth_o=[0]*T; own_e=[0]*T; oth_e=[0]*T
    seen={}
    for g in range(T):
        t=g%H; up=g>=H
        c_o=min(H-1-t,K) if up else min(t+1,K)
        c_e=min(H-1-t,K) if up else min(t,K)
        acc=0; save=0
        for s in range(K):
            if s==c_o: save,acc=acc,0
            p=(H-1-s) if up else s
            low=p<=t
            i,j=(p,2*t+1-p) if low else (t+1+p,T+t-p)
            assert 0<=i<j<T, (T,g,s,i,j)
            own=(low!=up)
            assert own==(s<c_o),(T,g,s)
            tgt=(t if low else t+H); assert i+j==2*tgt+1
            seen[(i,j)]=seen.get((i,j),0)+1
            acc+=D[i]*D[j]
        if c_o==K: save,acc=acc,0
        own_o[g],oth_o[g]=save,acc
        acc=0; save=0
        for e in range(K):          # K-1 full rounds + mid (e==K-1)
            if e==c_e: save,acc=acc,0
            mid=(e==K-1)
            p=(K-1) if mid else ((H-2-e) if up else e)
            low=p<t
            i,j=(p,2*t-p) if low else (t+1+p,T-1+t-p)
            assert 0<=i<j<T, (T,g,e,i,j)
            own=(low!=up)
            assert own==(e<c_e),(T,g,e,own,c_e)
            tgt=(t if low else t+H); assert i+j==2*tgt
            if mid:
                lo=D[j]&((1<<(32*ML))-1); hi=D[j]>>(32*ML)
                acc+= (D[i]*hi)<<(32*ML) if up else D[i]*lo
                seen[(i,j)]=seen.get((i,j),0)+0.5
            else:
                acc+=D[i]*D[j]; seen[(i,j)]=seen.get((i,j),0)+1
        if c_e==K: save,acc=acc,0
        own_e[g],oth_e[g]=save,acc
    for i in range(T):
        for j in range(i+1,T): assert seen.get((i,j),0)==1,(T,i,j,seen.get((i,j)))
    tot=0
    for g in range(T):
        O=own_o[g]+oth_o[g^H]; E=own_e[g]+oth_e[g^H]
        tot+=(E+O*B)<<(64*M*g)
    ref=sum(D[i]*D[j]<<(32*M*(i+j)) for i in range(T) for j in range(i+1,T))
    assert tot==ref
    # top lane odd sum is zero, sizes fit
    assert own_o[T-1]+oth_o[(T-1)^H]==0
for T in (4,8,16,32):
    for M in (2,3,4,5):
        for _ in range(20): check(T,M)
print("schedule OK")
