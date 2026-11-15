"""Exact leave-dyad-out best-path composition (2-3 edges) for any t-norm.
Naive enumeration O(n^4) vs (max,T)-product with top-2 exclusion O(n^3)."""
import numpy as np, time, sys, json
def T_prod(x,y): return x*y
def T_min(x,y): return np.minimum(x,y)
def T_luk(x,y): return np.maximum(0,x+y-1)
def T_ham(x,y):
    d=x+y-x*y
    return np.where(d>0, x*y/np.where(d>0,d,1), 0.0)
def T_yag(x,y): return np.maximum(0,1-np.sqrt((1-x)**2+(1-y)**2))
TN={'product':T_prod,'hamacher':T_ham,'minimum':T_min,'yager':T_yag,'lukasiewicz':T_luk}

def naive(A,T):
    n=len(A); out=np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            ks=[k for k in range(n) if k not in (i,j)]
            m=max(T(A[i,k],A[k,j]) for k in ks) if ks else 0.0
            for k in ks:
                for l in ks:
                    if l==k: continue
                    v=T(T(A[i,k],A[k,l]),A[l,j])
                    if v>m: m=v
            out[i,j]=m
    return out

def naive_vec(A,T):
    """O(n^4) enumeration, vectorised per source i."""
    n=len(A); out=np.zeros_like(A); idx=np.arange(n)
    for i in range(n):
        two=T(A[i,:,None],A)                      # [k,j]
        two[i,:]=0; 
        m2=two.copy(); m2[idx,idx]=0              # k==j
        best2=m2.max(0)
        first=T(A[i,:,None],A)                    # [k,l]
        first[i,:]=0; first[:,i]=0; first[idx,idx]=0
        three=T(first[:,:,None],A[None,:,:])      # [k,l,j]
        three[:,:,i]=0
        three[idx,:,idx]=0                        # k==j
        three[:,idx,idx]=0                        # l==j
        best3=three.max((0,1))
        r=np.maximum(best2,best3); r[i]=0; out[i]=r
    return out

def fast(A,T,chunk=64):
    """O(n^3): 2-hop (max,T) product; 3-hop with top-2 exclusion of k=j and mask l=i."""
    n=len(A); A=A.copy(); np.fill_diagonal(A,0)
    best=np.zeros((n,n)); arg=np.full((n,n),-1); sec=np.zeros((n,n))
    for s in range(0,n,chunk):                    # B[i,l]=max_k T(A[i,k],A[k,l]) with top-2
        V=T(A[s:s+chunk,:,None],A[None,:,:])      # [i,k,l]
        a=V.argmax(1); b=np.take_along_axis(V,a[:,None,:],1)[:,0,:]
        np.put_along_axis(V,a[:,None,:],-1.0,1)
        best[s:s+chunk]=b; arg[s:s+chunk]=a; sec[s:s+chunk]=V.max(1)
    two=best.copy()                                # k==i or k==j give T(0,.)=0 automatically
    idx=np.arange(n); out=np.zeros((n,n))
    for j in range(n):
        Bj=np.where(arg==j,sec,best)               # exclude intermediate k=j
        V=T(Bj,A[None,:,j])                        # [i,l]
        V[idx,idx]=0                               # exclude l=i
        V[:,j]=0
        out[:,j]=np.maximum(two[:,j],V.max(1))
    np.fill_diagonal(out,0)
    return out

def rand_graph(n,rng,density=0.85):
    A=rng.lognormal(np.log(0.08),1.0,(n,n)).clip(0,0.95)
    A*=rng.random((n,n))<density; np.fill_diagonal(A,0); return A

if __name__=='__main__':
    rng=np.random.default_rng(20261115); res={'exact':{},'timing':[]}
    # exactness: scalar enumeration on n=12, vectorised enumeration on n=24 and 60
    for name,T in TN.items():
        mx=0.0
        for r in range(20):
            A=rand_graph(12,rng); mx=max(mx,np.abs(naive(A,T)-fast(A,T)).max())
        for n in (24,60):
            for r in range(10):
                A=rand_graph(n,rng); mx=max(mx,np.abs(naive_vec(A,T)-fast(A,T)).max())
        # stress: uniform weights in [0,1], ties
        for r in range(10):
            A=np.round(rng.random((24,24)),1); np.fill_diagonal(A,0)
            mx=max(mx,np.abs(naive_vec(A,T)-fast(A,T)).max())
        res['exact'][name]=float(mx)
    for n in (24,50,100,200,400,800):
        A=rand_graph(n,rng); T=T_ham
        t0=time.perf_counter(); fast(A,T); tf=time.perf_counter()-t0
        tn=None
        if n<=200:
            t0=time.perf_counter(); naive_vec(A,T); tn=time.perf_counter()-t0
        res['timing'].append({'n':n,'fast_s':round(tf,4),'enum_s':None if tn is None else round(tn,3)})
    print(json.dumps(res,indent=1))
