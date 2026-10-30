import numpy as np, pandas as pd, itertools
from graph import *
df=pd.read_pickle('data_dev.pkl')
E=edges_df(data_edges(df)+literature_edges())
print('edges by rel/prov:'); print(E.groupby(['rel','prov']).mu.agg(['count','min','median','max']).round(3))
A=adj(E)
q,m,qs=best_q(A,range(20))
print('bestQ',round(q,4),'mean',round(qs.mean(),4),'ncomm',len(set(m)))
# check leidenalg directed quality matches mine: quality/(W) should equal qf
import igraph as ig, leidenalg as la
src,dst=np.nonzero(A); g=ig.Graph(n=A.shape[0],edges=list(zip(src,dst)),directed=True); g.es['weight']=A[src,dst]
p=la.RBConfigurationVertexPartition(g,initial_membership=m.tolist(),weights='weight')
print('la quality/W',p.quality()/A.sum(),'mine',qf(A,m))
# all-in-one and singleton
print('one comm',qf(A,np.zeros(len(NODES),int)),'by type',qf(A,np.array([['Symptom','Risk','Mechanism','Component','Outcome'].index(TYPE[n]) for n in NODES])))
for c,k in zip(NODES,m): pass
print({k:[n for n,mm in zip(NODES,m) if mm==k] for k in sorted(set(m))})
# null degree preservation
rng=np.random.default_rng(1); E1=n1_rewire(E,rng)
for col in ['src','dst']:
    a=E.groupby(['rel',col]).size(); b=E1.groupby(['rel',col]).size(); assert a.equals(b.reindex(a.index)), col
print('N1 preserves typed degrees; dup edges:',E1.duplicated(['src','dst']).sum(),'selfloops',(E1.src==E1.dst).sum())
# t-norm brute force check on a small profile
act={s:0.0 for s in SYM}; act['depressed_mood']=2/3; act['sleep']=1/3; act['concentration']=1.0
P=path_edges(E); adjl={}
for u,v,mu in zip(P.src,P.dst,P.mu): adjl.setdefault(u,[]).append((v,mu))
def allpaths(u,seen,prod):
    if u==OUT[0]: yield prod; return
    for v,mu in adjl.get(u,[]):
        if v not in seen: yield from allpaths(v,seen|{v},prod+[mu])
best={}
for s,a in act.items():
    if a<=0: continue
    def walk(u,seen,ws,comp):
        if u==OUT[0]:
            yield comp, ws; return
        for v,mu in adjl.get(u,[]):
            if v not in seen: yield from walk(v,seen|{v},ws+[mu], v if TYPE[v]=='Component' else comp)
    for comp,ws in walk(s,{s},[a],None):
        best[comp]=max(best.get(comp,0),float(np.prod(ws)))
sc=component_scores(E,act,'product')
print('max abs diff brute vs dijkstra (product):',max(abs(best.get(c,0)-sc[c]) for c in COMP))
