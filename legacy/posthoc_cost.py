"""POST-HOC (not pre-specified): binarisation cost under N2 null graphs."""
import json, numpy as np
from analysis import build, rq1, dev
from graph import n2_permute
rng=np.random.default_rng(777)
E=build(dev); obs,_,_=rq1(E,restarts=10,seed0=0)
costs={k:[] for k in obs['binarised']}
for b in range(200):
    o,_,_=rq1(n2_permute(E,rng),restarts=10,seed0=5000+b)
    for k,v in o['binarised'].items(): costs[k].append(v['cost'])
out={str(k):{'obs':obs['binarised'][k]['cost'],'null_mean':float(np.mean(v)),'null_q025':float(np.quantile(v,.025)),
   'null_q975':float(np.quantile(v,.975)),'p_ge':float((1+np.sum(np.array(v)>=obs['binarised'][k]['cost']))/201)} for k,v in costs.items()}
json.dump(out,open('results/posthoc_cost_null.json','w'),indent=1); print(json.dumps(out,indent=1))
