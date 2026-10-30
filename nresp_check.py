import json, numpy as np, pandas as pd
from graph2 import *
from analysis2 import graph, dev
rng=np.random.default_rng(99)
Ed,_=graph(dev); out={'obs_edges':len(Ed),'obs_total_weight':float(Ed.mu.sum()),'obs_max':float(Ed.mu.max())}
ne,tw,mx=[],[],[]
for b in range(50):
    En,_=graph(respondent_permute(dev,rng)); ne.append(len(En)); tw.append(En.mu.sum()); mx.append(En.mu.max() if len(En) else 0)
out.update({'null_edges_mean':float(np.mean(ne)),'null_edges_range':[int(min(ne)),int(max(ne))],'null_total_weight_mean':float(np.mean(tw)),'null_max_weight_mean':float(np.mean(mx)),'n':50})
json.dump(out,open('results2/nresp_density.json','w'),indent=1);print(out)
