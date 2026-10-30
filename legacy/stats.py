import json, numpy as np
R1=json.load(open('results/rq1.json')); R2=json.load(open('results/rq2.json')); P=json.load(open('results/pathways.json'))
import pandas as pd; V=pd.read_csv('results/vignettes.csv')
top1=V.top3_product.apply(lambda s: eval(s)[0]).value_counts().to_dict()
S={'rq1':{a:{'edges':v['edges_kept'],'Q':round(v['Q_on_GF_mean'],3),'cost':round(v['cost'],3),
   'ci':[round(x,3) for x in R1['rq1_boot_ci'][str(a)]],'NMI':round(v['NMI'],2),'ARI':round(v['ARI'],2),'ncomm':v['n_comm']}
   for a,v in R1['rq1_main']['binarised'].items()},
 'Qbar':round(R1['rq1_main']['Q_mean'],3),'ncomm':R1['rq1_main']['n_comm'],
 'rq2':{k:{kk:round(vv,4) for kk,vv in v.items()} for k,v in R2.items()},
 'heldout':{'Q':round(R1['heldout_rq1']['Q_mean'],3),'ncomm':R1['heldout_rq1']['n_comm'],'NMI':round(R1['heldout_NMI_with_dev'],2),
   'ARI':round(R1['heldout_ARI_with_dev'],2),'r':round(R1['heldout_weight_corr']['pearson'],3),'maxdiff':round(R1['heldout_weight_corr']['max_abs_diff'],3),
   'costs':{k:round(v['cost'],3) for k,v in R1['heldout_rq1']['binarised'].items()}},
 'boot_sd':[round(R1['edge_weight_boot_sd_median'],4),round(R1['edge_weight_boot_sd_max'],4)],
 'sens':{k:{'Q':round(v['Q_mean'],3),'ncomm':v['n_comm'],'costs':{kk:round(vv['cost'],3) for kk,vv in v['binarised'].items()}} for k,v in R1['sensitivity'].items()},
 'path':{'cmp':{k:{kk:round(vv,2) for kk,vv in v.items()} for k,v in P['comparison_vs_product'].items()},'flow':P['flow'],
   'distinct':[P['distinct_top3_product'],P['distinct_top3_binary']],'top1_product':top1,
   'binary_zero_reach':int((V.n_binary_reachable==0).sum()),'binary_one_reach':int((V.n_binary_reachable==1).sum())}}
json.dump(S,open('results/paper_stats.json','w'),indent=1,default=str); print(json.dumps(S,indent=1,default=str))
