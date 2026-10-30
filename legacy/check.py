import pandas as pd
need={'DPQ':['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090'],
'DEMO':['RIDAGEYR','INDFMPIR','WTMEC2YR','SDMVSTRA','SDMVPSU'],'SMQ':['SMQ020','SMQ040'],'BMX':['BMXBMI'],
'DIQ':['DIQ010'],'MCQ':['MCQ160B','MCQ160C','MCQ160E','MCQ160F','MCQ220'],'PAQ':['PAQ605','PAQ620','PAQ650','PAQ665']}
for s in 'EFGHIJ':
    for c,v in need.items():
        d=pd.read_sas(f'raw/{c}_{s}.xpt',format='xport')
        miss=[x for x in v if x not in d.columns]
        if miss: print(s,c,'MISSING',miss)
    print(s,'checked')
