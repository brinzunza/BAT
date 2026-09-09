"""Diagnose side imbalance; isolated-side runs are diagnostic, not pooled strategies."""
import json
import numpy as np
import causal_native as native
from prepare import DATA
from search import ts,compact

def main():
    root=DATA/'causal_discovery'/'benchmark'
    rows=[]
    for folder in ('base_policies','bounce_policies'):
        cfgs=json.loads((root/folder/'results.json').read_text())['results']
        for cfg in cfgs:
            a=np.load(root/folder/(cfg['name']+'_path'+str(cfg['path'])+'.npy'))
            for side in (-1,0,1):
                x=np.ascontiguousarray(a if side<0 else a[a[:,3]==side])
                d={'name':cfg['name'],'path':cfg['path'],'side':side}
                for label,start,end in [('train','2013-01-08','2016-01-01'),('validation','2016-01-08','2018-03-09')]:
                    d[label]=compact(native.account(x,np.empty((0,3)),.005,ts(start),ts(end)))
                rows.append(d)
    (root/'side_diagnostic.json').write_text(json.dumps(rows,indent=2))
    leaders=[x for x in rows if x['train']['pf']>1 and x['validation']['pf']>1]
    print(json.dumps({'cases':len(rows),'positive_both_periods':[
        {k:v for k,v in x.items()} for x in leaders]},indent=2),flush=True)

if __name__=='__main__':main()
