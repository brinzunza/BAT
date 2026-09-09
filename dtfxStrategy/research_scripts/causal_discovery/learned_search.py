"""Out-of-time synthetic expected-R score, trained by native histogram boosting."""
import json
import time
import numpy as np
import pandas as pd
import causal_native as native
from prepare import OUT,BASES
from search import ts,compact,score

def main():
    started=time.time();results=[]
    for base in BASES:
        source=np.load(OUT/(base+'_opportunities.npy'),mmap_mode='r')
        # The later period is physically excluded from all model/search calls.
        a=np.ascontiguousarray(source[source[:,0]<ts('2018-03-09')])
        fit1=a[a[:,1]<ts('2013-01-01')]
        early=a[(a[:,0]>=ts('2013-01-08'))&(a[:,1]<ts('2016-01-01'))]
        mean1,model1,pred1=native.learn(fit1,early,64)
        b=np.column_stack([early,pred1])
        thresholds=np.unique(np.quantile(pred1,[.1,.25,.5,.75,.9,.95,.99]))
        trials=[]
        for threshold in thresholds:
            for risk in [.0025,.005,.01,.02,.04,.06]:
                rules=np.array([[b.shape[1]-1,1,threshold]])
                d=compact(native.account(b,rules,risk,ts('2013-01-08'),ts('2016-01-01')))
                trials.append(dict(base=base,threshold=float(threshold),risk=risk,early=d,score=score(d)))
        trials.sort(key=lambda x:x['score'],reverse=True)
        chosen=trials[0]
        fit2=a[a[:,1]<ts('2016-01-01')]
        valid=a[(a[:,0]>=ts('2016-01-08'))&(a[:,1]<ts('2018-03-09'))]
        mean2,model2,pred2=native.learn(fit2,valid,64)
        b=np.column_stack([valid,pred2]);rules=np.array([[b.shape[1]-1,1,chosen['threshold']]])
        d=compact(native.account(b,rules,chosen['risk'],ts('2016-01-08'),ts('2018-03-09')))
        chosen['validation']=d;chosen['validation_score']=score(d)
        chosen['eligible']=chosen['early']['pf']>1 and d['pf']>1 and chosen['early']['n']>=200 and d['n']>=200
        chosen['mean']=mean2;chosen['model']=model2.reshape(-1,4).tolist()
        chosen['fit_ends']='2016-01-01';results.append(chosen)
        print(base,'early PF',round(chosen['early']['pf'],3),'validation',d,flush=True)
    results.sort(key=lambda r:r['validation_score'] if r['eligible'] else -1e6,reverse=True)
    (OUT/'learned_results.json').write_text(json.dumps(results,indent=2))
    if results[0]['eligible']:
        (OUT/'frozen_learned_candidate.json').write_text(json.dumps(results[0],indent=2))
        print('Eligible learned candidate frozen',flush=True)
    else:print('No eligible learned candidate',flush=True)
    print('Elapsed',round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
