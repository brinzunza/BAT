"""Test information from the currency-price identity, not own-price thresholds."""
import json
import numpy as np
import pandas as pd
import causal_native as native
from prepare import DATA,SYMBOLS
from search import compact,score,ts

OUT=DATA/'causal_discovery'/'basis';OUT.mkdir(exist_ok=True)

def main():
    markets=[np.load(DATA/'causal_discovery'/'market_cache'/(s+'.npy'),mmap_mode='r') for s in SYMBOLS]
    results=[];all_scores=[]
    for timeframe in [1,5,15]:
        folder=DATA/'causal_discovery'/'structural'
        if timeframe>1:folder=folder/f'{timeframe}min'
        for base in ['reclaim_1r','failure_2r','expansion_1r']:
            source=np.load(folder/(base+'_opportunities.npy'),mmap_mode='r')
            a=np.ascontiguousarray(source[source[:,0]<ts('2018-03-09')])
            values=native.basis_features(markets,np.ascontiguousarray(a[:,0]))
            inst=a[:,2].astype(int);sign=2*a[:,3]-1
            z=values[:,0]*np.where(inst==0,-1.,1.)*sign
            raw=values[np.arange(len(a)),np.minimum(inst,2)+1]*sign
            innovation=values[np.arange(len(a)),np.minimum(inst,2)+4]*sign
            z[inst==3]=raw[inst==3]=innovation[inst==3]=np.nan
            a=np.column_stack([a,z,raw,innovation])
            # USDJPY has no identified triangular counterpart in this four-pair graph;
            # it cannot pass these graph-specific rules (no fabricated feature value).
            rules=[]
            for f,levels in [(28,[0,1,2,3]),(29,[0,.5,1,1.6,2,3,5,8]),(30,[0,.5,1,1.6,2,3,5])]:
                for threshold in levels:
                    rules.append([[f,1,threshold]])
                    rules.append([[f,-1,-threshold]])
            # Demeaned dislocation + transaction-cost quality; no fit to future returns.
            rules += [[[30,1,x],[23,-1,c]] for x in [.5,1.6,3] for c in [.05,.10,.20]]
            trials=[]
            for rule in rules:
                for risk in [.005,.01,.02,.04,.06,.08,.10]:
                    d=compact(native.account(a,np.array(rule,dtype=float),risk,ts('2013-01-08'),ts('2016-01-01')))
                    r=dict(timeframe=timeframe,base=base,rules=rule,risk=risk,train=d,score=score(d));trials.append(r)
            trials.sort(key=lambda r:r['score'],reverse=True);all_scores+=trials
            shortlist=[];seen=set()
            for r in trials:
                key=json.dumps(r['rules'])
                if key in seen:continue
                seen.add(key);shortlist.append(r)
                if len(shortlist)==6:break
            for r in shortlist:
                d=compact(native.account(a,np.array(r['rules'],dtype=float),r['risk'],ts('2016-01-08'),ts('2018-03-09')))
                r['validation']=d;r['eligible']=r['train']['pf']>1 and d['pf']>1 and r['train']['n']>=200 and d['n']>=200
                results.append(r)
            np.save(OUT/f'{timeframe}min_{base}_pre2018.npy',a)
            best=max(shortlist,key=lambda r:score(r['validation']))
            print(timeframe,base,'train PF',best['train']['pf'],'validation PF',best['validation']['pf'],'n',best['validation']['n'],flush=True)
    (OUT/'shortlist.json').write_text(json.dumps(results,indent=2))
    (OUT/'discovery.json').write_text(json.dumps(all_scores,indent=2))
    eligible=[r for r in results if r['eligible']]
    if eligible:
        winner=max(eligible,key=lambda r:score(r['validation']));winner['frozen_at']=pd.Timestamp.now(tz='UTC').isoformat()
        (OUT/'frozen_candidate.json').write_text(json.dumps(winner,indent=2))
        print('FROZEN',winner,flush=True)
    else:print('No qualifying currency-graph candidate',flush=True)

if __name__=='__main__':main()
