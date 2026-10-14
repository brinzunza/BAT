"""Causal cross-scale state machine and shared regime rules, discovery only."""
import json
import itertools
import numpy as np
import pandas as pd
import causal_native as native
from prepare import DATA
from search import score,compact,ts

OUT=DATA/'causal_discovery'/'pooled';OUT.mkdir(exist_ok=True)

def main():
    pieces=[]
    for tf in [5,15,60]:
        source=np.load(DATA/'causal_discovery'/'structural'/f'{tf}min'/'failure_2r_opportunities.npy')
        # Only pre-2018 data enter this experiment at all.
        a=np.ascontiguousarray(source[source[:,0]<ts('2018-03-09')])
        fit=a[a[:,1]<ts('2013-01-01')]
        transformed=native.rank_features(fit,a)
        pieces.append(np.column_stack([transformed,np.full(len(a),tf)]))
    a=np.concatenate(pieces);a=a[np.argsort(a[:,0],kind='stable')]
    a=native.consensus(a)
    # Rank simultaneous signals by known cross-scale agreement/volatility, then instrument.
    a=np.ascontiguousarray(a[np.lexsort((a[:,2],-a[:,31],a[:,0]))])
    np.save(OUT/'pre2018_opportunities.npy',a)
    rules=[[]]
    # Feature percentiles are calibrated independently per scale on pre-2013 data.
    for col in [7,8,9,12,14,15,16,17,20,24,25,26,27]:
        for threshold in [.25,.5,.75,.9]:
            for sign in [-1,1]:rules.append([[col,sign,threshold]])
    rules += [[[29,1,x]] for x in [.4,.6,.8]]
    rules += [[[30,-1,x]] for x in [.05,.2,.4]]
    rules += [[[31,1,x]] for x in [0.,.2,.4,.6]]
    train=(ts('2013-01-08'),ts('2016-01-01'));valid=(ts('2016-01-08'),ts('2018-03-09'))
    all_results=[]
    def evaluate(candidate_rules):
        winners=[]
        for r in candidate_rules:
            rows=[]
            for risk in [.005,.01,.02,.04,.06,.08,.10]:
                d=compact(native.account(a,np.array(r,dtype=float).reshape(-1,3),risk,*train))
                row=dict(rules=r,risk=risk,train=d,score=score(d));rows.append(row);all_results.append(row)
            winners.append(max(rows,key=lambda r:r['score']))
        return sorted(winners,key=lambda r:r['score'],reverse=True)
    single=evaluate(rules);leaders=[];seen=set()
    for r in single:
        if r['rules'] and r['rules'][0][0] not in seen:
            seen.add(r['rules'][0][0]);leaders.append(r)
        if len(leaders)==10:break
    paired=evaluate([x['rules']+y['rules'] for x,y in itertools.combinations(leaders,2)])
    shortlist=sorted(single+paired,key=lambda r:r['score'],reverse=True)[:12]
    for r in shortlist:
        d=compact(native.account(a,np.array(r['rules'],dtype=float).reshape(-1,3),r['risk'],*valid))
        r['validation']=d;r['eligible']=r['train']['pf']>1 and d['pf']>1 and r['train']['n']>=200 and d['n']>=200
    (OUT/'shortlist.json').write_text(json.dumps(shortlist,indent=2))
    (OUT/'discovery.json').write_text(json.dumps(all_results,indent=2))
    for r in shortlist[:5]:print(r,flush=True)
    eligible=[r for r in shortlist if r['eligible']]
    if eligible:
        best=max(eligible,key=lambda r:score(r['validation']))
        best['frozen_at']=pd.Timestamp.now(tz='UTC').isoformat()
        (OUT/'frozen_candidate.json').write_text(json.dumps(best,indent=2))
        print('Candidate frozen; requires NAV replay and later-period validation',flush=True)
    else:print('No qualifying pooled candidate',flush=True)

if __name__=='__main__':main()
