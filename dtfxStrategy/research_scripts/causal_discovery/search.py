"""Discovery/validation only. No evaluation of entries on/after 2018-03-09."""
import itertools
import json
import time
import numpy as np
import pandas as pd
import causal_native as native
from prepare import OUT, FEATURES, BASES

def ts(s):return pd.Timestamp(s).timestamp()
TRAIN=(ts('2009-10-02'),ts('2016-01-01'))
VALID=(ts('2016-01-08'),ts('2018-03-09'))
RISKS=[.02,.04,.05,.06,.08,.10]

def compact(d):return {k:v for k,v in d.items() if k not in ['accepted','trace_time','trace_equity']}

def score(d):
    # Fixed discovery-only proxy for the leaderboard's four components. No post-2018 anchors fitted.
    if d['n']<200:return -1000.
    return (.30*np.clip(100*d['log_growth']/12,-100,100)+.30*np.clip((d['wr']-50)/45*100,0,100)
            +.20*min(100.,d['tpy']/24000*100)+.20*max(0,100+d['dd']))

def main():
    start=time.time(); all_rows=[]; finalists=[]
    for base in BASES:
        full=np.load(OUT/(base+'_opportunities.npy'),mmap_mode='r')
        # Physically restrict the data passed to discovery/validation code.
        a=np.ascontiguousarray(full[full[:,0]<VALID[1]])
        train=a[(a[:,0]>=TRAIN[0])&(a[:,1]<TRAIN[1])]
        conditions=[[]]
        for col,name in enumerate(FEATURES,7):
            vals=train[:,col];vals=vals[np.isfinite(vals)]
            for threshold in np.unique(np.quantile(vals,[.10,.25,.50,.75,.90])):
                for sign in [-1,1]:conditions.append([[col,sign,float(threshold)]])
        def run(conditions,stage):
            best=[]
            for rules in conditions:
                best_row=None
                for risk in RISKS:
                    d=compact(native.account(a,np.asarray(rules,dtype=float).reshape(-1,3),risk,*TRAIN))
                    row=dict(base=base,rules=rules,risk=risk,stage=stage,score=score(d),**d)
                    all_rows.append(row)
                    if best_row is None or row['score']>best_row['score']:best_row=row
                best.append(best_row)
            return sorted(best,key=lambda x:x['score'],reverse=True)
        singles=run(conditions,'single')
        # Select at most one threshold/direction per feature before combining mechanisms.
        leaders=[];seen=set()
        for row in singles:
            if not row['rules']:continue
            feature=row['rules'][0][0]
            if feature not in seen:
                leaders.append(row);seen.add(feature)
            if len(leaders)==10:break
        pairs=[x['rules']+y['rules'] for x,y in itertools.combinations(leaders,2)]
        paired=run(pairs,'pair')
        triplets=[]
        for pair in paired[:5]:
            for single in leaders[:5]:
                if single['rules'][0][0] not in [r[0] for r in pair['rules']]:
                    triplets.append(pair['rules']+single['rules'])
        triples=run(triplets,'triple')
        shortlist=sorted(singles+paired+triples,key=lambda x:x['score'],reverse=True)[:12]
        shortlist += [r for r in singles if not r['rules']]
        for row in shortlist:
            rules=np.asarray(row['rules'],dtype=float).reshape(-1,3)
            validation=compact(native.account(a,rules,row['risk'],*VALID))
            row={**row,'validation':validation,'validation_score':score(validation)}
            # Require positive results in both discovery subperiods, not just total.
            folds=[compact(native.account(a,rules,row['risk'],TRAIN[0],ts('2013-01-01'))),
                   compact(native.account(a,rules,row['risk'],ts('2013-01-08'),TRAIN[1]))]
            row['discovery_folds']=folds
            row['eligible']=all(x['pf']>1 and x['n']>=200 for x in folds) and validation['pf']>1
            finalists.append(row)
        print(base,'evaluated',len(conditions)+len(pairs)+len(triplets),'mechanisms with train-only risk optimization',flush=True)
    pd.DataFrame([{**r,'rules':json.dumps(r['rules'])} for r in all_rows]).to_csv(OUT/'discovery_scores.csv',index=False)
    finalists.sort(key=lambda r:r['validation_score'] if r['eligible'] else -1e6,reverse=True)
    (OUT/'validation_shortlist.json').write_text(json.dumps(finalists,indent=2))
    winner=finalists[0]
    if not winner['eligible']:
        (OUT/'rejected.json').write_text(json.dumps({'reason':'No candidate passed discovery fold and validation profitability gates',
                                                   'count':len(all_rows)},indent=2))
        print('No eligible candidate; saved rejected.json',flush=True)
        return
    winner['feature_names']=FEATURES
    winner['frozen_at_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    winner['selection']='pre-2016 mechanism/risk fit; 2016-2018 validation; post-2018 not opened'
    (OUT/'frozen_candidate.json').write_text(json.dumps(winner,indent=2))
    for r in finalists[:5]:
        print(r['base'],r['rules'],'risk',r['risk'],'validation',r['validation'],flush=True)
    print('FROZEN after',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
