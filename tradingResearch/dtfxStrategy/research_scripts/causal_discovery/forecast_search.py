"""Learn move direction/magnitude at confirmed DTFX zone births, not just WR filters."""
import json
import numpy as np
import pandas as pd
import causal_native as native
from prepare import DATA,SYMBOLS
from search import compact,score,ts

OUT=DATA/'causal_discovery'/'forecast';OUT.mkdir(exist_ok=True)

def main():
    markets=[np.load(DATA/'causal_discovery'/'market_cache'/(s+'.npy'),mmap_mode='r') for s in SYMBOLS]
    groups={(tf,h):[] for tf in [5,15] for h in [60,240]}
    for inst,(symbol,raw) in enumerate(zip(SYMBOLS,markets)):
        # No post-2018 bars even enter sample/label generation for this phase.
        raw=np.ascontiguousarray(raw[raw[:,0]<ts('2018-03-09')])
        t=np.rint(raw[:,0]*1e9).astype(np.int64);o,h,l,c=[np.ascontiguousarray(raw[:,j]) for j in [1,2,3,4]]
        for tf in [5,15]:
            at,ao,ah,al,ac=native.aggregate(t,o,h,l,c,tf)
            f=native.features(ao,ah,al,ac);z=native.zones(ao,ah,al,ac,at,2)
            for horizon in [60,240]:
                a=native.forecast_samples(*z[:4],z[5],z[4],at,ac,f,raw,horizon,tf,.016 if inst==3 else .00016)
                a[:,2]=inst
                a=np.column_stack([a,np.broadcast_to(np.eye(4)[inst],(len(a),4))])
                groups[tf,horizon].append(a)
        print(symbol,'forecast observations prepared',flush=True)
    results=[];trials_all=[]
    for (tf,horizon),pieces in groups.items():
        a=np.concatenate(pieces);a=np.ascontiguousarray(a[np.lexsort((a[:,2],a[:,0]))])
        np.save(OUT/f'{tf}min_h{horizon}_samples.npy',a)
        fit=a[a[:,1]<ts('2013-01-01')]
        early=np.ascontiguousarray(a[(a[:,0]>=ts('2013-01-08'))&(a[:,0]<ts('2016-01-01'))])
        mean,roots,model,pred=native.learn_trees(fit,early,80)
        candidates=[]
        for edge in [0.,.05,.10,.20,.40]:
            direction=np.where(np.abs(pred)>=early[:,29]+edge,np.sign(pred),0.)
            for rr in [.25,.5,1.]:
                trades=native.forecast_execute(early,direction,markets,horizon,rr)
                for risk in [.005,.01,.02,.04,.06,.08]:
                    d=compact(native.account(trades,np.empty((0,3)),risk,ts('2013-01-08'),ts('2016-01-01')))
                    r=dict(timeframe=tf,horizon=horizon,minimum_net_edge=edge,rr=rr,risk=risk,train=d,score=score(d));candidates.append(r)
        candidates.sort(key=lambda r:r['score'],reverse=True);trials_all+=candidates
        # Only one policy per (timeframe,horizon), selected before validation predictions.
        chosen=candidates[0]
        fit=a[a[:,1]<ts('2016-01-01')]
        valid=np.ascontiguousarray(a[a[:,0]>=ts('2016-01-08')])
        mean,roots,model,pred=native.learn_trees(fit,valid,80)
        direction=np.where(np.abs(pred)>=valid[:,29]+chosen['minimum_net_edge'],np.sign(pred),0.)
        trades=native.forecast_execute(valid,direction,markets,horizon,chosen['rr'])
        d=compact(native.account(trades,np.empty((0,3)),chosen['risk'],ts('2016-01-08'),ts('2018-03-09')))
        chosen['validation']=d
        chosen['eligible']=chosen['train']['pf']>1 and d['pf']>1 and chosen['train']['n']>=200 and d['n']>=200
        chosen['model_mean']=mean;chosen['model_roots']=roots.tolist();chosen['model_nodes']=model.reshape(-1,5).tolist()
        results.append(chosen)
        np.save(OUT/f'{tf}min_h{horizon}_validation_trades.npy',trades)
        print(tf,horizon,'train',chosen['train'],'validation',d,flush=True)
    (OUT/'discovery.json').write_text(json.dumps(trials_all,indent=2))
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    good=[r for r in results if r['eligible']]
    if good:
        best=max(good,key=lambda r:score(r['validation']));best['frozen_at']=pd.Timestamp.now(tz='UTC').isoformat()
        (OUT/'frozen_candidate.json').write_text(json.dumps(best,indent=2))
        print('Forecast candidate frozen; pending NAV and held-back evaluation',flush=True)
    else:print('No qualifying forecast policy',flush=True)

if __name__=='__main__':main()
