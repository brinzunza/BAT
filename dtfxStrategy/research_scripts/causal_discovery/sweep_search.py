"""False-break DTFX recapture search. Long loops are native C++."""
import json,itertools
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts,compact,score

def main():
    root=DATA/'causal_discovery';out=root/'sweep';out.mkdir(exist_ok=True)
    markets=[]
    for s in SYMBOLS:
        m=np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r')
        markets.append(m[:np.searchsorted(m[:,0],ts('2018-03-09'))])
    results=[]
    for tf in (1,5,15,60):
        samples=[]
        for inst,m in enumerate(markets):
            t=np.rint(m[:,0]*1e9).astype(np.int64)
            o,h,l,c=[np.ascontiguousarray(m[:,j]) for j in (1,2,3,4)]
            if tf!=1:t,o,h,l,c=native.aggregate(t,o,h,l,c,tf)
            f=native.features(o,h,l,c);z=native.zones(o,h,l,c,t,2)
            a=native.sweep_samples(z[0],z[1],z[3],z[5],t,o,l,h,c,np.ascontiguousarray(f[:,7]*c),.016 if inst==3 else .00016,tf)
            a[:,2]=inst;samples.append(a)
        a=np.concatenate(samples);a=np.ascontiguousarray(a[np.lexsort((a[:,3],a[:,2],a[:,0]))])
        for rr in (.5,1.,2.):
            x=native.forecast_execute(a,np.ones(len(a)),markets,240,rr)
            np.save(out/f'{tf}min_{rr}r.npy',x)
            train=(x[:,0]>=ts('2013-01-08'))&(x[:,1]<ts('2016-01-01'))
            rules=[[]]
            for col in range(7,12):
                for v in np.unique(np.quantile(x[train,col],[.1,.25,.5,.75,.9])):
                    for sign in (-1,1):rules.append([[col,sign,float(v)]])
            evaluated=[]
            def sweep(rs):
                for rule in rs:
                    for risk in (.005,.01,.02,.04):
                        d=compact(native.account(x,np.asarray(rule,dtype=float).reshape(-1,3),risk,ts('2013-01-08'),ts('2016-01-01')))
                        evaluated.append({'rules':rule,'risk':risk,'train':d,'score':score(d)})
            sweep(rules)
            leaders=[];seen=set()
            for row in sorted(evaluated,key=lambda r:r['score'],reverse=True):
                if row['rules'] and row['rules'][0][0] not in seen:
                    leaders.append(row['rules']);seen.add(row['rules'][0][0])
                if len(leaders)==4:break
            sweep([sum(pair,[]) for pair in itertools.combinations(leaders,2)])
            best=max(evaluated,key=lambda r:r['score']);rule=np.asarray(best['rules'],dtype=float).reshape(-1,3)
            best.update(timeframe=tf,rr=rr,training_evaluations=len(evaluated))
            best['validation']=compact(native.account(x,rule,best['risk'],ts('2016-01-08'),ts('2018-03-09')))
            best['eligible']=best['train']['pf']>1 and best['train']['n']>=200 and best['validation']['pf']>1 and best['validation']['n']>=200
            results.append(best);(out/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(best),flush=True)
    print('Eligible:',sum(x['eligible'] for x in results),flush=True)

if __name__=='__main__':main()
