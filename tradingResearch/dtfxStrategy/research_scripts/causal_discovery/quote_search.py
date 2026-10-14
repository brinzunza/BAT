"""Coverage-gated quote activity search with profitability before speed ranking."""
import json,itertools
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts,compact

def main():
    root=DATA/'causal_discovery';out=root/'tick_context'
    coverage=json.loads((out/'coverage.json').read_text())
    if not coverage['ready_for_search']:
        raise SystemExit('Coverage gate not met; no model fitted.')
    a=np.load(out/'context_pre2018.npy');a=np.ascontiguousarray(a[np.isfinite(a[:,28:]).all(axis=1)])
    markets=[]
    for s in SYMBOLS:
        m=np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r')
        markets.append(m[:np.searchsorted(m[:,0],ts('2018-03-09'))])
    results=[]
    for rr in (1.,2.):
        x=native.forecast_execute(a,np.ones(len(a)),markets,300,rr)
        train=(x[:,0]>=ts('2013-01-08'))&(x[:,1]<ts('2016-01-01'))
        rules=[[]]
        for col in range(28,48):
            for v in np.unique(np.quantile(x[train,col],[.1,.25,.5,.75,.9])):
                for sign in (-1,1):rules.append([[col,sign,float(v)]])
        evaluated=[]
        def sweep(rs):
            for rule in rs:
                q=np.asarray(rule,dtype=float).reshape(-1,3)
                for risk in (.005,.01,.02,.04,.06,.08,.12):
                    d=compact(native.account(x,q,risk,ts('2013-01-08'),ts('2016-01-01'),strict_units=True))
                    row={'rules':rule,'risk':risk,'train':d,'passes':False}
                    if d['n']>=200 and d['pf']>1 and d['log_growth']>0:
                        folds=[compact(native.account(x,q,risk,ts(b),ts(e),strict_units=True)) for b,e in
                            [('2013-01-08','2014-07-01'),('2014-07-08','2016-01-01')]]
                        row['folds']=folds
                        row['passes']=all(f['n']>=80 and f['pf']>1 and f['log_growth']>0 for f in folds)
                    evaluated.append(row)
        def ranking(r):return (r['passes'],r['train']['log_growth'] if r['train']['n']>=200 else -1e100)
        sweep(rules)
        leaders=[];seen=set()
        for r in sorted(evaluated,key=ranking,reverse=True):
            if r['rules'] and r['rules'][0][0] not in seen:
                leaders.append(r['rules']);seen.add(r['rules'][0][0])
            if len(leaders)==4:break
        sweep([sum(pair,[]) for pair in itertools.combinations(leaders,2)])
        passed=[r for r in evaluated if r['passes']]
        result={'rr':rr,'evaluations':len(evaluated),'profitable_fold_configs':len(passed),
                'validation_status':'Reused research period, not a pristine holdout'}
        if passed:
            best=max(passed,key=ranking);q=np.asarray(best['rules'],dtype=float).reshape(-1,3)
            best['validation']=compact(native.account(x,q,best['risk'],ts('2016-01-08'),ts('2018-03-09'),strict_units=True))
            d=best['validation'];best['eligible']=d['n']>=200 and d['pf']>1 and d['equity']>200
            result['selected']=best
        results.append(result);(out/'search_results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
