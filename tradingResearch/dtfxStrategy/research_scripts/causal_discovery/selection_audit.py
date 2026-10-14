"""Training-only audit of growth opportunities hidden by the WR-heavy proxy.

Reused validation is explicitly retrospective, not a fresh holdout.
"""
import json,itertools,os
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts,compact

def main():
    root=DATA/'causal_discovery';out=root/'selection_audit';out.mkdir(exist_ok=True)
    earlier=os.environ.get('AUDIT_EARLIER')=='1'
    if earlier:out=root/'earlier_selection_audit';out.mkdir(exist_ok=True)
    markets=[]
    if earlier:
        for s in SYMBOLS:
            m=np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r')
            markets.append(m[:np.searchsorted(m[:,0],ts('2018-03-09'))])
    results=[]
    folders=[('structural',7)] if earlier else [('sweep',7),('zone_field',28)]
    for folder,startcol in folders:
        paths=(root/folder).rglob('*_opportunities.npy') if earlier else (root/folder).glob('*.npy')
        for path in sorted(paths):
            x=np.load(path,mmap_mode='r');x=np.ascontiguousarray(x[x[:,0]<ts('2018-03-09')])
            if earlier:
                tf=1 if path.parent.name=='structural' else int(path.parent.name.removesuffix('min'))
                rr=.5 if 'half' in path.name else (2. if '2r' in path.name else 1.)
                x=native.forecast_execute(x,np.ones(len(x)),markets,(240//tf+1)*tf,rr)
            train=(x[:,0]>=ts('2013-01-08'))&(x[:,1]<ts('2016-01-01'))
            rules=[[]]
            for col in range(startcol,x.shape[1]):
                vals=x[train,col];vals=vals[np.isfinite(vals)]
                if not len(vals):continue
                for v in np.unique(np.quantile(vals,[.1,.25,.5,.75,.9])):
                    for sign in (-1,1):rules.append([[col,sign,float(v)]])
            evaluated=[]
            def sweep(rs):
                for rule in rs:
                    for risk in (.005,.01,.02,.04):
                        d=compact(native.account(x,np.asarray(rule,dtype=float).reshape(-1,3),risk,ts('2013-01-08'),ts('2016-01-01')))
                        evaluated.append({'rules':rule,'risk':risk,'train':d})
            def rank(r):return r['train']['log_growth'] if r['train']['n']>=200 else -1e100
            sweep(rules)
            leaders=[];seen=set()
            for r in sorted(evaluated,key=rank,reverse=True):
                if r['rules'] and r['rules'][0][0] not in seen:
                    leaders.append(r['rules']);seen.add(r['rules'][0][0])
                if len(leaders)==4:break
            sweep([sum(p,[]) for p in itertools.combinations(leaders,2)])
            viable=[r for r in evaluated if r['train']['n']>=200 and r['train']['pf']>1 and r['train']['log_growth']>0]
            passed=[]
            for r in viable:
                q=np.asarray(r['rules'],dtype=float).reshape(-1,3)
                r['folds']=[compact(native.account(x,q,r['risk'],ts(b),ts(e))) for b,e in
                    [('2013-01-08','2014-07-01'),('2014-07-08','2016-01-01')]]
                if all(f['n']>=80 and f['pf']>1 and f['log_growth']>0 for f in r['folds']):passed.append(r)
            row={'source':str(path.relative_to(root)),'evaluations':len(evaluated),'positive_training':len(viable),
                 'positive_both_training_folds':len(passed),'best_training':max(evaluated,key=rank)}
            if passed:
                best=max(passed,key=rank);q=np.asarray(best['rules'],dtype=float).reshape(-1,3)
                best['reused_validation']=compact(native.account(x,q,best['risk'],ts('2016-01-08'),ts('2018-03-09')))
                row['selected']=best
            results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2))
            print(row['source'],len(viable),len(passed),flush=True)

if __name__=='__main__':main()
