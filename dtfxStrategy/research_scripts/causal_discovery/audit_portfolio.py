"""Training-qualified structural portfolio; validation is explicitly reused."""
import json,itertools
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts,compact
from nav_validation import compact as nav_compact

def main():
    root=DATA/'causal_discovery';out=root/'earlier_selection_audit'
    rows=json.loads((out/'results.json').read_text());members=[r for r in rows if 'selected' in r]
    markets=[]
    for s in SYMBOLS:
        m=np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r')
        markets.append(m[:np.searchsorted(m[:,0],ts('2018-03-09'))])
    pieces=[];keys=[]
    for i,r in enumerate(members):
        src=root/r['source'];tf=1 if src.parent.name=='structural' else int(src.parent.name.removesuffix('min'))
        rr=.5 if 'half' in src.name else (2. if '2r' in src.name else 1.)
        a=np.load(src);a=np.ascontiguousarray(a[a[:,0]<ts('2018-03-09')])
        a=native.forecast_execute(a,np.ones(len(a)),markets,(240//tf+1)*tf,rr)
        keep=np.ones(len(a),dtype=bool)
        for col,sign,threshold in r['selected']['rules']:
            keep &= np.isfinite(a[:,int(col)]) & (sign*a[:,int(col)]>=sign*threshold)
        a=a[keep];priority=r['selected']['train']['log_growth']
        pieces.append(np.column_stack([a[:,:7],np.full(len(a),priority)]))
        keys.append((tf,src.name.split('_')[0]))
    candidates=[];best=None;best_data=None
    for size in range(1,len(members)+1):
        for inds in itertools.combinations(range(len(members)),size):
            if len({keys[i] for i in inds})!=size:continue
            a=np.concatenate([pieces[i] for i in inds]);a=np.ascontiguousarray(a[np.lexsort((a[:,3],a[:,2],-a[:,7],a[:,0]))])
            for risk in (.005,.01,.02,.04,.08,.12):
                d=compact(native.account(a,np.empty((0,3)),risk,ts('2013-01-08'),ts('2016-01-01')))
                if d['n']<200 or d['pf']<=1 or d['log_growth']<=0:continue
                folds=[compact(native.account(a,np.empty((0,3)),risk,ts(b),ts(e))) for b,e in
                    [('2013-01-08','2014-07-01'),('2014-07-08','2016-01-01')]]
                if not all(f['n']>=80 and f['pf']>1 and f['log_growth']>0 for f in folds):continue
                row={'sources':[members[i]['source'] for i in inds],'risk':risk,'train':d,'folds':folds}
                candidates.append(row)
                if best is None or d['log_growth']>best['train']['log_growth']:best=row;best_data=a
    if best is None:
        (out/'portfolio.json').write_text(json.dumps({'qualifying':0}));return
    best['reused_validation']=compact(native.account(best_data,np.empty((0,3)),best['risk'],ts('2016-01-08'),ts('2018-03-09')))
    best['qualifying_training_portfolios']=len(candidates)
    for label,b,e in [('train_nav','2013-01-08','2016-01-01'),('reused_validation_nav','2016-01-08','2018-03-09')]:
        d=native.nav_account(best_data,np.empty((0,3)),markets,best['risk'],ts(b),ts(e))
        assert d['max_initial_leverage']<=50+1e-9 and d['max_effective_risk_fraction']<=best['risk']+1e-9
        best[label]=nav_compact(d)
    (out/'portfolio.json').write_text(json.dumps(best,indent=2));np.save(out/'portfolio_pre2018.npy',best_data)
    print(json.dumps(best),flush=True)

if __name__=='__main__':main()
