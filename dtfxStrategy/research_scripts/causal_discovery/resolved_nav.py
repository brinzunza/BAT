"""NAV verification of eligible pre-2018 resolved-evidence candidates."""
import json
import os
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts
from nav_validation import compact

def main():
    root=DATA/'causal_discovery';out=root/'resolved'
    raw_replay=os.environ.get('RESOLVED_RAW_REPLAY')=='1'
    cases=json.loads((out/'results.json').read_text())
    markets=[]
    for s in SYMBOLS:
        a=np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r')
        markets.append(a[:np.searchsorted(a[:,0],ts('2018-03-09'))])
    results=[]
    for case in cases:
        if not case['eligible']:continue
        a=np.load(root/case['source']);a=np.ascontiguousarray(a[a[:,0]<ts('2018-03-09')]);a=a[np.argsort(a[:,0],kind='stable')]
        if raw_replay:
            # Diagnostic execution variant: 300 raw observations replace five
            # hourly bars. Missing bars make the time limit different; report it.
            a=native.forecast_execute(a,np.ones(len(a)),markets,300,1.)
        cols=a.shape[1];x=native.resolved_features(a,float(case['memory']))
        x[:,cols]-=case['penalty']*x[:,cols+1]/np.sqrt(np.maximum(1.,x[:,cols+2]))
        rules=np.array([[cols,1,case['threshold']],[cols+3,1,32.]])
        r={'selection':case,'rules':rules.tolist()}
        for label,start,end in [('train','2013-01-08','2016-01-01'),('validation','2016-01-08','2018-03-09')]:
            d=native.nav_account(x,rules,markets,case['risk'],ts(start),ts(end))
            assert d['max_initial_leverage']<=50+1e-9 and d['max_effective_risk_fraction']<=case['risk']+1e-9
            r[label]=compact(d)
            np.savez(out/(label+('_raw' if raw_replay else '')+'_nav.npz'),time=d['nav_times'],nav=d['nav_values'],accepted=d['accepted'],pnl=d['close_pnl'])
        results.append(r);print(json.dumps(r),flush=True)
    (out/('raw_nav_results.json' if raw_replay else 'nav_results.json')).write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
