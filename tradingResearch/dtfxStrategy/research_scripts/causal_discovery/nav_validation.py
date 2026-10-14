"""Apply NAV accounting to pre-2018 research leads, not the reserved later data."""
import json
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts

def compact(d):return {k:v for k,v in d.items() if not isinstance(v,np.ndarray)}

def main():
    root=DATA/'causal_discovery'
    markets=[np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r') for s in SYMBOLS]
    cases=[]
    pool=json.loads((root/'pooled'/'shortlist.json').read_text())
    for i,r in enumerate(pool[:3]):cases.append((f'pooled_{i}',np.load(root/'pooled'/'pre2018_opportunities.npy'),r['rules'],r['risk']))
    singles=json.loads((root/'structural'/'15min'/'validation_shortlist.json').read_text())
    lead=max(singles,key=lambda r:r['validation']['pf'])
    cases.append(('15min_sparse_lead',np.load(root/'structural'/'15min'/(lead['base']+'_opportunities.npy')),lead['rules'],lead['risk']))
    results=[]
    for name,a,rules,risk in cases:
        a=np.ascontiguousarray(a[a[:,0]<ts('2018-03-09')])
        r={'name':name,'risk':risk,'rules':rules}
        for label,begin,end in [('discovery','2013-01-08','2016-01-01'),('validation','2016-01-08','2018-03-09')]:
            d=native.nav_account(a,np.asarray(rules,dtype=float).reshape(-1,3),markets,risk,ts(begin),ts(end))
            assert d['max_initial_leverage']<=50+1e-9 and d['max_effective_risk_fraction']<=risk+1e-9
            r[label]=compact(d)
            np.savez(root/'pooled'/f'{name}_{label}_nav.npz',time=d['nav_times'],nav=d['nav_values'],
                     closes=d['close_times'],pnl=d['close_pnl'],accepted=d['accepted'])
        print(r,flush=True);results.append(r)
    (root/'nav_validation.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
