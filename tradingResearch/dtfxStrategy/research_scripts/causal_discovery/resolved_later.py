"""One frozen later-period evaluation; no optimization or automatic promotion."""
import json,hashlib
from pathlib import Path
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts
from nav_validation import compact

def main():
    root=DATA/'causal_discovery';out=root/'resolved'
    frozen=out/'frozen_candidate.json'
    if not frozen.exists():
        case=json.loads((out/'raw_nav_results.json').read_text())[0]['selection']
        cfg={'selection':case,'execution':'raw M1 bid/ask, SL first within ambiguous minute',
             'horizon_observations':300,'rr':1.,'minimum_resolved':32,
             'evidence':'EW net R of strictly earlier resolved shadows, separate instrument/side',
             'evaluation_begin':'2018-03-09','later_period_pristine_project_wide':False,
             'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.hpp')}}
        frozen.write_text(json.dumps(cfg,indent=2))
    cfg=json.loads(frozen.read_text());case=cfg['selection']
    markets=[np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r') for s in SYMBOLS]
    end=min(float(m[-1,0])+60 for m in markets)
    markets=[m[:np.searchsorted(m[:,0],end)] for m in markets]
    a=np.load(root/case['source']);a=np.ascontiguousarray(a[a[:,0]<end]);a=a[np.argsort(a[:,0],kind='stable')]
    a=native.forecast_execute(a,np.ones(len(a)),markets,cfg['horizon_observations'],cfg['rr'])
    cols=a.shape[1];x=native.resolved_features(a,float(case['memory']))
    x[:,cols]-=case['penalty']*x[:,cols+1]/np.sqrt(np.maximum(1.,x[:,cols+2]))
    rules=np.array([[cols,1,case['threshold']],[cols+3,1,float(cfg['minimum_resolved'])]])
    d=native.nav_account(x,rules,markets,case['risk'],ts(cfg['evaluation_begin']),end)
    assert d['max_initial_leverage']<=50+1e-9 and d['max_effective_risk_fraction']<=case['risk']+1e-9
    r={'frozen_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'end_seconds':end,'result':compact(d)}
    (out/'later_result.json').write_text(json.dumps(r,indent=2))
    np.savez(out/'later_nav.npz',time=d['nav_times'],nav=d['nav_values'],accepted=d['accepted'],pnl=d['close_pnl'])
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
