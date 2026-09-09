"""Evaluate the frozen quote policy only after later coverage passes; no fitting."""
import json, hashlib, datetime
from pathlib import Path
import numpy as np
import causal_native as native
from prepare import DATA, SYMBOLS

def main():
    root=DATA/'causal_discovery'; out=root/'tick_context/later'
    frozen=root/'tick_context/frozen_candidate.json'
    cfg=json.loads(frozen.read_text()); manifest=json.loads((out/'acquisition_manifest.json').read_text())
    digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert digest==manifest['frozen_sha256'], 'Frozen policy changed'
    a=np.load(manifest['source'])
    begin,end=manifest['begin_seconds'],manifest['end_seconds']
    a=np.ascontiguousarray(a[(a[:,0]>=begin)&(a[:,0]<end)])
    pieces=[[] for _ in SYMBOLS]
    record_digest=hashlib.sha256()
    for path in sorted((out/'records').glob('*.json')):
        record_digest.update(path.name.encode()); record_digest.update(path.read_bytes())
        r=json.loads(path.read_text()); key=r['symbol']+'_'+str(r['hour_start_s'])
        pieces[SYMBOLS.index(r['symbol'])].append(np.load(out/'minutes'/(key+'.npy')))
    minutes=[]
    for parts in pieces:
        m=np.concatenate(parts) if parts else np.empty((0,11))
        m=m[np.argsort(m[:,0],kind='stable')]
        assert np.all(np.diff(m[:,0])>0)
        minutes.append(np.ascontiguousarray(m))
    fields=native.quote_context(a,minutes); valid=np.isfinite(fields).all(axis=1)
    groups=[]
    for i,s in enumerate(SYMBOLS):
        k=a[:,2]==i
        groups.append({'symbol':s,'signals':int(k.sum()),'complete':int((k&valid).sum()),'coverage':float(valid[k].mean()) if k.any() else None})
    years=np.array([datetime.datetime.fromtimestamp(t,datetime.timezone.utc).year for t in a[:,0]])
    annual=[]
    for i,symbol in enumerate(SYMBOLS):
        for year in np.unique(years):
            k=(a[:,2]==i)&(years==year)
            if k.any():
                annual.append({'symbol':symbol,'year':int(year),'signals':int(k.sum()),
                               'complete':int((k&valid).sum()),'coverage':float(valid[k].mean())})
    coverage={'groups':groups,'annual_groups':annual,
              'ready':all(g['coverage'] is None or g['coverage']>=.95 for g in groups+annual)}
    (out/'coverage.json').write_text(json.dumps(coverage,indent=2))
    if not coverage['ready']:
        print(json.dumps(coverage));return
    markets=[]
    for s in SYMBOLS:
        m=np.load(root/'market_cache'/(s+'.npy'),mmap_mode='r'); markets.append(m[:np.searchsorted(m[:,0],end)])
    x=np.ascontiguousarray(np.column_stack([a,fields])[valid])
    x=native.forecast_execute(x,np.ones(len(x)),markets,cfg['horizon_observations'],cfg['rr'])
    d=native.nav_account(x,np.array(cfg['rules'],float),markets,cfg['risk'],begin,end)
    assert d['max_initial_leverage']<=50+1e-9
    assert d['max_effective_risk_fraction']<=cfg['risk']+1e-9
    provenance={'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'native_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                'quote_records_sha256':record_digest.hexdigest(),
                'manifest_sha256':hashlib.sha256((out/'acquisition_manifest.json').read_bytes()).hexdigest(),
                'evaluated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    result={'provenance':provenance,'frozen_sha256':digest,'coverage':coverage,'result':{k:v for k,v in d.items() if np.isscalar(v)},'limitations':'Project-reused later period; four-pair research, not current single-pair leaderboard comparison.'}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    np.savez(out/'evaluation_nav.npz',time=d['nav_times'],nav=d['nav_values'],accepted=d['accepted'],pnl=d['close_pnl'])
    print(json.dumps(result))

if __name__=='__main__': main()
