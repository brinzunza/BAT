"""Build causal quote-context rows and coverage report; never fit on partial downloads."""
import json
import numpy as np
import causal_native as native
from prepare import DATA,SYMBOLS
from search import ts

def main():
    root=DATA/'causal_discovery';out=root/'tick_context'
    manifest=json.loads((out/'acquisition_manifest.json').read_text())
    a=np.load(manifest['source']);a=np.ascontiguousarray(a[(a[:,0]>=ts('2013-01-08'))&(a[:,0]<ts('2018-03-09'))])
    pieces=[[] for _ in SYMBOLS]
    for record in sorted((out/'records').glob('*.json')):
        r=json.loads(record.read_text());k=r['symbol']+'_'+str(r['hour_start_s'])
        pieces[SYMBOLS.index(r['symbol'])].append(np.load(out/'minutes'/(k+'.npy')))
    minutes=[]
    for parts in pieces:
        m=np.concatenate(parts) if parts else np.empty((0,11))
        if len(m):
            m=m[np.argsort(m[:,0],kind='stable')]
            assert np.all(np.diff(m[:,0])>0),'Duplicated provider minute'
        minutes.append(np.ascontiguousarray(m))
    fields=native.quote_context(a,minutes);valid=np.isfinite(fields).all(axis=1)
    np.save(out/'context_pre2018.npy',np.column_stack([a,fields]))
    rows=[]
    for i,s in enumerate(SYMBOLS):
        for label,begin,end in [('train','2013-01-08','2016-01-01'),('validation','2016-01-08','2018-03-09')]:
            k=(a[:,2]==i)&(a[:,0]>=ts(begin))&(a[:,0]<ts(end));n=int(k.sum())
            rows.append({'symbol':s,'period':label,'signals':n,'complete':int((k&valid).sum()),
                'coverage':float(valid[k].mean()) if n else None})
    report={'signals':len(a),'complete':int(valid.sum()),'coverage':float(valid.mean()),'by_instrument_period':rows,
        'ready_for_search':all(r['coverage'] is None or r['coverage']>=.95 for r in rows),
        'missing_policy':'Missing 15-minute context rejects the signal; no zero filling or retrospective availability selection',
        'columns_added':['quote_rate','directional_repricing','available_size_imbalance','microprice_deviation_pips',
            'observed_spread_pips','mid_variation_pips']*3+['activity_1m_over_15m','imbalance_1m_minus_15m']}
    (out/'coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
