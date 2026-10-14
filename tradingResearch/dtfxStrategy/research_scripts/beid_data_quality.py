"""Audit raw FX bars, without modifying market data or silently deleting observations."""
import json
import numpy as np
import pandas as pd
from beid_conservative_audit import PATHS,OUT

def flags(d):
    p=d[['open','high','low','close']]
    finite=np.isfinite(p).all(axis=1)
    positive=p.gt(0).all(axis=1)
    consistent=(d.high>=p.max(axis=1))&(d.low<=p.min(axis=1))
    # Broad diagnostic only: not an automatic data-correction or trading filter.
    reference=d.close.where(d.close>0).rolling(60,min_periods=10).median().shift(1)
    extreme=p.div(reference,axis=0).gt(2).any(axis=1)|p.div(reference,axis=0).lt(.5).any(axis=1)
    return pd.DataFrame(dict(nonfinite=~finite,nonpositive=~positive,inconsistent=~consistent,extreme_vs_prior_median=extreme))

def main():
    results=[]
    for symbol,path in PATHS.items():
        d=pd.read_csv(path)
        d['time']=pd.to_datetime(d['date'] if 'date' in d else d['time'])
        d=d.sort_values('time').reset_index(drop=True)
        f=flags(d); bad=f.any(axis=1)
        rows=pd.concat([d.loc[bad,['time','open','high','low','close']],f[bad]],axis=1)
        rows.to_csv(OUT/f'{symbol}_suspect_input.csv',index=False)
        item=dict(symbol=symbol,rows=len(d),suspect_rows=int(bad.sum()),counts=f.sum().to_dict(),
            years=d.loc[bad,'time'].dt.year.value_counts().sort_index().to_dict(),
            examples=json.loads(rows.head(10).to_json(orient='records',date_format='iso')))
        results.append(item); print(json.dumps(item),flush=True)
    (OUT/'data_quality.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
