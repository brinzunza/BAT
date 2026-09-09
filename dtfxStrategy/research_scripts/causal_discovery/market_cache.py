"""One-time I/O conversion to native replay arrays; preserves original CSV files."""
import json
import numpy as np
import pandas as pd
from prepare import DATA,FILES,SYMBOLS

def main():
    out=DATA/'causal_discovery'/'market_cache';out.mkdir(exist_ok=True)
    manifest={}
    for symbol,file in zip(SYMBOLS,FILES):
        source=DATA/file;target=out/(symbol+'.npy')
        info={'bytes':source.stat().st_size,'mtime_ns':source.stat().st_mtime_ns}
        f=pd.read_csv(source,parse_dates=['date']).sort_values('date').drop_duplicates('date')
        a=np.column_stack([f.date.to_numpy(dtype='datetime64[ns]').astype(np.int64)/1e9,
                           f[['open','high','low','close']].to_numpy(dtype=float)])
        np.save(target,a);info['bars']=len(a);manifest[symbol]=info
        print(symbol,len(a),'bars cached',flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
