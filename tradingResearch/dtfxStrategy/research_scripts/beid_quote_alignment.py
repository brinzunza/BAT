"""Compare downloaded entry-hour quote bars with the exact master input prices.
Diagnostic only; never retrospectively shift prices to improve P&L.
"""
import json,lzma
import numpy as np
import pandas as pd
from beid_quote_cost_probe import DT
from beid_conservative_audit import DATA,PATHS,get_pip_size
from beid_clean_htf_study import OUT

def main():
    probe=json.loads((OUT/'quote_cost_probe.json').read_text())
    summaries=[]
    for symbol,path in PATHS.items():
        selected=[r for r in probe['records'] if r['symbol']==symbol and r['status']=='complete']
        frames=[];hours=set()
        for r in selected:
            d=pd.Timestamp(r['hour']);key=symbol+'_'+d.strftime('%Y%m%d_%H')
            a=np.frombuffer(lzma.decompress((DATA/'beid_quote_cost_probe'/(key+'.bi5')).read_bytes()),DT)
            scale=1000 if symbol=='USDJPY' else 100000
            q=pd.DataFrame(dict(time=d+pd.to_timedelta(a['ms'].astype('int64'),unit='ms'),bid=a['bid']/scale,ask=a['ask']/scale))
            q['mid']=(q.bid+q.ask)/2
            frames.append(q.set_index('time').resample('1min').last().dropna())
            for offset in [-120,-60,-1,0,1,60,120]:
                hours.add((d+pd.Timedelta(minutes=offset)).strftime('%Y-%m-%d %H'))
                hours.add((d+pd.Timedelta(minutes=offset+59)).strftime('%Y-%m-%d %H'))
        if not frames:continue
        chunks=[]
        for d in pd.read_csv(path,chunksize=500000):
            tc='date' if 'date' in d else 'time'
            m=d[tc].str.slice(0,13).isin(hours)
            if m.any():chunks.append(d[m].rename(columns={tc:'time'}))
        master=pd.concat(chunks);master['time']=pd.to_datetime(master.time);master=master.set_index('time')
        q=pd.concat(frames).sort_index()
        for lag in [-120,-60,-1,0,1,60,120]:
            shifted=q.copy();shifted.index+=pd.Timedelta(minutes=lag)
            match=shifted.join(master[['close']],how='inner')
            for field in ['bid','mid','ask']:
                delta=(match.close-match[field])/get_pip_size(symbol)
                summaries.append(dict(symbol=symbol,quote_timestamp_shift_minutes=lag,quote_field=field,n=len(delta),
                    median_gap_pips=float(delta.median()),median_absolute_gap_pips=float(delta.abs().median()),
                    p95_absolute_gap_pips=float(delta.abs().quantile(.95))))
            if lag==0:
                match.to_csv(OUT/f'{symbol}_source_quote_alignment.csv')
    (OUT/'quote_alignment.json').write_text(json.dumps(summaries,indent=2))
    print(pd.DataFrame(summaries).query('quote_timestamp_shift_minutes == 0').to_string(index=False),flush=True)

if __name__=='__main__':main()
