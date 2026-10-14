"""Public cross-source entry-hour quote probe; costs only, not a strategy backtest.
Hash-ordered acquisition is independent of P&L. No trading or credentials.
"""
import argparse,concurrent.futures,hashlib,json,lzma,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from beid_clean_htf_study import OUT
from beid_conservative_audit import DATA,get_pip_size
DT=np.dtype([('ms','>u4'),('ask','>u4'),('bid','>u4'),('av','>f4'),('bv','>f4')])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=32);args=parser.parse_args()
    root=DATA/'beid_quote_cost_probe';root.mkdir(exist_ok=True)
    t=pd.read_csv(OUT/'train_selected_portfolio.csv',parse_dates=['fill_time','exit_time'])
    t=t[t.fill_time>='2016-01-01']
    requests=t.assign(hour=t.fill_time.dt.floor('h'))[['ticker','hour']].drop_duplicates()
    schedule=[dict(symbol=r.ticker,hour=str(r.hour)) for r in requests.itertuples()]
    schedule.sort(key=lambda r:hashlib.sha256((r['symbol']+r['hour']).encode()).hexdigest())
    (OUT/'quote_probe_schedule.json').write_text(json.dumps(schedule,indent=2))
    def fetch(r):
        d=pd.Timestamp(r['hour']);s=r['symbol'];key=s+'_'+d.strftime('%Y%m%d_%H')
        path=root/(key+'.bi5');result=dict(**r)
        try:
            if path.exists():blob=path.read_bytes()
            else:
                url=f'https://datafeed.dukascopy.com/datafeed/{s}/{d.year}/{d.month-1:02d}/{d.day:02d}/{d.hour:02d}h_ticks.bi5'
                p=subprocess.run(['curl','--fail','--silent','--show-error','--max-time','20',url],capture_output=True)
                if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
                blob=p.stdout
            a=np.frombuffer(lzma.decompress(blob),DT)
            if not len(a) or not ((a['ask']>=a['bid']).all() and (a['ms']<3600000).all() and (np.diff(a['ms'].astype('int64'))>=0).all()):raise ValueError('Quote integrity failed')
            scale=1000 if s=='USDJPY' else 100000
            spreads=(a['ask'].astype(float)-a['bid'].astype(float))/scale/get_pip_size(s)
            events=t[(t.ticker==s)&(t.fill_time.dt.floor('h')==d)]
            near=[]
            for event in events.itertuples():
                ms=(event.fill_time-d).total_seconds()*1000
                mask=(a['ms']>=ms)&(a['ms']<ms+60000)
                if mask.any():near.append(dict(fill_time=str(event.fill_time),median_spread_pips=float(np.median(spreads[mask])),p95_spread_pips=float(np.quantile(spreads[mask],.95))))
            path.write_bytes(blob)
            result.update(status='complete',quotes=len(a),events=near,sha256=hashlib.sha256(blob).hexdigest())
        except Exception as exc:result.update(status='failed',error=str(exc))
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,schedule[:args.limit] if args.limit else schedule))
    events=[e for r in results if r['status']=='complete' for e in r['events']]
    summary=dict(status='cross-source costs only; no inference of executable BEID profitability',scheduled_hours=len(schedule),attempted_hours=len(results),complete_hours=sum(r['status']=='complete' for r in results),observed_events=len(events),records=results)
    if events:summary['entry_minute_spread_quantiles_pips']=np.quantile([e['median_spread_pips'] for e in events],[0,.5,.9,1]).tolist()
    (OUT/'quote_cost_probe.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__':main()
