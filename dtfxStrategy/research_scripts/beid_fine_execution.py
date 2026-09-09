"""Refine execution of the frozen conservative 3-minute BEID order set.
Keeps signal, admission, coarse expiry eligibility and single-touch policy unchanged.
One-minute OHLC still cannot prove all within-minute ordering or executable spreads.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from numba import njit
import beid_conservative_audit as base

OUT=base.OUT/'one_minute'

@njit(cache=True)
def replay(times, prices, starts, entries, stops, targets, sides):
    result=np.zeros((len(starts),5))
    for k in range(len(starts)):
        fill=-1
        for j in range(starts[k],len(times)):
            o,h,l,c=prices[j]
            if fill<0:
                # The known coarse fill bar must contain the fine fill.
                if times[j]>=times[starts[k]]+180_000_000_000:
                    raise ValueError('Fine fill missing within coarse fill bar')
                if (sides[k]==1 and l<=entries[k]) or (sides[k]==-1 and h>=entries[k]):
                    fill=j
                else:
                    continue
            hit_sl=l<=stops[k] if sides[k]==1 else h>=stops[k]
            if fill==j:
                hit_tp=c>=targets[k] if sides[k]==1 else c<=targets[k]
            else:
                hit_tp=h>=targets[k] if sides[k]==1 else l<=targets[k]
            reason=0
            if hit_sl:
                price=min(stops[k],o) if sides[k]==1 else max(stops[k],o)
                reason=1
            elif hit_tp:
                price=targets[k]; reason=2
            elif j==len(times)-1:
                price=c; reason=3
            if reason:
                result[k]=np.array([fill,j,price,reason,int(fill==j and hit_sl)],dtype=np.float64)
                break
    return result

def main():
    OUT.mkdir(exist_ok=True)
    frames=[]; comparisons=[]
    for symbol,path in base.PATHS.items():
        print('Replay',symbol,flush=True)
        t=pd.read_csv(base.OUT/f'{symbol}_trades.csv',parse_dates=['fill_time','exit_time'])
        d=pd.read_csv(path)
        dates=pd.to_datetime(d['date'] if 'date' in d else d['time']).to_numpy()
        from beid_data_quality import flags
        if flags(d).any(axis=None):
            raise ValueError(f'{symbol}: suspect raw bars; audit sources before replay')
        order=np.argsort(dates); dates=dates[order]
        prices=d[['open','high','low','close']].to_numpy(dtype=float)[order]
        starts=np.searchsorted(dates,t.fill_time.to_numpy())
        r=replay(dates.astype('int64'),prices,starts,t.entry.to_numpy(),t.sl.to_numpy(),t.tp.to_numpy(),np.where(t.side=='BUY',1,-1))
        coarse=t.copy()
        t.fill_time=dates[r[:,0].astype(int)]
        t.exit_time=dates[r[:,1].astype(int)]
        t['reason']=pd.Series(r[:,3]).map({1:'STOP',2:'TARGET',3:'END_MARK'})
        t['fill_bar_stop']=r[:,4].astype(bool)
        t.net_pips=np.where(t.side=='BUY',1,-1)*(r[:,2]-t.entry)/base.get_pip_size(symbol)-1.6
        t.net_r=t.net_pips/t.half_pips
        t.to_csv(OUT/f'{symbol}_trades.csv',index=False)
        comparisons.append(dict(symbol=symbol,coarse=base.metrics(coarse),fine=base.metrics(t),
            changed_exit_count=int((coarse.reason!=t.reason).sum()),fill_minute_stops=int(t.fill_bar_stop.sum())))
        frames.append(t)
        print(comparisons[-1],flush=True)
    (OUT/'resolution_comparison.json').write_text(json.dumps(comparisons,indent=2))
    base.OUT=OUT
    base.evaluate(pd.concat(frames,ignore_index=True))
    evaluation=pd.read_csv(OUT/'evaluation.csv')
    (OUT/'evaluation.json').write_text(evaluation.to_json(orient='records',indent=2))

if __name__=='__main__': main()
