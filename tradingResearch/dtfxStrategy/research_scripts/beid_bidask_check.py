"""Frozen 40-pip flip + closed-hour trend candidate on available five-second quotes.
Not a new tuning sample; short, previously available historical execution check only.
"""
import json
import numpy as np
import pandas as pd
import beid_conservative_audit as base
from beid_clean_htf_study import hour_direction,OUT

def quote_fill(q,entry,stop,target,side):
    """Return executed quote-bar indices, price and reason; expire pending orders only."""
    filled=False
    for r in q.itertuples():
        if not filled:
            if r.time>r.expiry:break
            if (r.ask_l<=entry if side==1 else r.bid_h>=entry):
                fill=r.time;filled=True;fresh=True
            else:continue
        else:fresh=False
        hit_stop=r.bid_l<=stop if side==1 else r.ask_h>=stop
        hit_target=(r.bid_c>=target if side==1 else r.ask_c<=target) if fresh else (r.bid_h>=target if side==1 else r.ask_l<=target)
        if hit_stop:
            price=min(stop,r.bid_o) if side==1 else max(stop,r.ask_o)
            return fill,r.time,price,'STOP'
        if hit_target:return fill,r.time,target,'TARGET'
    if filled:
        return fill,r.time,(r.bid_c if side==1 else r.ask_c),'END_MARK'
    return None

def main():
    frames=[];coverage=[]
    for symbol in base.PATHS:
        q=pd.read_parquet(base.DATA/'s5_cache'/f'{symbol}_s5.parquet')
        q['time']=pd.to_datetime(q.time,utc=True).dt.tz_localize(None)
        q=q.sort_values('time').reset_index(drop=True)
        if (q.bid_c>q.ask_c).any() or q.time.duplicated().any():raise ValueError('Invalid quotes')
        raw=q.set_index('time')[['open','high','low','close']]
        bars=raw.resample('3min',closed='left',label='left').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        # Drop final incomplete signal bar; quote replay can still exit there.
        bars=bars[bars.time+pd.Timedelta(minutes=3)<=q.time.max()]
        t=base.backtest_beid_conservative(bars,symbol,3)
        count=0
        if len(t):
            t=t[(t.zone_pips>=40)&t.type.str.contains('FLIP')]
            for z in t.itertuples():
                side=1 if z.side=='BUY' else -1
                window=q[q.time>=z.created_time].copy()
                window['expiry']=z.created_time+pd.Timedelta(hours=48)
                result=quote_fill(window,z.entry,z.sl,z.tp,side)
                if result is None:continue
                fill,exit,price,reason=result
                # Cancel the order at its first executable touch if hourly direction fails.
                if hour_direction(q[['time','close']],pd.Series([fill]))[0]!=side:continue
                if fill<q.time.min()+pd.Timedelta(hours=24):continue
                net=side*(price-z.entry)/base.get_pip_size(symbol)
                frames.append(dict(ticker=symbol,side=z.side,type=z.type,fill_time=fill,exit_time=exit,entry=z.entry,sl=z.sl,tp=z.tp,zone_pips=z.zone_pips,half_pips=z.half_pips,net_pips=net,net_r=net/z.half_pips,reason=reason))
                count+=1
        coverage.append(dict(symbol=symbol,first=str(q.time.min()),last=str(q.time.max()),quote_bars=len(q),candidate_trades=count))
    t=pd.DataFrame(frames)
    rows=[]
    if len(t):
        t=base.one_position(t)
        t.to_csv(OUT/'five_second_bidask_trades.csv',index=False)
        for commission_slip in [0,.2,.5]:
            p=t.copy();p.net_pips-=commission_slip;p.net_r=p.net_pips/p.half_pips
            rows.append(dict(extra_roundtrip_pips=commission_slip,**base.metrics(p)))
    summary=dict(status='short historical quote check; not sufficient for statistical validation',coverage=coverage,results=rows,spread_accounting='embedded in bid/ask fill and exit triggers; no extra spread deduction')
    (OUT/'five_second_bidask.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
