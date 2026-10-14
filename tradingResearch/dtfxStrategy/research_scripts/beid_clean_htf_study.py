"""Preregistered clean-period 3m BEID / 1m execution / completed-hour trend study."""
import json
import numpy as np
import pandas as pd
import beid_conservative_audit as base
from beid_data_quality import flags
from beid_fine_execution import replay
OUT=base.OUT/'clean_2003_htf'

def hour_direction(raw, entries):
    hourly=raw.set_index('time').close.resample('1h',label='right',closed='left').last().dropna()
    ema=hourly.ewm(span=20,adjust=False,min_periods=20).mean()
    direction=pd.Series(np.where((hourly>ema)&(ema>ema.shift()),1,np.where((hourly<ema)&(ema<ema.shift()),-1,0)),index=hourly.index)
    return direction.reindex(pd.DatetimeIndex(entries),method='ffill').fillna(0).to_numpy()

def evaluate(t):
    rows=[]
    for floor in [20,30,40]:
        for kind in ['all','continuation','flip']:
            for session in ['all','07-17']:
                for htf in ['any','aligned']:
                    s=base.subset(t,floor,kind,session)
                    if htf=='aligned':s=s[s.aligned]
                    train=base.one_position(s[(s.fill_time>='2004-01-01')&(s.fill_time<'2016-01-01')&(s.exit_time<'2016-01-01')])
                    rows.append(dict(floor=floor,kind=kind,session=session,htf=htf,**base.metrics(train,2.4)))
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'training_grid.csv',index=False)
    viable=grid[(grid.n>=500)&(grid.mean_r>0)].sort_values('mean_r',ascending=False)
    configs=[dict(floor=20,kind='all',session='all',htf='any',label='baseline')]
    if len(viable):
        w=viable.iloc[0];configs.append(dict(floor=int(w.floor),kind=w.kind,session=w.session,htf=w.htf,label='train_selected'))
    results=[]
    for cfg in configs:
        s=base.subset(t,cfg['floor'],cfg['kind'],cfg['session'])
        if cfg['htf']=='aligned':s=s[s.aligned]
        s=base.one_position(s[s.fill_time>='2004-01-01'])
        s.to_csv(OUT/f"{cfg['label']}_portfolio.csv",index=False)
        for label,start,end in [('train','2004-01-01','2016-01-01'),('validation','2016-01-01','2021-01-01'),('recent','2021-01-01','2026-04-01'),('reused_late','2026-04-01','2026-09-10')]:
            p=s[(s.fill_time>=start)&(s.fill_time<end)&(s.exit_time<end)]
            for cost in [1.6,2.4,3.2]:results.append(dict(**cfg,period=label,cost=cost,**base.metrics(p,cost)))
    e=pd.DataFrame(results);e.to_csv(OUT/'evaluation.csv',index=False)
    (OUT/'evaluation.json').write_text(e.to_json(orient='records',indent=2))
    print(e.to_string(index=False),flush=True)

def main():
    OUT.mkdir(exist_ok=True)
    frames=[]
    for symbol,path in base.PATHS.items():
        print('Build',symbol,flush=True)
        d=pd.read_csv(path);d['time']=pd.to_datetime(d['date'] if 'date' in d else d['time'])
        d=d[d.time>='2003-01-01'].sort_values('time').reset_index(drop=True)
        if flags(d).any(axis=None) or d.time.duplicated().any():raise ValueError('Data audit failure: '+symbol)
        bars=d.set_index('time').resample('3min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        t=base.backtest_beid_conservative(bars,symbol,3)
        dates=d.time.to_numpy()
        r=replay(dates.astype('int64'),d[['open','high','low','close']].to_numpy(float),np.searchsorted(dates,t.fill_time.to_numpy()),t.entry.to_numpy(),t.sl.to_numpy(),t.tp.to_numpy(),np.where(t.side=='BUY',1,-1))
        t.fill_time=dates[r[:,0].astype(int)];t.exit_time=dates[r[:,1].astype(int)]
        t.reason=pd.Series(r[:,3]).map({1:'STOP',2:'TARGET',3:'END_MARK'});t.fill_bar_stop=r[:,4].astype(bool)
        t.net_pips=np.where(t.side=='BUY',1,-1)*(r[:,2]-t.entry)/base.get_pip_size(symbol)-1.6
        t.net_r=t.net_pips/t.half_pips
        t['htf_direction']=hour_direction(d,t.fill_time)
        t['aligned']=t.htf_direction==np.where(t.side=='BUY',1,-1)
        t.to_csv(OUT/f'{symbol}_trades.csv',index=False);frames.append(t)
        print(symbol,base.metrics(t),flush=True)
    evaluate(pd.concat(frames,ignore_index=True))

if __name__=='__main__': main()
