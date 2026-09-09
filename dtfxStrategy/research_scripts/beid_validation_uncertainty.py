"""Calendar-block validation uncertainty for the frozen clean-period candidate."""
import json
import numpy as np
import pandas as pd
from beid_clean_htf_study import OUT

def block_interval(part,cost,days,seed=42,n_sims=2000):
    daily=part.assign(r=part.net_r-(cost-1.6)/part.half_pips).groupby(part.fill_time.dt.floor('D')).agg(r=('r','sum'),n=('r','count'))
    calendar=pd.date_range(daily.index.min(),daily.index.max(),freq='D')
    x=daily.reindex(calendar,fill_value=0).to_numpy()
    rng=np.random.default_rng(seed); means=[]
    for _ in range(n_sims):
        starts=rng.integers(0,len(x),size=(len(x)+days-1)//days)
        idx=((starts[:,None]+np.arange(days))%len(x)).ravel()[:len(x)]
        sums=x[idx].sum(axis=0)
        if sums[1]:means.append(sums[0]/sums[1])
    return np.quantile(means,[.025,.5,.975]).tolist()

def main():
    path=OUT/'train_selected_portfolio.csv'
    if not path.exists():return
    t=pd.read_csv(path,parse_dates=['fill_time','exit_time'])
    rows=[]
    for label,start,end in [('validation','2016-01-01','2021-01-01'),('recent','2021-01-01','2026-04-01'),('all_later','2016-01-01','2026-09-10')]:
        p=t[(t.fill_time>=start)&(t.fill_time<end)&(t.exit_time<end)]
        for cost in [1.6,2.4]:
            for days in [7,28]:
                ci=block_interval(p,cost,days)
                rows.append(dict(period=label,cost=cost,calendar_block_days=days,n=len(p),mean_r_95pct_interval=ci))
    (OUT/'validation_uncertainty.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
