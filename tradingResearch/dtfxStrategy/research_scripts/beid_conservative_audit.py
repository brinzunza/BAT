"""BEID execution audit. Research only; does not place orders.
Run: python3 dtfxStrategy/research_scripts/beid_conservative_audit.py
Legacy swing detection copied from verify_beid_full_history_fixed.py on 2026-09-09.
No imports from trading/broker modules. All timestamps are treated as source UTC;
timezone provenance needs independent verification before session rules go live.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "trading-data"
OUT = ROOT / "dtfxStrategy/research_results/beid_conservative_audit"
PATHS = {s: DATA / n for s,n in {
    "EURUSD":"eurusd_1min_2000_2026_master.csv",
    "GBPUSD":"gbpusd_1min_2000_2026_master.csv",
    "USDJPY":"usdjpy_1min_2000_2026_master.csv",
    "EURGBP":"lse_eurgbp_1min_full.csv"}.items()}

def get_pip_size(ticker):
    return .01 if "JPY" in ticker else .0001

def backtest_beid_conservative(df, ticker_name, timeframe_min, max_touches=5, spread_pips=1.6,
                         pullback_timeout_hours=48.0, zone_expiry_hours=48.0):
    """Preserve the prior swing generator; replace execution with conservative OHLC rules.
    Single touch, wall-clock expiry, no active-zone cap, >=20 pip admission.
    Fill-bar TP is credited only if its close proves a post-entry crossing.
    Otherwise an ambiguous TP is deferred; stop always wins ambiguity.
    """
    pip_size = get_pip_size(ticker_name)
    time_col = "time" if "time" in df.columns else "date"
    dates = df[time_col].values
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(opens)


    trend = 1
    pHH = opens[0]; pHH_idx = 0; cHH = np.nan
    pHL = opens[0]; pHL_idx = 0; cHL = np.nan; last_cHL = np.nan
    pLL = opens[0]; pLL_idx = 0; cLL = np.nan
    pLH = opens[0]; pLH_idx = 0; cLH = np.nan; last_cLH = np.nan
    in_pullback = False
    pullback_started_idx = -1
    prev_green_open = np.nan
    prev_red_open = np.nan

    active_zones, trades = [], []

    for i in range(n):
        o, h, l, cl = opens[i], highs[i], lows[i], closes[i]
        is_green = cl >= o
        is_red = cl < o

        retained = []
        for z in active_zones:
            side, entry, tp, sl, half_pips = z["side"], z["entry"], z["tp"], z["sl"], z["half_pips"]
            just_filled = False
            if not z["filled"]:
                if dates[i] - dates[z["created_idx"]] > np.timedelta64(int(zone_expiry_hours*3600), 's'):
                    continue
                touched = l <= entry if side == 1 else h >= entry
                if not touched:
                    retained.append(z)
                    continue
                z["filled"] = True
                z["fill_idx"] = i
                z["touch_count"] = 1
                just_filled = True
            hit_sl = l <= sl if side == 1 else h >= sl
            # Close beyond TP proves TP followed entry on a fill bar.
            hit_tp = (cl >= tp if side == 1 else cl <= tp) if just_filled else (h >= tp if side == 1 else l <= tp)
            terminal = i == n-1
            if hit_sl or hit_tp or terminal:
                if hit_sl:
                    exit_price = min(sl, o) if side == 1 else max(sl, o)
                    reason = "STOP"
                elif hit_tp:
                    exit_price, reason = tp, "TARGET"
                else:
                    exit_price, reason = cl, "END_MARK"
                net_pips = side * (exit_price-entry)/pip_size - spread_pips
                trades.append(dict(ticker=ticker_name, side="BUY" if side == 1 else "SELL",
                    type=z["type"], created_time=dates[z["created_idx"]]+np.timedelta64(timeframe_min,"m"),
                    fill_time=dates[z["fill_idx"]], exit_time=dates[i],
                    entry=entry, sl=sl, tp=tp, zone_pips=z["zone_pips"], half_pips=half_pips,
                    touch_num=1, reason=reason, fill_bar_stop=bool(just_filled and hit_sl),
                    net_pips=net_pips, net_r=net_pips/half_pips))
            else:
                retained.append(z)
        active_zones = retained

        # FIX: pullback timeout, scaled to this timeframe's actual bar duration. Resetting
        # only in_pullback isn't enough -- pLL/pHH (the primary extreme the NEXT pullback
        # attempt re-anchors from) stays frozen at the ancient value forever too (it only
        # advances via "if l<pLL" / "if h>pHH", which never fires again once price has
        # moved away from that extreme for good), so a bare in_pullback reset just lets the
        # very next bar restart an identical, equally-doomed pullback attempt with the same
        # stale reference -- an infinite retry loop instead of a one-time freeze. Confirmed
        # by direct instrumentation: EURUSD hit this reset 20+ times a few months apart
        # before still permanently locking up in 2002. Fix: abandon the stale swing
        # entirely on timeout and re-anchor from the current bar, same as giving up on a
        # zone that's expired unfilled.
        if in_pullback and (dates[i] - dates[pullback_started_idx] > np.timedelta64(int(pullback_timeout_hours*3600), "s")):
            in_pullback = False
            if trend == 1:
                pHH = h; pHH_idx = i
            else:
                pLL = l; pLL_idx = i

        if trend == 1:
            if h > pHH or np.isnan(pHH):
                pHH = h; pHH_idx = i
            if not np.isnan(last_cHL) and cl < last_cHL:
                old_floor = last_cHL; top_peak = pHH
                trend = -1; cLH = top_peak; last_cLH = np.nan
                pLL = l; pLL_idx = i; cLL = np.nan; in_pullback = False
                top, bottom = top_peak, old_floor
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                if top > bottom and zone_pips >= 20.0:
                    active_zones.append(dict(side=-1, type="SELL_FLIP", top=top, bottom=bottom, entry=mid,
                                              tp=bottom, sl=top, zone_pips=zone_pips, half_pips=half_pips,
                                              created_idx=i, filled=False, fill_idx=-1, touch_count=0))
            else:
                if not in_pullback and is_red and not np.isnan(prev_green_open) and cl < prev_green_open:
                    in_pullback = True; pullback_started_idx = i
                    cHH = pHH; pHL = l; pHL_idx = i
                elif in_pullback:
                    if l < pHL:
                        pHL = l; pHL_idx = i
                    if cl > cHH:
                        in_pullback = False
                        cHL = pHL; last_cHL = cHL
                        top, bottom = cHH, cHL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                        pHH = h; pHH_idx = i
                        if top > bottom and zone_pips >= 20.0:
                            active_zones.append(dict(side=1, type="BUY", top=top, bottom=bottom, entry=mid,
                                                      tp=top, sl=bottom, zone_pips=zone_pips, half_pips=half_pips,
                                                      created_idx=i, filled=False, fill_idx=-1, touch_count=0))
        elif trend == -1:
            if l < pLL or np.isnan(pLL):
                pLL = l; pLL_idx = i
            if not np.isnan(last_cLH) and cl > last_cLH:
                old_ceiling = last_cLH; bottom_trough = pLL
                trend = 1; cHL = bottom_trough; last_cHL = np.nan
                pHH = h; pHH_idx = i; cHH = np.nan; in_pullback = False
                top, bottom = old_ceiling, bottom_trough
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                if top > bottom and zone_pips >= 20.0:
                    active_zones.append(dict(side=1, type="BUY_FLIP", top=top, bottom=bottom, entry=mid,
                                              tp=top, sl=bottom, zone_pips=zone_pips, half_pips=half_pips,
                                              created_idx=i, filled=False, fill_idx=-1, touch_count=0))
            else:
                if not in_pullback and is_green and not np.isnan(prev_red_open) and cl > prev_red_open:
                    in_pullback = True; pullback_started_idx = i
                    cLL = pLL; pLH = h; pLH_idx = i
                elif in_pullback:
                    if h > pLH:
                        pLH = h; pLH_idx = i
                    if cl < cLL:
                        in_pullback = False
                        cLH = pLH; last_cLH = cLH
                        top, bottom = cLH, cLL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                        pLL = l; pLL_idx = i
                        if top > bottom and zone_pips >= 20.0:
                            active_zones.append(dict(side=-1, type="SELL", top=top, bottom=bottom, entry=mid,
                                                      tp=bottom, sl=top, zone_pips=zone_pips, half_pips=half_pips,
                                                      created_idx=i, filled=False, fill_idx=-1, touch_count=0))
        if is_green:
            prev_green_open = o
        elif is_red:
            prev_red_open = o

    return pd.DataFrame(trades)


def metrics(t, cost=1.6):
    r = t.net_r.to_numpy() - (cost-1.6)/t.half_pips.to_numpy()
    return dict(n=len(r), mean_r=float(r.mean()) if len(r) else None,
        sum_r=float(r.sum()), pf_r=float(r[r>0].sum()/-r[r<0].sum()) if (r<0).any() else None,
        net_pips=float((t.net_pips-(cost-1.6)).sum()))

def subset(t, floor, kind, session):
    mask = t.zone_pips.ge(floor)
    if kind == 'continuation': mask &= ~t.type.str.contains('FLIP')
    if kind == 'flip': mask &= t.type.str.contains('FLIP')
    if session == '07-17': mask &= t.fill_time.dt.hour.ge(7)&t.fill_time.dt.hour.lt(17)
    return t[mask]

def one_position(t):
    # Deterministic ties; slots release only on a later bar, avoiding intrabar ordering assumptions.
    t=t.sort_values(['fill_time','ticker','entry','sl','tp','type'], kind='stable')
    keep=[]; free=pd.Timestamp.min
    for row in t.itertuples():
        if row.fill_time > free:
            keep.append(row.Index); free=row.exit_time
    return t.loc[keep]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames=[]; manifest=[]
    for symbol,path in PATHS.items():
        print(f'Loading {symbol}',flush=True)
        d=pd.read_csv(path)
        tc='date' if 'date' in d else 'time'
        d['time']=pd.to_datetime(d[tc])
        dup=int(d.time.duplicated().sum())
        if dup: raise ValueError(f'{path}: {dup} duplicate timestamps; resolve provenance first')
        d=d.sort_values('time').set_index('time')
        from beid_data_quality import flags
        if flags(d).any(axis=None):
            raise ValueError(f'{symbol}: suspect raw bars; run beid_data_quality.py and resolve provenance first')
        if not ((d.high>=d[['open','close','low']].max(axis=1)) & (d.low<=d[['open','close','high']].min(axis=1))).all():
            raise ValueError(f'Invalid OHLC: {symbol}')
        bars=d.resample('3min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
        manifest.append(dict(symbol=symbol,path=str(path),bytes=path.stat().st_size,
            first=str(d.index.min()),last=str(d.index.max()),rows=len(d),bars=len(bars),duplicates=dup))
        t=backtest_beid_conservative(bars,symbol,3,max_touches=1)
        t.to_csv(OUT/f'{symbol}_trades.csv',index=False)
        frames.append(t)
        print(symbol,metrics(t),flush=True)
    t=pd.concat(frames,ignore_index=True)
    for col in ['fill_time','exit_time']: t[col]=pd.to_datetime(t[col])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    evaluate(t)

def evaluate(t):
    # Small, predeclared grid: no risk or TP/SL optimization. Selection uses pre-2016 only.
    rows=[]
    for floor in [20,30,40]:
        for kind in ['all','continuation','flip']:
            for session in ['all','07-17']:
                s=subset(t,floor,kind,session)
                train=one_position(s[(s.fill_time<'2016-01-01')&(s.exit_time<'2016-01-01')])
                row=dict(floor=floor,kind=kind,session=session,**metrics(train,2.4))
                rows.append(row)
    grid=pd.DataFrame(rows)
    grid.to_csv(OUT/'training_grid.csv',index=False)
    viable=grid[(grid.n>=500)&(grid.mean_r>0)].sort_values('mean_r',ascending=False)
    # Always evaluate the fixed >=20p baseline, even if no candidate passes training.
    configs=[dict(floor=20,kind='all',session='all',label='baseline')]
    if len(viable):
        winner=viable.iloc[0]
        configs.append(dict(floor=int(winner.floor),kind=winner.kind,session=winner.session,label='train_selected'))
    results=[]
    for cfg in configs:
        s=one_position(subset(t,cfg['floor'],cfg['kind'],cfg['session']))
        s.to_csv(OUT/f"{cfg['label']}_portfolio.csv",index=False)
        periods=[('train','2000-01-01','2016-01-01'),('validation','2016-01-01','2021-01-01'),
                 ('recent','2021-01-01','2026-04-01'),('reused_late','2026-04-01','2026-09-10')]
        for name,start,end in periods:
            part=s[(s.fill_time>=start)&(s.fill_time<end)&(s.exit_time<end)]
            for cost in [1.6,2.4,3.2]:
                results.append(dict(**cfg,period=name,cost=cost,**metrics(part,cost)))
    pd.DataFrame(results).to_csv(OUT/'evaluation.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print('Fill-bar stops:', int(t.fill_bar_stop.sum()),'of',len(t),flush=True)

if __name__ == '__main__':
    import sys
    if '--evaluate-cache' in sys.argv:
        cached=pd.concat([pd.read_csv(OUT/f'{symbol}_trades.csv',parse_dates=['fill_time','exit_time']) for symbol in PATHS],ignore_index=True)
        evaluate(cached)
    else:
        main()
