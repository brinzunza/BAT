import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research_scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import goal29_rerank_1_per_side as g29
import pandas as pd, numpy as np
from swing_zigzag import detect_confirmed_swings

rs, pg = g29.rs, g29.pg
times, lows, highs, df = rs.load_instrument('GBPUSD')
df = df.copy()
df.index = pd.to_datetime(times)

end = df.index.max()
start = end - pd.Timedelta(hours=24)
w = df.loc[start:end].reset_index().rename(columns={"index":"time"})
print(f"window: {w.time.iloc[0]} -> {w.time.iloc[-1]}  ({len(w)} 1-min bars)")

open_, high_, low_, close_ = w.open.values, w.high.values, w.low.values, w.close.values
n = len(w)
pip = pg.PIP_SIZE['GBPUSD']
spread_price = pg.SPREAD_PIPS_TYPICAL * pip

pivots = detect_confirmed_swings(high_, low_)
print(f"confirmed swing pivots: {len(pivots)}")

last_high = None
last_low = None
trend = None
zones = []
for confirm_idx, pivot_idx, typ, price in pivots:
    if typ == "H":
        if last_high is not None:
            if price > last_high[1]:
                if trend == "down" and last_low is not None:
                    zones.append(dict(created_idx=confirm_idx, top=float(last_high[1]), bottom=float(last_low[1]), is_buy=True, kind="flip"))
                trend = "up"
            else:
                if last_low is not None:
                    zones.append(dict(created_idx=confirm_idx, top=float(price), bottom=float(last_low[1]), is_buy=False, kind="pullback"))
                trend = "down"
        last_high = (pivot_idx, price)
    else:
        if last_low is not None:
            if price < last_low[1]:
                if trend == "up" and last_high is not None:
                    zones.append(dict(created_idx=confirm_idx, top=float(last_high[1]), bottom=float(last_low[1]), is_buy=False, kind="flip"))
                trend = "down"
            else:
                if last_high is not None:
                    zones.append(dict(created_idx=confirm_idx, top=float(last_high[1]), bottom=float(price), is_buy=True, kind="pullback"))
                trend = "up"
        last_low = (pivot_idx, price)

print(f"zones: {len(zones)}  (pullback: {sum(1 for z in zones if z['kind']=='pullback')}, flip: {sum(1 for z in zones if z['kind']=='flip')})")

MAXW = 240
MIN_STOP_PIPS = 3.0
rows = []
for z in zones:
    z["mid"] = (z["top"] + z["bottom"]) / 2.0
    stop_dist_pips = (z["top"] - z["bottom"]) / 2.0 / pip
    if stop_dist_pips < MIN_STOP_PIPS:
        continue
    c = z["created_idx"]
    end_i = min(c+MAXW, n)
    touch_idx = None
    for j in range(c+1, end_i):
        touched = (low_[j] <= z["mid"]) if z["is_buy"] else (high_[j] >= z["mid"])
        if touched:
            touch_idx = j
            break
    if touch_idx is None:
        continue
    if z["is_buy"]:
        deep = low_[touch_idx] <= z["mid"] - spread_price
        entry_price = z["mid"] if deep else z["mid"] + spread_price
    else:
        deep = high_[touch_idx] >= z["mid"] + spread_price
        entry_price = z["mid"] if deep else z["mid"] - spread_price
    stop_price = z["bottom"] if z["is_buy"] else z["top"]
    target_price = z["top"] if z["is_buy"] else z["bottom"]
    eff_risk = abs(entry_price - stop_price)
    if eff_risk <= 0:
        continue
    win_r = abs(target_price - entry_price) / eff_risk
    outcome, exit_idx = None, None
    for k in range(touch_idx, end_i):
        hit_sl = (low_[k] <= stop_price) if z["is_buy"] else (high_[k] >= stop_price)
        hit_tp = (high_[k] >= target_price) if z["is_buy"] else (low_[k] <= target_price)
        if hit_sl:
            outcome, exit_idx = "loss", k; break
        elif hit_tp:
            outcome, exit_idx = "win", k; break
    if outcome is None:
        continue
    net_r = win_r if outcome == "win" else -1.0
    rows.append(dict(created_idx=c, kind=z["kind"], is_buy=z["is_buy"], entry_time=w.time[touch_idx],
                      exit_time=w.time[exit_idx], outcome=outcome, net_r=net_r, win_r=win_r))

t = pd.DataFrame(rows)
print(f"\nresolved trades: {len(t)}")
if len(t):
    wr = 100*(t.net_r>0).mean()
    gw = t[t.net_r>0].net_r.sum()
    gl = -t[t.net_r<=0].net_r.sum()
    pf = gw/gl if gl>0 else float('inf')
    print(f"WR={wr:.1f}%  mean win_r={t.win_r.mean():.2f}  PF={pf:.3f}  sum net_r={t.net_r.sum():.2f}")
    print(t.groupby('kind').net_r.agg(['count','sum','mean']))
    print(f"wins={int((t.net_r>0).sum())} losses={int((t.net_r<=0).sum())}")
