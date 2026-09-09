import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research_scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import goal29_rerank_1_per_side as g29
import pandas as pd, numpy as np, json
from swing_zigzag import detect_confirmed_swings

rs, pg = g29.rs, g29.pg
times, lows, highs, df = rs.load_instrument('GBPUSD')
df = df.copy()
df.index = pd.to_datetime(times)

window = df.loc["2024-01-01":"2024-04-15"]
h4 = window.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
h4 = h4.reset_index().rename(columns={"index":"time"})
print("4H bars:", len(h4))

open_, high_, low_, close_ = h4.open.values, h4.high.values, h4.low.values, h4.close.values
n = len(h4)

raw_pivots = detect_confirmed_swings(high_, low_)
# raw_pivots: (confirm_idx, pivot_idx, type, price) -- already chronological by confirm_idx
print("pivots:", len(raw_pivots))

last_high = None
last_low = None
trend = None
zones = []
for confirm_idx, pivot_idx, typ, price in raw_pivots:
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

print(f"zones: {len(zones)} (pullback: {sum(1 for z in zones if z['kind']=='pullback')}, flip: {sum(1 for z in zones if z['kind']=='flip')})")

MAXW = 60
for z in zones:
    z["mid"] = (z["top"] + z["bottom"]) / 2.0
    c = z["created_idx"]
    end = min(c+MAXW, n)
    touch_idx = None
    for j in range(c+1, end):
        touched = (low_[j] <= z["mid"]) if z["is_buy"] else (high_[j] >= z["mid"])
        if touched:
            touch_idx = j
            break
    z["touch_idx"] = touch_idx
    if touch_idx is None:
        z["outcome"] = "no_touch"
        continue
    outcome, exit_idx = None, None
    for k in range(touch_idx, end):
        if z["is_buy"]:
            hit_sl = low_[k] <= z["bottom"]; hit_tp = high_[k] >= z["top"]
        else:
            hit_sl = high_[k] >= z["top"]; hit_tp = low_[k] <= z["bottom"]
        if hit_sl:
            outcome, exit_idx = "loss", k; break
        elif hit_tp:
            outcome, exit_idx = "win", k; break
    z["outcome"] = outcome or "unresolved"
    z["exit_idx"] = exit_idx

n_resolved = sum(1 for z in zones if z.get("outcome") in ("win","loss"))
n_win = sum(1 for z in zones if z.get("outcome")=="win")
n_loss = sum(1 for z in zones if z.get("outcome")=="loss")
print(f"resolved: {n_resolved}  win: {n_win}  loss: {n_loss}  WR: {100*n_win/max(n_resolved,1):.1f}%")

bars_json = [dict(t=h4.time[i].strftime("%Y-%m-%dT%H:%M"), o=round(float(open_[i]),5),
                   h=round(float(high_[i]),5), l=round(float(low_[i]),5), c=round(float(close_[i]),5))
             for i in range(n)]
zones_out = []
for z in zones:
    zones_out.append(dict(created_idx=z["created_idx"], top=round(z["top"],5), bottom=round(z["bottom"],5),
                           mid=round(z["mid"],5), is_buy=z["is_buy"], kind=z["kind"],
                           touch_idx=z.get("touch_idx"), exit_idx=z.get("exit_idx"), outcome=z["outcome"]))

with open(os.path.join(os.path.dirname(__file__), "zigzag_zones_data.json"), "w") as f:
    json.dump(dict(bars=bars_json, zones=zones_out), f, separators=(",",":"))
print("exported", len(bars_json), "bars and", len(zones_out), "zones")
