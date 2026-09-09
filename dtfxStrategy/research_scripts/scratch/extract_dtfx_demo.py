import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research_scripts"))
import goal29_rerank_1_per_side as g29
import pandas as pd, numpy as np, json

rs, pg = g29.rs, g29.pg
times, lows, highs, df = rs.load_instrument('GBPUSD')
df = df.copy()
df.index = pd.to_datetime(times)

# pick a window and resample to 4H bars for visual clarity
window = df.loc["2024-01-01":"2024-04-15"]
h4 = window.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
h4 = h4.reset_index().rename(columns={"index":"time"})
print("4H bars:", len(h4))

open_, high_, low_, close_ = h4.open.values, h4.high.values, h4.low.values, h4.close.values
n = len(h4)
LB = 3

# fractal swing detection (confirmed LB bars after the pivot bar)
is_sh = np.zeros(n, dtype=bool)
is_sl = np.zeros(n, dtype=bool)
for i in range(LB, n-LB):
    if high_[i] == high_[i-LB:i+LB+1].max() and (high_[i-LB:i]<high_[i]).all() and (high_[i+1:i+LB+1]<high_[i]).all():
        is_sh[i] = True
    if low_[i] == low_[i-LB:i+LB+1].min() and (low_[i-LB:i]>low_[i]).all() and (low_[i+1:i+LB+1]>low_[i]).all():
        is_sl[i] = True

# chronological confirmed swings (confirmed at i+LB)
swings = []  # (confirm_idx, pivot_idx, type, price)
for i in range(n):
    if is_sh[i] and i+LB < n:
        swings.append((i+LB, i, "H", high_[i]))
    if is_sl[i] and i+LB < n:
        swings.append((i+LB, i, "L", low_[i]))
swings.sort(key=lambda x: x[0])

zones = []
last_sh = None  # (price, pivot_idx)
last_sl = None
broken_sh_idx = set()
broken_sl_idx = set()

# walk bar by bar, updating confirmed swings and checking breaks
swing_ptr = 0
for i in range(n):
    while swing_ptr < len(swings) and swings[swing_ptr][0] == i:
        _, pivot_idx, typ, price = swings[swing_ptr]
        if typ == "H":
            last_sh = (price, pivot_idx)
        else:
            last_sl = (price, pivot_idx)
        swing_ptr += 1
    # check bullish break: close beyond last confirmed swing high, not yet used
    if last_sh is not None and last_sl is not None and last_sh[1] not in broken_sh_idx:
        if close_[i] > last_sh[0] and last_sl[1] < last_sh[1]:
            top, bottom = last_sh[0], last_sl[0]
            if top > bottom:
                mid = (top+bottom)/2.0
                zones.append(dict(created_idx=i, top=float(top), bottom=float(bottom), mid=float(mid), is_buy=True))
                broken_sh_idx.add(last_sh[1])
    if last_sl is not None and last_sh is not None and last_sl[1] not in broken_sl_idx:
        if close_[i] < last_sl[0] and last_sh[1] < last_sl[1]:
            top, bottom = last_sh[0], last_sl[0]
            if top > bottom:
                mid = (top+bottom)/2.0
                zones.append(dict(created_idx=i, top=float(top), bottom=float(bottom), mid=float(mid), is_buy=False))
                broken_sl_idx.add(last_sl[1])

print("zones found:", len(zones))

# simulate the simple rule: entry at mid on touch, SL at opposite bound, TP at near bound
MAXW = 60
for z in zones:
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
            hit_sl = low_[k] <= z["bottom"]
            hit_tp = high_[k] >= z["top"]
        else:
            hit_sl = high_[k] >= z["top"]
            hit_tp = low_[k] <= z["bottom"]
        if hit_sl:
            outcome, exit_idx = "loss", k; break
        elif hit_tp:
            outcome, exit_idx = "win", k; break
    z["outcome"] = outcome or "unresolved"
    z["exit_idx"] = exit_idx

for z in zones:
    print(z["created_idx"], h4.time[z["created_idx"]], "buy" if z["is_buy"] else "sell",
          round(z["bottom"],5), round(z["mid"],5), round(z["top"],5),
          z.get("touch_idx"), z["outcome"])

# export bars + zones as compact JSON
bars_json = [dict(t=h4.time[i].strftime("%Y-%m-%dT%H:%M"), o=round(float(open_[i]),5),
                   h=round(float(high_[i]),5), l=round(float(low_[i]),5), c=round(float(close_[i]),5))
             for i in range(n)]
zones_out = []
for z in zones:
    zones_out.append(dict(created_idx=z["created_idx"], top=round(z["top"],5), bottom=round(z["bottom"],5),
                           mid=round(z["mid"],5), is_buy=z["is_buy"], touch_idx=z.get("touch_idx"),
                           exit_idx=z.get("exit_idx"), outcome=z["outcome"]))

with open(os.path.join(os.path.dirname(__file__), "dtfx_demo_data.json"), "w") as f:
    json.dump(dict(bars=bars_json, zones=zones_out), f)
print("exported", len(bars_json), "bars and", len(zones_out), "zones")
