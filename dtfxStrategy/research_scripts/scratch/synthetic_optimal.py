import json
import numpy as np

base = 1.2600
# Strict alternation (every waypoint reverses direction from the last) -- small "confirm"
# bounces are real steps, not skippable, since the zigzag only registers a pivot once price
# genuinely reverses off it.
seq = [
    base,               # start                     1.2600
    base + 0.0150,      # H1 bootstrap              1.2750
    base + 0.0050,      # L1 bootstrap              1.2650
    base + 0.0200,      # H2 advance                1.2800
    base + 0.0100,      # L2 -> ZONE A [2700,2800]  1.2700
    base + 0.0120,      # confirm bounce (tiny)     1.2720
    base - 0.0100,      # L3 advance, breaches A    1.2500
    base + 0.0050,      # H -> ZONE B [2500,2650]   1.2650   (confirms L3; vs last_high 2720 -> lower high)
    base + 0.0030,      # confirm bounce (tiny)     1.2630
    base + 0.0250,      # H4 advance, breaches B    1.2850
    base + 0.0000,      # L -> ZONE C [2600,2850]   1.2600   (confirms H4; vs last_low 2500 -> higher low)
    base - 0.0020,      # confirm bounce (tiny)     1.2580
    base - 0.0200,      # L5 advance, breaches C    1.2400
    base + 0.0100,      # H -> ZONE D [2400,2700]   1.2700   (confirms L5; vs last_high 2850 -> lower high)
    base + 0.0080,      # confirm bounce (tiny)     1.2680
    base + 0.0350,      # H6 advance, breaches D    1.2950
]
BARS_PER_LEG = 5

def ramp(a, b, n):
    return list(np.linspace(a, b, n + 1))[1:]

prices = [seq[0]]
for i in range(1, len(seq)):
    prices.extend(ramp(seq[i-1], seq[i], BARS_PER_LEG))

rng = np.random.RandomState(3)
bars = []
for i in range(1, len(prices)):
    o, c = prices[i-1], prices[i]
    wick = abs(c - o) * 0.12 + 0.00015
    h = max(o, c) + wick * rng.uniform(0.15, 0.6)
    l = min(o, c) - wick * rng.uniform(0.15, 0.6)
    bars.append(dict(o=round(o,5), h=round(h,5), l=round(l,5), c=round(c,5)))

print("bars:", len(bars))
high_ = np.array([b["h"] for b in bars])
low_ = np.array([b["l"] for b in bars])
n = len(bars)

def zigzag(high, low):
    n = len(high)
    pivots = []
    direction_up = True
    cand_price, cand_idx = high[0], 0
    for i in range(1, n):
        if direction_up:
            if high[i] > cand_price:
                cand_price, cand_idx = high[i], i
            else:
                pivots.append((i, cand_idx, "H", cand_price))
                direction_up = False
                cand_price, cand_idx = low[i], i
        else:
            if low[i] < cand_price:
                cand_price, cand_idx = low[i], i
            else:
                pivots.append((i, cand_idx, "L", cand_price))
                direction_up = True
                cand_price, cand_idx = high[i], i
    return pivots

pivots = zigzag(high_, low_)
print("pivots:", len(pivots))
for p in pivots:
    print(" ", p)

last_high, last_low, trend = None, None, None
zones = []
for confirm_idx, pivot_idx, typ, price in pivots:
    if typ == "H":
        if last_high is not None:
            if price > last_high[1]:
                trend = "up"
            else:
                if last_low is not None:
                    zones.append(dict(created_idx=confirm_idx, top=float(price), bottom=float(last_low[1]), is_buy=True))
                trend = "down"
        last_high = (pivot_idx, price)
    else:
        if last_low is not None:
            if price < last_low[1]:
                trend = "down"
            else:
                if last_high is not None:
                    zones.append(dict(created_idx=confirm_idx, top=float(last_high[1]), bottom=float(price), is_buy=False))
                trend = "up"
        last_low = (pivot_idx, price)

print("zones:", len(zones))

pip = 0.0001
spread_price = 1.6 * pip
MAXW = 40
for z in zones:
    z["mid"] = (z["top"] + z["bottom"]) / 2.0
    c = z["created_idx"]
    end = min(c+MAXW, n)
    touch_idx = None
    orig_is_buy = not z["is_buy"]
    for j in range(c+1, end):
        touched = (low_[j] <= z["mid"]) if orig_is_buy else (high_[j] >= z["mid"])
        if touched:
            touch_idx = j
            break
    z["touch_idx"] = touch_idx
    if touch_idx is None:
        z["outcome"] = "no_touch"
        continue
    if orig_is_buy:
        deep = low_[touch_idx] <= z["mid"] - spread_price
        entry_price = z["mid"] if deep else z["mid"] + spread_price
    else:
        deep = high_[touch_idx] >= z["mid"] + spread_price
        entry_price = z["mid"] if deep else z["mid"] - spread_price
    stop_price = z["bottom"] if z["is_buy"] else z["top"]
    target_price = z["top"] if z["is_buy"] else z["bottom"]
    outcome, exit_idx = None, None
    for k in range(touch_idx, end):
        if z["is_buy"]:
            hit_sl = low_[k] <= stop_price; hit_tp = high_[k] >= target_price
        else:
            hit_sl = high_[k] >= stop_price; hit_tp = low_[k] <= target_price
        if hit_sl:
            outcome, exit_idx = "loss", k; break
        elif hit_tp:
            outcome, exit_idx = "win", k; break
    z["outcome"] = outcome or "unresolved"
    z["exit_idx"] = exit_idx
    z["height_pips"] = (z["top"] - z["bottom"]) / pip

for z in zones:
    print(f"  created={z['created_idx']} {'buy' if z['is_buy'] else 'sell'} "
          f"bottom={z['bottom']:.5f} mid={z['mid']:.5f} top={z['top']:.5f} "
          f"height={z.get('height_pips',0):.0f}pips touch={z.get('touch_idx')} outcome={z['outcome']}")

MIN_HEIGHT_PIPS = 50
big_zones = [z for z in zones if z.get("height_pips", 0) >= MIN_HEIGHT_PIPS]
print(f"\nbig zones (>= {MIN_HEIGHT_PIPS} pips): {len(big_zones)}")
for z in big_zones:
    print(f"  {'buy' if z['is_buy'] else 'sell'} outcome={z['outcome']} height={z['height_pips']:.0f}pips")

bars_json = [dict(t=f"2024-01-{1+i//6:02d}T{(i%6)*4:02d}:00", **b) for i, b in enumerate(bars)]
zones_out = [dict(created_idx=z["created_idx"], top=round(z["top"],5), bottom=round(z["bottom"],5),
                   mid=round(z["mid"],5), is_buy=z["is_buy"], touch_idx=z.get("touch_idx"),
                   exit_idx=z.get("exit_idx"), outcome=z["outcome"], height_pips=z.get("height_pips",0))
             for z in zones]
with open("synthetic_optimal_data.json", "w") as f:
    json.dump(dict(bars=bars_json, zones=zones_out), f, separators=(",",":"))
print("exported", len(bars_json), "bars,", len(zones_out), "zones")
