import json

# Waypoints: strictly alternating H/L, designed for 5 trend runs / 4 reversals,
# each run with >=5 swing pivots. (label, price)
WAYPOINTS = [
    ("L", 100),  # trend 1 (UP) start
    ("H", 110),
    ("L", 104),
    ("H", 118),
    ("L", 112),
    ("H", 128),   # last high of trend 1 -> flip zone (up->down) uses (L=112 .. H=128)
    ("L", 105),   # breaks 112 -> confirms DOWN (trend 2 start)
    ("H", 115),
    ("L", 98),
    ("H", 108),
    ("L", 90),
    ("H", 112),   # breaks last high -> confirms UP (trend 3 start) -> flip zone (down->up) uses (L=90..H=last)
    ("L", 104),
    ("H", 120),
    ("L", 112),
    ("H", 134),
    ("L", 120),
    ("H", 148),   # last high of trend 3 -> flip zone (up->down) uses (L=120..H=148)
    ("L", 118),   # breaks 120 -> confirms DOWN (trend 4 start)
    ("H", 130),
    ("L", 105),
    ("H", 122),
    ("L", 95),
    ("H", 112),
    ("L", 85),
    ("H", 115),   # breaks last high -> confirms UP (trend 5 start) -> flip zone (down->up) uses (L=85..H=last)
    ("L", 104),
    ("H", 126),
    ("L", 112),
    ("H", 138),
    ("L", 128),
    ("H", 145),
]

BARS_PER_LEG = 3

def build_bars(waypoints, bars_per_leg=BARS_PER_LEG):
    highs, lows, opens, closes = [], [], [], []
    prev_price = waypoints[0][1]
    # seed bar
    highs.append(prev_price + 1); lows.append(prev_price - 1)
    opens.append(prev_price); closes.append(prev_price)
    for i in range(1, len(waypoints)):
        typ, target = waypoints[i]
        start = waypoints[i - 1][1]
        for b in range(1, bars_per_leg + 1):
            frac = b / bars_per_leg
            level = start + (target - start) * frac
            if typ == "H":
                # rising leg: each bar's high pushes further; low stays a modest pullback inside
                highs.append(level)
                lows.append(level - abs(target - start) * 0.15 - 1)
            else:
                # falling leg: each bar's low pushes further; high stays a modest bounce inside
                lows.append(level)
                highs.append(level + abs(target - start) * 0.15 + 1)
            opens.append(level); closes.append(level)
    return highs, lows, opens, closes


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


def build_zones(pivots):
    last_high, last_low, trend = None, None, None
    zones = []
    seq = []  # chronological (pivot_idx, typ, price, trend_after)
    for confirm_idx, pivot_idx, typ, price in pivots:
        if typ == "H":
            if last_high is not None:
                if price > last_high[1]:
                    if trend == "down" and last_low is not None:
                        zones.append(dict(created_idx=confirm_idx, top=last_high[1], bottom=last_low[1],
                                           top_idx=last_high[0], bottom_idx=last_low[0],
                                           is_buy=True, kind="flip"))
                    trend = "up"
                else:
                    if last_low is not None:
                        zones.append(dict(created_idx=confirm_idx, top=price, bottom=last_low[1],
                                           top_idx=pivot_idx, bottom_idx=last_low[0],
                                           is_buy=False, kind="pullback"))
                    trend = "down"
            last_high = (pivot_idx, price)
        else:
            if last_low is not None:
                if price < last_low[1]:
                    if trend == "up" and last_high is not None:
                        zones.append(dict(created_idx=confirm_idx, top=last_high[1], bottom=last_low[1],
                                           top_idx=last_high[0], bottom_idx=last_low[0],
                                           is_buy=False, kind="flip"))
                    trend = "down"
                else:
                    if last_high is not None:
                        zones.append(dict(created_idx=confirm_idx, top=last_high[1], bottom=price,
                                           top_idx=last_high[0], bottom_idx=pivot_idx,
                                           is_buy=True, kind="pullback"))
                    trend = "up"
            last_low = (pivot_idx, price)
        seq.append((pivot_idx, typ, price, trend))
    return zones, seq


highs, lows, opens, closes = build_bars(WAYPOINTS)
pivots = zigzag(highs, lows)
zones, seq = build_zones(pivots)

print("n bars:", len(highs))
print("n pivots:", len(pivots))
for p in pivots:
    print("  confirm@", p[0], "pivot@", p[1], p[2], p[3])
print("n zones:", len(zones), "(pullback:", sum(1 for z in zones if z["kind"] == "pullback"),
      ", flip:", sum(1 for z in zones if z["kind"] == "flip"), ")")

# identify trend runs (segments between reversals) by walking seq and detecting trend changes
runs = []
cur_trend = None
cur_start = 0
for i, (pivot_idx, typ, price, trend) in enumerate(seq):
    if trend != cur_trend:
        if cur_trend is not None:
            runs.append((cur_trend, cur_start, i))
        cur_trend = trend
        cur_start = i
runs.append((cur_trend, cur_start, len(seq)))
print("\ntrend runs (direction, pivot-count):")
for direction, s, e in runs:
    print(" ", direction, e - s + (1 if s == 0 else 0), [seq[j][1] + str(round(seq[j][2])) for j in range(s, e)])

reversals = sum(1 for z in zones if z["kind"] == "flip")
print("\nreversals (flip zones):", reversals)

out = dict(highs=highs, lows=lows, opens=opens, closes=closes,
           pivots=[dict(confirm_idx=p[0], pivot_idx=p[1], typ=p[2], price=p[3]) for p in pivots],
           zones=zones)
with open("extended_trend_data.json", "w") as f:
    json.dump(out, f)
print("\nwrote extended_trend_data.json")

# ---- pixel-mapped SVG coordinates ----
N = len(highs)
BAR_W = 15
MARGIN_L, MARGIN_R = 20, 20
W = MARGIN_L + (N - 1) * BAR_W + MARGIN_R
MARGIN_T, MARGIN_B = 34, 34
H = 340
plotH = H - MARGIN_T - MARGIN_B
minP = min(lows) - 6
maxP = max(highs) + 6

def X(i):
    return round(MARGIN_L + i * BAR_W, 1)

def Y(p):
    return round(MARGIN_T + (maxP - p) / (maxP - minP) * plotH, 1)

print(f"\nSVG viewBox: 0 0 {W} {H}   (minP={minP} maxP={maxP})")

path_pts = " ".join(f"{X(p['pivot_idx'])},{Y(p['price'])}" for p in
                     [dict(pivot_idx=0, price=highs[0])] +
                     [dict(pivot_idx=pp[1], price=pp[3]) for pp in pivots])
print("\npolyline points:\n", path_pts)

print("\npivot dots:")
for p in pivots:
    print(f'  <circle cx="{X(p[1])}" cy="{Y(p[3])}" r="3.2"/>  <!-- {p[2]} {p[3]} -->')

print("\nzones (rects):")
for z in zones:
    x0, x1 = X(min(z["top_idx"], z["bottom_idx"])), X(max(z["top_idx"], z["bottom_idx"]))
    y0, y1 = Y(z["top"]), Y(z["bottom"])
    print(f'  kind={z["kind"]:9s} is_buy={str(z["is_buy"]):5s} x0={x0} x1={x1} y0={y0} y1={y1} top={z["top"]} bottom={z["bottom"]}')

print("\nreversal markers (flip zone creation points):")
for z in zones:
    if z["kind"] == "flip":
        x = X(max(z["top_idx"], z["bottom_idx"]))
        print(f'  flip at x={x}, top={z["top"]}, bottom={z["bottom"]}, new_dir={"up" if z["is_buy"] else "down"}')
