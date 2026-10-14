import json, os

SPREAD_PRICE = 0.00016  # GBPUSD, 1.6 pips, matches goal77 / extract_all_zones_inverted.py
PIP = 0.0001

d = json.load(open(os.path.join(os.path.dirname(__file__), "all_zones_inverted_data.json")))
bars = d["bars"]
zones = d["zones"]


def find(created_idx):
    for z in zones:
        if z["created_idx"] == created_idx:
            return z
    raise KeyError(created_idx)


PICKS = [57, 76, 140, 107]
LABELS = {
    57: "Pullback zone — win",
    76: "Pullback zone — loss",
    140: "Flip zone — win",
    107: "Flip zone — loss",
}

out = []
for c in PICKS:
    z = find(c)
    trade_is_buy = z["is_buy"]          # exported is_buy IS the traded (inverted) direction
    orig_is_buy = not trade_is_buy       # true zone context, used for touch/entry
    top, bottom, mid = z["top"], z["bottom"], z["mid"]

    touch_idx = z["touch_idx"]
    exit_idx = z.get("exit_idx")

    tb = bars[touch_idx]
    if orig_is_buy:
        deep = tb["l"] <= mid - SPREAD_PRICE
        entry = mid if deep else mid + SPREAD_PRICE
    else:
        deep = tb["h"] >= mid + SPREAD_PRICE
        entry = mid if deep else mid - SPREAD_PRICE

    stop = bottom if trade_is_buy else top
    target = top if trade_is_buy else bottom
    risk_pips = round(abs(entry - stop) / PIP, 1)
    reward_pips = round(abs(target - entry) / PIP, 1)

    lo = max(0, z["created_idx"] - 6)
    hi = min(len(bars), (exit_idx if exit_idx is not None else z["created_idx"]) + 3)
    window = bars[lo:hi]

    out.append(dict(
        label=LABELS[c],
        kind=z["kind"],
        is_buy=trade_is_buy,
        outcome=z["outcome"],
        top=top, bottom=bottom, mid=mid,
        entry=round(entry, 5), stop=round(stop, 5), target=round(target, 5),
        risk_pips=risk_pips, reward_pips=reward_pips,
        created_t=bars[z["created_idx"]]["t"],
        touch_t=bars[touch_idx]["t"],
        exit_t=bars[exit_idx]["t"] if exit_idx is not None else None,
        created_idx=z["created_idx"] - lo,
        touch_idx=touch_idx - lo,
        exit_idx=(exit_idx - lo) if exit_idx is not None else None,
        bars=window,
    ))

with open(os.path.join(os.path.dirname(__file__), "playbook_examples.json"), "w") as f:
    json.dump(out, f, separators=(",", ":"))

for e in out:
    print(e["label"], "| bars:", len(e["bars"]), "| entry", e["entry"], "stop", e["stop"], "target", e["target"],
          "| risk", e["risk_pips"], "pips | outcome", e["outcome"])
