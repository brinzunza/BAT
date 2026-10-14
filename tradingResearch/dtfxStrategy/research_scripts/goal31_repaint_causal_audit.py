"""Look-ahead / repainting audit (robustness tests #29, #30): does the backtest's zone
bar-walk start scanning for a fill from a bar that could not actually have been known to
exist yet in live operation?

`_prep_zones`'s `start_idx = searchsorted(times, created_at)` (dtfx_robust_search.py, and
identically dtfx_engine.detect_zones in C++) sets the bar-walk's starting point to the
zone's OWN `created_at` timestamp. But `created_at` is the timestamp of the confirming
PIVOT's own historical extreme bar (find_pivots_strict appends `pd.Timestamp(t[candidate_i])`
-- the candidate/extreme bar's own time -- not the bar where the streak of
`min_retracement` opposite-colored candles actually finished, which is when the pivot
becomes knowable). Structurally, a zone additionally needs the OPPOSITE-side pivot to also
be confirmed before run_structure_fsm emits it at all. Both delays are real and, in a live
system processing bars strictly forward, unavoidable -- PRODLIVETRADING's own engine.py
never rescans price between `created_at` and "now" for a fill that already silently
happened (only the one-time seed_history/warmup replay does that, by design, for state
initialization). If the historical backtest's bar-walk starts at `created_at`'s own bar
instead of the bar where the zone actually became knowable, it can register fills using
price action from a window that live trading could never have acted on -- a genuine
look-ahead discrepancy, distinct from (though related in spirit to) the entry_idx/exit_idx
mislabeling bug found and fixed earlier in this project.

This re-implements find_pivots_strict + run_structure_fsm (exact port, PRODLIVETRADING/
strategy/pivots.py + structure_fsm.py) with ONE addition: each pivot also carries the bar
index at which it was ACTUALLY confirmed (the loop position where the min_retracement
streak completed), separate from the extreme bar's own (repainted) index. Because
find_pivots_strict emits pivots in increasing confirmation-index order (a single forward
pass), and run_structure_fsm consumes them in that same order, the CURRENT event being
processed when a zone is created always carries the correct (maximum) confirmation index
for that zone -- no O(n^2) bar-by-bar re-simulation needed, just this one O(n) pass.

Compares Speed-King's full trade population under:
  (a) the naive start_idx everywhere else in this project uses (searchsorted on created_at)
  (b) the causally-correct start_idx (confirm_idx) -- what a live system could actually see
`created_at` itself (used for the MIN_FILL_AGE gate) is left unchanged in both cases --
isolating exactly one variable: where the bar-walk is allowed to start looking for a fill.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import dtfx_robust_search as rs
import prod_filter_grid_search as pg
import goal20_speed_king_margin_report as g20

TRADEABLE = g20.TRADEABLE
SK_N, SK_RR, SK_AGE_H, SK_BOUNCE, SK_TOUCHES = g20.SK_N, g20.SK_RR, g20.SK_AGE_H, g20.SK_BOUNCE, g20.SK_TOUCHES


def find_pivots_with_confirm(df, min_retracement):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    pivots = []  # (extreme_idx, price, typ, confirm_idx)
    looking_for = "high"
    candidate_price, candidate_i = h[0], 0
    streak = 0
    n = len(df)
    for i in range(1, n):
        is_red = c[i] < o[i]
        is_green = c[i] > o[i]
        if looking_for == "high":
            if c[i] > candidate_price:
                candidate_price, candidate_i, streak = h[i], i, 0
            elif is_red:
                streak += 1
                if streak >= min_retracement:
                    pivots.append((candidate_i, candidate_price, "H", i))
                    looking_for, candidate_price, candidate_i, streak = "low", l[i], i, 0
            else:
                streak = 0
        else:
            if c[i] < candidate_price:
                candidate_price, candidate_i, streak = l[i], i, 0
            elif is_green:
                streak += 1
                if streak >= min_retracement:
                    pivots.append((candidate_i, candidate_price, "L", i))
                    looking_for, candidate_price, candidate_i, streak = "high", h[i], i, 0
            else:
                streak = 0
    return pivots


def run_structure_fsm_with_confirm(zigzag):
    events, zones = [], []
    trend = confirmed_high = confirmed_low = None
    pending_low_idx = pending_high_idx = None
    i = 0
    seen_low = seen_high = None
    while i < len(zigzag) and (seen_low is None or seen_high is None):
        ei, v, typ, ci = zigzag[i]
        if typ == "L" and seen_low is None:
            seen_low = (ei, v, ci)
        if typ == "H" and seen_high is None:
            seen_high = (ei, v, ci)
        i += 1
    if not (seen_low and seen_high):
        return events, zones
    confirmed_low, confirmed_high = seen_low[1], seen_high[1]
    trend = "up"
    events.append([seen_low[0], seen_low[1], "L", "bootstrap", seen_low[2]])
    events.append([seen_high[0], seen_high[1], "H", "bootstrap", seen_high[2]])

    for ei, v, typ, ci in zigzag[i:]:
        if typ == "L":
            if trend == "up":
                if v < confirmed_low:
                    confirmed_high, confirmed_low = confirmed_high, v
                    trend = "down"; pending_high_idx = None
                    events.append([ei, v, "L", "flip_down", ci])
                else:
                    pending_low_idx = len(events)
                    events.append([ei, v, "L", "pending", ci])
            else:
                if confirmed_low is None or v < confirmed_low:
                    old_low = confirmed_low
                    zone_top = events[pending_high_idx][1] if pending_high_idx is not None else confirmed_high
                    confirmed_low = v
                    if pending_high_idx is not None:
                        confirmed_high = events[pending_high_idx][1]
                        events[pending_high_idx][3] = "confirmed"
                        pending_high_idx = None
                    events.append([ei, v, "L", "confirmed", ci])
                    if old_low is not None and zone_top is not None and zone_top > old_low:
                        zones.append(dict(top=zone_top, bottom=old_low, mid=(zone_top + old_low) / 2,
                                           extreme_idx=ei, confirm_idx=ci, side="sell"))
                else:
                    events.append([ei, v, "L", "pending", ci])
        else:
            if trend == "down":
                if v > confirmed_high:
                    confirmed_low, confirmed_high = confirmed_low, v
                    trend = "up"; pending_low_idx = None
                    events.append([ei, v, "H", "flip_up", ci])
                else:
                    pending_high_idx = len(events)
                    events.append([ei, v, "H", "pending", ci])
            else:
                if confirmed_high is None or v > confirmed_high:
                    old_high = confirmed_high
                    zone_bottom = events[pending_low_idx][1] if pending_low_idx is not None else confirmed_low
                    confirmed_high = v
                    if pending_low_idx is not None:
                        confirmed_low = events[pending_low_idx][1]
                        events[pending_low_idx][3] = "confirmed"
                        pending_low_idx = None
                    events.append([ei, v, "H", "confirmed", ci])
                    if old_high is not None and zone_bottom is not None and old_high > zone_bottom:
                        zones.append(dict(top=old_high, bottom=zone_bottom, mid=(old_high + zone_bottom) / 2,
                                           extreme_idx=ei, confirm_idx=ci, side="buy"))
                else:
                    events.append([ei, v, "H", "pending", ci])
    return events, zones


def zones_with_confirm_idx(instrument):
    times, lows, highs, df = rs.load_instrument(instrument)
    zigzag = find_pivots_with_confirm(df, pg.MIN_RETRACEMENT)
    _, zones = run_structure_fsm_with_confirm(zigzag)
    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    extreme_idx = np.array([z["extreme_idx"] for z in zones], dtype=np.int64)
    confirm_idx = np.array([z["confirm_idx"] for z in zones], dtype=np.int64)
    created_at = times[extreme_idx]  # == the naive start_idx's own timestamp, by construction
    return dict(top=top, bottom=bottom, mid=mid, is_buy=is_buy,
                extreme_idx=extreme_idx, confirm_idx=confirm_idx, created_at=created_at)


def run_backtest(inst, z, start_idx_field):
    times, lows, highs, df = rs.load_instrument(inst)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
    risk_pct = (SK_N * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    ok = risk_pct >= pg.MIN_RISK_PCT
    top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
    is_buy, start_idx, c_ns = z["is_buy"][ok], z[start_idx_field][ok], created_ns[ok]
    touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
        top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(SK_AGE_H * 3600 * 1e9), SK_N, SK_N * SK_RR, SK_BOUNCE,
        pg.PROBE_DEPTH, 1.0, SK_TOUCHES, pg.MAX_WINDOW_BARS)
    pip = pg.PIP_SIZE[inst]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip
    r_gross = np.where(is_win, SK_RR, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = r_gross - cost_in_r
    return pd.DataFrame({
        "instrument": inst, "entry_time": pd.to_datetime(times[entry_idx]),
        "exit_time": pd.to_datetime(times[exit_idx]), "net_r": net_r, "is_win": is_win,
    })


def summarize(t, label):
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = len(t) / years
    wr = 100 * t.is_win.mean()
    w = t.loc[t.net_r > 0, "net_r"].sum(); l = -t.loc[t.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"  {label:32s} n={len(t):7,}  ({tpy:6,.0f}/yr)  WR={wr:5.2f}%  PF={pf:.3f}")
    return dict(label=label, n=len(t), tpy=tpy, wr=wr, pf=pf)


def main():
    all_zones = {}
    lag_summaries = []
    for inst in TRADEABLE:
        print(f"Computing causal confirm-index for {inst} (pure-Python pivots+FSM re-walk)...", flush=True)
        z = zones_with_confirm_idx(inst)
        all_zones[inst] = z
        times, lows, highs, df = rs.load_instrument(inst)
        naive_start_idx = np.searchsorted(times, z["created_at"], side="right")
        lag_bars = z["confirm_idx"] - naive_start_idx  # minutes, since bars are 1-min
        lag_summaries.append(dict(
            instrument=inst, n_zones=len(lag_bars),
            median_lag_min=float(np.median(lag_bars)), mean_lag_min=float(np.mean(lag_bars)),
            p90_lag_min=float(np.percentile(lag_bars, 90)), max_lag_min=float(lag_bars.max()),
            pct_lag_gt_sk_age=float(100 * (lag_bars > SK_AGE_H * 60).mean()),
            pct_lag_zero=float(100 * (lag_bars == 0).mean()),
        ))
        print(f"  {inst}: {len(lag_bars):,} zones, confirmation lag (confirm_idx - naive start_idx): "
              f"median={np.median(lag_bars):.0f}min, mean={np.mean(lag_bars):.1f}min, "
              f"P90={np.percentile(lag_bars,90):.0f}min, max={lag_bars.max():,}min; "
              f"{100*(lag_bars > SK_AGE_H*60).mean():.1f}%% of zones have lag exceeding Speed-King's "
              f"own {SK_AGE_H*60:.1f}-min fill-age gate")

    lag_df = pd.DataFrame(lag_summaries)
    lag_df.to_csv(os.path.join(DATA_DIR, "goal31_repaint_lag_summary.csv"), index=False)
    print("\n" + lag_df.to_string(index=False))

    print(f"\n{'='*94}\nSpeed-King full population: naive start_idx vs. causally-correct confirm_idx\n{'='*94}")
    naive_frames, causal_frames = [], []
    for inst in TRADEABLE:
        z = all_zones[inst]
        naive_frames.append(run_backtest(inst, z, "extreme_idx"))
        causal_frames.append(run_backtest(inst, z, "confirm_idx"))
    naive = pd.concat(naive_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    causal = pd.concat(causal_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    naive.to_csv(os.path.join(DATA_DIR, "goal31_speedking_naive.csv"), index=False)
    causal.to_csv(os.path.join(DATA_DIR, "goal31_speedking_causal.csv"), index=False)

    r_naive = summarize(naive, "Naive (start_idx = created_at)")
    r_causal = summarize(causal, "Causal (start_idx = confirm_idx)")

    pct_trade_delta = 100 * (r_causal["n"] - r_naive["n"]) / r_naive["n"]
    print(f"\nTrade count change: {pct_trade_delta:+.1f}%   "
          f"WR change: {r_causal['wr']-r_naive['wr']:+.2f}pp   PF change: {r_causal['pf']-r_naive['pf']:+.3f}")

    # how many of the naive population's trades used a start_idx that a live system
    # could not have known about yet (i.e., a fill was found using the pre-confirmation
    # window) -- direct headline number
    print(f"\nOf {len(naive):,} naive trades, comparing directly against the causal re-walk on the same")
    print(f"zone set shows the population-level impact above. See goal31_speedking_{{naive,causal}}.csv")
    print(f"for the full trade-level detail.")


if __name__ == "__main__":
    main()
