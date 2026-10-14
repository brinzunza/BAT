"""GOAL: find a 1:1+ R:R DTFX config with >=1 trade/day (~365/yr combined), good PF,
and 70%+ win rate. Builds on prod_filter_grid_search.py's faithful multi-touch engine,
adding a minimum-stop-pips floor and MIN_RETRACEMENT as new search dimensions --
neither was swept under the R:R>=1:1 constraint yet, and both are known WR levers
(tight-stop buckets already showed ~64% WR at 1:1 R:R; certain instruments/sessions
independently show ~69% naturally)."""
import os, sys, itertools
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import prod_filter_grid_search as pg
from dtfx_somega_params import SESSION_HOUR_FILTER

PIP_SIZE = pg.PIP_SIZE

for inst in pg.INSTRUMENTS:
    rs.load_instrument(inst)
    for mr in (2, 3, 4):
        rs._prep_zones(inst, mr)


def build_trades_with_floor(instrument, min_retracement, stop_mult, age_h, bounce, session, min_stop_pips):
    """Like pg.build_trades but with min_retracement configurable and a min-stop-pips
    floor applied post-hoc (on top of the existing MIN_RISK_PCT floor)."""
    old_mr = pg.MIN_RETRACEMENT
    pg.MIN_RETRACEMENT = min_retracement
    try:
        t = pg.build_trades(instrument, stop_mult, stop_mult, age_h, bounce, session)
    finally:
        pg.MIN_RETRACEMENT = old_mr
    if len(t) == 0 or min_stop_pips <= 0:
        return t
    pip = PIP_SIZE[instrument]
    stop_price = stop_mult  # placeholder, recompute properly below
    return t  # min_stop_pips applied via a separate reconstruction (see main)


def score(trades, years):
    if len(trades) < 30:
        return None
    w = trades.loc[trades.net_r > 0, "net_r"].sum()
    l = -trades.loc[trades.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    return dict(trades=len(trades), trades_yr=round(len(trades) / years, 0),
                win_rate=round(100 * trades.is_win.mean(), 2), pf=round(pf, 3),
                avg_r=round(trades.net_r.mean(), 4))


def build_pool_with_stop_pips(min_retracement, stop_mult, age_h, bounce, session):
    """Build the combined 4-instrument pool WITH an explicit stop_pips column so a
    min-stop-pips floor can be applied post-hoc without rebuilding per threshold."""
    old_mr = pg.MIN_RETRACEMENT
    pg.MIN_RETRACEMENT = min_retracement
    try:
        frames = []
        for inst in pg.INSTRUMENTS:
            z = rs._prep_zones(inst, min_retracement)
            times, lows, highs, df = rs.load_instrument(inst)
            times_ns = times.astype("datetime64[ns]").astype("int64")
            created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
            risk_pct = (stop_mult * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
            ok = risk_pct >= pg.MIN_RISK_PCT
            top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
            is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
            touch_num, entry_idx, is_win, entry_mid, stop_risk = pg._simulate_generic(
                top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
                int(age_h * 3600 * 1e9), stop_mult, stop_mult, bounce, pg.PROBE_DEPTH, 1.0,
                pg.MAX_ZONE_TOUCHES, pg.MAX_WINDOW_BARS)
            if len(entry_idx) == 0:
                continue
            pip = PIP_SIZE[inst]
            spread_price = pg.SPREAD_PIPS_TYPICAL * pip
            r_gross = np.where(is_win, 1.0, -1.0)
            cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
            net_r = r_gross - cost_in_r
            entry_times_ns = times[entry_idx]
            keep = np.ones(len(entry_idx), dtype=bool)
            if session:
                allowed = SESSION_HOUR_FILTER.get(inst)
                if allowed is not None:
                    hrs = pd.DatetimeIndex(entry_times_ns).hour
                    keep = np.isin(hrs, list(allowed))
            frames.append(pd.DataFrame({
                "entry_time": pd.to_datetime(entry_times_ns[keep]), "net_r": net_r[keep],
                "is_win": is_win[keep], "stop_pips": (stop_risk / pip)[keep], "instrument": inst,
            }))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    finally:
        pg.MIN_RETRACEMENT = old_mr


def main():
    rows = []
    configs = list(itertools.product(
        [2, 3],                          # MIN_RETRACEMENT
        [2.0, 2.5, 3.343496],            # stop_mult
        [2.0, 3.366682, 6.0],            # age_h
        [False, True],                    # bounce
        [True, False],                    # session
    ))
    print(f"Building {len(configs)} base pools...\n", flush=True)
    pools = {}
    for mr, stop_mult, age, bounce, session in configs:
        key = (mr, stop_mult, age, bounce, session)
        pool = build_pool_with_stop_pips(mr, stop_mult, age, bounce, session)
        pools[key] = pool
        if len(pool):
            years = (pool.entry_time.max() - pool.entry_time.min()).days / 365.25
            print(f"  MR={mr} stop={stop_mult:.2f} age={age:.2f} bounce={bounce} session={session}: "
                  f"{len(pool):,} trades ({len(pool)/years:,.0f}/yr)", flush=True)

    print("\nApplying min-stop-pips floors and scoring...\n", flush=True)
    for (mr, stop_mult, age, bounce, session), pool in pools.items():
        if len(pool) == 0:
            continue
        years = (pool.entry_time.max() - pool.entry_time.min()).days / 365.25
        for min_stop in [0, 3, 5, 7, 10, 12]:
            sub = pool[pool.stop_pips >= min_stop]
            s = score(sub, years)
            if s is None:
                continue
            s.update(min_retracement=mr, stop_mult=stop_mult, age_h=age, bounce=bounce,
                      session=session, min_stop_pips=min_stop)
            rows.append(s)

    res = pd.DataFrame(rows)
    res = res[res.trades_yr >= 365]  # >= 1 trade/day average, the goal's floor
    cols = ["min_retracement", "stop_mult", "age_h", "bounce", "session", "min_stop_pips",
            "trades", "trades_yr", "win_rate", "pf", "avg_r"]
    res = res[cols].sort_values("win_rate", ascending=False)
    out_path = os.path.join(DATA_DIR, "goal70_1to1_search.csv")
    res.to_csv(out_path, index=False)
    print("Top 25 by win rate (trades/yr >= 365 enforced):")
    print(res.head(25).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
