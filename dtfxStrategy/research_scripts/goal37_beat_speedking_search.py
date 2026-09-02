"""/goal: find a strategy that beats Speed-King on BOTH the composite score (goal12) and
the margin score (goal23) specifically.

Key structural insight from goal12_strategy_scores.csv: Speed-King currently scores a
PERFECT 100/100 on 4 of the 6 composite factors (speed, WR, frequency, drawdown -- 87% of
the total weight) and near-worst on margin (6.9/100, second-worst of 14) and R:R (0.0/100,
worst of 14). Composite/margin are MIN-MAX NORMALIZED across the candidate pool, which
means: since Speed-King is the current #1 (best) on those 4 factors, any new candidate can
only exceed Speed-King's OWN composite by being better than Speed-King on at least one of
them -- which simultaneously pulls Speed-King's own normalized score DOWN (no longer the
max) while pulling the new candidate's UP. A candidate that's merely "close" to Speed-King
everywhere can't win; it has to actually beat it somewhere real.

Candidate direction: narrower R:R than Speed-King's 0.20, holding N=7.0/age_h=0.175 fixed.
A nearer target should (a) push win rate and trade frequency even higher than Speed-King's
already-best-in-class numbers (more of the goal12 dominant-weight factors), and (b) shrink
average WINNING-trade duration (target is reached faster), which should REDUCE concurrency
and thus IMPROVE margin capacity too -- the one factor Speed-King is weak on. Losing-trade
duration (governed by N, held fixed) is unchanged, but losers become a smaller fraction of
the population as WR rises, so average duration should still fall.

Phase 1: cheap screen (no MC) over a narrow-R:R grid to find the best candidate(s).
Phase 2: full goal12/goal23-methodology evaluation (MC speed/drawdown at N_SIMS=50000,
margin-constrained accept rate) of the top candidate(s), reusing the other 13 strategies'
already-computed raw values from goal12_strategy_scores.csv, then renormalizing the full
15-candidate pool and comparing directly against Speed-King's own (now-renormalized) score.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import dtfx_robust_search as rs
import prod_filter_grid_search as pg
import goal20_speed_king_margin_report as g20
from goal10_cana_nt_mc_capped import run_mc_capped, BLOCK, MAX_TRADES

TRADEABLE = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
SPLIT = pd.Timestamp("2018-03-09")
SK = dict(n=7.0, rr=0.20, age_h=0.175, bounce=False, touches=4)  # Speed-King's own config


def build_full(n, rr, age_h, bounce=False, touches=4):
    """Full-history population, true entry/exit times (needed for margin/concurrency)."""
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (n * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(age_h * 3600 * 1e9), n, n * rr, bounce, pg.PROBE_DEPTH, 1.0, touches, pg.MAX_WINDOW_BARS)
        if len(entry_idx) == 0:
            continue
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, rr, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({
            "instrument": inst, "entry_time": pd.to_datetime(times[entry_idx]),
            "exit_time": pd.to_datetime(times[exit_idx]), "entry_price": entry_mid,
            "stop_risk": stop_risk, "net_r": net_r, "is_win": is_win,
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def quick_stats(t):
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = len(t) / years
    wr = 100 * t.is_win.mean()
    w = t.loc[t.net_r > 0, "net_r"].sum(); l = -t.loc[t.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    return dict(n=len(t), tpy=tpy, wr=wr, pf=pf)


def concurrency_quick(t):
    t = t.sort_values("entry_time")
    starts = t.entry_time.values.astype("datetime64[ns]")
    ends = t.exit_time.values.astype("datetime64[ns]")
    events = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(len(starts)), -np.ones(len(ends))])
    order = np.argsort(events, kind="stable")
    concurrent = np.cumsum(deltas[order])
    dur_min = (ends - starts).astype("timedelta64[s]").astype(float) / 60.0
    return dict(max_concurrent=int(concurrent.max()), median_dur_min=float(np.median(dur_min)),
                mean_dur_min=float(np.mean(dur_min)))


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Building Speed-King's own config as the baseline for comparison...", flush=True)
    sk_full = build_full(**SK)
    sk_holdout = sk_full[sk_full.entry_time >= SPLIT]
    sk_stats = quick_stats(sk_holdout)
    sk_conc = concurrency_quick(sk_full)
    print(f"  Speed-King (baseline): {sk_stats}  conc={sk_conc}")

    print(f"\n{'='*94}\nPHASE 1: cheap screen, narrower R:R than Speed-King's 0.20, N=7.0/age=0.175 fixed\n{'='*94}")
    rr_grid = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
    rows = []
    for rr in rr_grid:
        full = build_full(SK["n"], rr, SK["age_h"], SK["bounce"], SK["touches"])
        holdout = full[full.entry_time >= SPLIT]
        s = quick_stats(holdout)
        c = concurrency_quick(full)
        rows.append(dict(rr=rr, **s, **{f"conc_{k}": v for k, v in c.items()}))
        print(f"  rr={rr:<6} n={s['n']:6,} ({s['tpy']:7,.0f}/yr) WR={s['wr']:5.2f}% PF={s['pf']:.3f}  "
              f"max_conc={c['max_concurrent']:3d}  median_dur={c['median_dur_min']:5.1f}min  "
              f"mean_dur={c['mean_dur_min']:6.1f}min")
    screen_df = pd.DataFrame(rows)
    screen_df.to_csv(os.path.join(DATA_DIR, "goal37_rr_screen.csv"), index=False)


if __name__ == "__main__":
    main()
