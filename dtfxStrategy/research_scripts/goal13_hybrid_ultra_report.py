"""Full report for Hybrid-Ultra (N=6.0, R:R=0.4, 2.0h fill-age, unconditional re-arm,
touches<=4): spread sensitivity (0.8/1.6/2.0 pips), $2M-capped Monte Carlo to $1M and
$10M, period win-rate stats (day/week/month/year), and last month's trade log.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal5_rr_sweep as g5
from goal10_cana_nt_mc_capped import run_mc_capped, BLOCK, MAX_TRADES

N_MULT = 6.0
RR = 0.4
AGE_H = 2.0
REQUIRE_BOUNCE = False
TOUCHES = 4
START = 200.0
RISK_FRAC = 0.05
CAP = 2_000_000.0
N_SIMS = 25_000


def build_hybrid_ultra(spread_pips):
    frames = []
    for inst in g5.TRADEABLE:
        z = g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)
        times, lows, highs, df = g5.rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (N_MULT * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= g5.pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, is_win, entry_mid, stop_risk = g5.pg._simulate_generic(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(AGE_H * 3600 * 1e9), N_MULT, N_MULT * RR, REQUIRE_BOUNCE, g5.pg.PROBE_DEPTH,
            1.0, TOUCHES, g5.pg.MAX_WINDOW_BARS)
        pip = g5.pg.PIP_SIZE[inst]
        spread_price = spread_pips * pip
        r_gross = np.where(is_win, RR, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]),
                                     "net_r": net_r, "is_win": is_win}))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def score(t, label):
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = len(t) / years
    wr = 100 * t.is_win.mean()
    w = t.loc[t.net_r > 0, "net_r"].sum(); l = -t.loc[t.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    avg_r = t.net_r.mean()
    print(f"{label:14s} n={len(t):7,} ({tpy:6.0f}/yr) WR={wr:5.2f}% PF={pf:.3f} avg_r={avg_r:.4f}")
    return dict(label=label, trades=len(t), tpy=tpy, wr=wr, pf=pf, avg_r=avg_r)


def max_streak(bool_array):
    m = r = 0
    for v in bool_array:
        r = r + 1 if v else 0
        m = max(m, r)
    return m


def main():
    for inst in g5.TRADEABLE:
        g5.rs.load_instrument(inst)
        g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)

    # ---------- Part 1: spread sensitivity ----------
    print("=" * 70)
    print("PART 1: Spread sensitivity ($200 start, 5% risk/trade)")
    print("=" * 70)
    spread_results = []
    trade_logs = {}
    for spread in [0.8, 1.6, 2.0]:
        t = build_hybrid_ultra(spread)
        trade_logs[spread] = t
        spread_results.append(score(t, f"spread={spread}"))
        t.to_csv(os.path.join(DATA_DIR, f"goal13_hybrid_ultra_spread{str(spread).replace('.','p')}.csv"), index=False)

    baseline = trade_logs[1.6]

    # ---------- Part 2: $2M-capped Monte Carlo, 25,000 sims ----------
    print("\n" + "=" * 70)
    print(f"PART 2: Monte Carlo ({N_SIMS:,} sims, $2M position-sizing cap, 1.6-pip spread)")
    print("=" * 70)
    net_r = baseline.net_r.values.astype(np.float64)
    years = (baseline.entry_time.max() - baseline.entry_time.min()).days / 365.25
    tpy = len(baseline) / years
    t1_raw, t10_raw, max_dd, ruined = run_mc_capped(net_r, N_SIMS, BLOCK, MAX_TRADES, seed=99)
    t1_hit = t1_raw[t1_raw > 0]; t10_hit = t10_raw[t10_raw > 0]
    y1 = t1_hit / tpy; y10 = t10_hit / tpy
    print(f"Time to $1M: quickest={y1.min():.3f} slowest={y1.max():.3f} avg={y1.mean():.3f} median={np.median(y1):.3f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(y1,p):.3f} yrs")
    print(f"\nTime to $10M (capped at $2M sizing): quickest={y10.min():.3f} slowest={y10.max():.3f} "
          f"avg={y10.mean():.3f} median={np.median(y10):.3f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(y10,p):.3f} yrs")
    dd_pct = max_dd * 100
    print(f"\nMax drawdown: median={np.median(dd_pct):.1f}% mean={dd_pct.mean():.1f}% worst={dd_pct.min():.1f}% best={dd_pct.max():.1f}%")
    print(f"%% reaching $1M: {100*len(t1_hit)/N_SIMS:.2f}%  %% reaching $10M: {100*len(t10_hit)/N_SIMS:.2f}%  %% ruined: {100*ruined.mean():.3f}%")

    mc_out = pd.DataFrame({"years_to_1m": np.where(t1_raw>0, t1_raw/tpy, np.nan),
                            "years_to_10m": np.where(t10_raw>0, t10_raw/tpy, np.nan),
                            "max_drawdown_pct": dd_pct, "ruined": ruined})
    mc_out.to_csv(os.path.join(DATA_DIR, "goal13_hybrid_ultra_mc_25000.csv"), index=False)

    # ---------- Part 3: period win-rate stats ----------
    print("\n" + "=" * 70)
    print("PART 3: Period stats (1.6-pip spread, full history)")
    print("=" * 70)
    t = baseline.copy()
    for freq, label in [("D", "Day"), ("W", "Week"), ("M", "Month"), ("Y", "Year")]:
        g = t.copy()
        g["period"] = g["entry_time"].dt.to_period(freq)
        agg = g.groupby("period").agg(trades=("net_r", "size"), total_r=("net_r", "sum")).reset_index()
        agg["positive"] = agg.total_r > 0
        pct_pos = 100 * agg.positive.mean()
        avg_win = agg.loc[agg.positive, "total_r"].mean()
        win_streak = max_streak(agg.sort_values("period").positive.values)
        loss_streak = max_streak(~agg.sort_values("period").positive.values)
        print(f"{label:6s}: {len(agg):5,} periods, {pct_pos:5.1f}% positive, avg winning-{label.lower()} = {avg_win:+.2f}R, "
              f"win streak={win_streak}, loss streak={loss_streak}")

    trade_win_streak = max_streak(t.is_win.values)
    trade_loss_streak = max_streak(~t.is_win.values)
    print(f"\nPer-trade streaks: longest win={trade_win_streak}, longest loss={trade_loss_streak}")

    # zero trade days
    all_days = pd.date_range(t.entry_time.min().normalize(), t.entry_time.max().normalize(), freq="D")
    traded = set(t.entry_time.dt.date)
    zero_all = sum(1 for d in all_days if d.date() not in traded)
    zero_wd = sum(1 for d in all_days if d.date() not in traded and d.weekday() < 5)
    total_wd = sum(1 for d in all_days if d.weekday() < 5)
    print(f"Zero-trade days: {zero_all:,}/{len(all_days):,} total ({100*zero_all/len(all_days):.1f}%); "
          f"weekdays only: {zero_wd:,}/{total_wd:,} ({100*zero_wd/total_wd:.1f}%)")

    # ---------- Part 4: last month's trades ----------
    last_period = t.entry_time.max().to_period("M")
    last_month = t[t.entry_time.dt.to_period("M") == last_period]
    last_month[["entry_time", "net_r", "is_win"]].to_csv(
        os.path.join(DATA_DIR, "goal13_hybrid_ultra_last_month.csv"), index=False)
    prev_period = last_period - 1
    prev_month = t[t.entry_time.dt.to_period("M") == prev_period]
    prev_month[["entry_time", "net_r", "is_win"]].to_csv(
        os.path.join(DATA_DIR, "goal13_hybrid_ultra_prev_full_month.csv"), index=False)
    print(f"\nLast month in data ({last_period}, partial): {len(last_month)} trades")
    print(f"Last FULL month ({prev_period}): {len(prev_month)} trades, "
          f"WR={100*prev_month.is_win.mean():.2f}%, total_r={prev_month.net_r.sum():+.1f}R")


if __name__ == "__main__":
    main()
