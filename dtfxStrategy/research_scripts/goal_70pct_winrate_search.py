"""GOAL: find a config approaching 70% win rate with high profit factor, high trade
frequency, and out-of-sample robustness -- and report honestly where that's NOT jointly
achievable rather than cherry-picking one axis.

Context from same-day prior research (dtfxStrategy/data/*.csv, scripts not retained):
  - risk_reward_sweep_results.csv: WIDENING the target (1:1 -> 1:5) monotonically LOWERS
    win rate (60.3% -> 20.3%) AND profit factor (1.51 -> 1.26) AND wrecks max drawdown
    (30.8% -> 88.2%). Never tested NARROWING the target below 1:1 -- that's the missing
    data point for a 70% WR target, tested here.
  - filter_permutation_study_results.csv: tighter stop alone (N=2.0 vs prod's 3.34, same
    1:1 R:R) raised win rate 58.79% -> 61.10% at THE SAME frequency (1162 vs 1163/yr) and
    a comparable PF (1.235 vs 1.203) -- the best free lunch found so far, but still nowhere
    near 70%.
  - Elite Bounce Requirement (SOMEGA session + re-touch bounce gate) reached 60.43% WR /
    1.282 PF but cut frequency to 681/yr (~59% of baseline). CANA's actual deployed config
    (24h, bounce gate on re-touches only) is closest to that file's row 3 (24h, no bounce
    on touch 1): 57.69% WR, 1.155 PF, 2577/yr -- i.e. CANA traded WR/PF away for frequency,
    the opposite direction from this goal.

This script fills the actual gap: sweeps target R:R BELOW 1:1 (narrower target = higher
win rate almost by construction, since the exit level is easier to reach) crossed with the
two known-good stop sizes (N=2.0, N=3.343496), on top of SOMEGA session + 3.366682h age
(the two already-validated, still-live filters), using the CORRECTED asymmetric cost model
(spread only paid on stop-outs -- target and soft-stop exits are passive limit fills, per
the hybrid-stop design actually running in PRODLIVETRADING; dtfx_robust_search.py's own
IBKR-era cost model charges spread on every trade uniformly, which is the OLDER, superseded
assumption -- not repeated here).

Then takes the best 1-2 candidates and actually checks robustness: chronological
train (2009-2020) / test (2020-2026) split, reporting the OOS numbers even if they're worse.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import kurtosis as scipy_kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
from dtfx_somega_params import SESSION_HOUR_FILTER

INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
MIN_RETRACEMENT = 2
MIN_FILL_AGE_H = 3.366682          # keep the validated age filter fixed across this sweep
SPREAD_PIPS_TYPICAL = 1.6           # OANDA-standard retail spread, in pips, applied ONLY to stop-outs
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}


@njit(cache=False)
def _simulate_asym(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                    lows, highs, stop_mult, target_mult, fast_bars, max_bars):
    n = len(zone_top)
    n_times = len(lows)
    entry_idx_out = np.full(n, -1, dtype=np.int64)
    status_out = np.zeros(n, dtype=np.int8)  # 0=unresolved 1=stop 2=target

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        half_width = (top - bottom) / 2.0
        invalidation = bottom if is_buy else top
        start = zone_start_idx[i]
        if start >= n_times:
            continue

        fast_end = min(start + fast_bars, n_times)
        breach_idx, mid_idx = -1, -1
        for j in range(start, fast_end):
            if is_buy:
                if breach_idx == -1 and lows[j] <= invalidation:
                    breach_idx = j
                if mid_idx == -1 and lows[j] <= mid:
                    mid_idx = j
            else:
                if breach_idx == -1 and highs[j] >= invalidation:
                    breach_idx = j
                if mid_idx == -1 and highs[j] >= mid:
                    mid_idx = j
            if breach_idx != -1 and mid_idx != -1:
                break
        if mid_idx == -1 and breach_idx == -1 and fast_end < n_times:
            for j in range(fast_end, n_times):
                if is_buy:
                    if breach_idx == -1 and lows[j] <= invalidation:
                        breach_idx = j
                    if mid_idx == -1 and lows[j] <= mid:
                        mid_idx = j
                else:
                    if breach_idx == -1 and highs[j] >= invalidation:
                        breach_idx = j
                    if mid_idx == -1 and highs[j] >= mid:
                        mid_idx = j
                if breach_idx != -1 and mid_idx != -1:
                    break

        if mid_idx == -1:
            continue
        if breach_idx != -1 and breach_idx <= mid_idx:
            continue

        entry_idx = mid_idx
        after_start = entry_idx + 1
        after_end = min(after_start + max_bars, n_times)
        if after_start >= n_times:
            continue

        stop_dist = stop_mult * half_width
        target_dist = target_mult * half_width
        stop_level = mid - stop_dist if is_buy else mid + stop_dist
        target_level = mid + target_dist if is_buy else mid - target_dist
        sl_idx, tp_idx = -1, -1
        for j in range(after_start, after_end):
            if is_buy:
                if sl_idx == -1 and lows[j] <= stop_level:
                    sl_idx = j
                if tp_idx == -1 and highs[j] >= target_level:
                    tp_idx = j
            else:
                if sl_idx == -1 and highs[j] >= stop_level:
                    sl_idx = j
                if tp_idx == -1 and lows[j] <= target_level:
                    tp_idx = j
            if sl_idx != -1 and tp_idx != -1:
                break

        if sl_idx == -1 and tp_idx == -1:
            continue
        entry_idx_out[i] = entry_idx
        if sl_idx != -1 and (tp_idx == -1 or sl_idx <= tp_idx):
            status_out[i] = 1
        else:
            status_out[i] = 2

    return entry_idx_out, status_out


def build_trades_asym(instrument, stop_mult, target_mult, min_fill_age_hours, somega_only):
    z = rs._prep_zones(instrument, MIN_RETRACEMENT)
    times, lows, highs, df = rs.load_instrument(instrument)
    entry_idx, status = _simulate_asym(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(stop_mult), float(target_mult),
        rs.FILL_SEARCH_FAST_BARS, rs.MAX_BARS_TO_RESOLVE)

    filled = entry_idx >= 0
    if not filled.any():
        return pd.DataFrame(columns=["entry_time", "net_r", "is_win"])

    half_width = (z["top"] - z["bottom"]) / 2.0
    stop_dist = stop_mult * half_width
    risk_pct_of_price = stop_dist / z["mid"]          # risk floor still defined off the STOP distance
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - z["created_at"]) / np.timedelta64(1, "h")

    keep = filled & (risk_pct_of_price >= rs.MIN_RISK_PCT if False else risk_pct_of_price >= 0.000170) \
           & (fill_hours >= min_fill_age_hours)

    if somega_only:
        allowed_hours = SESSION_HOUR_FILTER.get(instrument)
        if allowed_hours is not None:
            entry_hour = pd.DatetimeIndex(entry_times_ns).hour
            in_hours = np.isin(entry_hour, list(allowed_hours))
            keep = keep & in_hours

    if not keep.any():
        return pd.DataFrame(columns=["entry_time", "net_r", "is_win"])

    idx = np.where(keep)[0]
    is_win = status[idx] == 2
    # r in units of the STOP distance (1R = stop_dist): win = +target_mult/stop_mult, loss = -1.0
    r_gross = np.where(is_win, target_mult / stop_mult, -1.0)

    # Corrected asymmetric cost model: spread only paid crossing it on a STOP-OUT (market exit);
    # target and (live) soft-stop exits are passive limit fills, cost-free -- matches the hybrid
    # stop actually deployed in PRODLIVETRADING, not the older uniform-cost IBKR assumption.
    pip = PIP_SIZE[instrument]
    spread_price = SPREAD_PIPS_TYPICAL * pip
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_dist[idx])
    net_r = r_gross - cost_in_r

    out = pd.DataFrame({
        "entry_time": pd.to_datetime(entry_times_ns[idx]),
        "net_r": net_r,
        "is_win": is_win,
    }).sort_values("entry_time").reset_index(drop=True)
    return out


def stats_for(trades, years):
    if len(trades) < 30:
        return None
    wins = trades[trades["is_win"]]
    losses = trades[~trades["is_win"]]
    wr = len(wins) / len(trades) * 100
    gross_win = wins["net_r"].sum()
    gross_loss = -losses["net_r"].sum()
    pf = gross_win / gross_loss if gross_loss > 1e-9 else float("inf")
    avg_r = trades["net_r"].mean()
    trades_per_yr = len(trades) / years

    equity = 100.0
    peak = 100.0
    maxdd = 0.0
    curve = []
    for r in trades["net_r"].values:
        equity *= (1.0 + 0.01 * r)   # 1% risk, fixed-fraction, for an apples-to-apples Sharpe/DD read
        peak = max(peak, equity)
        maxdd = max(maxdd, (peak - equity) / peak)
        curve.append(equity)
    daily = pd.Series(curve, index=trades["entry_time"]).resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    ret = ret[ret != 0]
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if len(ret) > 10 and ret.std() > 0 else float("nan")
    exk = scipy_kurtosis(ret) if len(ret) > 10 else float("nan")

    return dict(trades=len(trades), trades_per_yr=round(trades_per_yr, 1), win_rate=round(wr, 2),
                profit_factor=round(pf, 3), avg_r=round(avg_r, 4), sharpe=round(sharpe, 2),
                maxdd_pct=round(maxdd * 100, 1), excess_kurtosis=round(exk, 2))


def main():
    print("Loading instruments...", flush=True)
    for inst in INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, MIN_RETRACEMENT)

    stop_mults = [2.0, 3.343496]
    rr_ratios = [1.0, 0.75, 0.5, 0.35, 0.25]   # target_mult = stop_mult * rr_ratio

    rows = []
    for somega in (False, True):
        for stop_mult in stop_mults:
            for rr in rr_ratios:
                target_mult = stop_mult * rr
                all_trades = []
                for inst in INSTRUMENTS:
                    t = build_trades_asym(inst, stop_mult, target_mult, MIN_FILL_AGE_H, somega)
                    all_trades.append(t)
                combined = pd.concat(all_trades, ignore_index=True).sort_values("entry_time")
                if len(combined) < 30:
                    continue
                years = (combined["entry_time"].max() - combined["entry_time"].min()).days / 365.25
                s = stats_for(combined, years)
                if s is None:
                    continue
                s.update(session_filter="SOMEGA" if somega else "NONE (24h)",
                         stop_mult=round(stop_mult, 3), target_mult=round(target_mult, 3),
                         rr_ratio=rr)
                rows.append(s)
                print(f"  session={s['session_filter']:10s} stop={stop_mult:.3f} rr={rr:.2f} "
                      f"-> WR={s['win_rate']:.2f}% PF={s['profit_factor']:.3f} "
                      f"trades/yr={s['trades_per_yr']:.0f} Sharpe={s['sharpe']:.2f} "
                      f"maxDD={s['maxdd_pct']:.1f}%", flush=True)

    result = pd.DataFrame(rows)
    cols = ["session_filter", "stop_mult", "target_mult", "rr_ratio", "trades", "trades_per_yr",
            "win_rate", "profit_factor", "avg_r", "sharpe", "maxdd_pct", "excess_kurtosis"]
    result = result[cols].sort_values("win_rate", ascending=False)
    out_path = os.path.join(DATA_DIR, "goal_70pct_winrate_sweep.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\n" + "=" * 100)
    print("FULL PORTFOLIO (EURUSD+GBPUSD+EURGBP+USDJPY), sorted by win rate")
    print("=" * 100)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()


def robustness_check(candidates):
    """Chronological train(<2020)/test(>=2020) split for the shortlisted candidates --
    the check CANA skipped. Reports whether the win rate / PF actually hold up OOS,
    not just in the full-history blend."""
    print("\n" + "=" * 100)
    print("ROBUSTNESS CHECK: chronological TRAIN (2009-2020) vs TEST (2020-2026), held out")
    print("=" * 100)
    rows = []
    for name, stop_mult, target_mult, somega in candidates:
        all_trades = []
        for inst in INSTRUMENTS:
            t = build_trades_asym(inst, stop_mult, target_mult, MIN_FILL_AGE_H, somega)
            all_trades.append(t)
        combined = pd.concat(all_trades, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        split = pd.Timestamp("2020-01-01")
        train = combined[combined["entry_time"] < split]
        test = combined[combined["entry_time"] >= split]
        train_years = (train["entry_time"].max() - train["entry_time"].min()).days / 365.25
        test_years = (test["entry_time"].max() - test["entry_time"].min()).days / 365.25
        s_train = stats_for(train, train_years)
        s_test = stats_for(test, test_years)
        for label, s in (("TRAIN 2009-2020", s_train), ("TEST 2020-2026 (OOS)", s_test)):
            if s is None:
                continue
            s["candidate"] = name
            s["split"] = label
            rows.append(s)
            print(f"  {name:32s} {label:22s} WR={s['win_rate']:.2f}% PF={s['profit_factor']:.3f} "
                  f"avg_r={s['avg_r']:.4f} trades/yr={s['trades_per_yr']:.0f} "
                  f"Sharpe={s['sharpe']:.2f} maxDD={s['maxdd_pct']:.1f}%")
    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "goal_70pct_winrate_robustness_check.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "robustness":
    for inst in INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, MIN_RETRACEMENT)
    robustness_check([
        ("SOMEGA stop3.34 rr1.00 (current baseline)", 3.343496, 3.343496, True),
        ("SOMEGA stop3.34 rr0.50", 3.343496, 3.343496 * 0.50, True),
        ("SOMEGA stop3.34 rr0.35", 3.343496, 3.343496 * 0.35, True),
        ("24h    stop3.34 rr0.35", 3.343496, 3.343496 * 0.35, False),
        ("24h    stop3.34 rr0.25", 3.343496, 3.343496 * 0.25, False),
    ])
