"""Speed-King robustness diagnostics (tests #4-7, #13-19, #35-37): stability, confidence
intervals, significance, outlier sensitivity, regime/cross-asset/cross-session breakdowns,
rolling windows, losing streaks, cross-instrument correlation, and loss clustering.

Uses ONLY the TRUE-holdout population (entry_time >= 2018-03-09, 141,154 trades) for every
test in this file, NOT the full 2009-2026 history -- Speed-King's (N, R:R, age_h) were
chosen by a grid search that used the first half of history, so using that same data again
to claim "stability" would be circular. Everything here is genuinely out-of-sample.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

SPLIT = pd.Timestamp("2018-03-09")


def load_holdout():
    full = pd.read_csv(os.path.join(DATA_DIR, "goal20_speed_king_full_true_times.csv"),
                        parse_dates=["entry_time", "exit_time"])
    return full[full.entry_time >= SPLIT].sort_values("entry_time").reset_index(drop=True)


def pf_wr(t):
    wr = 100 * t.is_win.mean()
    w = t.loc[t.net_r > 0, "net_r"].sum(); l = -t.loc[t.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    return wr, pf


def section(title):
    print(f"\n{'='*94}\n{title}\n{'='*94}")


def main():
    t = load_holdout()
    print(f"TRUE-holdout population: {len(t):,} trades, {t.entry_time.min().date()} -> "
          f"{t.entry_time.max().date()} ({(t.entry_time.max()-t.entry_time.min()).days/365.25:.1f}yr)")
    wr0, pf0 = pf_wr(t)
    mean_r, median_r = t.net_r.mean(), t.net_r.median()
    print(f"Overall: WR={wr0:.2f}%  PF={pf0:.3f}  mean net_r={mean_r:+.4f}R  median net_r={median_r:+.4f}R")

    # ---- #4/#5/#14: year-by-year stability -----------------------------------------
    section("#4 / #5 / #14 -- Year-by-year win rate and mean R/trade")
    t["year"] = t.entry_time.dt.year
    yearly = []
    for yr, sub in t.groupby("year"):
        if len(sub) < 50:
            continue
        wr, pf = pf_wr(sub)
        yearly.append(dict(year=yr, n=len(sub), wr=wr, pf=pf, mean_r=sub.net_r.mean()))
        print(f"  {yr}: n={len(sub):6,}  WR={wr:5.2f}%  PF={pf:.3f}  mean_r={sub.net_r.mean():+.4f}R")
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(os.path.join(DATA_DIR, "goal32_yearly.csv"), index=False)
    losing_years = ydf[ydf.pf < 1.0]
    print(f"\n  Win-rate range across years: {ydf.wr.min():.2f}% - {ydf.wr.max():.2f}% "
          f"(std={ydf.wr.std():.2f}pp)")
    print(f"  Mean-R range across years:   {ydf.mean_r.min():+.4f}R - {ydf.mean_r.max():+.4f}R "
          f"(std={ydf.mean_r.std():.4f}R)")
    print(f"  Years with PF < 1.0: {len(losing_years)} of {len(ydf)} "
          f"({list(losing_years.year) if len(losing_years) else 'none'})")

    # ---- #6/#7: confidence intervals + significance --------------------------------
    section("#6 / #7 -- Confidence intervals and statistical significance")
    rng = np.random.default_rng(42)
    n_boot = 5000
    r = t.net_r.values
    win = t.is_win.values.astype(float)
    boot_mean_r = np.empty(n_boot)
    boot_wr = np.empty(n_boot)
    n = len(t)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_mean_r[i] = r[idx].mean()
        boot_wr[i] = 100 * win[idx].mean()
    ci_r = np.percentile(boot_mean_r, [2.5, 97.5])
    ci_wr = np.percentile(boot_wr, [2.5, 97.5])
    print(f"  Mean net R/trade: {mean_r:+.4f}R, bootstrap 95% CI [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}] "
          f"(width {ci_r[1]-ci_r[0]:.4f}R, {100*(ci_r[1]-ci_r[0])/abs(mean_r):.1f}% of the point estimate)")
    print(f"  Win rate: {wr0:.2f}%, bootstrap 95% CI [{ci_wr[0]:.2f}%, {ci_wr[1]:.2f}%] "
          f"(width {ci_wr[1]-ci_wr[0]:.2f}pp)")
    z_wr = (win.mean() - 0.5) / np.sqrt(0.25 / n)
    t_r, p_r = stats.ttest_1samp(r, 0.0)
    print(f"  WR vs 50% (naive z-test, IID assumption): z={z_wr:.1f} (p << 1e-300)")
    print(f"  Mean R vs 0 (naive one-sample t-test, IID assumption): t={t_r:.1f}, p={p_r:.2e}")
    print(f"  CAVEAT: both tests assume independent trades. Trades overlap in time and share")
    print(f"  4 correlated instruments and one strategy logic, so the true effective sample size")
    print(f"  is smaller than n={n:,} -- treat the bootstrap CI above (which doesn't assume IID)")
    print(f"  as the more honest uncertainty estimate; the p-values are directionally correct")
    print(f"  (the edge is not noise) but overstated in magnitude.")

    # ---- #13: outlier-trim robustness -----------------------------------------------
    section("#13 -- Outlier-removal robustness (trim best/worst N% of trades by net R)")
    for trim_pct in [0, 1, 2, 5, 10]:
        k = int(len(t) * trim_pct / 100)
        srt = t.sort_values("net_r")
        trimmed = srt.iloc[k: len(srt) - k] if k > 0 else srt
        wr, pf = pf_wr(trimmed)
        print(f"  Trim {trim_pct:2d}% each tail: n={len(trimmed):6,}  WR={wr:5.2f}%  PF={pf:.3f}")

    # ---- #15: regime breakdown --------------------------------------------------------
    section("#15 -- Market-regime breakdown")
    t["risk_tercile"] = pd.qcut(t.stop_risk / t.groupby("instrument").stop_risk.transform("median"),
                                 3, labels=["low-vol", "mid-vol", "high-vol"])
    print("  By local-volatility tercile (zone stop distance relative to that instrument's own median):")
    for tercile, sub in t.groupby("risk_tercile", observed=True):
        wr, pf = pf_wr(sub)
        print(f"    {tercile:9s}: n={len(sub):6,}  WR={wr:5.2f}%  PF={pf:.3f}  mean_r={sub.net_r.mean():+.4f}R")
    print("\n  By hand-labeled macro window:")
    windows = [
        ("2018 baseline", "2018-03-09", "2018-12-31"),
        ("2020 COVID crash", "2020-02-15", "2020-04-15"),
        ("2020 full year", "2020-01-01", "2020-12-31"),
        ("2022 rate-hike cycle", "2022-01-01", "2022-12-31"),
        ("2023-2024 calm", "2023-01-01", "2024-12-31"),
        ("2025-2026 recent", "2025-01-01", "2026-08-31"),
    ]
    for label, s, e in windows:
        sub = t[(t.entry_time >= s) & (t.entry_time <= e)]
        if len(sub) < 50:
            continue
        wr, pf = pf_wr(sub)
        print(f"    {label:22s}: n={len(sub):6,}  WR={wr:5.2f}%  PF={pf:.3f}  mean_r={sub.net_r.mean():+.4f}R")

    # ---- #16/#17: rolling 1yr / 3yr stability ----------------------------------------
    section("#16 / #17 -- Rolling window stability")
    for months, tag in [(12, "1-year"), (36, "3-year")]:
        rows = []
        m0 = t.entry_time.min()
        m_end = t.entry_time.max()
        cursor = m0 + pd.DateOffset(months=months)
        while cursor <= m_end:
            win_start = cursor - pd.DateOffset(months=months)
            sub = t[(t.entry_time >= win_start) & (t.entry_time < cursor)]
            if len(sub) >= 50:
                wr, pf = pf_wr(sub)
                rows.append(dict(window_end=cursor, n=len(sub), wr=wr, pf=pf, mean_r=sub.net_r.mean()))
            cursor += pd.DateOffset(months=1)
        rdf = pd.DataFrame(rows)
        rdf.to_csv(os.path.join(DATA_DIR, f"goal32_rolling_{months}m.csv"), index=False)
        n_losing = int((rdf.pf < 1.0).sum())
        print(f"  Rolling {tag} (stepped monthly, {len(rdf)} windows): "
              f"PF range [{rdf.pf.min():.3f}, {rdf.pf.max():.3f}], "
              f"WR range [{rdf.wr.min():.2f}%, {rdf.wr.max():.2f}%], "
              f"windows with PF<1.0: {n_losing}/{len(rdf)}")

    # ---- #18: cross-asset ---------------------------------------------------------
    section("#18 -- Cross-asset (per-instrument) robustness")
    for inst, sub in t.groupby("instrument"):
        wr, pf = pf_wr(sub)
        years = (sub.entry_time.max() - sub.entry_time.min()).days / 365.25
        print(f"  {inst:8s}: n={len(sub):6,} ({len(sub)/years:6,.0f}/yr)  WR={wr:5.2f}%  PF={pf:.3f}  "
              f"mean_r={sub.net_r.mean():+.4f}R")

    # ---- #19: cross-session ---------------------------------------------------------
    section("#19 -- Cross-session robustness")
    def session_of(h):
        if 22 <= h or h < 6: return "Asian (22-06 UTC)"
        if 6 <= h < 12: return "London AM (06-12 UTC)"
        if 12 <= h < 16: return "London/NY overlap (12-16 UTC)"
        return "NY PM (16-22 UTC)"
    t["session"] = t.entry_time.dt.hour.map(session_of)
    for sess, sub in t.groupby("session"):
        wr, pf = pf_wr(sub)
        print(f"  {sess:30s}: n={len(sub):6,}  WR={wr:5.2f}%  PF={pf:.3f}  mean_r={sub.net_r.mean():+.4f}R")

    # ---- #35: losing-streak analysis -------------------------------------------------
    section("#35 -- Losing-streak analysis")
    losses = (~t.is_win.values).astype(int)
    streaks = []
    cur = 0
    for x in losses:
        if x: cur += 1
        else:
            if cur > 0: streaks.append(cur)
            cur = 0
    if cur > 0: streaks.append(cur)
    streaks = np.array(streaks)
    p_loss = 1 - t.is_win.mean()
    print(f"  Longest losing streak: {streaks.max()} consecutive losses (in chronological entry order)")
    print(f"  Streak-length distribution: mean={streaks.mean():.2f}, median={np.median(streaks):.0f}, "
          f"count>=5: {(streaks>=5).sum()}, count>=8: {(streaks>=8).sum()}, count>=10: {(streaks>=10).sum()}")
    expected_max = np.log(len(t)) / np.log(1 / p_loss) if p_loss > 0 else float("nan")
    print(f"  Expected longest streak under an IID model with p(loss)={p_loss:.3f} and n={len(t):,} trades: "
          f"~{expected_max:.1f} (observed {streaks.max()} -- "
          f"{'consistent with' if abs(streaks.max()-expected_max) < 5 else 'notably different from'} IID)")

    # ---- #36: cross-instrument correlation ------------------------------------------
    section("#36 -- Portfolio correlation across instruments")
    daily = t.set_index("entry_time").groupby("instrument").resample("D").net_r.sum().unstack("instrument").fillna(0.0)
    corr = daily.corr()
    print(corr.round(3).to_string())
    corr.to_csv(os.path.join(DATA_DIR, "goal32_instrument_correlation.csv"))

    # ---- #37: tail-risk / clustered-loss analysis ------------------------------------
    section("#37 -- Tail-risk / clustered-loss analysis")
    is_win_int = t.is_win.astype(int).values
    ac1 = np.corrcoef(is_win_int[:-1], is_win_int[1:])[0, 1]
    print(f"  Lag-1 autocorrelation of win/loss sequence: {ac1:+.4f} "
          f"({'no meaningful serial dependence' if abs(ac1) < 0.02 else 'some serial dependence present'})")
    daily_r = t.set_index("entry_time").net_r.resample("D").sum()
    roll30 = daily_r.rolling(30).sum()
    worst_idx = roll30.idxmin()
    print(f"  Worst rolling 30-day realized R: {roll30.min():+.1f}R, ending {worst_idx.date()} "
          f"(window {(worst_idx - pd.Timedelta(days=29)).date()} -> {worst_idx.date()})")
    print(f"  For comparison, an average 30-day period nets "
          f"{(len(t)/((t.entry_time.max()-t.entry_time.min()).days)*30*mean_r):+.1f}R")


if __name__ == "__main__":
    main()
