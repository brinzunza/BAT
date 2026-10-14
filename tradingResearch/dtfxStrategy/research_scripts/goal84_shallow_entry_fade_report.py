"""Full backtest report for the corrected Shallow-Entry Fade strategy (30% retracement,
inverted, goal82's entry-price bug fix + goal85's flat-cost spread convention applied) --
direct user request (2026-09-22): "write down this strategy, return backtest results ...
with plots on account growth".

CORRECTION (2026-09-23): the first version of this report built trades from goal82's
build_frac_trades_fixed(), which reprices entry_price by up to one spread depending on
whether the touch bar was "deep" enough, then defines "1R" off that shifted entry -- an
asymmetric cost model where losing trades never actually pay any spread cost (a stop-out
is tautologically -1.0*eff_risk regardless of entry repricing). Per direct user
specification, this now builds trades from goal85_shallow_entry_fade_flatcost.py instead:
entry/stop/target held at the exact raw zone levels (no repricing at all), with exactly
one spread's worth of R-cost subtracted from every trade's raw result, win or loss alike.
This changes the headline numbers (PF 1.366 -> 1.188; 100% of months profitable no longer
holds, 201/204 does) -- see goal85's own docstring for the full comparison.

Reuses, unchanged, every other already-audited piece of this session's pipeline:
  - goal85.build_frac_trades_flatcost()    -- corrected zone detection + entry/exit resolution
  - goal79.simulate_two_account()          -- two-sub-account event-driven $ walk
  - goal79.run_monte_carlo_two_account()   -- block-bootstrap MC, race to $1M
  - goal81.thirds_robustness() logic       -- reimplemented locally to capture numbers as data
  - goal83's per-year, fresh-$200 methodology -- reimplemented locally (per_year_backtest)
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal79_two_account_portfolio as g79
from goal85_shallow_entry_fade_flatcost import build_frac_trades_flatcost as build_frac_trades_fixed

simulate_two_account = g79.simulate_two_account
run_monte_carlo_two_account = g79.run_monte_carlo_two_account
period_stats = g79.period_stats

FRAC = 0.30
RISK_PCT = g79.RISK_PCT          # 0.05
START_PER_SIDE = g79.START_PER_SIDE  # 100.0
MAX_LEVERAGE = g79.MAX_LEVERAGE  # 50.0
CAP = g79.ACCOUNT_CAP            # 2_000_000.0
N_MC_SIMS = 3000
MC_TARGET = 1_000_000.0

OUT_DIR = os.path.join(SCRIPT_DIR, "..", "research_results", "goal84_shallow_entry_fade")
os.makedirs(OUT_DIR, exist_ok=True)


def thirds_robustness(accepted):
    span_start, span_end = accepted["entry_time"].min(), accepted["entry_time"].max()
    edges = pd.date_range(span_start, span_end, periods=4)
    rows = []
    for i in range(3):
        if i < 2:
            mask = (accepted["entry_time"] >= edges[i]) & (accepted["entry_time"] < edges[i + 1])
        else:
            mask = (accepted["entry_time"] >= edges[i]) & (accepted["entry_time"] <= edges[i + 1])
        chunk = accepted[mask]
        r = chunk["net_r"].values
        gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
        pf = gw / gl if gl > 0 else float("inf")
        wr = 100 * (r > 0).mean() if len(r) else float("nan")
        rows.append(dict(third=i + 1, start=str(edges[i].date()), end=str(edges[i + 1].date()),
                          n=int(len(chunk)), wr=wr, pf=pf, profitable=bool(pf > 1.0)))
    return rows


def per_year_backtest(trades, risk_pct):
    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades.entry_time).dt.year
    rows = []
    for year, grp in trades.groupby("year"):
        grp = grp.sort_values("entry_time").reset_index(drop=True)
        _, eq_buy, eq_sell, max_dd, _ = simulate_two_account(
            grp, risk_pct=risk_pct, start_per_side=START_PER_SIDE,
            max_leverage=MAX_LEVERAGE, cap=CAP, track_detail=False)
        end_total = eq_buy + eq_sell
        rows.append(dict(year=int(year), n_signals=len(grp), end_total=end_total,
                          return_pct=100.0 * (end_total / (2 * START_PER_SIDE) - 1.0),
                          max_dd=max_dd))
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def main():
    t0 = time.time()
    print(f"Building corrected frac={FRAC} trade candidates (all 4 pairs, inverted, entry-price bug fixed)...")
    candidates = build_frac_trades_fixed(FRAC)
    print(f"  {len(candidates):,} candidates ({time.time()-t0:.0f}s)")

    print("Running full-history two-account simulation (with equity trace)...")
    result, eq_buy, eq_sell, max_dd, trace = simulate_two_account(
        candidates, risk_pct=RISK_PCT, start_per_side=START_PER_SIDE,
        max_leverage=MAX_LEVERAGE, cap=CAP, track_detail=True)
    accepted = result[result["accepted"]].reset_index(drop=True)
    print(f"  Accepted {len(accepted):,} / {len(candidates):,} candidates ({time.time()-t0:.0f}s)")

    # ---- headline trade stats ----
    net_r = accepted["net_r"].values
    wins, losses = net_r[net_r > 0], net_r[net_r <= 0]
    wr = 100.0 * len(wins) / len(net_r)
    avg_win, avg_loss = float(wins.mean()), float(losses.mean())
    rr = avg_win / abs(avg_loss)
    pf = float(wins.sum() / abs(losses.sum()))
    span_start, span_end = accepted["entry_time"].min(), accepted["exit_time"].max()
    years = (span_end - span_start) / np.timedelta64(365, "D")
    days = (span_end - span_start) / np.timedelta64(1, "D")

    per_symbol = {}
    for inst, g in accepted.groupby("instrument"):
        gw, gl = g.loc[g.net_r > 0, "net_r"].sum(), -g.loc[g.net_r <= 0, "net_r"].sum()
        per_symbol[inst] = dict(n=int(len(g)), wr=float(100 * (g.net_r > 0).mean()),
                                 pf=float(gw / gl if gl > 0 else float("inf")))

    r_skew = float(sps.skew(net_r, bias=False))
    r_kurt = float(sps.kurtosis(net_r, fisher=True, bias=False))

    # ---- equity curve (daily-resampled for stats + plots) ----
    equity = trace.set_index("close_time")["equity_combined"]
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    daily = equity.resample("D").last().ffill().dropna()
    daily_ret = daily.pct_change().dropna()
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    sharpe = float(daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252)) if daily_ret.std(ddof=1) > 0 else float("nan")
    eq_skew = float(sps.skew(daily_ret.values, bias=False))
    eq_kurt = float(sps.kurtosis(daily_ret.values, fisher=True, bias=False))

    period_pcts = {}
    for freq, label in [("D", "day"), ("W-FRI", "week"), ("M", "month"), ("A", "year")]:
        ps = period_stats(equity, freq, label)
        period_pcts[label] = ps

    thirds = thirds_robustness(accepted)

    print(f"Running Monte Carlo ({N_MC_SIMS:,} sims, race to ${MC_TARGET:,.0f})...")
    mc_days, mc_trades_ct, mc_dds, mc_ruin_rate = run_monte_carlo_two_account(
        accepted, N_MC_SIMS, target=MC_TARGET)
    mc = dict(
        pct_reached=float(100 * len(mc_days) / N_MC_SIMS),
        median_days=float(np.median(mc_days)) if len(mc_days) else None,
        p5_days=float(np.percentile(mc_days, 5)) if len(mc_days) else None,
        p95_days=float(np.percentile(mc_days, 95)) if len(mc_days) else None,
        ruin_rate_pct=float(100 * mc_ruin_rate),
        dd_median_pct=float(np.median(mc_dds)),
        dd_worst_p5_pct=float(np.percentile(mc_dds, 5)),
    )
    print(f"  done ({time.time()-t0:.0f}s)")

    print("Running per-year (fresh $200 each year) backtest for the growth-by-year chart...")
    per_year = per_year_backtest(candidates, RISK_PCT)
    print(f"  done ({time.time()-t0:.0f}s)")

    stats = dict(
        frac=FRAC, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE, cap=CAP,
        n_candidates=int(len(candidates)), n_accepted=int(len(accepted)),
        span_start=str(span_start.date()), span_end=str(span_end.date()), years=float(years),
        win_rate_pct=float(wr), avg_win_r=avg_win, avg_loss_r=avg_loss, rr=float(rr),
        profit_factor=pf, trades_per_year=float(len(accepted) / years),
        trades_per_day=float(len(accepted) / days),
        net_r_skew=r_skew, net_r_excess_kurtosis=r_kurt,
        per_instrument=per_symbol,
        combined_final_equity=float(eq_buy + eq_sell), eq_buy_final=float(eq_buy), eq_sell_final=float(eq_sell),
        max_drawdown_pct=float(max_dd),
        sharpe_daily_annualized=sharpe, daily_return_skew=eq_skew, daily_return_excess_kurtosis=eq_kurt,
        pct_profitable_days=period_pcts["day"]["pct_profitable"],
        pct_profitable_weeks=period_pcts["week"]["pct_profitable"],
        pct_profitable_months=period_pcts["month"]["pct_profitable"],
        pct_profitable_years=period_pcts["year"]["pct_profitable"],
        n_days=period_pcts["day"]["n_periods"], n_weeks=period_pcts["week"]["n_periods"],
        n_months=period_pcts["month"]["n_periods"], n_years=period_pcts["year"]["n_periods"],
        thirds_robustness=thirds, monte_carlo=mc,
    )
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    per_year.to_json(os.path.join(OUT_DIR, "per_year.json"), orient="records", indent=2)

    print("\n" + "=" * 100)
    print("HEADLINE RESULTS -- Shallow-Entry Fade, 30% entry, inverted, entry-price bug fixed")
    print("=" * 100)
    for k, v in stats.items():
        if k in ("per_instrument", "thirds_robustness", "monte_carlo"):
            continue
        print(f"  {k}: {v}")
    print("  per_instrument:", json.dumps(per_symbol, indent=2))
    print("  thirds_robustness:", json.dumps(thirds, indent=2))
    print("  monte_carlo:", json.dumps(mc, indent=2))

    # ================= PLOTS =================
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                          "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # 1. Equity curve, log scale, with cap-crossing annotation
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(daily.index, daily.values, color="#1a6b3c", lw=1.2)
    ax.set_yscale("log")
    ax.set_title(f"Shallow-Entry Fade (30% entry) — Combined Two-Account Equity, {span_start.date()}..{span_end.date()}")
    ax.set_ylabel("Combined equity ($, log scale)")
    cap_cross = daily[daily >= 2 * CAP]
    if len(cap_cross):
        cx = cap_cross.index[0]
        ax.axvline(cx, color="firebrick", ls="--", lw=1)
        ax.annotate(f"position-sizing cap reached ({cx.date()})\ndollar growth beyond this point is a\nmodeling artifact of the fixed $2M cap,\nnot a realistic forecast",
                    xy=(cx, daily.loc[cx:].iloc[0]), xytext=(0.03, 0.80), textcoords="axes fraction",
                    fontsize=8, color="firebrick", arrowprops=dict(arrowstyle="->", color="firebrick", lw=0.8))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_equity_curve.png"), dpi=150)
    plt.close(fig)

    # 2. Drawdown (underwater) curve
    running_peak = daily.cummax()
    dd = 100.0 * (daily - running_peak) / running_peak
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.6)
    ax.set_title("Drawdown from running peak (combined equity, daily)")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_drawdown.png"), dpi=150)
    plt.close(fig)

    # 3. Per-year, fresh $200 each year -- return %
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#1a6b3c" if v > 0 else "#b3382c" for v in per_year["return_pct"]]
    ax.bar(per_year["year"].astype(str), per_year["return_pct"], color=colors)
    ax.set_yscale("symlog")
    ax.set_title("Per-year return, starting fresh at $200 each calendar year (5% risk/trade)")
    ax.set_ylabel("Return (%, symlog scale)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_per_year_returns.png"), dpi=150)
    plt.close(fig)

    # 4. Monthly returns heatmap
    monthly = equity.resample("M").last().ffill().dropna()
    monthly_ret = monthly.pct_change().dropna() * 100.0
    dfm = monthly_ret.to_frame("ret")
    dfm["year"] = dfm.index.year
    dfm["month"] = dfm.index.month
    pivot = dfm.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(pivot) + 2))
    vmax = np.nanpercentile(np.abs(pivot.values), 95)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly return heatmap (combined equity, % change)")
    for yi in range(pivot.shape[0]):
        for mi in range(pivot.shape[1]):
            v = pivot.values[yi, mi]
            if np.isfinite(v):
                ax.text(mi, yi, f"{v:+.0f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="% return", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_monthly_heatmap.png"), dpi=150)
    plt.close(fig)

    # 5. R-multiple distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(-1.2, max(3.0, np.percentile(net_r, 99)), 60)
    ax.hist(net_r[net_r <= 0], bins=bins, color="#b3382c", alpha=0.8, label=f"losses (n={len(losses):,})")
    ax.hist(net_r[net_r > 0], bins=bins, color="#1a6b3c", alpha=0.8, label=f"wins (n={len(wins):,})")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Per-trade R-multiple distribution (accepted trades)")
    ax.set_xlabel("Net R"); ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_r_multiple_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots + results.json + per_year.json saved to {OUT_DIR}")
    print(f"(total: {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
