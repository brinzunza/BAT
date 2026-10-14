"""Three-way "all metrics" comparison: CANA (faithful), SOMEGA (live), and Speed-King --
the ORIGINAL/base config of each strategy family (not CANA+/SOMEGA Robust/SOMEGA-Ultra,
which are later-derived variants), at IDENTICAL assumptions throughout: 3.00% risk
(Speed-King's live .env value), 50x leverage margin realism (goal20's exact model --
position sizing/margin computed the same way PRODLIVETRADING's broker layer does it),
$200 start, 1.6-pip spread (already in net_r via prod_filter_grid_search's cost model).
Using one shared risk/leverage/spread setting for all three isolates strategy
differences instead of confounding them with different risk settings.

Each strategy is scored on ITS OWN fair population, per this project's standing
"TRUE holdout" convention (see STRATEGY_RANKINGS.md): CANA and SOMEGA (live) have no
fitted/optimized rule (nothing was ever chosen by a grid search on this data), so
their full 16.9yr history is the fair test. Speed-King's (N, R:R, age_h) WAS chosen by
a grid search using only the FIRST half of history, so only the untouched second half
(entry_time >= 2018-03-09) is fair for it. Comparing "CANA's full history" against
"Speed-King's holdout-only" is the methodologically correct comparison here, not an
oversight -- exactly how each one's own published numbers elsewhere in this project
were derived. A second, common-window (>= 2018-03-09) cut is ALSO computed for all
three so there is one genuinely apples-to-apples side-by-side too.

Definitions (STRATEGY_RANKINGS.md's "Exact parameters" table):
  CANA (faithful):  N=3.00, R:R=1.0, fill-age=2.0h,      bounce-gated,   touches<=4, no session filter
  SOMEGA (live):    N=3.343496, R:R=1.0, fill-age=3.366682h, unconditional, touches<=4,
                    session filter -- EURGBP hr>=18, GBPUSD {20,21,22}, USDJPY {19,20,21,22}, EURUSD none
  Speed-King:       N=7.00, R:R=0.20, fill-age=0.175h,   unconditional,  touches<=4, no session filter

For each strategy, on its own fair population AND on the common post-2018-03-09 window:
  1. Standard stats: trades/yr, win rate, profit factor, weekly Sharpe (of the margin-
     constrained $200-start equity curve, not the unconstrained 5%-risk convention used
     elsewhere in this project -- consistent with everything else in this comparison)
  2. Concurrency: max/avg simultaneous open positions, % time with >=2 open
  3. Margin-constrained walk (goal20's model): signal-acceptance rate, $200-start single
     continuous compounding stream, realized max drawdown
  4. Forward Monte Carlo (25,000 sims, same 3% risk, $2M sizing cap): median years to
     $1M, drawdown percentiles, ruin rate
  5. Month-by-month, EACH MONTH RESTARTS AT $200 (goal27's model): % of months
     profitable, mean/median monthly return, best/worst month
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
import goal20_speed_king_margin_report as g20
import goal23_margin_capacity_all_strategies as g23
from goal10_cana_nt_mc_capped import BLOCK, MAX_TRADES, CAP, T1, T10, RUIN_FLOOR

RISK_PCT = 0.03
SPLIT = g23.SPLIT  # 2018-03-09, Speed-King's TRUE-holdout split point
MIN_TRADES_FOR_STATS = 20

CONFIGS = {
    "CANA (faithful)": g23.STRATEGIES["CANA (faithful)"],
    "SOMEGA (live)": g23.STRATEGIES["SOMEGA (live)"],
    "Speed-King": g23.STRATEGIES["Speed-King"],
}


def build_population(cfg):
    """Full-history trade population for this config's own session filter (None of
    these three use the VOTE mechanism, so g23.build_trades directly applies)."""
    return g23.build_trades(cfg["n"], cfg["rr"], cfg["age_h"], cfg["bounce"],
                             cfg["touches"], cfg["session"])


def concurrency_check(t):
    t = t.sort_values("entry_time").reset_index(drop=True)
    starts = t.entry_time.values.astype("datetime64[ns]")
    ends = t.exit_time.values.astype("datetime64[ns]")
    events = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(len(starts), dtype=int), -np.ones(len(ends), dtype=int)])
    order = np.argsort(events, kind="stable")
    events_sorted, deltas_sorted = events[order], deltas[order]
    concurrent = np.cumsum(deltas_sorted)
    max_concurrent = int(concurrent.max())
    dt_seconds = np.diff(events_sorted).astype("timedelta64[s]").astype(float)
    w = np.maximum(dt_seconds, 0)
    avg_concurrent = np.average(concurrent[:-1], weights=w) if len(w) and w.sum() > 0 else 0.0
    frac_ge2 = (concurrent[:-1] >= 2).astype(float)
    frac_ge2_time = np.average(frac_ge2, weights=w) if len(w) and w.sum() > 0 else 0.0
    return dict(max_concurrent=max_concurrent, avg_concurrent=avg_concurrent,
                frac_ge2_time=100 * frac_ge2_time)


def weekly_sharpe_from_trace(trace):
    s = trace.set_index("close_time")["equity"].resample("W").last().ffill()
    wk_ret = s.pct_change().dropna()
    if len(wk_ret) < 10 or wk_ret.std() == 0:
        return float("nan")
    return (wk_ret.mean() / wk_ret.std()) * np.sqrt(52)


def profile_population(label, pop, window_label):
    """Runs the full metric suite (margin walk, concurrency, MC, monthly restart) on
    one (strategy, window) population. Mirrors goal27's part1/part2 exactly, at 3%
    risk, generalized to any strategy's trade population."""
    print(f"\n{'#'*94}\n# {label}  --  {window_label}  (n={len(pop):,} candidates, "
          f"{pop.entry_time.min().date()} -> {pop.entry_time.max().date()})\n{'#'*94}")

    years = (pop.entry_time.max() - pop.entry_time.min()).days / 365.25
    tpy = len(pop) / years
    wr_raw = 100 * pop.is_win.mean()
    w_ = pop.loc[pop.net_r > 0, "net_r"].sum(); l_ = -pop.loc[pop.net_r <= 0, "net_r"].sum()
    pf_raw = w_ / l_ if l_ > 0 else float("inf")
    print(f"Raw population: {tpy:,.0f} trades/yr, WR={wr_raw:.2f}%, PF={pf_raw:.3f} "
          f"(before any margin acceptance/rejection)")

    conc = concurrency_check(pop)
    print(f"Concurrency (unconstrained signal population): max simultaneous={conc['max_concurrent']}, "
          f"avg={conc['avg_concurrent']:.2f}, %% time >=2 open={conc['frac_ge2_time']:.1f}%%")

    t, final_eq, max_dd_pct, accepted, trace = g20.simulate_margin_constrained(
        pop, start=g20.START, risk_pct=RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
        cap=g20.CAP, track_detail=True)
    acc = t[t.accepted]
    wr = 100 * acc.is_win.mean() if len(acc) else float("nan")
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    sharpe = weekly_sharpe_from_trace(trace)
    accept_pct = 100 * len(acc) / len(t) if len(t) else float("nan")
    print(f"Margin-constrained ({100*RISK_PCT:.0f}% risk, 50x leverage): accepted {len(acc):,}/{len(t):,} "
          f"({accept_pct:.1f}%), WR={wr:.2f}%, PF={pf:.3f}, weekly Sharpe={sharpe:.2f}")
    print(f"  $200 start -> ${final_eq:,.2f} over {years:.1f}yr (single continuous compounding stream, "
          f"$2M sizing cap); realized max drawdown: {max_dd_pct:.1f}%")

    net_r_all = pop.net_r.values.astype(np.float64)
    t1_raw, t10_raw, mc_dd, ruined = dtfx_engine.run_mc_capped(
        net_r_all, 25_000, BLOCK, MAX_TRADES, 22, start=g20.START, risk_frac=RISK_PCT,
        cap=CAP, target1=T1, target2=T10, ruin_floor=RUIN_FLOOR)
    t1_hit = t1_raw[t1_raw > 0]
    y1 = t1_hit / tpy
    dd_pct = mc_dd * 100
    print(f"Monte Carlo (25,000 sims, unconstrained-concurrency, same {100*RISK_PCT:.0f}% risk): "
          f"median {np.median(y1):.3f}yr to $1M (P10={np.percentile(y1,10):.3f}, "
          f"P90={np.percentile(y1,90):.3f}); max-DD median={np.median(dd_pct):.1f}% "
          f"(P10={np.percentile(dd_pct,10):.1f}%, P90={np.percentile(dd_pct,90):.1f}%); "
          f"ruin={100*ruined.mean():.3f}%")

    months = pd.period_range(pop.entry_time.min().to_period("M"), pop.entry_time.max().to_period("M"), freq="M")
    data_end = pop.entry_time.max()
    rows = []
    for m in months:
        m_start, m_end = m.start_time, min(m.end_time, data_end)
        window = pop[(pop.entry_time >= m_start) & (pop.entry_time <= m_end)]
        partial = m_end < m.end_time
        if len(window) < MIN_TRADES_FOR_STATS:
            continue
        _, feq, mdd, _, _ = g20.simulate_margin_constrained(
            window, start=g20.START, risk_pct=RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
            cap=None, track_detail=False)
        rows.append(dict(month=str(m), return_pct=100 * (feq / g20.START - 1),
                          max_dd_pct=mdd, partial=partial))
    mdf = pd.DataFrame(rows)
    complete = mdf[~mdf.partial]
    pct_profit = 100 * (complete.return_pct > 0).mean() if len(complete) else float("nan")
    print(f"Month-by-month, restart at $200 ({len(complete)} complete months): "
          f"{pct_profit:.1f}% profitable, mean={complete.return_pct.mean():+.1f}%, "
          f"median={complete.return_pct.median():+.1f}%, "
          f"best={complete.return_pct.max():+.1f}% ({complete.loc[complete.return_pct.idxmax(),'month']}), "
          f"worst={complete.return_pct.min():+.1f}% ({complete.loc[complete.return_pct.idxmin(),'month']})")

    return dict(
        label=label, window=window_label, n=len(pop), years=years, tpy=tpy,
        wr_raw=wr_raw, pf_raw=pf_raw, max_concurrent=conc["max_concurrent"],
        avg_concurrent=conc["avg_concurrent"], frac_ge2_time=conc["frac_ge2_time"],
        accept_pct=accept_pct, wr_margin=wr, pf_margin=pf, sharpe=sharpe,
        final_eq=final_eq, max_dd_realized=max_dd_pct,
        mc_median_yrs_1m=float(np.median(y1)), mc_p10_yrs_1m=float(np.percentile(y1, 10)),
        mc_p90_yrs_1m=float(np.percentile(y1, 90)), mc_median_dd=float(np.median(dd_pct)),
        mc_ruin_pct=float(100 * ruined.mean()),
        month_pct_profitable=pct_profit, month_mean_return=complete.return_pct.mean() if len(complete) else np.nan,
        month_median_return=complete.return_pct.median() if len(complete) else np.nan,
        month_best=complete.return_pct.max() if len(complete) else np.nan,
        month_worst=complete.return_pct.min() if len(complete) else np.nan,
    )


def main():
    for inst in g23.TRADEABLE:
        g23.rs.load_instrument(inst)
        g23.rs._prep_zones(inst, g23.pg.MIN_RETRACEMENT)

    print("Building all three strategies' full-history trade populations (C++ engine)...")
    pops = {label: build_population(cfg) for label, cfg in CONFIGS.items()}
    for label, pop in pops.items():
        pop.to_csv(os.path.join(DATA_DIR, f"goal28_{label.split()[0].lower()}_full.csv"), index=False)
        print(f"  {label}: {len(pop):,} trades, {pop.entry_time.min()} -> {pop.entry_time.max()}")

    results_own = {}
    for label, cfg in CONFIGS.items():
        pop = pops[label]
        fair = pop[pop.entry_time >= SPLIT] if cfg["holdout"] else pop
        wlabel = "own fair window (TRUE holdout, >=2018-03-09)" if cfg["holdout"] else "own fair window (full 2009-2026 history)"
        results_own[label] = profile_population(label, fair, wlabel)

    print(f"\n\n{'='*94}\nCOMMON WINDOW: all three strategies, entry_time >= {SPLIT.date()} "
          f"(the only period all three can be compared on head-to-head)\n{'='*94}")
    results_common = {}
    for label in CONFIGS:
        common_pop = pops[label][pops[label].entry_time >= SPLIT]
        results_common[label] = profile_population(label, common_pop, f"common window (>= {SPLIT.date()})")

    summary_own = pd.DataFrame(results_own.values())
    summary_common = pd.DataFrame(results_common.values())
    summary_own.to_csv(os.path.join(DATA_DIR, "goal28_summary_own_window.csv"), index=False)
    summary_common.to_csv(os.path.join(DATA_DIR, "goal28_summary_common_window.csv"), index=False)

    print(f"\n\n{'='*94}\nSUMMARY (own fair window per strategy)\n{'='*94}")
    print(summary_own.to_string(index=False))
    print(f"\n\n{'='*94}\nSUMMARY (common window, >= {SPLIT.date()})\n{'='*94}")
    print(summary_common.to_string(index=False))


if __name__ == "__main__":
    main()
