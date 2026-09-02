"""Speed-King: ROLLING 3-week backtest, each window independently restarting at $200 --
2.0-pip spread, 3.00% risk, 50x leverage, $500,000 position-sizing cap (same assumptions
as goal34/goal35). Stepped weekly (3-week window / 1-week step = 3x overlap, the same
overlap ratio goal35 used for its 3-month/1-month version). Reports full distributional
stats on FINAL BALANCE (dollars), not just % return, per this request.
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

import goal20_speed_king_margin_report as g20
from goal29_speed_king_spread_sensitivity import build_raw, with_spread

rs, pg, TRADEABLE = g20.rs, g20.pg, g20.TRADEABLE

START = 200.0
SPREAD_PIPS = 2.0
RISK_PCT = 0.03
MAX_LEVERAGE = 50.0
CAP = 500_000.0
WINDOW_WEEKS = 3
STEP_WEEKS = 1
MIN_TRADES_FOR_STATS = 20


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Building Speed-King's population at 2.0-pip spread...", flush=True)
    raw = build_raw()
    full = with_spread(raw, SPREAD_PIPS)
    data_start, data_end = full.entry_time.min(), full.entry_time.max()
    print(f"  {len(full):,} trades, {data_start} -> {data_end}\n")

    rows = []
    cursor = data_start
    win_len = pd.Timedelta(weeks=WINDOW_WEEKS)
    step = pd.Timedelta(weeks=STEP_WEEKS)
    while cursor + win_len <= data_end + pd.Timedelta(days=1):
        win_start = cursor
        win_end = min(cursor + win_len, data_end)
        window = full[(full.entry_time >= win_start) & (full.entry_time <= win_end)]
        partial = win_end < cursor + win_len
        if len(window) >= MIN_TRADES_FOR_STATS:
            t, final_eq, max_dd_pct, accepted, _ = g20.simulate_margin_constrained(
                window, start=START, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE,
                cap=CAP, track_detail=True)
            acc = t[t.accepted]
            if len(acc):
                wr = 100 * acc.is_win.mean()
                w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
                pf = w / l if l > 0 else float("inf")
                max_conc = int(t.concurrent_at_entry.max())
                accept_pct = 100 * len(acc) / len(t)
            else:
                wr = pf = float("nan"); max_conc = 0; accept_pct = float("nan")
            rows.append(dict(
                window_start=win_start.date(), window_end=win_end.date(),
                trades=len(t), accepted=len(acc), accept_pct=round(accept_pct, 1),
                win_rate=round(wr, 2), pf=round(pf, 3) if np.isfinite(pf) else pf,
                end_balance=round(final_eq, 2), return_pct=round(100 * (final_eq / START - 1), 2),
                max_dd_pct=round(max_dd_pct, 1), max_concurrent=max_conc, partial=partial,
            ))
        cursor += step

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal36_rolling_3w_restart.csv"), index=False)
    complete = df[~df.partial]
    eb = complete.end_balance

    print("=" * 94)
    print(f"ROLLING {WINDOW_WEEKS}-WEEK BACKTEST, restart at $200 each window "
          f"(stepped every {STEP_WEEKS} week)")
    print(f"${START:.0f} start, {SPREAD_PIPS}-pip spread, {100*RISK_PCT:.1f}% risk, {MAX_LEVERAGE:.0f}x "
          f"leverage, ${CAP:,.0f} sizing cap")
    print("=" * 94)
    print(f"\n{len(df)} rolling windows spanned ({len(complete)} complete, "
          f"{len(df)-len(complete)} partial/truncated at the end of data)")

    pct_profitable = 100 * (eb > START).mean()
    print(f"\n%% windows profitable (end balance > $200): {pct_profitable:.1f}% "
          f"({int((eb>START).sum())} up, {int((eb<=START).sum())} down)")

    print(f"\n{'='*94}\nFINAL-BALANCE STATISTICS (dollars, {len(complete)} windows)\n{'='*94}")
    print(f"  Mean:    ${eb.mean():,.2f}")
    print(f"  Median:  ${eb.median():,.2f}")
    print(f"  Std dev: ${eb.std():,.2f}")
    print(f"  Min:     ${eb.min():,.2f}  ({complete.loc[eb.idxmin(),'window_start']} -> "
          f"{complete.loc[eb.idxmin(),'window_end']})")
    print(f"  Max:     ${eb.max():,.2f}  ({complete.loc[eb.idxmax(),'window_start']} -> "
          f"{complete.loc[eb.idxmax(),'window_end']})")
    print(f"\n  Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:2d}: ${np.percentile(eb, p):,.2f}")

    print(f"\n  Bucketed outcome distribution:")
    buckets = [
        ("Lost money (< $200)", (eb < START)),
        ("Flat-ish ($200-$300)", (eb >= START) & (eb < 300)),
        ("Doubled+ ($300-$1,000)", (eb >= 300) & (eb < 1000)),
        ("10x+ ($1,000-$10,000)", (eb >= 1000) & (eb < 10_000)),
        ("100x+ ($10,000-$100,000)", (eb >= 10_000) & (eb < 100_000)),
        ("Near/at cap ($100,000-$500,000)", (eb >= 100_000) & (eb < 500_000)),
        ("Past the $500k cap", (eb >= 500_000)),
    ]
    for label, mask in buckets:
        n = int(mask.sum())
        print(f"    {label:32s}: {n:4d} windows ({100*n/len(complete):5.1f}%)")

    print(f"\n  For context, % return: mean={complete.return_pct.mean():+.1f}%  "
          f"median={complete.return_pct.median():+.1f}%")
    print(f"  Max drawdown within a window: mean={complete.max_dd_pct.mean():.1f}%  "
          f"worst={complete.max_dd_pct.min():.1f}%")
    print(f"  Mean accept rate: {complete.accept_pct.mean():.1f}%   "
          f"Mean max concurrent: {complete.max_concurrent.mean():.1f} (overall max: "
          f"{complete.max_concurrent.max()})")

    print(f"\n  Worst 10 windows by end balance:")
    for _, r in complete.nsmallest(10, "end_balance").iterrows():
        print(f"    {r.window_start} -> {r.window_end}: ${r.end_balance:,.2f} "
              f"({r.return_pct:+.1f}%, max DD {r.max_dd_pct:.1f}%)")

    print(f"\nFull table: data/goal36_rolling_3w_restart.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
