"""Per-calendar-year backtest of the corrected Shallow-Entry Fade strategy (goal82's fix:
entry_price repriced for the side actually traded), resetting to a fresh $200 ($100/$100
buy/sell, matching this project's real two-account architecture) at the start of every year,
across a range of entry fractions and risk percentages.

Reuses goal79's own simulate_two_account() unchanged for the per-year event-driven $ walk --
this is the same, already-audited two-account mechanic used for the whole-history numbers in
goal81/goal82, just re-run on one calendar year of trades at a time instead of the full span.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal79_two_account_portfolio as g79
from goal82_audit_inversion_entry_bug import build_frac_trades_fixed

simulate_two_account = g79.simulate_two_account

FRACS = [0.20, 0.25, 0.30, 0.35]
RISK_LEVELS = [0.02, 0.05, 0.10, 0.15]
START_PER_SIDE = 100.0
MAX_LEVERAGE = 50.0
CAP = 2_000_000.0


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
                          max_dd=max_dd,
                          span=f"{grp.entry_time.min().date()}..{grp.entry_time.max().date()}"))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    trades_by_frac = {}
    for frac in FRACS:
        print(f"Building corrected trades for frac={frac} ...", flush=True)
        trades_by_frac[frac] = build_frac_trades_fixed(frac)
        print(f"  {len(trades_by_frac[frac]):,} candidate trades ({time.time()-t0:.0f}s)", flush=True)

    all_rows = []
    for frac in FRACS:
        print(f"\n{'='*100}\nENTRY = {int(frac*100)}%\n{'='*100}")
        for risk in RISK_LEVELS:
            tab = per_year_backtest(trades_by_frac[frac], risk)
            tab["frac"] = frac
            tab["risk_pct"] = risk
            all_rows.append(tab)
            n_years = len(tab)
            n_pos = int((tab.return_pct > 0).sum())
            print(f"\n--- risk={int(risk*100)}% per trade --- "
                  f"({n_pos}/{n_years} years profitable)")
            print(f"{'year':>6s} {'signals':>8s} {'end_$200->':>12s} {'return':>9s} {'maxDD':>8s}  span")
            for _, r in tab.sort_values("year").iterrows():
                print(f"{r['year']:>6d} {r['n_signals']:>8,} {r['end_total']:>12,.2f} "
                      f"{r['return_pct']:>+8.1f}% {r['max_dd']:>7.2f}%  [{r['span']}]")

    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(SCRIPT_DIR, "scratch", "goal83_per_year_entry_risk_sweep.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n\n{'='*100}\nSUMMARY: years profitable out of total, by entry% x risk%\n{'='*100}")
    print(f"{'entry':>7s}", end="")
    for risk in RISK_LEVELS:
        print(f"{'risk='+str(int(risk*100))+'%':>12s}", end="")
    print()
    for frac in FRACS:
        print(f"{int(frac*100):>6d}%", end="")
        for risk in RISK_LEVELS:
            sub = out[(out.frac == frac) & (out.risk_pct == risk)]
            n_pos = int((sub.return_pct > 0).sum())
            n_tot = len(sub)
            print(f"{f'{n_pos}/{n_tot}':>12s}", end="")
        print()

    print(f"\nSaved to {out_path}")
    print(f"(total: {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
