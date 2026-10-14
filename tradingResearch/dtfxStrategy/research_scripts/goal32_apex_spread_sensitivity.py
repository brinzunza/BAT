"""How spread-resistant is Apex-King? Sweeps spread from 0 pips up past the point PF
drops under 1.0, at this deployment's fixed settings (two $100 accounts, 6% risk, 50x
leverage, 1-per-side cap) -- everything else held constant, spread is the only thing
that moves.

Key simplification that makes this cheap: which trades get ACCEPTED (margin + 1-per-side
gating) depends only on stop_risk/entry_price/risk_pct -- never on spread, since spread
only ever subtracts from a LOSING trade's net_r, after the accept/reject decision is
already made. So the accepted trade SET (and therefore win rate and trade count) is
IDENTICAL at every spread level; only average loss size, PF, expectancy, and the
resulting equity curve change. That means net_r can be recomputed fresh for each spread
value from data already built once (no need to re-run zone detection or the C++ fill
simulation per spread level) -- each spread point costs one cheap two-account backtest
pass, not a full rebuild.

net_r formula (unchanged from build_trades_with_side): wins net_r=+rr always (limit-
filled, no spread paid); losses net_r=-1.0 - spread_pips*PIP_SIZE[inst]/stop_risk (the
stop exit crosses the spread). ratio = PIP_SIZE[inst]/stop_risk is a fixed per-trade
property, spread-independent -- computed once, reused for every spread level.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30

RISK_PCT = 0.06  # this deployment's fixed risk -- spread is the only thing varied
SPREAD_GRID = [0.0, 0.8, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0, 11.0, 12.0]


def build_base_trades():
    """Apex-King's true trade stream with rr and the per-trade spread-cost ratio
    (PIP_SIZE[inst]/stop_risk) attached -- everything needed to recompute net_r for
    any spread value without re-running zone detection or the fill simulation."""
    trades = g30.build_apex_trades()  # net_r baked in at the module's SPREAD_PIPS_TYPICAL (1.6)
    pg = g30.g29.pg
    rr = g30.APEX_CFG["rr"]
    pip = trades.instrument.map(pg.PIP_SIZE)
    ratio = pip / trades.stop_risk  # PIP_SIZE[inst] / stop_risk, spread-independent
    trades = trades.assign(rr=rr, ratio=ratio)
    return trades


def net_r_at_spread(trades, spread_pips):
    r_gross = np.where(trades.is_win, trades.rr, -1.0)
    cost_in_r = np.where(trades.is_win, 0.0, spread_pips * trades.ratio)
    return r_gross - cost_in_r


def main():
    print("Building Apex-King TRUE holdout trade stream (spread-independent ingredients)...")
    base = build_base_trades()
    print(f"  {len(base):,} signal candidates\n")

    rows = []
    print(f"{'spread':>7}  {'WR':>7}  {'PF':>7}  {'avg_loss_R':>11}  {'expectancy_R':>13}  "
          f"{'accepted':>9}  {'end_$':>15}  {'maxDD':>7}  {'yrs->1M':>8}")
    for spread in SPREAD_GRID:
        t = base.copy()
        t["net_r"] = net_r_at_spread(base, spread)

        wr = 100 * t.is_win.mean()
        wins = t[t.is_win]; losses = t[~t.is_win]
        avg_loss_r = losses.net_r.mean()
        gross_win = wins.net_r.sum(); gross_loss = -losses.net_r.sum()
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        expectancy = t.net_r.mean()  # per-signal expectancy (pre-margin-filter, matches WR/PF above)

        tsim, eq_buy, eq_sell, max_dd, trace = g30.simulate_two_account(t, RISK_PCT)
        acc = tsim[tsim.accepted]
        n_acc = len(acc)
        combined_end = eq_buy + eq_sell

        # Years to $1M from this single historical path's own trace (cheap proxy for
        # speed impact -- not a full MC, but the accept/reject set and relative ordering
        # of trades is identical across spread levels, so this comparison is meaningful
        # even though any single path carries the usual small-sample caveat).
        yrs_to_1m = float("nan")
        if len(trace):
            hit = trace[trace.equity_combined >= 1_000_000]
            if len(hit):
                first_hit = hit.iloc[0].close_time
                yrs_to_1m = (first_hit - t.entry_time.min()).days / 365.25

        status = "" if pf >= 1.0 else "  <-- FAILS (PF<1)"
        print(f"{spread:7.1f}  {wr:6.2f}%  {pf:7.3f}  {avg_loss_r:11.4f}  {expectancy:13.4f}  "
              f"{n_acc:9,}  ${combined_end:14,.0f}  {max_dd:6.1f}%  "
              f"{yrs_to_1m:7.2f}{status}")
        rows.append(dict(spread_pips=spread, wr=wr, pf=pf, avg_loss_r=avg_loss_r,
                          expectancy_r=expectancy, accepted=n_acc, end_combined=combined_end,
                          max_dd_pct=max_dd, years_to_1m=yrs_to_1m))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal32_apex_spread_sensitivity.csv"), index=False)

    # Exact breakeven spread (PF=1, closed form since ratio is fixed per losing trade):
    # gross_win = n_wins*rr (constant); gross_loss(S) = sum_losses(1 + S*ratio_i).
    # Solve gross_win = gross_loss(S) for S.
    wins = base[base.is_win]; losses = base[~base.is_win]
    gross_win = wins.rr.sum()
    sum_ratio = losses.ratio.sum()
    n_losses = len(losses)
    breakeven_spread = (gross_win - n_losses) / sum_ratio
    print(f"\nExact breakeven spread (PF=1.0): {breakeven_spread:.2f} pips")
    print(f"Currently deployed at 1.6 pips -- {breakeven_spread/1.6:.1f}x headroom before PF crosses 1.0")


if __name__ == "__main__":
    main()
