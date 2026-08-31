"""Sweep R:R <= 1 (narrower target than stop) on the CANA-style base (no session
filter, age=2.0h, bounce-gated, MAX_ZONE_TOUCHES=8, tradeable 4-pair universe only --
no gold) to find the best joint trade-off across frequency, win rate, Sharpe, and
median years to $1M at $200/5%-risk/1.6-pip-spread.

Narrower rr raises win rate mechanically (a closer target is easier to hit before the
wider stop) but each win pays less (net_r = +rr instead of +1.0) -- the classic
trade-off from earlier in this project. Sharpe here is computed on WEEKLY equity-curve
returns (not per-trade), annualized by sqrt(52) -- the standard, comparable definition,
avoiding the misleadingly huge numbers a per-trade Sharpe produces at high frequency.
"""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import prod_filter_grid_search as pg
import goal2_mc_race as mc

TRADEABLE = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
STOP_MULT = 2.0
AGE_H = 2.0
MAX_TOUCHES = 8
RISK_FRAC = 0.05
START = 200.0


def build_pool(rr, session_filter=False):
    target_mult = STOP_MULT * rr
    frames = [pg.build_trades(inst, STOP_MULT, target_mult, AGE_H, require_bounce=True,
                               session_filter=session_filter) for inst in TRADEABLE]
    full = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return full


def weekly_sharpe(trades):
    """Weekly-return Sharpe of the $200-start, 5%-risk compounding equity curve."""
    eq = np.empty(len(trades) + 1)
    eq[0] = START
    for i, r in enumerate(trades.net_r.values):
        risk = eq[i] * RISK_FRAC
        eq[i + 1] = max(eq[i] + risk * r, 0.01)
    trades = trades.copy()
    trades["equity"] = eq[1:]
    weekly = trades.set_index("entry_time")["equity"].resample("W").last().ffill()
    wk_ret = weekly.pct_change().dropna()
    if wk_ret.std() == 0 or len(wk_ret) < 10:
        return float("nan")
    return (wk_ret.mean() / wk_ret.std()) * np.sqrt(52)


def score_row(trades, label):
    years = (trades.entry_time.max() - trades.entry_time.min()).days / 365.25
    tpy = len(trades) / years
    wr = 100 * trades.is_win.mean()
    w = trades.loc[trades.net_r > 0, "net_r"].sum(); l = -trades.loc[trades.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    sharpe = weekly_sharpe(trades)
    t1_arr, t10_arr, ruined = mc.run_mc(trades.net_r.values, tpy, seed=hash(label) % 100000)
    y1m = np.percentile(t1_arr, 50) / tpy if len(t1_arr) else float("inf")
    y1m_p10 = np.percentile(t1_arr, 10) / tpy if len(t1_arr) else float("inf")
    y1m_p90 = np.percentile(t1_arr, 90) / tpy if len(t1_arr) else float("inf")
    return dict(label=label, trades=len(trades), tpy=tpy, wr=wr, pf=pf, sharpe=sharpe,
                y1m_p10=y1m_p10, y1m=y1m, y1m_p90=y1m_p90, ruin_pct=100 * ruined / mc.N_SIMS)


def print_row(r):
    print(f"{r['label']:10s} n={r['trades']:7,} ({r['tpy']:6.0f}/yr) WR={r['wr']:5.2f}% PF={r['pf']:.3f} "
          f"Sharpe={r['sharpe']:5.2f}  yrs->1M[P10/P50/P90]={r['y1m_p10']:.2f}/{r['y1m']:.2f}/{r['y1m_p90']:.2f}  "
          f"ruin={r['ruin_pct']:.1f}%")


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    rows = []
    for rr in [1.0, 0.75, 0.5, 0.35, 0.25, 0.15]:
        trades = build_pool(rr, session_filter=False)
        r = score_row(trades, f"rr={rr}")
        rows.append(r)
        print_row(r)
        trades.to_csv(os.path.join(DATA_DIR, f"goal5_rr_{str(rr).replace('.','p')}_trades.csv"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "goal5_rr_sweep_results.csv"), index=False)


if __name__ == "__main__":
    main()
