import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import grid_search_optimal_sl_tp_mc_race as g5
import grid_search_10pip_sl_tp_mc_race as g10

cache = g5.load_cache()

# Date range
all_times = []
for inst, d in cache.items():
    all_times.append(d["times"].min())
    all_times.append(d["times"].max())
t_min = min(all_times)
t_max = max(all_times)
total_days = (t_max - t_min).total_seconds() / 86400.0

print(f"Dataset span: {t_min} to {t_max} ({total_days:.1f} days / {total_days/365.25:.1f} years)")

configs_5p = [
    (1.0, 1.0, "5-Pip: 1.0SL : 1.0TP"),
    (1.0, 1.25, "5-Pip: 1.0SL : 1.25TP"),
    (1.0, 0.8, "5-Pip: 1.0SL : 0.8TP"),
    (1.25, 1.0, "5-Pip: 1.25SL : 1.0TP"),
    (1.25, 1.25, "5-Pip: 1.25SL : 1.25TP"),
    (1.5, 1.0, "5-Pip: 1.5SL : 1.0TP"),
    (2.0, 1.0, "5-Pip: 2.0SL : 1.0TP"),
]

configs_10p = [
    (1.0, 1.0, "10-Pip: 1.0SL : 1.0TP"),
    (1.0, 0.8, "10-Pip: 1.0SL : 0.8TP"),
    (1.25, 1.0, "10-Pip: 1.25SL : 1.0TP"),
    (1.0, 1.25, "10-Pip: 1.0SL : 1.25TP"),
    (1.25, 0.8, "10-Pip: 1.25SL : 0.8TP"),
    (1.5, 1.0, "10-Pip: 1.5SL : 1.0TP"),
    (2.0, 0.8, "10-Pip: 2.0SL : 0.8TP"),
    (3.0, 0.8, "10-Pip: 3.0SL : 0.8TP"),
]

print("\n--- 5-PIP CONFIGS ---")
for sl, tp, name in configs_5p:
    g5.MIN_ZONE_PIPS = 5.0
    trades = g5.build_trades_for_pair(cache, sl, tp, spread_pips=1.6)
    res, _, _, _, _ = g5.g79.simulate_two_account(trades, risk_pct=0.03, start_per_side=100.0, max_leverage=50.0, cap=10_000_000.0, track_detail=False)
    acc = res[res["accepted"]]
    n_exec = len(acc)
    tpd = n_exec / (total_days * (5/7)) # trading days
    wr = 100.0 * (acc["net_r"] > 0).mean() if n_exec else 0.0
    avg_win = acc.loc[acc["net_r"] > 0, "net_r"].mean() if (acc["net_r"] > 0).any() else 0.0
    gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
    gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else 0
    
    # mc
    med_days, med_dd = g5.run_mc_fast(trades, n_sims=300)
    print(f"{name:<25} | Execs: {n_exec:<6} | Trades/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | Days to $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")

print("\n--- 10-PIP CONFIGS ---")
for sl, tp, name in configs_10p:
    g10.MIN_ZONE_PIPS = 10.0
    trades = g10.build_trades_for_pair(cache, sl, tp, spread_pips=1.6)
    res, _, _, _, _ = g10.g79.simulate_two_account(trades, risk_pct=0.03, start_per_side=100.0, max_leverage=50.0, cap=10_000_000.0, track_detail=False)
    acc = res[res["accepted"]]
    n_exec = len(acc)
    tpd = n_exec / (total_days * (5/7)) # trading days
    wr = 100.0 * (acc["net_r"] > 0).mean() if n_exec else 0.0
    avg_win = acc.loc[acc["net_r"] > 0, "net_r"].mean() if (acc["net_r"] > 0).any() else 0.0
    gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
    gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else 0
    
    # mc
    med_days, med_dd = g10.run_mc_fast(trades, n_sims=300)
    print(f"{name:<25} | Execs: {n_exec:<6} | Trades/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | Days to $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")

