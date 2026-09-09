"""Backtest with USDJPY = 20-Pip Minimum Zone:
Comparing:
1. Baseline PROD (USDJPY = 10p, EURUSD = 10p, GBPUSD = 10p, EURGBP = 10p)
2. USDJPY 20-Pip Hybrid (USDJPY = 20p, EURUSD = 10p, GBPUSD = 10p, EURGBP = 10p)
3. USDJPY 20-Pip with 5-Pip Majors (USDJPY = 20p, EURUSD = 5p, GBPUSD = 5p, EURGBP = 5p)
4. USDJPY Isolated Performance (USDJPY at 10p vs 20p)

Evaluated under Full Symmetric Spread (1.6p) and Fixed Margin Monte Carlo.
"""
import os
import sys
import time
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
ARTIFACTS_DIR = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05"

import goal79_two_account_portfolio as g79
import audit_and_retest_full_spread_and_fixed_mc as audited

SL_MULT = 1.0
TP_MULT = 0.8
SPREAD_PIPS = 1.6
RISK_PCT = 0.03


def run_experiment(cache, min_pips_dict, sl=1.0, tp=0.8, trading_days=6178.6):
    trades = audited.build_trades(cache, sl, tp, min_pips_dict, spread_pips=SPREAD_PIPS)
    
    # Portfolio simulation
    res, eq_b, eq_s, max_dd, _ = g79.simulate_two_account(
        trades, risk_pct=RISK_PCT, start_per_side=100.0,
        max_leverage=50.0, cap=10_000_000.0, track_detail=False
    )
    acc = res[res["accepted"]]
    n_exec = len(acc)
    tpd = n_exec / trading_days
    wr = 100.0 * (acc["net_r"] > 0).mean() if n_exec else 0.0
    avg_win = acc.loc[acc["net_r"] > 0, "net_r"].mean() if (acc["net_r"] > 0).any() else 0.0
    gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
    gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else 0.0
    
    med_days, med_dd = audited.run_mc_fixed(trades, n_sims=300)
    
    # Per-symbol breakdown
    sym_breakdown = {}
    for sym in min_pips_dict.keys():
        sub = acc[acc["symbol"] == sym]
        s_cnt = len(sub)
        s_wr = 100.0 * (sub["net_r"] > 0).mean() if s_cnt else 0.0
        s_avg_win = sub.loc[sub["net_r"] > 0, "net_r"].mean() if (sub["net_r"] > 0).any() else 0.0
        s_gw = sub.loc[sub["net_r"] > 0, "net_r"].sum()
        s_gl = -sub.loc[sub["net_r"] <= 0, "net_r"].sum()
        s_pf = s_gw / s_gl if s_gl > 0 else 0.0
        sym_breakdown[sym] = {
            "execs": s_cnt,
            "tpd": s_cnt / trading_days,
            "wr": s_wr,
            "avg_win": s_avg_win,
            "pf": s_pf
        }

    return {
        "n_exec": n_exec,
        "tpd": tpd,
        "wr": wr,
        "avg_win": avg_win,
        "pf": pf,
        "med_days": med_days,
        "med_dd": med_dd,
        "symbols": sym_breakdown,
        "trades": trades
    }


def main():
    print("Loading cache...")
    cache = audited.load_cache()
    all_times = [d["times"].min() for d in cache.values()] + [d["times"].max() for d in cache.values()]
    total_days = (max(all_times) - min(all_times)).total_seconds() / 86400.0
    trading_days = total_days * (5.0 / 7.0)

    configs_to_test = [
        ("Current PROD (All 10p)", {"USDJPY": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}, 1.0, 0.8),
        ("USDJPY=20p + Others=10p (1.0SL:0.8TP)", {"USDJPY": 20.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}, 1.0, 0.8),
        ("USDJPY=20p + Others=10p (1.0SL:1.0TP)", {"USDJPY": 20.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}, 1.0, 1.0),
        ("USDJPY=20p + Others=5p (1.0SL:0.8TP)", {"USDJPY": 20.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 0.8),
        ("USDJPY=20p + Others=5p (1.0SL:1.0TP)", {"USDJPY": 20.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 1.0),
    ]

    print("\n" + "=" * 115)
    print(f"  BACKTEST RESULTS: USDJPY = 20-PIP MINIMUM ZONE ({total_days/365.25:.1f} YRS DATA)")
    print("=" * 115)

    res_list = []
    for name, pips_dict, sl, tp in configs_to_test:
        r = run_experiment(cache, pips_dict, sl, tp, trading_days)
        res_list.append((name, r, pips_dict, sl, tp))
        print(f"\n>>> Configuration: {name}")
        print(f"  Total Execs: {r['n_exec']:,} | T/Day: {r['tpd']:.1f} | WR: {r['wr']:.2f}% | AvgWin: +{r['avg_win']:.2f}R | PF(1.6p): {r['pf']:.2f} | MC $1M: {r['med_days']:.1f}d | MC DD: {r['med_dd']:.1f}%")
        print("  Symbol Breakdown:")
        for sym, s_data in r['symbols'].items():
            print(f"    - {sym:<6} (min {pips_dict[sym]:>4.1f}p): {s_data['execs']:>6,} trades ({s_data['tpd']:>4.1f}/d) | WR: {s_data['wr']:>5.1f}% | AvgWin: +{s_data['avg_win']:.2f}R | PF: {s_data['pf']:.2f}")

    # Isolated USDJPY Comparison (10p vs 20p vs 30p)
    print("\n" + "=" * 115)
    print("  ISOLATED USDJPY DEEP DIVE (10p vs 15p vs 20p vs 25p vs 30p)")
    print("=" * 115)
    uj_pips = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    uj_cache = {"USDJPY": cache["USDJPY"]}
    
    for p in uj_pips:
        trades_uj = audited.build_trades(uj_cache, 1.0, 0.8, {"USDJPY": p}, spread_pips=1.6)
        res_uj, _, _, _, _ = g79.simulate_two_account(
            trades_uj, risk_pct=RISK_PCT, start_per_side=100.0,
            max_leverage=50.0, cap=10_000_000.0, track_detail=False
        )
        acc_uj = res_uj[res_uj["accepted"]]
        n_uj = len(acc_uj)
        wr_uj = 100.0 * (acc_uj["net_r"] > 0).mean() if n_uj else 0.0
        avg_w_uj = acc_uj.loc[acc_uj["net_r"] > 0, "net_r"].mean() if (acc_uj["net_r"] > 0).any() else 0.0
        gw_uj = acc_uj.loc[acc_uj["net_r"] > 0, "net_r"].sum()
        gl_uj = -acc_uj.loc[acc_uj["net_r"] <= 0, "net_r"].sum()
        pf_uj = gw_uj / gl_uj if gl_uj > 0 else 0.0
        print(f"  USDJPY min_zone={p:>4.1f}p | Execs: {n_uj:>6,} | Trades/Day: {n_uj/trading_days:>4.1f} | WR: {wr_uj:>5.1f}% | AvgWin: +{avg_w_uj:.2f}R | PF: {pf_uj:>4.2f}")


if __name__ == "__main__":
    main()
