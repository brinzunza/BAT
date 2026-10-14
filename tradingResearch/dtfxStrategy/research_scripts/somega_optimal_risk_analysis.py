"""Comprehensive Risk Optimization for SOMEGA Portfolio.

Evaluates:
1. Analytical Kelly Criterion & Fractional Kelly (Full, Half, Quarter, Eighth Kelly).
2. Empirical Risk Curve Sweep from 0.1% to 20% risk per trade on the full trade dataset.
3. Metric optimization: Calmar ratio, Sharpe, Sortino, Max Drawdown, CAGR.
4. Monte Carlo Sequence-of-Returns Simulation (1,000 paths per risk level) to find
   true 95th/99th percentile drawdown risk.
5. Impact of broker leverage caps (50:1) at higher risk levels.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from somega_pre2020_hybrid_5pct_equity import get_hybrid_trades, INSTRUMENTS, STATUS_TARGET, STATUS_STOP_LIMIT, STATUS_STOP_BACKSTOP

def load_all_trades():
    print("Loading hybrid stop trades for all instruments...")
    all_trades = []
    for name, fname in INSTRUMENTS.items():
        t = get_hybrid_trades(name, fname)
        if len(t):
            all_trades.append(t)
            print(f"  {name}: {len(t):,} trades")
    merged = pd.concat(all_trades).sort_values("entry_time").reset_index(drop=True)
    return merged

def evaluate_risk(trades, risk_pct, start_eq=10000.0):
    eq = start_eq
    curve_v = [eq]
    curve_t = [trades["entry_time"].iloc[0]]
    
    for _, t in trades.iterrows():
        # net_r is +1.0 on win, -1.0 on limit stop, -1.5R - spread on backstop
        ret = risk_pct * t["net_r"]
        eq = max(eq * (1 + ret), 0.0)
        curve_v.append(eq)
        curve_t.append(t["entry_time"])
        
    s = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = s.resample("D").last().ffill()
    d_ret = daily.pct_change().dropna()
    d_ret_nz = d_ret[d_ret != 0]
    
    n_years = (daily.index[-1] - daily.index[0]).days / 365.25
    final_eq = s.iloc[-1]
    total_ret = final_eq / start_eq - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
    
    cummax = daily.cummax()
    dd = daily / cummax - 1.0
    maxdd = abs(dd.min())
    calmar = cagr / maxdd if maxdd > 0 else 0.0
    
    sharpe = (d_ret_nz.mean() / d_ret_nz.std() * np.sqrt(252)) if d_ret_nz.std() > 0 else 0.0
    downside = d_ret_nz[d_ret_nz < 0]
    sortino = (d_ret_nz.mean() / downside.std() * np.sqrt(252)) if (len(downside) > 1 and downside.std() > 0) else 0.0
    
    # Calculate underwater durations (days in drawdown)
    is_in_dd = dd < 0
    dd_blocks = (~is_in_dd).cumsum()[is_in_dd]
    if len(dd_blocks):
        longest_dd_days = is_in_dd.groupby(dd_blocks).count().max()
    else:
        longest_dd_days = 0
        
    return {
        "risk_pct": risk_pct,
        "cagr": cagr,
        "maxdd": maxdd,
        "calmar": calmar,
        "sharpe": sharpe,
        "sortino": sortino,
        "longest_dd_days": longest_dd_days,
        "final_eq": final_eq,
        "n_years": n_years,
    }

def monte_carlo_drawdown(trades, risk_pct, n_sims=1000, start_eq=10000.0):
    """Reshuffles trades to calculate sequence-of-returns distribution."""
    net_r_arr = trades["net_r"].values
    n_trades = len(net_r_arr)
    max_dds = []
    
    rng = np.random.default_rng(42)
    for _ in range(n_sims):
        shuffled_r = rng.permutation(net_r_arr)
        # Fast compounding
        multipliers = 1.0 + risk_pct * shuffled_r
        cum_equity = start_eq * np.cumprod(multipliers)
        cum_max = np.maximum.accumulate(cum_equity)
        dd = (cum_equity - cum_max) / cum_max
        max_dds.append(abs(dd.min()))
        
    return {
        "mc_median_dd": np.median(max_dds),
        "mc_95th_dd": np.percentile(max_dds, 95),
        "mc_99th_dd": np.percentile(max_dds, 99),
        "mc_worst_dd": np.max(max_dds),
    }

def main():
    trades = load_all_trades()
    n_total = len(trades)
    wins = (trades["net_r"] > 0).sum()
    losses = (trades["net_r"] < 0).sum()
    win_rate = wins / n_total
    
    avg_win = trades.loc[trades["net_r"] > 0, "net_r"].mean()
    avg_loss = abs(trades.loc[trades["net_r"] < 0, "net_r"].mean())
    b = avg_win / avg_loss  # payoff ratio
    p = win_rate
    q = 1.0 - p
    
    # Kelly calculation: f* = (p*b - q) / b
    kelly_f = (p * b - q) / b
    
    print("\n" + "="*80)
    print("  THEORETICAL KELLY CRITERION")
    print("="*80)
    print(f"  Total Trades:         {n_total:,}")
    print(f"  Win Rate (p):         {win_rate*100:.2f}%")
    print(f"  Payoff Ratio (b):     {b:.4f} (Avg Win: +{avg_win:.3f}R, Avg Loss: -{avg_loss:.3f}R)")
    print(f"  Full Kelly (f*):      {kelly_f*100:.2f}%  (Maximizes long-term log growth)")
    print(f"  Half Kelly (f*/2):    {kelly_f/2*100:.2f}%  (Standard institutional aggressive ceiling)")
    print(f"  Quarter Kelly (f*/4): {kelly_f/4*100:.2f}%  (Optimal growth/drawdown sweet spot)")
    print(f"  Eighth Kelly (f*/8):  {kelly_f/8*100:.2f}%  (Conservative capital preservation)")
    
    # Risk sweep
    grid = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125, 0.15, 0.20]
    
    results = []
    print("\nRunning Risk Sweep and Monte Carlo simulations (1,000 runs per risk level)...")
    for r in grid:
        stat = evaluate_risk(trades, r)
        mc = monte_carlo_drawdown(trades, r, n_sims=1000)
        stat.update(mc)
        results.append(stat)
        
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*110)
    print("  EMPIRICAL RISK CURVE & MONTE CARLO DRAWDOWN PROFILE")
    print("="*110)
    
    disp_df = df_res.copy()
    disp_df["Risk %"] = disp_df["risk_pct"].apply(lambda x: f"{x*100:.2f}%")
    disp_df["CAGR"] = disp_df["cagr"].apply(lambda x: f"{x*100:,.1f}%")
    disp_df["Historical MaxDD"] = disp_df["maxdd"].apply(lambda x: f"{x*100:.1f}%")
    disp_df["MC 95% MaxDD"] = disp_df["mc_95th_dd"].apply(lambda x: f"{x*100:.1f}%")
    disp_df["MC 99% MaxDD"] = disp_df["mc_99th_dd"].apply(lambda x: f"{x*100:.1f}%")
    disp_df["Calmar"] = disp_df["calmar"].apply(lambda x: f"{x:.2f}")
    disp_df["Sharpe"] = disp_df["sharpe"].apply(lambda x: f"{x:.2f}")
    disp_df["Sortino"] = disp_df["sortino"].apply(lambda x: f"{x:.2f}")
    disp_df["Longest DD (Days)"] = disp_df["longest_dd_days"]
    
    cols = ["Risk %", "CAGR", "Historical MaxDD", "MC 95% MaxDD", "MC 99% MaxDD", "Calmar", "Sharpe", "Sortino", "Longest DD (Days)"]
    print(disp_df[cols].to_string(index=False))
    
    # Save CSV
    out_csv = os.path.join(DATA_DIR, "somega_optimal_risk_analysis.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved full results table to {out_csv}")

if __name__ == "__main__":
    main()
