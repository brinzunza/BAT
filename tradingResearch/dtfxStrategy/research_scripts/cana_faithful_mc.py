"""Same 25,000-path, $200-start, $2M-cap-with-vault-sweep, 5%-risk Monte Carlo, run on
the CORRECTED CANA trade pool from cana_faithful_backtest.py (both bugs fixed). Full
stats parity with goal_70pct_time_to_target_mc.py's report (percentiles included)."""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal_70pct_time_to_target_mc import simulate, N_PATHS, BLOCK_LEN, START_CAPITAL, CAP, \
    TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED

CANA_TRADES_PER_YEAR = 2848.0

pool_df = pd.read_csv(os.path.join(DATA_DIR, "cana_faithful_trades.csv"))
pool = np.ascontiguousarray(pool_df["net_r"].values, dtype=np.float64)

print("=" * 78)
print("CANA -- CORRECTED BACKTEST STATS (both bugs fixed)")
print("=" * 78)
years = (pd.to_datetime(pool_df["entry_time"]).max() - pd.to_datetime(pool_df["entry_time"]).min()).days / 365.25
wins = pool_df.loc[pool_df.net_r > 0, "net_r"].sum()
losses = -pool_df.loc[pool_df.net_r <= 0, "net_r"].sum()
pf = wins / losses
print(f"Trades: {len(pool_df):,}  ({len(pool_df)/years:,.0f}/yr over {years:.1f} yrs)")
print(f"Win rate: {100*pool_df['is_win'].mean():.2f}%   Profit factor: {pf:.3f}   "
      f"Avg R/trade: {pool_df['net_r'].mean():.4f}")
print("\nBy instrument:")
for inst, sub in pool_df.groupby("instrument"):
    w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
    yrs = (pd.to_datetime(sub.entry_time).max() - pd.to_datetime(sub.entry_time).min()).days / 365.25
    print(f"  {inst}: {len(sub):,} trades ({len(sub)/yrs:,.0f}/yr)  WR={100*sub.is_win.mean():.2f}%  PF={w/l:.3f}")
print("\nBy touch number:")
for t_num, sub in pool_df.groupby("touch_num"):
    w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
    pf_t = w / l if l > 0 else float("inf")
    print(f"  touch {t_num}: {len(sub):,} trades  WR={100*sub.is_win.mean():.2f}%  PF={pf_t:.3f}  "
          f"avg_r={sub.net_r.mean():.4f}")

print("\n" + "=" * 78)
print(f"MONTE CARLO: {N_PATHS:,} paths, ${START_CAPITAL:.0f} start, {RISK_PCT*100:.0f}% risk/trade, "
      f"1.6-pip spread (stop-outs only), ${CAP:,.0f} cap + vault sweep")
print(f"Frequency assumed: {CANA_TRADES_PER_YEAR:,.0f} trades/yr (this backtest's actual combined rate)")
print("=" * 78)

reach1, reach2, ruined = simulate(pool, len(pool), BLOCK_LEN, N_PATHS, START_CAPITAL, CAP,
                                   TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED)


def report(name, reach, target):
    hit = reach[reach > 0]
    n_hit = len(hit)
    n_ruined = int(((reach == -1) & ruined).sum())
    n_timeout = N_PATHS - n_hit - n_ruined
    months = hit / CANA_TRADES_PER_YEAR * 12.0
    print(f"\n--- Time to ${target:,.0f} net worth ---")
    print(f"  Reached: {n_hit:,}/{N_PATHS:,} ({100*n_hit/N_PATHS:.1f}%)   "
          f"Ruined: {n_ruined:,} ({100*n_ruined/N_PATHS:.2f}%)   "
          f"Timed out: {n_timeout:,} ({100*n_timeout/N_PATHS:.2f}%)")
    if n_hit == 0:
        return
    print(f"  Quickest (min):  {months.min():7.1f} months  ({months.min()/12:.2f} yr)")
    print(f"  Average (mean):  {months.mean():7.1f} months  ({months.mean()/12:.2f} yr)")
    print(f"  Median:          {np.median(months):7.1f} months  ({np.median(months)/12:.2f} yr)")
    print(f"  Slowest (max):   {months.max():7.1f} months  ({months.max()/12:.2f} yr)")
    print(f"  Range:           {months.max()-months.min():7.1f} months")
    for q in (5, 25, 75, 95):
        print(f"  p{q:<3d}:            {np.percentile(months, q):7.1f} months")


report("$1,000,000", reach1, TARGET1)
report("$10,000,000", reach2, TARGET2)

out = pd.DataFrame({"reach_1m_trades": reach1, "reach_10m_trades": reach2, "ruined": ruined})
out.to_csv(os.path.join(DATA_DIR, "cana_faithful_time_to_target_mc.csv"), index=False)
print(f"\nSaved: {os.path.join(DATA_DIR, 'cana_faithful_time_to_target_mc.csv')}")
