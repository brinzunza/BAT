"""Extend the faithful multi-touch framework to XAUUSD and BTCUSD -- untested levers
in the R:R>=1:1 search so far. Establishes baseline quality and each instrument's own
best hour-of-day window (analogous to how SOMEGA's hours were discovered for FX),
using pip/spread conventions already established elsewhere in this project
(dtfx_cross_asset_backtest.py): XAUUSD pip=0.1/spread=3.5pips ($0.35), BTCUSD
pip=1.0/spread=15pips ($15).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({
    "XAUUSD": "lse_xauusd_full_1min.csv",
    "BTCUSD": "lse_btcusd_1min_2018on.csv",
})
import prod_filter_grid_search as pg
pg.PIP_SIZE.update({"XAUUSD": 0.1, "BTCUSD": 1.0})
pg.SPREAD_PIPS_TYPICAL_OVERRIDE = {"XAUUSD": 3.5, "BTCUSD": 15.0}

# Patch build_trades to use per-instrument spread overrides
_orig_build = pg.build_trades
def build_trades_override(instrument, stop_mult, target_mult, min_fill_age_h, bounce, session, exc_mult=1.0):
    old_spread = pg.SPREAD_PIPS_TYPICAL
    pg.SPREAD_PIPS_TYPICAL = pg.SPREAD_PIPS_TYPICAL_OVERRIDE.get(instrument, old_spread)
    try:
        return _orig_build(instrument, stop_mult, target_mult, min_fill_age_h, bounce, session, exc_mult)
    finally:
        pg.SPREAD_PIPS_TYPICAL = old_spread
pg.build_trades = build_trades_override

NEW_INSTRUMENTS = ["XAUUSD", "BTCUSD"]
for inst in NEW_INSTRUMENTS:
    print(f"Loading {inst}...", flush=True)
    rs.load_instrument(inst)
    rs._prep_zones(inst, 2)

print("\n=== Baseline (1:1 R:R, N=2.0, age=3.37h, no session, no bounce) ===")
pools = {}
for inst in NEW_INSTRUMENTS:
    t = pg.build_trades(inst, 2.0, 2.0, 3.366682, False, False)
    pools[inst] = t
    years = (t.entry_time.max()-t.entry_time.min()).days/365.25
    w = t.loc[t.net_r>0,"net_r"].sum(); l = -t.loc[t.net_r<=0,"net_r"].sum()
    print(f"  {inst}: {len(t):,} trades ({len(t)/years:,.0f}/yr) WR={100*t.is_win.mean():.2f}% PF={w/l:.3f}")

print("\n=== Hour-of-day win rate breakdown (UTC) ===")
for inst in NEW_INSTRUMENTS:
    t = pools[inst]
    t = t.copy()
    t["hour"] = t.entry_time.dt.hour
    tab = t.groupby("hour").agg(trades=("is_win","size"), wr=("is_win","mean"))
    tab["wr"] = (tab.wr*100).round(1)
    print(f"\n{inst}:")
    print(tab.to_string())
    # chi-square test for hour-of-day significance
    ct = pd.crosstab(t["hour"], t["is_win"])
    chi2, p, dof, exp = chi2_contingency(ct)
    print(f"  chi2 p-value across all 24 hours: {p:.2e}")

print("\n=== XAUUSD: hours 12-17 UTC + stop ceiling (gold pips, 1 pip=$0.10) ===")
xau = pools["XAUUSD"].copy()
xau["hour"] = xau.entry_time.dt.hour
xau_session = xau[xau.hour.between(12, 17)]
years = (xau_session.entry_time.max()-xau_session.entry_time.min()).days/365.25
for ceiling in [None, 30, 20, 15, 10]:
    sub = xau_session if ceiling is None else xau_session[xau_session.stop_pips <= ceiling]
    if len(sub) < 30: continue
    w = sub.loc[sub.net_r>0,"net_r"].sum(); l = -sub.loc[sub.net_r<=0,"net_r"].sum()
    tag = "no ceiling" if ceiling is None else f"<={ceiling}pips(${ceiling*0.1:.2f})"
    print(f"  {tag:22s}: trades={len(sub):5,d} ({len(sub)/years:5,.0f}/yr) WR={100*sub.is_win.mean():.2f}% PF={(w/l if l>0 else float('inf')):.3f}")

print("\n=== BTCUSD: 24h (no session effect found) + stop ceiling (BTC pips, 1 pip=$1) ===")
btc = pools["BTCUSD"]
years = (btc.entry_time.max()-btc.entry_time.min()).days/365.25
for ceiling in [None, 200, 100, 50, 30]:
    sub = btc if ceiling is None else btc[btc.stop_pips <= ceiling]
    if len(sub) < 30: continue
    w = sub.loc[sub.net_r>0,"net_r"].sum(); l = -sub.loc[sub.net_r<=0,"net_r"].sum()
    tag = "no ceiling" if ceiling is None else f"<={ceiling}pips(${ceiling:.0f})"
    print(f"  {tag:22s}: trades={len(sub):5,d} ({len(sub)/years:5,.0f}/yr) WR={100*sub.is_win.mean():.2f}% PF={(w/l if l>0 else float('inf')):.3f}")
