"""Component-level speed comparison: Python/numba vs C++, on real EURUSD data."""
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(SCRIPT_DIR, "..", "research_scripts")
sys.path.insert(0, RESEARCH_DIR)

import dtfx_engine
import dtfx_robust_search as rs
import prod_filter_grid_search as pg
from goal10_cana_nt_mc_capped import run_mc_capped_numba, BLOCK, MAX_TRADES, START, RISK_FRAC, CAP, T1, T10, RUIN_FLOOR

times, lows, highs, df = rs.load_instrument("EURUSD")
times_ns = times.astype("datetime64[ns]").astype(np.int64)
o = df["open"].values.astype(np.float64)
h = df["high"].values.astype(np.float64)
l = df["low"].values.astype(np.float64)
c = df["close"].values.astype(np.float64)

print(f"EURUSD: {len(df):,} bars\n")

# --- 1. Zone detection ---
t0 = time.perf_counter()
zigzag = rs.find_pivots_strict(df, 2)
_, zones = rs.run_structure_fsm(zigzag)
t_py = time.perf_counter() - t0
print(f"Zone detection  python: {t_py:8.3f}s  ({len(zones):,} zones)")

t0 = time.perf_counter()
z = dtfx_engine.detect_zones(o, h, l, c, times_ns, 2)
t_cpp = time.perf_counter() - t0
print(f"Zone detection  C++:    {t_cpp:8.3f}s  ({len(z[0]):,} zones)  -> {t_py/t_cpp:.1f}x")

# --- 2. Backtest engine (Hybrid-Ultra config) ---
zp = rs._prep_zones("EURUSD", pg.MIN_RETRACEMENT)
created_ns = zp["created_at"].astype("datetime64[ns]").astype("int64")
stop_mult, target_mult, age_h, touches = 6.0, 2.4, 2.0, 4
risk_pct = (stop_mult * (zp["top"] - zp["bottom"]) / 2.0) / zp["mid"]
keep = risk_pct >= pg.MIN_RISK_PCT
top, bottom, mid = zp["top"][keep], zp["bottom"][keep], zp["mid"][keep]
is_buy, start_idx, c_ns = zp["is_buy"][keep], zp["start_idx"][keep], created_ns[keep]
min_fill_age_ns = int(age_h * 3600 * 1e9)
args = (top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs, min_fill_age_ns,
        stop_mult, target_mult, False, pg.PROBE_DEPTH, 1.0, touches, pg.MAX_WINDOW_BARS)

pg._simulate_generic_numba(*args)  # warm up numba JIT (first call compiles)
t0 = time.perf_counter()
r_py = pg._simulate_generic_numba(*args)
t_py = time.perf_counter() - t0
print(f"\nBacktest engine python (numba, warm): {t_py*1000:8.2f}ms  ({len(r_py[0]):,} trades)")

t0 = time.perf_counter()
r_cpp = dtfx_engine.simulate_generic(*args)
t_cpp = time.perf_counter() - t0
print(f"Backtest engine C++:                  {t_cpp*1000:8.2f}ms  ({len(r_cpp[0]):,} trades)  -> {t_py/t_cpp:.1f}x")

# --- 3. Monte Carlo ---
touch, entry_idx, is_win, mid_out, stop_risk = r_py
r_gross = np.where(is_win, target_mult / stop_mult, -1.0)
spread_price = pg.SPREAD_PIPS_TYPICAL * pg.PIP_SIZE["EURUSD"]
cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
net_r = (r_gross - cost_in_r).astype(np.float64)

n_sims = 25_000
run_mc_capped_numba(net_r, 10, BLOCK, MAX_TRADES, seed=1)  # warm up
t0 = time.perf_counter()
run_mc_capped_numba(net_r, n_sims, BLOCK, MAX_TRADES, seed=42)
t_py = time.perf_counter() - t0
print(f"\nMonte Carlo ({n_sims:,} sims) python (numba, warm, 1 thread): {t_py:8.3f}s")

t0 = time.perf_counter()
dtfx_engine.run_mc_capped(net_r, n_sims, BLOCK, MAX_TRADES, 42, start=START, risk_frac=RISK_FRAC,
                           cap=CAP, target1=T1, target2=T10, ruin_floor=RUIN_FLOOR)
t_cpp = time.perf_counter() - t0
import os as _os
print(f"Monte Carlo ({n_sims:,} sims) C++ ({_os.cpu_count()} threads):        {t_cpp:8.3f}s  -> {t_py/t_cpp:.1f}x")
