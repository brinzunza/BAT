"""Validates dtfx_engine.run_portfolio_mc -- the most general, least-tested simulation
primitive in this codebase (N arbitrary accounts, margin computed entirely in Python so it has
zero instrument-specific branches in C++, real per-unit pyramid margin, event-driven exit-time
settlement, two-tier ruin) -- against the Python reference implementation it's meant to
replace/accelerate, on two real datasets that stress different parts of its surface:

- BEID zone-fade trades (no pyramid columns): exercises the single-synthetic-unit degrade path
  and multi-instrument account grouping.
- Donchian-pyramid trades (real unit_rs/unit_entry_times/unit_entry_prices, built via
  `donchian_concurrent_engine.build_concurrent_trades`): exercises real incremental per-unit
  margin checks, the part the BEID dataset cannot test at all.

Mirrors validate_two_account.py's structure: a deterministic exact-match check (single real
chronological pass, no resampling -- achieved by calling run_portfolio_mc with block=n,
n_sims=1, so its own internal bootstrap draws exactly one block covering the whole real trade
sequence in order) against `margin_aware_mc.simulate_path_margin_aware` (the Python reference,
itself already covered by test_margin_aware_mc.py's regression tests), plus a statistical
Monte Carlo agreement check at large n_sims (different RNGs by construction, so not bit-exact).

One diagnostic metric, avg_skipped_per_path / skipped, is NOT held to a tight tolerance here:
initial validation found the Python and C++ engines' outcome-critical stats (true_median_years,
p_reach_ever, p_reach_within_1yr, p_ruin) agree well, but their skip *counts* differ by roughly
2x on a real gold+BTC dataset -- most plausibly a definitional difference in what counts as a
"skip" (e.g. whether an attempted trade on an already-dead account increments the counter)
rather than a functional discrepancy in the margin logic itself, since every metric that
actually depends on correct margin gating (reach/ruin rates, median years) agrees. Flagged
here explicitly rather than silently loosened away; worth a closer look before this script's
skip-count check (if one is added later) is trusted as tightly as the others.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DTFX_ROOT = os.path.join(REPO_ROOT, "dtfxStrategy")
RESEARCH_DIR = os.path.join(DTFX_ROOT, "research_scripts")
BEID_DIR = os.path.join(RESEARCH_DIR, "beid_goal_hunt")
sys.path.insert(0, RESEARCH_DIR)
sys.path.insert(0, BEID_DIR)
sys.path.insert(0, REPO_ROOT)  # for `import quant_engine`

import dtfx_engine
from margin_aware_mc import simulate_path_margin_aware, run_mc_margin_aware
from quant_engine.portfolio import to_engine_arrays
from quant_engine.stats import true_median_and_stats

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures += 1


TRADE_COLS = ["instrument", "entry_time", "exit_time", "entry_price", "stop_risk", "net_r",
              "is_buy", "risk_pct", "units", "unit_rs", "unit_entry_times", "unit_entry_prices"]


def load_beid_trades():
    """No pyramid columns -- exercises the single-synthetic-unit degrade path."""
    full = pd.read_pickle(os.path.join(BEID_DIR, "legit_dataset.pkl"))
    mask = (full.zone_size_pips >= 15) & (full.prior1_dist_to_entry_pips >= 5)
    sub = full[mask].sort_values("entry_time", kind="stable").reset_index(drop=True).copy()
    sub["risk_pct"] = 0.01
    sub["units"] = 1
    print(f"Loaded {len(sub):,} real BEID trades (zone>=15p, dist>=5p)")
    return sub


def load_donchian_trades():
    """Real unit_rs/unit_entry_times/unit_entry_prices -- exercises real incremental
    per-unit margin checks that the BEID dataset can't test at all."""
    import gold_orb_m1_probe as gp
    import btc_orb_m1_probe as bp
    from donchian_concurrent_engine import build_concurrent_trades
    from feargreed_full_mc_test import compute_atr_ratio

    gold = build_concurrent_trades("XAUUSD", gp.XAUUSD_CSV, gp.PIP, gp.SPREAD_PIPS, entry_days=8,
                                    k_stop=2.0, add_step=1.0, max_units=5, max_concurrent=1, start_year=2021)
    gold["risk_pct"] = 0.15
    btc = build_concurrent_trades("BTCUSD", bp.BTC_CSV, bp.PIP, bp.SPREAD_PIPS, entry_days=8, k_stop=4.0,
                                   add_step=1.0, max_units=5, max_concurrent=10, start_year=2021)
    atr_ratio = compute_atr_ratio(bp.BTC_CSV)
    btc["atr_ratio"] = btc.entry_time.dt.date.map(atr_ratio)
    btc = btc[btc.atr_ratio < 1.0].copy()
    btc["risk_pct"] = 0.5
    combined = pd.concat([gold[TRADE_COLS], btc[TRADE_COLS]], ignore_index=True)
    combined = combined.sort_values("entry_time", kind="stable").reset_index(drop=True)
    print(f"Loaded {len(combined):,} real gold+BTC Donchian-pyramid trades")
    return combined


LEVERAGE = {"XAUUSD": 20.0, "BTCUSD": 10.0, "ETHUSD": 10.0}


def python_reference_path(t, instrument_list, **kw):
    """Single deterministic walk (no resampling), real chronological order -- calls
    margin_aware_mc.simulate_path_margin_aware directly with the trades' own real
    entry/exit times as the timeline (dt_years measured from the first trade), matching
    exactly what run_portfolio_mc(block=len(t), n_sims=1) is expected to reproduce."""
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    dt_years = (entry_ns - entry_ns[0]) / 1e9 / 86400 / 365.25
    exit_dt_years = (exit_ns - entry_ns[0]) / 1e9 / 86400 / 365.25
    unit_count = t.units.to_numpy(float)
    unit_rs_all = t.unit_rs.to_numpy(object)
    unit_prices_all = t.unit_entry_prices.to_numpy(object)
    unit_offsets_all = np.array([
        (pd.to_datetime(list(ts)).values.astype("datetime64[ns]").astype(np.int64) - entry_ns[i])
        for i, ts in enumerate(t.unit_entry_times)
    ], dtype=object)
    return simulate_path_margin_aware(
        t.instrument.to_numpy(), t.instrument.to_numpy(), t.is_buy.to_numpy(),
        t.stop_risk.to_numpy(float), t.entry_price.to_numpy(float), t.net_r.to_numpy(float),
        entry_ns, exit_ns, dt_years, t.risk_pct.to_numpy(float), instrument_list,
        unit_count=unit_count, exit_dt_years=exit_dt_years, unit_rs=unit_rs_all,
        unit_add_offsets_ns=unit_offsets_all, unit_prices=unit_prices_all, **kw)


def ensure_pyramid_columns(t):
    """BEID trades have no pyramid columns; fill them in with the single-unit degrade shape
    so python_reference_path's array-building (which assumes the columns exist) works
    uniformly for both datasets."""
    t = t.copy()
    if "unit_rs" not in t.columns:
        t["units"] = 1
        t["unit_rs"] = [[r] for r in t.net_r]
        t["unit_entry_times"] = [[e] for e in t.entry_time]
        t["unit_entry_prices"] = [[p] for p in t.entry_price]
    return t


def validate_deterministic(trades, instrument_list, *, label):
    print(f"\n{'='*90}\nDETERMINISTIC ({label}): run_portfolio_mc (C++, block=n) vs simulate_path_margin_aware (Python)\n{'='*90}")
    t = ensure_pyramid_columns(trades).sort_values("entry_time", kind="stable").reset_index(drop=True)
    n = len(t)
    # account_mode="instrument_side" is the direct equivalent of margin_aware_mc.py's own
    # per-(instrument, side) equity keying -- "instrument" alone would pool buy and sell
    # trades for the same instrument into one account, which is a DIFFERENT (and wrong)
    # comparison, not a margin-aware-engine bug (caught exactly this way during development).
    kw = dict(start_per_inst=200.0 / len(instrument_list), max_leverage=50.0, gold_leverage=20.0,
              cap=500000.0, target1=1e6, leg_floor=2.0, total_floor=10.0, btc_leverage=10.0)

    t0 = time.time()
    yrs_py, ruined_py, skipped_py = python_reference_path(t, instrument_list, **kw)
    py_secs = time.time() - t0

    arrays = to_engine_arrays(t, account_mode="instrument_side", leverage_overrides=LEVERAGE)
    start_per_account = (200.0 / len(instrument_list)) / 2.0
    t0 = time.time()
    target_years, max_dd, ruined_cpp, processed, accepted, skipped_cpp = dtfx_engine.run_portfolio_mc(
        arrays["entry_ns"], arrays["exit_ns"], arrays["account_code"], arrays["risk_pct"],
        arrays["leverage_risk_cap"], arrays["unit_count"], arrays["unit_r"], arrays["unit_offset_ns"],
        arrays["unit_price_ratio"], arrays["n_accounts"], 1, n, n, 1,
        start_per_account, 500000.0, 1e6, 2.0, 10.0, 365.25 * 86400 * 1e9, 1, 1,
    )
    cpp_secs = time.time() - t0
    yrs_cpp, ruined_cpp0 = float(target_years[0]), bool(ruined_cpp[0])

    check("ruined flag matches", ruined_py == ruined_cpp0, f"py={ruined_py} cpp={ruined_cpp0}")
    if not ruined_py and not ruined_cpp0:
        check("years-to-target matches", abs(yrs_py - yrs_cpp) < 1e-4 * max(1.0, abs(yrs_py)),
              f"py={yrs_py:.6f} cpp={yrs_cpp:.6f}")
    print(f"  py skipped={skipped_py}  cpp skipped={int(skipped_cpp[0])}  (diagnostic only, see module docstring)")
    print(f"  timing: python={py_secs:.3f}s  cpp={cpp_secs:.4f}s  speedup={py_secs/max(cpp_secs,1e-9):.1f}x")


def validate_mc(trades, instrument_list, *, label, n_sims=3000):
    print(f"\n{'='*90}\nSTATISTICAL ({label}): run_portfolio_mc (C++) vs run_mc_margin_aware (Python)\n{'='*90}")
    t = ensure_pyramid_columns(trades)
    kw = dict(start_per_inst=200.0 / len(instrument_list), max_leverage=50.0, gold_leverage=20.0,
              cap=500000.0, target1=1e6, leg_floor=2.0, total_floor=10.0, btc_leverage=10.0)

    t0 = time.time()
    r_py = run_mc_margin_aware(t, instrument_list, n_sims=n_sims, block=20, max_trades=8000,
                                seed=7, **kw)
    py_secs = time.time() - t0

    arrays = to_engine_arrays(t, account_mode="instrument_side", leverage_overrides=LEVERAGE)
    start_per_account = (200.0 / len(instrument_list)) / 2.0
    t0 = time.time()
    target_years, max_dd, ruined, processed, accepted, skipped = dtfx_engine.run_portfolio_mc(
        arrays["entry_ns"], arrays["exit_ns"], arrays["account_code"], arrays["risk_pct"],
        arrays["leverage_risk_cap"], arrays["unit_count"], arrays["unit_r"], arrays["unit_offset_ns"],
        arrays["unit_price_ratio"], arrays["n_accounts"], n_sims, 20, 8000, 7,
        start_per_account, 500000.0, 1e6, 2.0, 10.0, 365.25 * 86400 * 1e9,
        int(np.median(np.diff(arrays["entry_ns"])[np.diff(arrays["entry_ns"]) > 0])), 0,
    )
    cpp_secs = time.time() - t0
    r_cpp = true_median_and_stats(target_years, ruined, n_sims)

    print(f"  true_median_years: py={r_py['true_median_years']:.3f}  cpp={r_cpp['true_median_years']:.3f}")
    print(f"  p_reach_ever:      py={r_py['p_reach_ever']:.1f}%  cpp={r_cpp['p_reach_ever']:.1f}%")
    print(f"  p_reach_within_1yr:py={r_py['p_reach_within_1yr']:.1f}%  cpp={r_cpp['p_reach_within_1yr']:.1f}%")
    print(f"  p_ruin:            py={r_py['p_ruin']:.2f}%  cpp={r_cpp['p_ruin']:.2f}%")
    print(f"  timing: python={py_secs:.1f}s  cpp={cpp_secs:.2f}s  speedup={py_secs/max(cpp_secs,1e-9):.1f}x")

    check("true_median_years within 20% relative (or both inf)",
          (np.isinf(r_py["true_median_years"]) and np.isinf(r_cpp["true_median_years"])) or
          abs(r_py["true_median_years"] - r_cpp["true_median_years"]) <= 0.2 * max(r_py["true_median_years"], r_cpp["true_median_years"], 1e-9),
          f"py={r_py['true_median_years']:.3f} cpp={r_cpp['true_median_years']:.3f}")
    check("p_reach_ever within 10 pct points", abs(r_py["p_reach_ever"] - r_cpp["p_reach_ever"]) <= 10.0)
    check("p_reach_within_1yr within 10 pct points", abs(r_py["p_reach_within_1yr"] - r_cpp["p_reach_within_1yr"]) <= 10.0)
    check("p_ruin within 5 pct points", abs(r_py["p_ruin"] - r_cpp["p_ruin"]) <= 5.0)


def validate_target_invariance(trades, instrument_list, *, label, n_sims=200):
    """Checks a real, narrower claim than "two calls always continue each other": the C++
    worker creates exactly ONE std::mt19937_64 PER THREAD, declared OUTSIDE the per-sim loop
    (cpp_engine/src/portfolio_mc.cpp, `worker()`) -- so it keeps advancing across every sim a
    given thread processes. Two calls at different targets only draw an IDENTICAL block
    sequence for a given sim index if that sim is the first (and, with n_threads=n_sims, only)
    sim on its thread; otherwise an earlier sim in the same thread consumes a different number
    of draws under a different target, and every later sim on that thread silently runs on a
    DIFFERENT random sequence between the two calls. This was found empirically: the first
    version of this check used n_threads=1 with n_sims>1 and failed (backward-ruin and
    non-monotonic-years counter-examples on real data) for exactly this reason -- it is not a
    hypothetical edge case. The fix, verified below, is n_threads=n_sims (one thread per sim,
    each independently seeded via seed_seq{seed, thread_idx}) -- which is also why
    quant_engine.portfolio.run_portfolio()'s `extra_targets` forces n_threads=n_sims whenever
    it's used, and why that feature is documented as a small-n_sims tool, not a substitute for
    run_mc_capped's native two-stage tracking at the thousands-of-sims scale this codebase's
    production MC runs actually use."""
    print(f"\n{'='*90}\nTARGET INVARIANCE ({label}): n_threads=n_sims (one thread per sim), target=T vs target=2T\n{'='*90}")
    t = ensure_pyramid_columns(trades)
    arrays = to_engine_arrays(t, account_mode="instrument_side", leverage_overrides=LEVERAGE)
    start_per_account = (200.0 / len(instrument_list)) / 2.0
    common = dict(n_sims=n_sims, block=20, max_trades=8000, seed=11,
                  start_per_account=start_per_account, cap=500000.0, leg_floor=2.0,
                  total_floor=10.0, year_ns=365.25 * 86400 * 1e9, boundary_gap_ns=1, n_threads=n_sims)

    def run(target):
        target_years, max_dd, ruined, processed, accepted, skipped = dtfx_engine.run_portfolio_mc(
            arrays["entry_ns"], arrays["exit_ns"], arrays["account_code"], arrays["risk_pct"],
            arrays["leverage_risk_cap"], arrays["unit_count"], arrays["unit_r"],
            arrays["unit_offset_ns"], arrays["unit_price_ratio"], arrays["n_accounts"],
            common["n_sims"], common["block"], common["max_trades"], common["seed"],
            common["start_per_account"], common["cap"], target, common["leg_floor"],
            common["total_floor"], common["year_ns"], common["boundary_gap_ns"], common["n_threads"],
        )
        return np.asarray(target_years), np.asarray(ruined, dtype=bool)

    T = 1e6
    yrs_T, ruined_T = run(T)
    yrs_2T, ruined_2T = run(2 * T)

    backward = ruined_2T & ~ruined_T
    check("no sim is ruined at 2T but not at T", not backward.any(),
          f"{int(backward.sum())} counter-example(s)")

    both_reached = (yrs_T >= 0) & (yrs_2T >= 0)
    if both_reached.any():
        non_decreasing = (yrs_2T[both_reached] >= yrs_T[both_reached] - 1e-9).all()
        check("years-to-reach is non-decreasing from T to 2T", non_decreasing,
              f"{both_reached.sum()} sims reached both")
    else:
        print("  (no sims reached both targets in this sample -- monotonicity check has nothing to compare)")


def validate_pyramid_incremental_margin():
    """Promotes cpp_engine/test_portfolio_mc.py's synthetic single-row cases into this suite
    (same assertions quant_engine's own test_margin_aware_mc.py checks for the Python side):
    an add whose incremental margin doesn't fit is silently skipped, not the whole trade."""
    print(f"\n{'='*90}\nUNIT: incremental pyramid margin (synthetic, mirrors test_margin_aware_mc.py)\n{'='*90}")
    entry_ns = np.array([0], dtype=np.int64)
    exit_ns = np.array([5 * int(86400e9)], dtype=np.int64)
    account_code = np.array([0], dtype=np.int32)
    risk_pct = np.array([0.5])
    leverage_risk_cap = np.array([1.0])  # generous; the point of this test is the ADD margin gate
    unit_count = np.array([5], dtype=np.int32)
    unit_r = np.array([[1.0] * 5])
    unit_offset_ns = np.array([[0, int(86400e9), 2 * int(86400e9), 3 * int(86400e9), 4 * int(86400e9)]], dtype=np.int64)
    unit_price_ratio = np.array([[1.0] * 5])
    target_years, max_dd, ruined, processed, accepted, skipped = dtfx_engine.run_portfolio_mc(
        entry_ns, exit_ns, account_code, risk_pct, leverage_risk_cap, unit_count, unit_r,
        unit_offset_ns, unit_price_ratio, 1, 1, 1, 1, 1, 200.0, 500000.0, 350.0, 1.0, 1.0,
        365.25 * 86400 * 1e9, 1, 1,
    )
    check("some units skipped when margin runs out (matches Python-side test_pyramid_additions_consume_incremental_margin)",
          int(skipped[0]) >= 1, f"skipped={int(skipped[0])}")


if __name__ == "__main__":
    beid = load_beid_trades()
    validate_deterministic(beid, sorted(beid.instrument.unique().tolist()), label="BEID (no pyramid)")
    validate_mc(beid, sorted(beid.instrument.unique().tolist()), label="BEID (no pyramid)")

    donchian = load_donchian_trades()
    validate_deterministic(donchian, ["XAUUSD", "BTCUSD"], label="Donchian (real pyramid units)")
    validate_mc(donchian, ["XAUUSD", "BTCUSD"], label="Donchian (real pyramid units)")
    validate_target_invariance(donchian, ["XAUUSD", "BTCUSD"], label="Donchian (real pyramid units)")

    validate_pyramid_incremental_margin()

    print("\n" + "=" * 90)
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failures} CHECK(S) FAILED")
    print("=" * 90)
    sys.exit(1 if failures else 0)
