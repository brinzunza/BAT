"""
Validates the new dtfx_engine.simulate_two_account / run_two_account_mc against the Python
originals in goal30_apex_king_deployment.py, on real BEID trade data (from this session's
beid_goal_hunt/ investigation -- a realistic, multi-instrument, multi-year trade set, not a
synthetic toy).

simulate_two_account is deterministic -> exact match required (accepted mask, equity_buy,
equity_sell, max_dd_pct all bit-for-bit... well, float equal within a tight tolerance for
max_dd_pct given summation-order differences are possible but should be negligible since both
implementations sum in the same chronological order).

run_two_account_mc uses a different RNG (std::mt19937_64 vs Python's np.random.default_rng) by
construction, so it's validated statistically: percentile agreement on median years-to-$1M,
P(reach), P(ruin), median drawdown at a large n_sims.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(SCRIPT_DIR, "..", "dtfxStrategy", "research_scripts")
BEID_DIR = os.path.join(RESEARCH_DIR, "beid_goal_hunt")
sys.path.insert(0, RESEARCH_DIR)
sys.path.insert(0, BEID_DIR)

import dtfx_engine
import goal30_apex_king_deployment as g30

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures += 1


INSTR_CODE = {"EURUSD": 0, "GBPUSD": 0, "EURGBP": 2, "USDJPY": 1}


def load_real_trades():
    full = pd.read_pickle(os.path.join(BEID_DIR, "legit_dataset.pkl"))
    mask = (full.zone_size_pips >= 15) & (full.prior1_dist_to_entry_pips >= 5)
    # kind='stable' to match goal30_apex_king_deployment.py's (now-fixed) internal re-sort --
    # tied entry_time values are common at 1-min granularity, and an unstable sort here would
    # make the comparison against the C++ port meaningless (each Python call could reorder
    # ties differently on its own, independent of the C++ port's correctness).
    sub = full[mask].sort_values("entry_time", kind="stable").reset_index(drop=True)
    print(f"Loaded {len(sub):,} real BEID trades for validation (zone>=15p, dist>=5p)")
    return sub


def to_engine_inputs(sub):
    entry_ns = sub.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = sub.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    instrument_code = sub.instrument.map(INSTR_CODE).values.astype(np.int32)
    entry_price = sub.entry_price.values.astype(np.float64)
    stop_risk = sub.stop_risk.values.astype(np.float64)
    net_r = sub.net_r.values.astype(np.float64)
    is_buy = sub.is_buy.values.astype(np.uint8)
    return entry_ns, exit_ns, instrument_code, entry_price, stop_risk, net_r, is_buy


def validate_deterministic(sub):
    print("\n" + "=" * 90 + "\nDETERMINISTIC: simulate_two_account (C++) vs simulate_two_account (Python)\n" + "=" * 90)
    risk_pct = 0.025

    t0 = time.time()
    t_acc, eq_buy_py, eq_sell_py, max_dd_py, trace_py = g30.simulate_two_account_python(sub, risk_pct)
    py_secs = time.time() - t0
    accepted_py = t_acc.accepted.values

    entry_ns, exit_ns, instrument_code, entry_price, stop_risk, net_r, is_buy = to_engine_inputs(sub)
    t0 = time.time()
    (accepted_cpp, eq_buy_cpp, eq_sell_cpp, max_dd_cpp,
     trace_ns_cpp, trace_buy_cpp, trace_sell_cpp) = dtfx_engine.simulate_two_account(
        entry_ns, exit_ns, instrument_code, entry_price, stop_risk, net_r, is_buy,
        risk_pct=risk_pct, start_per_side=100.0, max_leverage=50.0, cap=500000.0,
        gbp_usd_approx=1.36, track_detail=True)
    cpp_secs = time.time() - t0

    check("accepted mask matches exactly", np.array_equal(accepted_py, accepted_cpp),
          f"py sum={accepted_py.sum()}, cpp sum={accepted_cpp.sum()}")
    check("equity_buy matches", abs(eq_buy_py - eq_buy_cpp) < 1e-6 * max(1.0, abs(eq_buy_py)),
          f"py={eq_buy_py:.6f} cpp={eq_buy_cpp:.6f}")
    check("equity_sell matches", abs(eq_sell_py - eq_sell_cpp) < 1e-6 * max(1.0, abs(eq_sell_py)),
          f"py={eq_sell_py:.6f} cpp={eq_sell_cpp:.6f}")
    check("trace length matches", len(trace_py) == len(trace_ns_cpp),
          f"py={len(trace_py)} cpp={len(trace_ns_cpp)}")
    if len(trace_py) == len(trace_ns_cpp) and len(trace_py) > 0:
        py_close_ns = trace_py.close_time.values.astype("datetime64[ns]").astype(np.int64)
        check("trace close times match exactly", np.array_equal(py_close_ns, trace_ns_cpp))
        check("trace equity_buy matches", np.allclose(trace_py.equity_buy.values, trace_buy_cpp, atol=1e-6))
        check("trace equity_sell matches", np.allclose(trace_py.equity_sell.values, trace_sell_cpp, atol=1e-6))
    check("max_dd matches", abs(max_dd_py - max_dd_cpp) < 1e-4,
          f"py={max_dd_py:.6f}% cpp={max_dd_cpp:.6f}%")
    print(f"  timing: python={py_secs:.3f}s  cpp={cpp_secs:.4f}s  "
          f"speedup={py_secs/max(cpp_secs,1e-9):.1f}x")
    return t_acc[t_acc.accepted]


def validate_mc(accepted):
    print("\n" + "=" * 90 + "\nSTATISTICAL: run_two_account_mc (C++) vs run_mc (Python)\n" + "=" * 90)
    risk_pct = 0.025
    n_sims = 500
    max_trades = 200_000

    t0 = time.time()
    t1s_py, t10s_py, dds_py, ruins_py, accs_py = g30.run_mc_python(
        accepted, risk_pct, n_paths=n_sims, seed=42, max_trades=max_trades, ruin_floor=50.0)
    py_secs = time.time() - t0

    entry_ns, exit_ns, instrument_code, entry_price, stop_risk, net_r, is_buy = to_engine_inputs(accepted)
    t0 = time.time()
    t1s_cpp, t10s_cpp, dds_cpp, ruins_cpp, processed_cpp, accepted_n_cpp = dtfx_engine.run_two_account_mc(
        entry_ns, exit_ns, instrument_code, entry_price, stop_risk, net_r, is_buy,
        n_sims=n_sims, block=g30.BLOCK, max_trades=max_trades, seed=42, risk_pct=risk_pct,
        start_per_side=100.0, max_leverage=50.0, cap=500000.0, target1=1e6, target2=1e7,
        year_ns=g30.YEAR_NS, gbp_usd_approx=1.36, ruin_floor=50.0, n_threads=0)
    cpp_secs = time.time() - t0

    reached_py = t1s_py >= 0
    reached_cpp = t1s_cpp >= 0
    p_reach_py = 100 * reached_py.mean()
    p_reach_cpp = 100 * reached_cpp.mean()
    med_py = np.median(t1s_py[reached_py]) if reached_py.any() else float("inf")
    med_cpp = np.median(t1s_cpp[reached_cpp]) if reached_cpp.any() else float("inf")
    p_ruin_py = 100 * ruins_py.mean()
    p_ruin_cpp = 100 * ruins_cpp.mean()
    dd_med_py = np.median(dds_py)
    dd_med_cpp = np.median(dds_cpp)

    print(f"  P(reach $1M):   py={p_reach_py:.1f}%   cpp={p_reach_cpp:.1f}%")
    print(f"  median years:   py={med_py:.3f}   cpp={med_cpp:.3f}")
    print(f"  P(ruin):        py={p_ruin_py:.2f}%   cpp={p_ruin_cpp:.2f}%")
    print(f"  median MC DD:   py={dd_med_py:.2f}%   cpp={dd_med_cpp:.2f}%")
    print(f"  timing: python={py_secs:.1f}s  cpp={cpp_secs:.2f}s  speedup={py_secs/max(cpp_secs,1e-9):.1f}x")

    check("P(reach $1M) within 10 pct points", abs(p_reach_py - p_reach_cpp) <= 10.0,
          f"py={p_reach_py:.1f}% cpp={p_reach_cpp:.1f}%")
    if reached_py.any() and reached_cpp.any():
        check("median years within 20% relative", abs(med_py - med_cpp) <= 0.2 * max(med_py, med_cpp),
              f"py={med_py:.3f} cpp={med_cpp:.3f}")
    check("P(ruin) within 5 pct points", abs(p_ruin_py - p_ruin_cpp) <= 5.0,
          f"py={p_ruin_py:.2f}% cpp={p_ruin_cpp:.2f}%")
    check("median drawdown within 5 pct points", abs(dd_med_py - dd_med_cpp) <= 5.0,
          f"py={dd_med_py:.2f}% cpp={dd_med_cpp:.2f}%")


if __name__ == "__main__":
    sub = load_real_trades()
    accepted = validate_deterministic(sub)
    validate_mc(accepted)

    print("\n" + "=" * 90)
    if failures == 0:
        print(f"ALL CHECKS PASSED")
    else:
        print(f"{failures} CHECK(S) FAILED")
    print("=" * 90)
    sys.exit(1 if failures else 0)
