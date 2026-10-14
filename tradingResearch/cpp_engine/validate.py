"""Validates the C++ dtfx_engine extension against the existing Python/numba implementations
on real instrument data. Run after any change to cpp_engine/src/, and before trusting the
delegation wired into dtfx_robust_search.py / prod_filter_grid_search.py / goal10_cana_nt_mc_capped.py.

Deterministic checks (zone detection, backtest engine) require EXACT array equality -- these
are pure functions of the same inputs, any mismatch is a real bug. Monte Carlo uses a
different RNG (std::mt19937_64 vs numba/NumPy's MT19937) by construction, so it's validated
statistically instead: percentile/summary agreement at large n_sims, not bit-exact paths.
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DIR = os.path.join(SCRIPT_DIR, "..", "dtfxStrategy", "research_scripts")
sys.path.insert(0, RESEARCH_DIR)

import dtfx_engine
import dtfx_robust_search as rs
import prod_filter_grid_search as pg
from goal10_cana_nt_mc_capped import run_mc_capped_numba, BLOCK, MAX_TRADES

INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]  # pg import already registers USDJPY

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures += 1


def times_to_ns(times):
    return times.astype("datetime64[ns]").astype("int64")


def validate_zone_detection():
    print("\n=== 1. Zone detection (pivots + structure FSM) vs dtfx_engine.detect_zones ===")
    for inst in INSTRUMENTS:
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times_to_ns(times)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        for min_retr in [2, 3, 4, 5]:
            zigzag = rs.find_pivots_strict(df, min_retr)
            _, zones_py = rs.run_structure_fsm(zigzag)
            top_py = np.array([z["top"] for z in zones_py], dtype=np.float64)
            bottom_py = np.array([z["bottom"] for z in zones_py], dtype=np.float64)
            mid_py = np.array([z["mid"] for z in zones_py], dtype=np.float64)
            is_buy_py = np.array([z["side"] == "buy" for z in zones_py], dtype=np.bool_)
            created_at_py = np.array([np.datetime64(z["created_at"]) for z in zones_py])
            start_idx_py = np.searchsorted(times, created_at_py, side="right").astype(np.int64)
            created_at_ns_py = created_at_py.astype("datetime64[ns]").astype(np.int64)

            top_c, bottom_c, mid_c, is_buy_c, created_at_ns_c, start_idx_c = \
                dtfx_engine.detect_zones(o, h, l, c, times_ns, min_retr)

            ok = (len(top_py) == len(top_c) and
                  np.array_equal(top_py, top_c) and
                  np.array_equal(bottom_py, bottom_c) and
                  np.array_equal(mid_py, mid_c) and
                  np.array_equal(is_buy_py, is_buy_c) and
                  np.array_equal(created_at_ns_py, created_at_ns_c) and
                  np.array_equal(start_idx_py, start_idx_c))
            check(f"{inst} min_retracement={min_retr} ({len(top_py):,} zones)", ok,
                  "" if ok else f"py n={len(top_py)} cpp n={len(top_c)}")


# label, stop_mult, rr, min_fill_age_h, require_bounce, touches -- the configs actually in use
BACKTEST_CONFIGS = [
    ("Omega",        3.343496, 1.0, 3.366682, False, 4),
    ("Hybrid-Ultra",  6.0,     0.4, 2.0,      False, 4),
    ("DTFX-Ultra",    6.0,     0.4, 2.0,      True,  8),
    ("SOMEGA-Ultra",  6.0,     0.4, 3.366682, False, 4),
]


def validate_backtest_engine():
    print("\n=== 2. Backtest engine: numba _simulate_generic vs dtfx_engine.simulate_generic ===")
    for inst in INSTRUMENTS:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times_to_ns(times)
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        for label, stop_mult, rr, age_h, bounce, touches in BACKTEST_CONFIGS:
            target_mult = stop_mult * rr
            risk_pct = (stop_mult * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
            keep = risk_pct >= pg.MIN_RISK_PCT
            top, bottom, mid = z["top"][keep], z["bottom"][keep], z["mid"][keep]
            is_buy, start_idx, c_ns = z["is_buy"][keep], z["start_idx"][keep], created_ns[keep]
            min_fill_age_ns = int(age_h * 3600 * 1e9)

            touch_py, entry_py, win_py, mid_py, risk_py = pg._simulate_generic_numba(
                top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
                min_fill_age_ns, stop_mult, target_mult, bounce, pg.PROBE_DEPTH, 1.0,
                touches, pg.MAX_WINDOW_BARS)

            touch_c, entry_c, win_c, mid_c, risk_c = dtfx_engine.simulate_generic(
                top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
                min_fill_age_ns, stop_mult, target_mult, bounce, pg.PROBE_DEPTH, 1.0,
                touches, pg.MAX_WINDOW_BARS)

            ok = (len(touch_py) == len(touch_c) and
                  np.array_equal(touch_py, touch_c) and
                  np.array_equal(entry_py, entry_c) and
                  np.array_equal(win_py, win_c) and
                  np.array_equal(mid_py, mid_c) and
                  np.array_equal(risk_py, risk_c))
            check(f"{inst} {label} ({len(touch_py):,} trades)", ok,
                  "" if ok else f"py n={len(touch_py)} cpp n={len(touch_c)}")


def validate_monte_carlo():
    print("\n=== 3. Monte Carlo: numba run_mc_capped vs dtfx_engine.run_mc_capped (statistical) ===")
    z = rs._prep_zones("EURUSD", pg.MIN_RETRACEMENT)
    times, lows, highs, df = rs.load_instrument("EURUSD")
    times_ns = times_to_ns(times)
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
    stop_mult, target_mult, age_h = 6.0, 6.0 * 0.4, 2.0  # Hybrid-Ultra
    risk_pct = (stop_mult * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    keep = risk_pct >= pg.MIN_RISK_PCT
    top, bottom, mid = z["top"][keep], z["bottom"][keep], z["mid"][keep]
    is_buy, start_idx, c_ns = z["is_buy"][keep], z["start_idx"][keep], created_ns[keep]
    touch, entry_idx, is_win, mid_out, stop_risk = pg._simulate_generic_numba(
        top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(age_h * 3600 * 1e9), stop_mult, target_mult, False, pg.PROBE_DEPTH, 1.0, 4,
        pg.MAX_WINDOW_BARS)
    pip = pg.PIP_SIZE["EURUSD"]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip
    r_gross = np.where(is_win, target_mult / stop_mult, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = (r_gross - cost_in_r).astype(np.float64)

    n_sims = 100_000
    t1_py, t10_py, dd_py, ruin_py = run_mc_capped_numba(net_r, n_sims, BLOCK, MAX_TRADES, seed=42)
    t1_c, t10_c, dd_c, ruin_c = dtfx_engine.run_mc_capped(net_r, n_sims, BLOCK, MAX_TRADES, 42)

    years = (df.index.max() - df.index.min()).days / 365.25
    tpy = len(net_r) / years

    def summarize(t1, dd, ruin):
        t1_hit = t1[t1 > 0]
        return dict(
            median_yrs_1m=float(np.median(t1_hit / tpy)) if len(t1_hit) else float("nan"),
            p5_yrs_1m=float(np.percentile(t1_hit / tpy, 5)) if len(t1_hit) else float("nan"),
            p95_yrs_1m=float(np.percentile(t1_hit / tpy, 95)) if len(t1_hit) else float("nan"),
            median_dd_pct=float(np.median(dd) * 100),
            pct_ruined=float(100 * ruin.mean()),
            pct_hit_1m=float(100 * len(t1_hit) / len(t1)),
        )

    s_py = summarize(t1_py, dd_py, ruin_py)
    s_c = summarize(t1_c, dd_c, ruin_c)
    print(f"  py:  {s_py}")
    print(f"  cpp: {s_c}")
    for key in s_py:
        py_v, c_v = s_py[key], s_c[key]
        rel_diff = abs(py_v - c_v) / max(abs(py_v), 1e-9)
        check(f"{key}: py={py_v:.4f} cpp={c_v:.4f} (rel diff {rel_diff:.2%})", rel_diff < 0.05)


def validate_regime_features():
    """No numba/Python original exists (this is a new construct, 2026-09-04) -- validated
    against a direct pure-Python/NumPy re-derivation of the same formulas instead, at a
    handful of grid-aligned points spanning the full history. Deterministic -- exact
    equality expected."""
    print("\n=== Regime features (compute_regime_features) ===")
    inst = "EURUSD"
    times, lows, highs, df = rs.load_instrument(inst)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(close)
    window, stride, lookback = 500, 50, 200

    er, hurst, volpct, rvol = dtfx_engine.compute_regime_features(high, low, close, window,
                                                                   stride, lookback, 0)
    ret = np.diff(close, prepend=close[0])
    ret[0] = 0.0

    def rescaled_range(seg):
        if len(seg) < 2:
            return 0.0
        mean = seg.mean()
        cum = np.concatenate([[0.0], np.cumsum(seg - mean)])
        s = seg.std(ddof=1)
        return (cum.max() - cum.min()) / s if s > 0 else 0.0

    def py_point(i):
        ws = i - window + 1
        if ws < 1:
            return None
        seg = ret[ws:i + 1]
        net = abs(close[i] - close[ws - 1])
        path = np.abs(seg).sum()
        er_v = net / path if path > 0 else 0.0
        rs_full = rescaled_range(seg)
        half = window // 2
        rs_half = rescaled_range(ret[i - half + 1:i + 1])
        hurst_v = np.log(rs_full / rs_half) / np.log(2.0) if rs_full > 0 and rs_half > 0 else np.nan
        return er_v, hurst_v, seg.std(ddof=1)

    n_grid = (n - (window - 1)) // stride
    check_gs = np.linspace(5, n_grid - 5, 12).astype(int)
    max_diffs = {"er": 0.0, "hurst": 0.0, "rvol": 0.0}
    for g in check_gs:
        i = g * stride + (window - 1)
        ref = py_point(i)
        if ref is None:
            continue
        er_v, hurst_v, rvol_v = ref
        max_diffs["er"] = max(max_diffs["er"], abs(er[i] - er_v))
        max_diffs["rvol"] = max(max_diffs["rvol"], abs(rvol[i] - rvol_v))
        if not (np.isnan(hurst[i]) and np.isnan(hurst_v)):
            max_diffs["hurst"] = max(max_diffs["hurst"], abs(hurst[i] - hurst_v))

    for key, tol in [("er", 1e-9), ("hurst", 1e-9), ("rvol", 1e-9)]:
        check(f"{key} matches Python reference at 12 spot-checked points "
              f"(max abs diff {max_diffs[key]:.2e})", max_diffs[key] < tol)

    warm = window - 1 + stride * 2
    check("no NaN after warmup", not np.isnan(er[warm:]).any() and not np.isnan(volpct[warm:]).any())
    check("vol_percentile stays in [0,1]",
          bool(((volpct[warm:] >= 0) & (volpct[warm:] <= 1)).all()))
    block = er[warm + 1000:warm + 1000 + stride - 1]
    check("forward-fill holds constant within a stride block", bool(np.allclose(block, block[0])))


if __name__ == "__main__":
    validate_zone_detection()
    validate_backtest_engine()
    validate_monte_carlo()
    validate_regime_features()
    print()
    if failures == 0:
        print(f"ALL CHECKS PASSED ({PASS})")
    else:
        print(f"{failures} CHECK(S) FAILED")
        sys.exit(1)
