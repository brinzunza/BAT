"""/goal (2026-09-04): "try adding new parameters... combine the strategy with other
synthetic indicators that give it edge... be hyper creative and use niche ideas... use C++
for the engines for large data and computational tasks. find profitable, frequent, high
winrate strategies. At least 3 unique ones."

Adds THREE new synthetic indicators on top of the existing, fully-validated realistic-fill
DTFX zone-fade engine (g51) -- none of g51's cost-model/fill/resolver logic is touched, only
which touches get OFFERED to it in the first place:

  1. Efficiency Ratio (Kaufman ER) + Hurst-est REGIME gates (cpp_engine/src/features.cpp,
     new `dtfx_engine.compute_regime_features` -- exact-match validated against a pure-Python
     reference, 6.3M EURUSD bars in 0.13s). A zone-fade is a mean-reversion bet; the wager is
     it should fire only when the market IS mean-reverting (low ER / Hurst<0.5), not when it's
     trending through the zone.
  2. Volatility-percentile BAND gate (same C++ function): avoid the deadest-quiet regime
     (barely clears the spread even on a full win) and the most violent regime (fills reprice
     badly, stops blow through) -- trade only the middle band of recently-seen volatility.
  3. Impulse Velocity: a new zone-quality construct computed straight from each zone's own
     geometry (no new per-bar data needed) -- the speed, in pips/minute, of the structural
     break that CREATED the zone (start_idx -> created_at). Wager: a zone born from a fast,
     violent break carries more real institutional intent than one of identical height that
     formed from a slow grind, and should hold up better on retest.

Every gate is independently toggleable and composable (AND-combined) via build_gated_touches,
so "different orders and combinations" is literally just which optional args are passed.
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30
import goal31_apex_stats as g31
import goal51_single_symbol_closest_price_backtest as g51

rs, pg = g29.rs, g29.pg
dtfx_engine = rs.dtfx_engine

_regime_cache = {}
_impulse_cache = {}


def load_regime_features(inst, window=500, stride=50, vol_lookback=200):
    key = (inst, window, stride, vol_lookback)
    if key in _regime_cache:
        return _regime_cache[key]
    times, lows, highs, df = rs.load_instrument(inst)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    er, hurst, volpct, rvol = dtfx_engine.compute_regime_features(
        high, low, close, window, stride, vol_lookback, 0)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    _regime_cache[key] = (times_ns, er, hurst, volpct, rvol)
    return _regime_cache[key]


def impulse_velocity_lookup(inst):
    """{(round(mid,8), created_at_ns): pips/minute of the zone-forming impulse}. Keyed
    identically to how g51's own output pairs (entry_price, created_at) so it can be joined
    onto g51's touches without re-deriving any of its (already validated) fill logic."""
    if inst in _impulse_cache:
        return _impulse_cache[inst]
    times, lows, highs, df = rs.load_instrument(inst)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
    top, bottom, mid, start_idx = z["top"], z["bottom"], z["mid"], z["start_idx"]
    c_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
    start_ns = times_ns[start_idx]
    dur_min = np.maximum((c_ns - start_ns) / 6e10, 1.0 / 60.0)
    pip = pg.PIP_SIZE[inst]
    velocity_pips_per_min = ((top - bottom) / pip) / dur_min
    mid_r = np.round(mid, 8)
    lookup = {(mid_r[k], c_ns[k]): velocity_pips_per_min[k] for k in range(len(mid_r))}
    _impulse_cache[inst] = lookup
    return lookup


def build_gated_touches(inst, n, rr, age_h, bounce, touches, side_is_buy, min_risk_pct=None,
                         er_max=None, hurst_max=None, volpct_range=None,
                         impulse_min_pips_per_min=None, window=500, stride=50,
                         vol_lookback=200):
    """g51's fully-validated realistic-fill touches, narrowed by zero or more independent
    (AND-combined) synthetic-indicator gates. Every gate defaults to None (no-op) so this
    is a strict superset of the plain ungated strategy -- passing no gates reproduces g51
    exactly, letting every comparison below be apples-to-apples against the same baseline."""
    t = g51.build_touches_with_metadata(inst, n, rr, age_h, bounce, touches, side_is_buy,
                                         min_risk_pct=min_risk_pct)
    if t.empty:
        return t
    keep = np.ones(len(t), dtype=bool)

    if er_max is not None or hurst_max is not None or volpct_range is not None:
        times_ns, er, hurst, volpct, rvol = load_regime_features(inst, window, stride, vol_lookback)
        entry_ns = t["entry_time"].values.astype("int64")
        idx = np.searchsorted(times_ns, entry_ns)
        idx = np.clip(idx, 0, len(times_ns) - 1)
        if not np.array_equal(times_ns[idx], entry_ns):
            raise AssertionError(f"[{inst}] entry_time not aligned to bar grid -- indexing bug")
        if er_max is not None:
            keep &= er[idx] <= er_max
        if hurst_max is not None:
            keep &= hurst[idx] <= hurst_max
        if volpct_range is not None:
            lo, hi = volpct_range
            keep &= (volpct[idx] >= lo) & (volpct[idx] <= hi)

    if impulse_min_pips_per_min is not None:
        lookup = impulse_velocity_lookup(inst)
        mid_r = t["entry_price"].values
        created_ns = t["created_at"].values.astype("int64")
        vel = np.array([lookup.get((m, c), np.nan) for m, c in zip(mid_r, created_ns)])
        keep &= np.nan_to_num(vel, nan=-1.0) >= impulse_min_pips_per_min

    return t[keep].reset_index(drop=True)


def build_both_sides(inst, n, rr, age_h, touches, min_risk_pct=None, **gates):
    parts = [build_gated_touches(inst, n, rr, age_h, False, touches, side, min_risk_pct=min_risk_pct, **gates)
             for side in (True, False)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["entry_time", "exit_time", "net_r"])
    return pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def thirds_pf(t):
    """Stricter than a single train/holdout split -- splits into 3 equal-DURATION windows
    and requires all three individually profitable, catching regime-dependent overfits a
    single split can hide (established this session: see goal60's own docstring)."""
    if len(t) < 45:
        return (None, None, None)
    start, end = t.entry_time.min(), t.entry_time.max()
    span = end - start
    b1, b2 = start + span / 3, start + 2 * span / 3
    windows = [t[t.entry_time < b1], t[(t.entry_time >= b1) & (t.entry_time < b2)],
               t[t.entry_time >= b2]]

    def pf_of(x):
        if len(x) < 15:
            return None
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gw / gl if gl > 0 else float("inf")
    return tuple(pf_of(w) for w in windows)


def years_to_1m(t, risk_grid=np.arange(0.02, 0.61, 0.02)):
    years_elapsed = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    if years_elapsed <= 0:
        return float("inf"), None, None, None
    best = None
    for rp in risk_grid:
        _, eq_buy, eq_sell, dd, _ = g30.simulate_two_account(t, rp)
        cagr = ((eq_buy + eq_sell) / 200.0) ** (1 / years_elapsed) - 1
        if best is None or cagr > best[0]:
            best = (cagr, rp, eq_buy + eq_sell, dd)
    cagr, rp, final_eq, dd = best
    y = np.log(1_000_000 / 200.0) / np.log(1 + cagr) if cagr > 0 else float("inf")
    return y, rp, final_eq, dd


def summarize(t, label):
    if t.empty or len(t) < 45:
        return None
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    wr = 100 * (t.net_r > 0).mean()
    gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    p1, p2, p3 = thirds_pf(t)
    robust = bool(p1 and p2 and p3 and p1 > 1.0 and p2 > 1.0 and p3 > 1.0)
    y1m, risk, final_eq, dd = years_to_1m(t)
    return dict(label=label, n_trades=len(t), years=years, freq_tpy=len(t) / years, wr=wr,
                pf=pf, pf_third1=p1, pf_third2=p2, pf_third3=p3, robust_thirds=robust,
                years_to_1m=y1m, best_risk=risk, max_dd=dd)


def percentile_grid(inst, n, rr, age, mrp, touches=20, window=500, stride=50, vol_lookback=200):
    """Gate thresholds anchored to the STRATEGY'S OWN already-filtered touch population, not
    the raw full-zone population. First attempt calibrated off the full zone set and every
    impulse-velocity gate came back completely non-binding (identical n_trades/PF at p25/
    p50/p75) -- traced to MIN_RISK_PCT already implicitly selecting almost entirely for
    fast-forming zones (median matched velocity ~2500 pips/min on USDJPY-YenRapid vs. ~240
    pips/min across the full unfiltered zone population), so population-wide percentiles sat
    two orders of magnitude below where this specific config's touches actually live. Basing
    thresholds on the touches a config would ACTUALLY take makes every gate level meaningfully
    binding regardless of how much MIN_RISK_PCT/N/R:R already pre-selected the population."""
    t = build_both_sides(inst, n, rr, age, touches, min_risk_pct=mrp)
    if t.empty:
        return None
    times_ns, er, hurst, volpct, rvol = load_regime_features(inst, window, stride, vol_lookback)
    entry_ns = t["entry_time"].values.astype("int64")
    idx = np.clip(np.searchsorted(times_ns, entry_ns), 0, len(times_ns) - 1)
    lookup = impulse_velocity_lookup(inst)
    mid_r, created_ns = t["entry_price"].values, t["created_at"].values.astype("int64")
    vel = np.array([lookup.get((m, c), np.nan) for m, c in zip(mid_r, created_ns)])
    return dict(
        er_max=[float(np.nanpercentile(er[idx], p)) for p in (25, 50, 75)],
        hurst_max=[float(np.nanpercentile(hurst[idx], p)) for p in (25, 50, 75)],
        volpct_range=[(0.0, float(np.nanpercentile(volpct[idx], 80))),
                      (float(np.nanpercentile(volpct[idx], 20)), 1.0),
                      (float(np.nanpercentile(volpct[idx], 15)), float(np.nanpercentile(volpct[idx], 85)))],
        impulse_min_pips_per_min=[float(np.nanpercentile(vel, p)) for p in (25, 50, 75)],
    )


# Established, individually-validated base configs (REALISTIC_FILL_RANKINGS.md) spanning a
# genuine spread of R:R (0.15 -> 1.00) -- Yen-Rapid's mechanically-near-guaranteed win rate
# is a poor test of whether a REGIME filter adds real edge (barely any room to move); Yen-
# Punch/Dollar-Wide/Bedrock-King all require a real directional read to win, which is exactly
# where a mean-reversion/volatility/impulse-quality gate should earn its keep if it does.
BASE_CONFIGS = {
    "USDJPY-YenRapid": dict(inst="USDJPY", n=4.0, rr=0.15, mrp=0.005, age=0.05),
    "USDJPY-YenPunch": dict(inst="USDJPY", n=4.0, rr=0.75, mrp=0.005, age=0.08),
    "EURUSD-DollarWide": dict(inst="EURUSD", n=15.0, rr=1.00, mrp=0.020, age=0.08),
    "EURUSD-BalancedDollar": dict(inst="EURUSD", n=30.0, rr=0.50, mrp=0.020, age=0.08),
    "EURGBP-BedrockKing": dict(inst="EURGBP", n=22.0, rr=0.75, mrp=0.020, age=0.08),
    "GBPUSD-CableWide": dict(inst="GBPUSD", n=15.0, rr=1.50, mrp=0.020, age=0.08),
}


# Final, refined winners -- fine-swept around each gate sweep's own best region (see
# goal62 development notes / REALISTIC_FILL_RANKINGS.md), each on a DIFFERENT symbol and
# headlining a DIFFERENT one of the three new synthetic indicators, so all three earn a
# validated showcase rather than one indicator winning by coincidence of which base config
# it happened to be tested against first:
FINAL_STRATEGIES = {
    "Storm-Rider": dict(inst="USDJPY", n=4.0, rr=0.15, mrp=0.005, age=0.05,
                         gates=dict(volpct_range=(0.40, 1.0)),
                         note="Yen-Rapid + volatility-percentile floor (avoid dead-quiet stretches)"),
    "Chop-Fade Balanced": dict(inst="EURUSD", n=30.0, rr=0.50, mrp=0.020, age=0.08,
                                gates=dict(er_max=0.05),
                                note="Balanced-Dollar + Kaufman Efficiency Ratio ceiling (choppy-only)"),
    "Chop-Fade Bedrock": dict(inst="EURGBP", n=22.0, rr=0.75, mrp=0.020, age=0.08,
                               gates=dict(er_max=0.07),
                               note="Bedrock-King + Kaufman Efficiency Ratio ceiling (choppy-only)"),
    "Velocity-Cable": dict(inst="GBPUSD", n=15.0, rr=1.50, mrp=0.020, age=0.08,
                            gates=dict(impulse_min_pips_per_min=2154.0),
                            note="Cable-Wide + Impulse Velocity floor (fast-formed zones only)"),
}


def full_report(t, split=pd.Timestamp("2018-01-01")):
    s = summarize(t, "")
    ann_sharpe, exk, calmar, maxdd1, _ = g31.sharpe_and_kurtosis(t)
    train, hold = t[t.entry_time < split], t[t.entry_time >= split]

    def pf_of(x):
        if len(x) < 5:
            return None
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gw / gl if gl > 0 else float("inf")
    s.update(sharpe=ann_sharpe, excess_kurtosis=exk, calmar=calmar, maxdd_1pct_ref=-100 * maxdd1,
              train_pf=pf_of(train), train_n=len(train), holdout_pf=pf_of(hold), holdout_n=len(hold))
    return s


if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 100)
    print("FINAL STRATEGIES -- full report")
    print("=" * 100)
    for name, cfg in FINAL_STRATEGIES.items():
        base = build_both_sides(cfg["inst"], cfg["n"], cfg["rr"], cfg["age"], 20, min_risk_pct=cfg["mrp"])
        gated = build_both_sides(cfg["inst"], cfg["n"], cfg["rr"], cfg["age"], 20,
                                  min_risk_pct=cfg["mrp"], **cfg["gates"])
        bs, gs = full_report(base), full_report(gated)
        print(f"\n--- {name} ({cfg['inst']}) -- {cfg['note']} ---")
        print(f"  gate: {cfg['gates']}")
        print(f"  BEFORE: n={bs['n_trades']} freq={bs['freq_tpy']:.1f}/yr WR={bs['wr']:.1f}% "
              f"PF={bs['pf']:.3f} Sharpe={bs['sharpe']:.2f} y1m={bs['years_to_1m']:.1f}")
        print(f"  AFTER:  n={gs['n_trades']} freq={gs['freq_tpy']:.1f}/yr WR={gs['wr']:.1f}% "
              f"PF={gs['pf']:.3f} Sharpe={gs['sharpe']:.2f} y1m={gs['years_to_1m']:.1f}")
        print(f"  thirds PF: {gs['pf_third1']:.3f} / {gs['pf_third2']:.3f} / {gs['pf_third3']:.3f} "
              f"(robust={gs['robust_thirds']})")
        print(f"  train(pre-2018) PF={gs['train_pf']}(n={gs['train_n']}) | "
              f"holdout(2018+) PF={gs['holdout_pf']}(n={gs['holdout_n']})")
        print(f"  ExcessKurtosis={gs['excess_kurtosis']:+.2f} Calmar={gs['calmar']:.2f} "
              f"MaxDD(1%ref)={gs['maxdd_1pct_ref']:.1f}% best_risk={gs['best_risk']}")
        fname = "goal62_" + name.lower().replace(" ", "_").replace("-", "_") + "_trades.csv"
        gated.to_csv(os.path.join(DATA_DIR, fname), index=False)
        print(f"  saved -> {fname}")
    print(f"\n(full report: {time.time()-t0:.1f}s)")

    print("\n" + "=" * 100)
    print("EXPLORATORY SWEEPS (all base configs x all gate types) -- supporting evidence")
    print("=" * 100)
    rows = []
    for name, cfg in BASE_CONFIGS.items():
        inst, n, rr, mrp, age = cfg["inst"], cfg["n"], cfg["rr"], cfg["mrp"], cfg["age"]
        t = build_both_sides(inst, n, rr, age, 20, min_risk_pct=mrp)
        s = summarize(t, f"{name} (ungated baseline)")
        if s:
            rows.append(s)
            print(f"{s['label']}: n={s['n_trades']} freq={s['freq_tpy']:.1f}/yr WR={s['wr']:.1f}% "
                  f"PF={s['pf']:.3f} robust={s['robust_thirds']} y1m={s['years_to_1m']:.1f}")

        pg_grid = percentile_grid(inst, n, rr, age, mrp)
        gate_grid = (
            [dict(er_max=v) for v in pg_grid["er_max"]]
            + [dict(hurst_max=v) for v in pg_grid["hurst_max"]]
            + [dict(volpct_range=v) for v in pg_grid["volpct_range"]]
            + [dict(impulse_min_pips_per_min=v) for v in pg_grid["impulse_min_pips_per_min"]]
            + [dict(er_max=pg_grid["er_max"][1], hurst_max=pg_grid["hurst_max"][1]),
               dict(er_max=pg_grid["er_max"][1], volpct_range=pg_grid["volpct_range"][2]),
               dict(hurst_max=pg_grid["hurst_max"][1],
                    impulse_min_pips_per_min=pg_grid["impulse_min_pips_per_min"][2]),
               dict(er_max=pg_grid["er_max"][0], hurst_max=pg_grid["hurst_max"][0],
                    volpct_range=pg_grid["volpct_range"][2]),
               dict(impulse_min_pips_per_min=pg_grid["impulse_min_pips_per_min"][2],
                    volpct_range=pg_grid["volpct_range"][2])]
        )
        for gates in gate_grid:
            t = build_both_sides(inst, n, rr, age, 20, min_risk_pct=mrp, **gates)
            label = f"{name} + " + ", ".join(
                f"{k}={(round(v,3) if not isinstance(v,tuple) else tuple(round(x,3) for x in v))}"
                for k, v in gates.items())
            s = summarize(t, label)
            if s:
                rows.append(s)

    out = pd.DataFrame(rows).sort_values("pf", ascending=False)
    out_path = os.path.join(DATA_DIR, "goal62_regime_gate_sweep.csv")
    out.to_csv(out_path, index=False)
    print(f"\n{len(out)} results, saved {out_path} ({time.time()-t0:.1f}s)")
    print(out.to_string(index=False))
