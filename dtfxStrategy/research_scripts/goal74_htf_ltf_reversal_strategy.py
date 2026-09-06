"""Direct user request (2026-09-05): "develop and backtest this strat" -- a YouTube video
transcript (channel: Stoic Trader) describing a price-action / market-structure strategy.
Rules extracted from the transcript:

1. Identify the higher-timeframe (HTF) trend via swing structure: higher-highs/higher-lows
   = bullish, lower-highs/lower-lows = bearish (the video's "the last low that created the
   high" bookkeeping -- exactly this project's own trend_state()/structure_fsm concept).
2. Only trade WITH the HTF trend -- the video is explicit that counter-trend trades get a
   worse R:R and should be avoided.
3. Wait for price to retrace into the 50% of the most recent unmitigated impulse leg that
   built the current structure -- this is exactly a DTFX zone's own [bottom, top, mid]
   geometry (already built by this project's zone detector: a zone's mid IS the 50%
   retracement of the leg that created it).
4. At that zone, on a LOWER time frame, require a "manipulation" pattern before entering:
   a liquidity sweep of a local swing in the zone's own bias direction (the video's "sweep
   the low, then sweep/break the high" reversal signature) -- reusing this project's own
   liquidity-sweep detector (goal70) at a short lookback as the LTF proxy.
5. Entry at the zone's mid (the standard zone-fade convention throughout this project),
   stop at the zone's own boundary (N=1: risk = half the zone height = the manipulation
   swing's own extreme), target a multiple of that risk (R:R swept -- the video's own
   "-2 to -4" measured-move projections are ~2-4x the swing's size, a fixed-R:R target is
   the direct, defensible translation of that into this project's existing framework).

The genuinely new ingredient versus every earlier DTFX+ICT strategy this session: an
explicit HTF-trend-ALIGNMENT gate on which zones are even eligible. goal62-73 gated zones
by regime/momentum/sweep-confluence but never by swing-structure trend direction. Built
and backtested with the exact same realistic-fill + true-accepted pipeline as everything
else in this project for direct comparability.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30
import goal51_single_symbol_closest_price_backtest as g51
import goal70_ict_liquidity_fvg as g70

rs, pg = g29.rs, g29.pg

_feature_cache = {}


def build_htf_ltf_features(inst, htf_lookback=240, ltf_lookback=15, sweep_window_bars=60):
    """trend: this project's own trend_state() (+1/-1/0), forward-filled, zero lookahead.
    had_ltf_sweep_{bull,bear}: whether a genuine liquidity sweep of a LOCAL (ltf_lookback)
    swing occurred within the trailing sweep_window_bars -- the video's "manipulation" leg,
    at a much shorter lookback than the HTF trend itself, standing in for a lower time
    frame's own swing structure the same way goal72 used a trailing-window proxy instead of
    reconstructing exact impulse-leg boundaries (see that script's own docstring)."""
    key = (inst, htf_lookback, ltf_lookback, sweep_window_bars)
    if key in _feature_cache:
        return _feature_cache[key]
    times, lows, highs, df = rs.load_instrument(inst)
    open_, high_, low_, close_ = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    trend = g70.trend_state(high_, low_, swing_lookback=htf_lookback)
    bull_sweep, bear_sweep, _, _ = g70.detect_liquidity_sweeps(open_, high_, low_, close_, ltf_lookback)

    def trailing_any(mask, window):
        s = pd.Series(mask.astype(int))
        return (s.rolling(window, min_periods=1).sum() > 0).values

    out = dict(times=times, trend=trend,
               had_ltf_sweep_bull=trailing_any(bull_sweep, sweep_window_bars),
               had_ltf_sweep_bear=trailing_any(bear_sweep, sweep_window_bars))
    _feature_cache[key] = out
    return out


def build_htf_ltf_touches(inst, rr, age_h, touches, min_risk_pct, htf_lookback=240,
                           ltf_lookback=15, sweep_window_bars=60, require_htf_trend=True,
                           require_ltf_sweep=True, n=1.0, min_stop_pips=5.0):
    """n=1.0 (fixed): stop sits exactly at the zone's own boundary -- the manipulation
    swing's extreme, per the video's own stop placement, not some wider multiple of it.
    At N=1, risk_pct runs ~2 orders of magnitude smaller than the N=15-45 zone-fade work
    elsewhere in this project (median ~0.00018 vs the 0.005-0.02 thresholds used there) --
    min_risk_pct must be recalibrated for this N, and min_stop_pips guards against the same
    tiny-stop/spread-dominates failure mode goal70 hit and fixed earlier this session."""
    feats = build_htf_ltf_features(inst, htf_lookback, ltf_lookback, sweep_window_bars)
    times_ns = feats["times"].astype("datetime64[ns]").astype("int64")
    pip = pg.PIP_SIZE[inst]
    parts = []
    for side_is_buy in (True, False):
        t = g51.build_touches_with_metadata(inst, n, rr, age_h, False, touches, side_is_buy,
                                             min_risk_pct=min_risk_pct)
        if t.empty:
            continue
        t = t[t["stop_risk"].values / pip >= min_stop_pips].reset_index(drop=True)
        if t.empty:
            continue
        created_ns = t["created_at"].values.astype("int64")
        idx = np.clip(np.searchsorted(times_ns, created_ns), 0, len(times_ns) - 1)
        t["htf_trend"] = feats["trend"][idx]
        t["had_ltf_sweep"] = (feats["had_ltf_sweep_bull"][idx] if side_is_buy
                               else feats["had_ltf_sweep_bear"][idx])
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    if require_htf_trend:
        want = np.where(out["is_buy"].values, 1.0, -1.0)
        out = out[out["htf_trend"].values == want]
    if require_ltf_sweep:
        out = out[out["had_ltf_sweep"].values]
    return out.reset_index(drop=True)


def evaluate(inst, rr, age_h, touches, min_risk_pct, **kwargs):
    t = build_htf_ltf_touches(inst, rr, age_h, touches, min_risk_pct, **kwargs)
    if len(t) < 15:
        return None
    price_at = g51._price_lookup(inst)
    resolved = g51.resolve_closest_price(t, age_h, price_at)
    if len(resolved) < 15:
        return None
    acc_t, _, _, _, _ = g30.simulate_two_account(resolved, 0.06)
    acc = acc_t[acc_t.accepted].sort_values("entry_time").reset_index(drop=True)
    if len(acc) < 15:
        return None
    years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
    wr = 100 * (acc.net_r > 0).mean()
    gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    return dict(acc=acc, n=len(acc), freq=len(acc) / years, wr=wr, pf=pf)


def thirds_pf(acc):
    if len(acc) < 45:
        return (None, None, None)
    start, end = acc.entry_time.min(), acc.entry_time.max()
    span = end - start
    b1, b2 = start + span / 3, start + 2 * span / 3
    windows = [acc[acc.entry_time < b1], acc[(acc.entry_time >= b1) & (acc.entry_time < b2)],
               acc[acc.entry_time >= b2]]

    def pf_of(x):
        if len(x) < 10:
            return None
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gw / gl if gl > 0 else float("inf")
    return tuple(pf_of(w) for w in windows)


if __name__ == "__main__":
    import time
    t0 = time.time()
    INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]

    print("=== Step 1: does each ingredient matter? (GBPUSD, RR=2.0, N=1.0, mrp=0.0001) ===")
    for require_htf_trend, require_ltf_sweep, label in [
        (False, False, "baseline (plain zone-fade, N=1, no gates)"),
        (True, False, "+ HTF trend alignment only"),
        (False, True, "+ LTF sweep confluence only"),
        (True, True, "+ BOTH (the full video recipe)"),
    ]:
        r = evaluate("GBPUSD", 2.0, 0.08, 20, 0.0001,
                     require_htf_trend=require_htf_trend, require_ltf_sweep=require_ltf_sweep)
        if r is None:
            print(f"  {label}: too few")
            continue
        print(f"  {label}: n={r['n']} freq={r['freq']:.1f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"({time.time()-t0:.0f}s)")

    print("\n=== Step 2: R:R sweep, full recipe, GBPUSD ===")
    for rr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        r = evaluate("GBPUSD", rr, 0.08, 20, 0.0001, require_htf_trend=True, require_ltf_sweep=True)
        if r is None:
            print(f"  RR={rr}: too few")
            continue
        p1, p2, p3 = thirds_pf(r["acc"])
        print(f"  RR={rr}: n={r['n']} freq={r['freq']:.1f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"thirds={p1}/{p2}/{p3} ({time.time()-t0:.0f}s)")

    print("\n=== Step 3: HTF/LTF lookback sensitivity, full recipe, GBPUSD, RR=2.5 ===")
    for htf_lb, ltf_lb in [(120, 10), (240, 15), (480, 15), (240, 30), (480, 30)]:
        r = evaluate("GBPUSD", 2.5, 0.08, 20, 0.0001, require_htf_trend=True, require_ltf_sweep=True,
                     htf_lookback=htf_lb, ltf_lookback=ltf_lb)
        if r is None:
            print(f"  htf={htf_lb} ltf={ltf_lb}: too few")
            continue
        print(f"  htf={htf_lb} ltf={ltf_lb}: n={r['n']} freq={r['freq']:.1f}/yr WR={r['wr']:.1f}% "
              f"PF={r['pf']:.3f} ({time.time()-t0:.0f}s)")

    print("\n=== Step 4: per-symbol R:R sweep -- does the recipe generalize as-is? ===")
    for inst in ["EURUSD", "EURGBP", "USDJPY"]:
        print(f"  {inst}:")
        for rr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            r = evaluate(inst, rr, 0.08, 20, 0.0001, require_htf_trend=True, require_ltf_sweep=True)
            if r is None:
                print(f"    RR={rr}: too few")
                continue
            p1, p2, p3 = thirds_pf(r["acc"])
            print(f"    RR={rr}: n={r['n']} freq={r['freq']:.1f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
                  f"thirds={p1}/{p2}/{p3} ({time.time()-t0:.0f}s)")

    print("\n=== Final recipe: HTF trend + LTF sweep confluence, symbols where it's robust ===")
    print("(EURUSD found no thirds-clean R:R; USDJPY never crossed PF=1.0 at any R:R 1.5-4.0 --")
    print(" both excluded, disclosed honestly rather than forced into the pool.)")
    FINAL_CONFIGS = [("GBPUSD", 2.5), ("EURGBP", 4.0)]
    all_acc = []
    for inst, rr in FINAL_CONFIGS:
        r = evaluate(inst, rr, 0.08, 20, 0.0001, require_htf_trend=True, require_ltf_sweep=True)
        p1, p2, p3 = thirds_pf(r["acc"])
        print(f"  {inst} RR={rr}: n={r['n']} freq={r['freq']:.1f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"thirds={p1}/{p2}/{p3}")
        all_acc.append(r["acc"])
        r["acc"].to_csv(os.path.join(DATA_DIR, f"goal74_htf_ltf_reversal_{inst.lower()}_trades.csv"),
                         index=False)

    pooled = pd.concat(all_acc, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    years = (pooled.entry_time.max() - pooled.entry_time.min()).days / 365.25
    wr = 100 * (pooled.net_r > 0).mean()
    gw, gl = pooled[pooled.net_r > 0].net_r.sum(), -pooled[pooled.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    print(f"POOLED (GBPUSD+EURGBP): n={len(pooled)} freq={len(pooled)/years:.1f}/yr WR={wr:.1f}% PF={pf:.3f}")

    split = pd.Timestamp("2018-01-01")
    train, hold = pooled[pooled.entry_time < split], pooled[pooled.entry_time >= split]

    def pf_of(x):
        if len(x) < 5:
            return None, len(x)
        gwx, glx = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return (gwx / glx if glx > 0 else float("inf")), len(x)
    print(f"  train (pre-2018): PF={pf_of(train)[0]} n={pf_of(train)[1]}")
    print(f"  holdout (2018+): PF={pf_of(hold)[0]} n={pf_of(hold)[1]}")
    p1, p2, p3 = thirds_pf(pooled)
    print(f"  thirds: {p1}/{p2}/{p3}")
    pooled.to_csv(os.path.join(DATA_DIR, "goal74_htf_ltf_reversal_pooled_trades.csv"), index=False)

    print(f"\n(total: {time.time()-t0:.0f}s)")
