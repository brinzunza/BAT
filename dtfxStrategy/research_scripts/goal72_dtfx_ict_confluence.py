"""/goal (2026-09-04) continued: rather than build liquidity-sweep+FVG as an entirely
separate signal (goal70/71, which found a real but structurally modest PF~1.0-1.1 ceiling
given its own 25-37% win rate), this applies the SAME new concepts as CONFIRMATION FILTERS
on top of an already-strong DTFX zone-fade candidate at a genuinely reasonable R:R --
Velocity-Cable (GBPUSD, N=15, R:R=1.50, already gated by Impulse Velocity, PF=1.622 per
goal62's own validated result). Checks whether a zone whose formation was ALSO accompanied
by a genuine liquidity sweep and/or a Fair Value Gap in the bars just before its creation
is higher quality than an unconfirmed zone -- combining DTFX's own trend-following/
institutional-order thesis with the new concepts, rather than discarding it.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal51_single_symbol_closest_price_backtest as g51
import goal62_regime_feature_strategies as g62
import goal70_ict_liquidity_fvg as g70
import goal63_hyperfreq_portfolio as g63

rs, pg = g29.rs, g29.pg


def build_confluence_features(inst, lookback_bars=60, swing_lookback=15):
    """Per-bar: was there a liquidity sweep (either direction) and/or an FVG (either
    direction) within the trailing `lookback_bars` window ending at this bar -- used to
    tag each DTFX zone's own `created_at` bar with whether its formation was accompanied
    by a genuine sweep/FVG just before confirmation."""
    times, lows, highs, df = rs.load_instrument(inst)
    open_, high_, low_, close_ = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    bullish_sweep, bearish_sweep, _, _ = g70.detect_liquidity_sweeps(open_, high_, low_, close_, swing_lookback)
    is_bull_fvg, _, _, is_bear_fvg, _, _ = g70.detect_fvg(high_, low_)

    any_sweep = bullish_sweep | bearish_sweep
    any_fvg = is_bull_fvg | is_bear_fvg
    bull_signal = bullish_sweep | is_bull_fvg
    bear_signal = bearish_sweep | is_bear_fvg

    def trailing_any(mask, window):
        s = pd.Series(mask.astype(int))
        return (s.rolling(window, min_periods=1).sum() > 0).values

    return dict(
        times=times,
        had_sweep_bull=trailing_any(bullish_sweep, lookback_bars),
        had_sweep_bear=trailing_any(bearish_sweep, lookback_bars),
        had_fvg_bull=trailing_any(is_bull_fvg, lookback_bars),
        had_fvg_bear=trailing_any(is_bear_fvg, lookback_bars),
    )


def build_confluence_touches(inst, n, rr, age_h, touches, gates, lookback_bars=60,
                              require_sweep=False, require_fvg=False):
    """Same realistic-fill zone-fade touches as goal62's own gated builder, tagged with
    whether the zone's OWN created_at bar was preceded (within lookback_bars) by a
    same-direction liquidity sweep and/or FVG -- then optionally filtered on it."""
    conf = build_confluence_features(inst, lookback_bars)
    times_ns = conf["times"].astype("datetime64[ns]").astype("int64")

    parts = []
    for side_is_buy in (True, False):
        t = g62.build_gated_touches(inst, n, rr, age_h, False, touches, side_is_buy,
                                     min_risk_pct=gates.get("mrp"), **{k: v for k, v in gates.items() if k != "mrp"})
        if t.empty:
            continue
        created_ns = t["created_at"].values.astype("int64")
        idx = np.clip(np.searchsorted(times_ns, created_ns), 0, len(times_ns) - 1)
        if side_is_buy:
            t["had_sweep"] = conf["had_sweep_bull"][idx]
            t["had_fvg"] = conf["had_fvg_bull"][idx]
        else:
            t["had_sweep"] = conf["had_sweep_bear"][idx]
            t["had_fvg"] = conf["had_fvg_bear"][idx]
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    if require_sweep:
        out = out[out.had_sweep]
    if require_fvg:
        out = out[out.had_fvg]
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import time
    t0 = time.time()

    cfg = g62.FINAL_STRATEGIES["Velocity-Cable"]
    inst, n, rr, age = cfg["inst"], cfg["n"], cfg["rr"], cfg["age"]
    gates = dict(cfg["gates"])
    gates["mrp"] = cfg["mrp"]

    print(f"=== Velocity-Cable base ({inst} N={n} R:R={rr}) + confluence checks ===")
    import goal30_apex_king_deployment as g30

    def evaluate(inst, n, rr, age, gates, require_sweep, require_fvg=False, lookback_bars=60):
        t = build_confluence_touches(inst, n, rr, age, 20, gates, lookback_bars=lookback_bars,
                                      require_sweep=require_sweep, require_fvg=require_fvg)
        if len(t) < 15:
            return None
        price_at = g51._price_lookup(inst)
        resolved = g51.resolve_closest_price(t, age, price_at)
        acc_t, _, _, _, _ = g30.simulate_two_account(resolved, 0.06)
        acc = acc_t[acc_t.accepted].sort_values("entry_time").reset_index(drop=True)
        if len(acc) < 15:
            return None
        return acc

    for require_sweep, require_fvg, label in [
        (False, False, "baseline (no confluence filter)"),
        (True, False, "+ liquidity sweep in formation window"),
        (False, True, "+ FVG in formation window"),
        (True, True, "+ BOTH sweep and FVG"),
    ]:
        acc = evaluate(inst, n, rr, age, gates, require_sweep, require_fvg)
        if acc is None:
            print(f"{label}: too few")
            continue
        years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
        wr = 100 * (acc.net_r > 0).mean()
        gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"{label}: n={len(acc)} freq={len(acc)/years:.1f}/yr WR={wr:.1f}% PF={pf:.3f} "
              f"({time.time()-t0:.0f}s)")

    print("\n=== Final recipe: momentum-gated (Impulse Velocity) + sweep confluence, "
          "pooled across EURUSD + GBPUSD ===")
    final_configs = [
        ("GBPUSD", 15.0, 1.5, 0.08, dict(mrp=0.020, impulse_min_pips_per_min=2154.0)),
        ("EURUSD", 15.0, 1.5, 0.08, dict(mrp=0.020, impulse_min_pips_per_min=2154.0)),
    ]
    all_acc = []
    for f_inst, f_n, f_rr, f_age, f_gates in final_configs:
        acc = evaluate(f_inst, f_n, f_rr, f_age, f_gates, require_sweep=True)
        if acc is None:
            print(f"{f_inst}: too few")
            continue
        all_acc.append(acc)
        years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
        wr = 100 * (acc.net_r > 0).mean()
        gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"  {f_inst}: n={len(acc)} freq={len(acc)/years:.1f}/yr WR={wr:.1f}% PF={pf:.3f}")
        acc.to_csv(os.path.join(DATA_DIR, f"goal72_dtfx_ict_confluence_{f_inst.lower()}_trades.csv"),
                    index=False)

    pooled = pd.concat(all_acc, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    years = (pooled.entry_time.max() - pooled.entry_time.min()).days / 365.25
    wr = 100 * (pooled.net_r > 0).mean()
    gw, gl = pooled[pooled.net_r > 0].net_r.sum(), -pooled[pooled.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    print(f"POOLED: n={len(pooled)} freq={len(pooled)/years:.1f}/yr WR={wr:.1f}% PF={pf:.3f}")

    split = pd.Timestamp("2018-01-01")
    train, hold = pooled[pooled.entry_time < split], pooled[pooled.entry_time >= split]

    def pf_of(x):
        if len(x) < 5:
            return None, len(x)
        gwx, glx = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return (gwx / glx if glx > 0 else float("inf")), len(x)
    print(f"  train (pre-2018): PF={pf_of(train)[0]} n={pf_of(train)[1]}")
    print(f"  holdout (2018+): PF={pf_of(hold)[0]} n={pf_of(hold)[1]}")
    print(f"\n(total: {time.time()-t0:.0f}s)")
