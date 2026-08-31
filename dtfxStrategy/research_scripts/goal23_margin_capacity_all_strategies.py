"""Margin-capacity factor for the composite score.

goal20 found that at $200 start / 5% risk / 50x leverage, most of a strategy's designed
trade population gets rejected because concurrent positions run out of margin (Speed-King:
only ~20-38% of signals actually get taken, depending on the window). That's a real,
strategy-specific property -- a strategy with more overlap opportunities (higher
frequency, tighter stops relative to price) loses more of its edge to this than one that
doesn't. This computes it for all 14 composite-score candidates and adds it as a new
factor: how much of each strategy's edge actually survives realistic capital constraints
at a small account, not just its raw (unconstrained) trade frequency.

Rebuilds TRUE entry/exit trade streams for all 14 directly from STRATEGY_RANKINGS.md's
"Exact parameters per strategy" table -- that table's own stated purpose is full
reproducibility from prose alone, so this doubles as a check that the table is still
accurate. Uses dtfx_engine.simulate_generic_ex (C++) for zone detection + the bar-walk,
applies each strategy's own session filter (if any) to entry AND exit together, then runs
goal20's margin-constrained walk ($200 start, 5% risk, 50x leverage, $2M cap for
long-run sanity) to get each strategy's average signal-acceptance rate on its own TRUE
holdout population (full history for the 3 strategies with no fitted rule -- CANA
faithful, SOMEGA live, goal70_final_winner -- matching goal12's existing convention).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal5_rr_sweep as g5
import goal20_speed_king_margin_report as g20
import cana_faithful_backtest as cana
import goal2_hour_opt as hopt
import goal2_robust_hours as rh
import goal8_unified_holdout as g8

rs, pg, TRADEABLE = g5.rs, g5.pg, g5.TRADEABLE
SPLIT = pd.Timestamp("2018-03-09")


def get_wf_hours_and_split(pools):
    """Exact replica of goal8_unified_holdout.py's true_holdout() selection step: the
    session-block filter for CANA+/DTFX Robust/SOMEGA Robust is the OUTPUT of a
    data-driven per-instrument vote (lenient: keep a 6hr block if net-positive in >=2
    of 4 first-half chronological quarters), not a fixed set safe to hand-transcribe
    from STRATEGY_RANKINGS.md's prose table -- confirmed by a real ~2x trade-count
    mismatch for CANA+ when this was first tried by hand. `pools` must carry the same
    `hour` (block index 0-3, via entry_time.hour//6) and `glog` columns goal8 uses."""
    tmin = min(t.entry_time.min() for t in pools.values())
    tmax = max(t.entry_time.max() for t in pools.values())
    half = tmin + (tmax - tmin) / 2
    first_half_quarters = rh.make_quarters(tmin, half, k=4)
    orig_vote = rh.bucket_vote
    rh.bucket_vote = g8.vote_lenient
    wf_hours, _ = rh.select_hours(pools, first_half_quarters)
    rh.bucket_vote = orig_vote
    return wf_hours, half


def hrs(*h):
    return set(h)


def hrange(lo, hi):
    return set(range(lo, hi + 1))


def hge(lo):
    return set(range(lo, 24))


# label -> (n, rr, age_h, bounce, touches, session, holdout)
#   n, age_h: float (same for all instruments) or {instrument: float}
#   session: None (no filter at all) or {instrument: set-of-hours-or-None}
#   holdout: True -> filter to entry_time >= SPLIT (matches goal12's *_true_holdout.csv);
#            False -> full history (the 3 strategies with no fitted rule)
STRATEGIES = {
    "Speed-King":          dict(n=7.0, rr=0.20, age_h=0.175, bounce=False, touches=4,
                                 session=None, holdout=True),
    "Quality-King":        dict(n=7.0, rr=0.35, age_h=0.40, bounce=True, touches=8,
                                 session=None, holdout=True),
    "Hybrid-Ultra":        dict(n=6.0, rr=0.4, age_h=2.0, bounce=False, touches=4,
                                 session=None, holdout=True),
    "DTFX-Ultra":          dict(n=6.0, rr=0.4, age_h=2.0, bounce=True, touches=8,
                                 session=None, holdout=True),
    "SOMEGA-Ultra":        dict(n=6.0, rr=0.4, age_h=3.366682, bounce=False, touches=4,
                                 session=None, holdout=True),
    "CANA-NarrowTarget":   dict(n=3.00, rr=0.8, age_h=2.0, bounce=True, touches=4,
                                 session=None, holdout=True),
    "SOMEGA-NarrowTarget": dict(n=3.343496, rr=0.7, age_h=3.366682, bounce=False, touches=4,
                                 session=None, holdout=True),
    "DTFX-NarrowTarget":   dict(n=2.0, rr=0.7, age_h=2.0, bounce=True, touches=4,
                                 session=None, holdout=True),
    # CANA+, DTFX Robust, SOMEGA Robust: session="VOTE" -- their block filter is the
    # OUTPUT of goal8_unified_holdout.py's data-driven lenient vote, not a fixed set;
    # handled specially in main() by calling the real vote logic directly (see
    # get_wf_hours_and_split above -- a hand-transcribed fixed set was tried first and
    # was off by ~2x for CANA+, caught by cross-checking against the published numbers).
    "CANA+":               dict(n=3.00, rr=1.0, age_h=2.0, bounce=True, touches=4,
                                 session="VOTE", pool_fn="cana", holdout=True),
    "CANA (faithful)":     dict(n=3.00, rr=1.0, age_h=2.0, bounce=True, touches=4,
                                 session=None, holdout=False),
    "DTFX Robust":         dict(n=2.0, rr=1.0, age_h=2.0, bounce=True, touches=8,
                                 session="VOTE", pool_fn="dtfx_robust", holdout=True),
    "SOMEGA (live)":       dict(n=3.343496, rr=1.0, age_h=3.366682, bounce=False, touches=4,
                                 session={"EURUSD": None, "EURGBP": hge(18),
                                          "GBPUSD": hrs(20, 21, 22), "USDJPY": hrs(19, 20, 21, 22)},
                                 holdout=False),
    "SOMEGA Robust":       dict(n=3.343496, rr=1.0, age_h=3.366682, bounce=False, touches=4,
                                 session="VOTE", pool_fn="somega_robust", holdout=True),
    "goal70_final_winner": dict(n=2.0, rr=1.0,
                                 age_h={"EURGBP": 2.5, "GBPUSD": 2.5, "USDJPY": 2.5, "EURUSD": 3.366682},
                                 bounce=True, touches=4,
                                 session={"EURGBP": hge(18), "GBPUSD": hrs(20, 21, 22),
                                          "USDJPY": hrs(19, 20, 21, 22), "EURUSD": hrs(19, 20, 21, 22)},
                                 holdout=False),
}


def build_trades(n, rr, age_h, bounce, touches, session):
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

        n_i = n[inst] if isinstance(n, dict) else n
        age_i = age_h[inst] if isinstance(age_h, dict) else age_h
        target_mult = n_i * rr

        risk_pct = (n_i * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]

        touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(age_i * 3600 * 1e9), n_i, target_mult, bounce, pg.PROBE_DEPTH, 1.0,
            touches, pg.MAX_WINDOW_BARS)

        if len(entry_idx) == 0:
            continue

        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, rr, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        entry_time = pd.to_datetime(times[entry_idx])
        exit_time = pd.to_datetime(times[exit_idx])

        keep = np.ones(len(entry_idx), dtype=bool)
        if isinstance(session, dict):
            allowed = session.get(inst)
            if allowed is not None:
                keep = np.isin(entry_time.hour, list(allowed))
        # session == "VOTE" (CANA+/DTFX Robust/SOMEGA Robust) or None: no static filter
        # here -- "VOTE" is applied afterward in main() using the real vote's own output.

        frames.append(pd.DataFrame({
            "instrument": inst, "entry_time": entry_time[keep], "exit_time": exit_time[keep],
            "entry_price": entry_mid[keep], "stop_risk": stop_risk[keep],
            "net_r": net_r[keep], "is_win": is_win[keep],
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def sanity_check(label, t):
    """Cross-check against goal12_strategy_scores.csv's already-published trades/tpy/wr."""
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = len(t) / years
    wr = 100 * t.is_win.mean()
    print(f"  {label:22s} n={len(t):7,} ({tpy:6.0f}/yr) WR={wr:5.2f}%")


# Pool builders matching goal8_unified_holdout.py::main()'s three vote-based families
# exactly (same base trade population the real vote is computed on).
POOL_BUILDERS = {
    "cana": lambda inst: cana.build_cana_trades(inst),
    "dtfx_robust": lambda inst: hopt.build_all(inst),
    "somega_robust": lambda inst: pg.build_trades(
        inst, 3.343496, 3.343496, 3.366682, require_bounce=False, session_filter=False),
}


def vote_filtered_trades(cfg):
    """CANA+/DTFX Robust/SOMEGA Robust: get the real vote's wf_hours + split point from
    the canonical pool-building functions, then apply that to OUR OWN true-entry/exit
    reconstruction (built with the same zone params) instead of a hand-transcribed set."""
    pools = {}
    for inst in TRADEABLE:
        t = POOL_BUILDERS[cfg["pool_fn"]](inst)
        t["glog"] = np.log1p(0.05 * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools[inst] = t
    wf_hours, half = get_wf_hours_and_split(pools)

    raw = build_trades(cfg["n"], cfg["rr"], cfg["age_h"], cfg["bounce"], cfg["touches"], session=None)
    block = raw.entry_time.dt.hour // 6
    keep = np.zeros(len(raw), dtype=bool)
    for inst in TRADEABLE:
        m = (raw.instrument.values == inst) & block.isin(wf_hours[inst]).values & (raw.entry_time >= half).values
        keep |= m
    return raw[keep].sort_values("entry_time").reset_index(drop=True)


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Rebuilding TRUE entry/exit trade streams for all 14 strategies (C++ engine)...")
    rows = []
    for label, cfg in STRATEGIES.items():
        if cfg["session"] == "VOTE":
            sub = vote_filtered_trades(cfg)
        else:
            t = build_trades(cfg["n"], cfg["rr"], cfg["age_h"], cfg["bounce"], cfg["touches"], cfg["session"])
            sub = t[t.entry_time >= SPLIT] if cfg["holdout"] else t
        sanity_check(label, sub)

        _, final_eq, max_dd, accepted, _ = g20.simulate_margin_constrained(
            sub, start=g20.START, risk_pct=g20.RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
            cap=g20.CAP, track_detail=False)
        accept_pct = 100 * accepted.mean()
        rows.append(dict(strategy=label, n_trades=len(sub), accept_pct=accept_pct,
                          final_equity_200start=final_eq, max_dd_pct=max_dd))
        print(f"    -> margin accept rate ($200 start, 5% risk, 50x leverage): {accept_pct:5.1f}%")

    df = pd.DataFrame(rows).sort_values("accept_pct", ascending=False)
    df.to_csv(os.path.join(DATA_DIR, "goal23_margin_capacity.csv"), index=False)
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved: goal23_margin_capacity.csv")


if __name__ == "__main__":
    main()
