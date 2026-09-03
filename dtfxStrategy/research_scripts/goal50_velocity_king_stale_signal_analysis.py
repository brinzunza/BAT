"""Velocity-King: does trading a zone LATE (after it's already sat mature for a long time)
produce worse outcomes than trading it fresh? Directly motivated by a live-ops finding
(2026-09-03): with no staleness cap in strategy/engine.py, a zone that keeps losing the
MAX_CONCURRENT_POSITIONS / MAX_CONCURRENT_PER_SIDE race via undo_dispatch() retries FOREVER
-- confirmed live, zones up to 250+ hours (10+ days) old were still "dispatching" every
20s poll. This never shows up in any margin-capacity backtest in this project, because
every one of them (goal20/goal23/goal28/goal29/goal30) modeled a margin-rejected signal as
PERMANENTLY LOST, not retried -- the live bot's actual behavior has never been tested for
edge quality before.

This can't be answered with the existing single-touch-immediate-fill trade data (entry_idx
there is always "first touch after maturity", which conflates zone quality with how long it
organically took price to return to mid -- not the same thing as a concurrency-forced delay,
but the natural variation IS a real, valid proxy for the underlying question: is trading a
zone a long time after it matured associated with worse outcomes, regardless of WHY the
delay happened. Buckets trades by (actual fill time - zone maturity time) and compares
WR/PF/mean-R across buckets.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal29_rerank_1_per_side as g29

rs, pg, TRADEABLE = g29.rs, g29.pg, g29.TRADEABLE
VELOCITY_CFG = dict(n=5.0, rr=0.20, age_h=0.20, bounce=False, touches=16)
MIN_FILL_AGE_H = VELOCITY_CFG["age_h"]


def build_with_created_at():
    n, rr, age_h, bounce, touches = (VELOCITY_CFG["n"], VELOCITY_CFG["rr"], VELOCITY_CFG["age_h"],
                                      VELOCITY_CFG["bounce"], VELOCITY_CFG["touches"])
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (n * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(age_h * 3600 * 1e9), n, n * rr, bounce, pg.PROBE_DEPTH, 1.0, touches, pg.MAX_WINDOW_BARS)
        if len(entry_idx) == 0:
            continue
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, rr, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        # created_at per OUTPUT trade: entry_mid in the output is always an EXACT copy of
        # the source zone's own `mid` (DTFX entry is always the zone's 50% level, and a
        # re-touch re-arms from the SAME zone boundaries) -- not computed or rounded, a
        # direct passthrough, so it's a safe, exact join key back to the input zone array.
        # Not all input zones produce output rows (never touched / invalidated first), so
        # a positional/cumsum reconstruction over touch_num resets silently misaligns the
        # moment any zone is skipped; joining on mid sidesteps that entirely.
        mid_to_created = dict(zip(mid, c_ns))
        created_ns_per_row = np.array([mid_to_created[m] for m in entry_mid], dtype="int64")
        created_at_per_row = pd.to_datetime(created_ns_per_row.astype("datetime64[ns]"))
        frames.append(pd.DataFrame({
            "instrument": inst, "entry_time": pd.to_datetime(times[entry_idx]),
            "exit_time": pd.to_datetime(times[exit_idx]), "created_at": created_at_per_row,
            "touch_num": touch_num, "net_r": net_r, "is_win": is_win,
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Building Velocity-King trades with zone created_at attached...", flush=True)
    t = build_with_created_at()
    print(f"  {len(t):,} trades, {t.entry_time.min()} -> {t.entry_time.max()}")

    # Only touch_num==1 rows are meaningful for "time since zone matured" -- touch #2+
    # re-arm from the PRIOR touch's resolution, not from the original zone creation, so
    # created_at doesn't represent "how long has this specific opportunity been waiting"
    # for them. Sanity check this restriction doesn't gut the sample.
    first_touch = t[t.touch_num == 1].copy()
    print(f"  {len(first_touch):,} are first-touch (the ones this question applies to)")

    first_touch["wait_h"] = (first_touch.entry_time - first_touch.created_at).dt.total_seconds() / 3600
    first_touch["extra_wait_h"] = first_touch.wait_h - MIN_FILL_AGE_H

    print(f"\nExtra-wait distribution (beyond the {MIN_FILL_AGE_H}h mandatory gate):")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  P{p}: {np.percentile(first_touch.extra_wait_h, p):.2f}h")
    print(f"  Max: {first_touch.extra_wait_h.max():.1f}h ({first_touch.extra_wait_h.max()/24:.1f} days)")

    bins = [-0.01, 0, 1, 6, 24, 24*3, 24*10, 24*30, np.inf]
    labels = ["0h (instant)", "0-1h", "1-6h", "6-24h", "1-3d", "3-10d", "10-30d", "30d+"]
    first_touch["bucket"] = pd.cut(first_touch.extra_wait_h, bins=bins, labels=labels)

    print(f"\n{'Bucket':>14} {'n':>8} {'WR':>8} {'PF':>8} {'mean_r':>9} {'median_r':>10}")
    rows = []
    for label in labels:
        sub = first_touch[first_touch.bucket == label]
        if len(sub) < 5:
            print(f"{label:>14} {len(sub):>8}  (too few to report)")
            continue
        wr = 100 * sub.is_win.mean()
        w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        rows.append(dict(bucket=label, n=len(sub), wr=wr, pf=pf, mean_r=sub.net_r.mean(), median_r=sub.net_r.median()))
        print(f"{label:>14} {len(sub):>8,} {wr:>7.2f}% {pf:>8.3f} {sub.net_r.mean():>+9.4f} {sub.net_r.median():>+10.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "goal50_stale_signal_buckets.csv"), index=False)

    # Also: does the FIRST bucket a live retry-cap decision would care about (concurrency-
    # driven delays, which in the live system are typically minutes to a few hours, not
    # days) look meaningfully different from "instant/near-instant" fills?
    overall_wr = 100 * first_touch.is_win.mean()
    w = first_touch.loc[first_touch.net_r > 0, "net_r"].sum(); l = -first_touch.loc[first_touch.net_r <= 0, "net_r"].sum()
    overall_pf = w / l if l > 0 else float("inf")
    print(f"\nOverall (all first-touch trades): WR={overall_wr:.2f}%  PF={overall_pf:.3f}")


if __name__ == "__main__":
    main()
