"""Direct user request (2026-09-06): "backtest different min pip zone size to make sure it
stays profitable given a 1.6 pip spread on every entry" -- a robustness sweep flagged as a
natural next step in dtfx.md's Status section after Head to Head found variant C's
theoretically-better R:R gets erased by spread on this "every swing" mechanism's mostly-tiny
zones.

Sweeps a RAW zone-height floor (checked before any spread adjustment -- distinct from the
existing min_stop_price, which floors the post-spread effective risk) across both target
conventions, all four pairs, full history each. Loads and computes swings/zones ONCE per
instrument, then re-runs the (cheap, <1s) resolver at each threshold against those same
cached zone arrays -- avoids re-reading four multi-hundred-MB CSVs per threshold.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0     # unchanged post-spread risk floor, same as Head to Head
MAX_WINDOW = 1440
MIN_ZONE_PIPS_SWEEP = [0, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]


def load_and_build(inst, fname, pip):
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, parse_dates=["date"])
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    close_ = df["close"].values.astype(np.float64)
    n_bars = len(df)

    confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_, low_, close_)
    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
     ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
    last_high_of_bar, last_low_of_bar = g99.build_last_confirmed(confirm_idx, is_high, price, n_bars)
    last_hh_of_bar, last_ll_of_bar = g99.build_confirmed_hh_ll(confirm_idx, is_high, price, n_bars)

    return dict(high=high_, low=low_, z_created=z_created, z_top=z_top, z_bottom=z_bottom,
                z_is_buy=z_is_buy, last_high_of_bar=last_high_of_bar, last_low_of_bar=last_low_of_bar,
                last_hh_of_bar=last_hh_of_bar, last_ll_of_bar=last_ll_of_bar, n_zones=len(z_created))


def stats_from(outcome, net_r):
    resolved = outcome > 0
    n = int(resolved.sum())
    if n == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), avg_r=float("nan"))
    r = net_r[resolved]
    wr = 100.0 * (outcome[resolved] == 1).mean()
    gl = -r[r <= 0].sum()
    pf = r[r > 0].sum() / gl if gl > 0 else float("inf")
    return dict(n=n, wr=wr, pf=pf, avg_r=r.mean())


def main():
    t0 = time.time()
    cache = {}
    for inst, (fname, pip) in INSTRUMENTS.items():
        t1 = time.time()
        cache[inst] = (load_and_build(inst, fname, pip), pip)
        print(f"{inst}: loaded + zones built, {cache[inst][0]['n_zones']:,} zones  ({time.time()-t1:.1f}s)")

    rows = []
    for min_pips in MIN_ZONE_PIPS_SWEEP:
        pooled_a_n, pooled_a_r = [], []
        pooled_c_n, pooled_c_r = [], []
        for inst, (d, pip) in cache.items():
            min_zone_height = min_pips * pip
            spread_price = SPREAD_PIPS * pip
            min_stop_price = MIN_STOP_PIPS * pip

            _, _, outcome_a, net_r_a, _ = g99.resolve_base_strategy(
                d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
                d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price,
                MAX_WINDOW, False, min_zone_height)
            _, _, outcome_c, net_r_c, _, _, _ = g99.resolve_fixed_structural_target(
                d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
                d["last_hh_of_bar"], d["last_ll_of_bar"], spread_price, min_stop_price,
                MAX_WINDOW, min_zone_height)

            sa = stats_from(outcome_a, net_r_a)
            sc = stats_from(outcome_c, net_r_c)
            rows.append(dict(min_pips=min_pips, instrument=inst, variant="A", **sa))
            rows.append(dict(min_pips=min_pips, instrument=inst, variant="C", **sc))

            resolved_a = outcome_a > 0
            pooled_a_n.append(sa["n"]); pooled_a_r.append(net_r_a[resolved_a])
            resolved_c = outcome_c > 0
            pooled_c_n.append(sc["n"]); pooled_c_r.append(net_r_c[resolved_c])

        ra = np.concatenate(pooled_a_r) if pooled_a_n else np.array([])
        rc = np.concatenate(pooled_c_r) if pooled_c_n else np.array([])
        for label, r in [("A", ra), ("C", rc)]:
            n = len(r)
            if n == 0:
                rows.append(dict(min_pips=min_pips, instrument="ALL", variant=label, n=0, wr=float("nan"), pf=float("nan"), avg_r=float("nan")))
                continue
            wr = 100.0 * (r > 0).mean()
            gl = -r[r <= 0].sum()
            pf = r[r > 0].sum() / gl if gl > 0 else float("inf")
            rows.append(dict(min_pips=min_pips, instrument="ALL", variant=label, n=n, wr=wr, pf=pf, avg_r=r.mean()))
        print(f"min_pips={min_pips:>4}  A: n={len(ra):>10,} WR={100*(ra>0).mean() if len(ra) else float('nan'):.2f}% PF={(ra[ra>0].sum()/-ra[ra<=0].sum() if len(ra) and (ra<=0).any() else float('nan')):.3f}   "
              f"C: n={len(rc):>10,} WR={100*(rc>0).mean() if len(rc) else float('nan'):.2f}% PF={(rc[rc>0].sum()/-rc[rc<=0].sum() if len(rc) and (rc<=0).any() else float('nan')):.3f}  ({time.time()-t0:.0f}s elapsed)")

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "goal103_min_zone_size_sweep.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
