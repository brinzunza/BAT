"""Direct user request (2026-09-07): "backtest 5 pip vs 10 pip deeper to see if 5 pip can
realistically work given 1.6 spread on every entry or higher. show average R on win,
average R on loss, winrate, trade count, etc."

Deeper than goal103's single-spread sweep: crosses BOTH candidate minimum zone heights
(5 and 10 pips) against a spread stress ladder, decomposing win/loss into their own
average R rather than just a blended avgR -- which matters here specifically because of
how spread interacts with a fixed 1:1R target (dtfx.md #7 variant A):

  entry_price = mid +/- spread_price   (spread paid on the entry side only, dtfx.md #8)
  stop_price  = the zone's OPPOSITE edge (no spread)
  eff_risk    = |entry_price - stop_price|          -- this is HALF the raw zone height,
                                                         PLUS one spread (for a normal,
                                                         non-"deep" fill)
  win:  net_r = |target - entry_price| / eff_risk = (half_height - spread) / (half_height + spread)
  loss: net_r = -1.0 exactly, always (by construction of a fixed 1:1R target)

So avg-R-on-loss is trivially -1.0000 in every single row below -- shown anyway, plainly,
rather than silently omitted, so the reader can see that's true by construction and not a
result. Avg-R-on-win is the number that actually moves, and it shrinks fast as spread
approaches a small zone's own half-height -- exactly the mechanism this test is checking.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal99_dtfx_base_backtest as g99
import goal104_optimal_risk_two_account as g104

PRE_END = pd.Timestamp("2021-01-01")
POST_START = pd.Timestamp("2026-04-01")
ZONE_HEIGHTS_PIPS = [5.0, 10.0]
SPREAD_SWEEP_PIPS = [1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440


def resolve_all(cache, min_zone_pips, spread_pips):
    """Returns per-instrument (entry_time, net_r, is_buy) arrays pooled together, for one
    (zone height, spread) combo, across all 4 pairs."""
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = spread_pips * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price,
            MAX_WINDOW, False, min_zone_height)

        resolved = outcome > 0
        ei = entry_idx[resolved]
        frames.append(pd.DataFrame({
            "instrument": inst,
            "entry_time": d["times"].values[ei],
            "net_r": net_r[resolved],
            "is_win": outcome[resolved] == 1,
        }))
    return pd.concat(frames, ignore_index=True)


def stats_row(df, label):
    n = len(df)
    if n == 0:
        return dict(label=label, n=0, wr=float("nan"), avg_r_win=float("nan"),
                     avg_r_loss=float("nan"), pf=float("nan"), avg_r=float("nan"))
    r = df["net_r"].values
    wins = df["is_win"].values
    wr = 100.0 * wins.mean()
    avg_r_win = r[wins].mean() if wins.any() else float("nan")
    avg_r_loss = r[~wins].mean() if (~wins).any() else float("nan")
    gross_win = r[r > 0].sum()
    gross_loss = -r[r <= 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return dict(label=label, n=n, wr=wr, avg_r_win=avg_r_win, avg_r_loss=avg_r_loss,
                pf=pf, avg_r=r.mean())


def print_row(row):
    print(f"  {row['label']:<28} n={row['n']:>8,}  WR={row['wr']:>6.2f}%  "
          f"avgR(win)={row['avg_r_win']:>+7.4f}  avgR(loss)={row['avg_r_loss']:>+7.4f}  "
          f"PF={row['pf']:>6.3f}  avgR(all)={row['avg_r']:>+7.4f}")


def main():
    t0 = time.time()
    print("Loading + building zones once (shared across every height/spread combo)...")
    cache = g104.load_and_build_zones()

    all_rows = []
    for min_zone_pips in ZONE_HEIGHTS_PIPS:
        print(f"\n{'='*90}\nmin_zone = {min_zone_pips:.0f} pips -- spread stress, PRE-2021 data only "
              f"(honest/trustworthy window, dtfx.md convention)\n{'='*90}")
        for spread_pips in SPREAD_SWEEP_PIPS:
            df = resolve_all(cache, min_zone_pips, spread_pips)
            pre = df[df["entry_time"] < np.datetime64(PRE_END)]
            row = stats_row(pre, f"spread={spread_pips:.1f}p")
            row.update(min_zone_pips=min_zone_pips, spread_pips=spread_pips, regime="PRE-2021")
            print_row(row)
            all_rows.append(row)

        # Walk-forward honesty check at the baseline spread: does the height's edge survive
        # on the genuinely-never-touched POST-2026-04 holdout, same as dtfx.md's methodology?
        print(f"\n  -- walk-forward holdout check, min_zone={min_zone_pips:.0f}p, spread=1.6p --")
        df16 = resolve_all(cache, min_zone_pips, 1.6)
        smoothed = df16[(df16["entry_time"] >= PRE_END) & (df16["entry_time"] < POST_START)]
        post = df16[df16["entry_time"] >= POST_START]
        for chunk, tag in [(smoothed, "SMOOTHED (2021-2026.03, excluded)"), (post, "POST-2026-04 (genuine holdout)")]:
            row = stats_row(chunk, tag)
            row.update(min_zone_pips=min_zone_pips, spread_pips=1.6, regime=tag)
            print_row(row)
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(DATA_DIR, "goal111_5pip_vs_10pip_depth.csv"), index=False)
    print(f"\nSaved -> data/goal111_5pip_vs_10pip_depth.csv")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
