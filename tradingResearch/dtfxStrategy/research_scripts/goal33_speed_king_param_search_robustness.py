"""Speed-King parameter-plateau, joint-parameter, and walk-forward robustness (tests
#9, #11, #12): is (N=7.0, R:R=0.20, age_h=0.175) a narrow spike that happened to score
well, or does performance stay strong across a broad neighborhood -- and does
re-optimizing on expanding windows of history keep landing in the same region?

g-score (this project's standing screening metric from the original age-optimization
discovery, goal16-19): g = trades_per_year * mean(log(1 + 0.05*net_r)) -- rewards both
frequency and quality, used here for grid selection; PF/WR are reported for interpretability
against the user's own PF >= 1.2-1.3 threshold (test #2).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import dtfx_robust_search as rs
import prod_filter_grid_search as pg

TRADEABLE = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
BASE = dict(n=7.0, rr=0.20, age_h=0.175, bounce=False, touches=4)
SPLIT = pd.Timestamp("2018-03-09")


def build(n, rr, age_h, bounce=False, touches=4):
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
        touch_num, entry_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(age_h * 3600 * 1e9), n, n * rr, bounce, pg.PROBE_DEPTH, 1.0, touches, pg.MAX_WINDOW_BARS)
        if len(entry_idx) == 0:
            continue
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, rr, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({"instrument": inst, "entry_time": pd.to_datetime(times[entry_idx]),
                                     "net_r": net_r, "is_win": is_win}))
    if not frames:
        return pd.DataFrame(columns=["instrument", "entry_time", "net_r", "is_win"])
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def stats(t):
    if len(t) < 30:
        return dict(n=len(t), tpy=0.0, wr=float("nan"), pf=float("nan"), g=-999.0)
    years = max((t.entry_time.max() - t.entry_time.min()).days / 365.25, 0.01)
    tpy = len(t) / years
    wr = 100 * t.is_win.mean()
    w = t.loc[t.net_r > 0, "net_r"].sum(); l = -t.loc[t.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    g = tpy * np.log1p(0.05 * t.net_r).mean()
    return dict(n=len(t), tpy=tpy, wr=wr, pf=pf, g=g)


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    # ---- #11: one-at-a-time parameter plateau (full history, holding others fixed) --
    print("=" * 94)
    print("#11 -- Parameter plateau: one-at-a-time sweeps around Speed-King's chosen point")
    print(f"       (N={BASE['n']}, R:R={BASE['rr']}, age_h={BASE['age_h']})")
    print("=" * 94)

    sweeps = {
        "N (stop mult)": ("n", [4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0]),
        "R:R (target frac)": ("rr", [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30, 0.35]),
        "age_h (fill age, hours)": ("age_h", [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00]),
    }
    plateau_rows = []
    for label, (key, values) in sweeps.items():
        print(f"\n  {label}:")
        for v in values:
            cfg = dict(BASE); cfg[key] = v
            t = build(cfg["n"], cfg["rr"], cfg["age_h"], cfg["bounce"], cfg["touches"])
            s = stats(t)
            marker = "  <-- chosen" if abs(v - BASE[key]) < 1e-9 else ""
            print(f"    {key}={v:<7} n={s['n']:7,} ({s['tpy']:6,.0f}/yr)  WR={s['wr']:5.2f}%  "
                  f"PF={s['pf']:.3f}  g={s['g']:.3f}{marker}")
            plateau_rows.append(dict(sweep=label, key=key, value=v, **s))
    pdf = pd.DataFrame(plateau_rows)
    pdf.to_csv(os.path.join(DATA_DIR, "goal33_plateau_sweep.csv"), index=False)

    print("\n  Plateau width (contiguous range around the chosen point where PF stays >= 1.3,")
    print("  the user's own upper robustness threshold):")
    for label, (key, values) in sweeps.items():
        sub = pdf[pdf.key == key].sort_values("value")
        ok = sub.pf >= 1.3
        # find the contiguous run of `ok` that contains the chosen value
        chosen_pos = (sub.value - BASE[key]).abs().idxmin()
        vals = sub.value.tolist(); oks = ok.tolist()
        ci = sub.index.get_loc(chosen_pos)
        lo = ci
        while lo > 0 and oks[lo - 1]: lo -= 1
        hi = ci
        while hi < len(oks) - 1 and oks[hi + 1]: hi += 1
        print(f"    {label}: PF>=1.3 holds from {vals[lo]} to {vals[hi]} "
              f"({'chosen value inside this range' if oks[ci] else 'chosen value NOT in a >=1.3 run'})")

    # ---- #12: joint multi-parameter grid --------------------------------------------
    print(f"\n{'='*94}\n#12 -- Joint multi-parameter robustness (3x3x3 grid around the chosen point)\n{'='*94}")
    n_grid = [6.0, 7.0, 8.0]
    rr_grid = [0.15, 0.20, 0.25]
    age_grid = [0.125, 0.175, 0.225]
    joint_rows = []
    for n in n_grid:
        for rr in rr_grid:
            for age in age_grid:
                t = build(n, rr, age)
                s = stats(t)
                joint_rows.append(dict(n_mult=n, rr=rr, age_h=age, **s))
    jdf = pd.DataFrame(joint_rows)
    jdf.to_csv(os.path.join(DATA_DIR, "goal33_joint_grid.csv"), index=False)
    n_pass = int((jdf.pf >= 1.2).sum())
    print(f"  {n_pass}/{len(jdf)} of the 27 combinations in this neighborhood have PF >= 1.2 "
          f"(range: PF {jdf.pf.min():.3f} - {jdf.pf.max():.3f})")
    worst = jdf.loc[jdf.pf.idxmin()]
    print(f"  Worst combo in the neighborhood: N={worst.n_mult}, R:R={worst.rr}, age_h={worst.age_h} "
          f"-> PF={worst.pf:.3f}, WR={worst.wr:.2f}%")

    # ---- #9: lightweight walk-forward (expanding window re-selection) ---------------
    print(f"\n{'='*94}\n#9 -- Walk-forward: re-select (N,R:R,age_h) on an EXPANDING window, test on the next fold\n{'='*94}")
    print("  Modest 3x3x4=36-point grid per fold (not the full Optuna search used for the")
    print("  Robust family elsewhere in this project, but enough to test regional stability).")
    wf_n = [6.0, 7.0, 8.0]
    wf_rr = [0.15, 0.20, 0.25]
    wf_age = [0.125, 0.175, 0.225, 0.30]
    folds = [
        ("2009-2018 -> 2018-2020", "2018-03-09", "2018-01-01", "2020-01-01"),
        ("2009-2020 -> 2020-2022", "2020-01-01", "2020-01-01", "2022-01-01"),
        ("2009-2022 -> 2022-2024", "2022-01-01", "2022-01-01", "2024-01-01"),
        ("2009-2024 -> 2024-2026", "2024-01-01", "2024-01-01", "2026-08-31"),
    ]
    # cache builds across folds since the same (n,rr,age) combo is reused per fold boundary
    build_cache = {}
    def cached_build(n, rr, age):
        key = (n, rr, age)
        if key not in build_cache:
            build_cache[key] = build(n, rr, age)
        return build_cache[key]

    wf_rows = []
    for label, train_end, test_start, test_end in folds:
        best_g, best_cfg, best_train_stats = -1e18, None, None
        for n in wf_n:
            for rr in wf_rr:
                for age in wf_age:
                    full = cached_build(n, rr, age)
                    train = full[full.entry_time < train_end]
                    s = stats(train)
                    if s["g"] > best_g and s["n"] >= 200:
                        best_g, best_cfg, best_train_stats = s["g"], (n, rr, age), s
        n, rr, age = best_cfg
        full = cached_build(n, rr, age)
        test = full[(full.entry_time >= test_start) & (full.entry_time < test_end)]
        s_test = stats(test)
        print(f"  Fold {label}: train-selected N={n}, R:R={rr}, age_h={age} "
              f"(train g={best_train_stats['g']:.3f}, PF={best_train_stats['pf']:.3f})")
        print(f"    -> OOS test: n={s_test['n']:,}  WR={s_test['wr']:.2f}%  PF={s_test['pf']:.3f}  "
              f"g={s_test['g']:.3f}")
        wf_rows.append(dict(fold=label, n=n, rr=rr, age_h=age, train_g=best_train_stats["g"],
                             train_pf=best_train_stats["pf"], test_n=s_test["n"], test_wr=s_test["wr"],
                             test_pf=s_test["pf"], test_g=s_test["g"]))
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(os.path.join(DATA_DIR, "goal33_walk_forward.csv"), index=False)
    print(f"\n  Summary: {int((wf_df.test_pf >= 1.0).sum())}/{len(wf_df)} folds profitable OOS "
          f"(PF>=1.0); region selected: N in {sorted(set(wf_df.n))}, R:R in {sorted(set(wf_df.rr))}, "
          f"age_h in {sorted(set(wf_df.age_h))} (Speed-King's actual choice: N=7.0, R:R=0.20, age_h=0.175)")


if __name__ == "__main__":
    main()
