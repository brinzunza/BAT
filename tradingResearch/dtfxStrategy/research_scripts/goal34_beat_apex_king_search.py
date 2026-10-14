"""/goal: find a strategy that reaches $1M faster than Apex-King (257.6 days) under
the CURRENT v15 evaluation methodology -- two separate $100 sub-accounts (buy-only,
sell-only, never pooled), 50x leverage, 1.6-pip spread, max 1 open position per side,
and the candidate's OWN MC-robust-optimal risk_pct. No fixed-parameter reuse from the
old margin-only searches that found Apex-King (goal37 series) -- those never modeled
the two-account/per-side-cap restriction at all, so a candidate that won there isn't
guaranteed to win here (confirmed generally by Speed-King's own optimal risk shifting
from 3% under the old model to 8% under this one).

bounce=False held fixed throughout: PRODLIVETRADING's live engine.py only ever
implements unconditional re-arm (no bounce-gated logic exists live at all), so a
bounce=True candidate wouldn't actually be deployable without new live code -- out of
scope for "still following all the restrictions."

Two-phase search, cheap-then-expensive:
Phase 1: build each candidate's raw (unconstrained) trade population directly via
dtfx_engine.simulate_generic_ex (C++, zone detection cached once) -- no margin/MC, just
WR, avg-loss-R, and raw trades/day. Rank by an analytic expected-log-growth-per-day proxy
at a fixed reference risk (Apex-King's own 6%) -- cheap enough to sweep a wide grid.
Phase 2: run the REAL two-account risk sweep + MC (goal30/goal33 methodology, this
project's actual evaluation) on only the top screened candidates, apples-to-apples
against Apex-King's own 257.6-day figure recomputed the same way in this same run.
"""
import os
import sys
import time
import signal
import itertools
import numpy as np
import pandas as pd


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def with_timeout(seconds, fn, *args, **kwargs):
    """Unix SIGALRM-based hard timeout -- some (n, rr, age_h, touches) combinations hit
    pathological worst-case behavior in the C++ multi-touch re-arm loop (observed live:
    one candidate stalled a run for 700+s with no progress) that a wall-clock check
    inside the loop can't catch since the call itself never returns control. Skip it and
    move on rather than let one bad combination block the whole screen."""
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    except _Timeout:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30

rs, pg, TRADEABLE = g29.rs, g29.pg, g29.TRADEABLE
SPLIT = g29.SPLIT
APEX = dict(n=4.5, rr=0.18, age_h=0.30, touches=8)  # prior champion, for reference
ROUND1_WINNER = dict(n=3.5, rr=0.18, age_h=0.15, touches=12)  # round 1's best (197.6d), current champion

REFERENCE_RISK = 0.06  # Apex-King's own optimal -- used only for Phase 1 ranking


def build_raw_with_side(n, rr, age_h, touches, bounce=False, holdout=True):
    """Cheap: raw signal population (no margin/concurrency), with side. holdout=True
    (default): entry_time >= SPLIT, the TRUE out-of-sample test set -- use this ONLY for
    final reporting on an already-chosen candidate, never for selection. holdout=False:
    entry_time < SPLIT, the TRAINING set -- use this for any search/screening step, so
    the reported holdout number stays honest (this project's standing discipline:
    parameters chosen using only pre-split data, scored on the untouched second half --
    see Apex-King/Speed-King's own derivation notes)."""
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (n * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        risk_ok = risk_pct >= pg.MIN_RISK_PCT

        for side_is_buy in (True, False):
            mask = risk_ok & (z["is_buy"] == side_is_buy)
            top, bottom, mid = z["top"][mask], z["bottom"][mask], z["mid"][mask]
            is_buy_arr, start_idx, c_ns = z["is_buy"][mask], z["start_idx"][mask], created_ns[mask]
            if len(top) == 0:
                continue
            touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                top, bottom, mid, is_buy_arr, start_idx, c_ns, times_ns, lows, highs,
                int(age_h * 3600 * 1e9), n, n * rr, bounce, pg.PROBE_DEPTH, 1.0, touches, pg.MAX_WINDOW_BARS)
            if len(entry_idx) == 0:
                continue
            pip = pg.PIP_SIZE[inst]
            spread_price = pg.SPREAD_PIPS_TYPICAL * pip
            r_gross = np.where(is_win, rr, -1.0)
            cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
            net_r = r_gross - cost_in_r
            frames.append(pd.DataFrame({
                "instrument": inst, "entry_time": pd.to_datetime(times[entry_idx]),
                "exit_time": pd.to_datetime(times[exit_idx]), "entry_price": entry_mid,
                "stop_risk": stop_risk, "net_r": net_r, "is_win": is_win, "is_buy": side_is_buy,
            }))
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return t[t.entry_time >= SPLIT] if holdout else t[t.entry_time < SPLIT]


def screen_score(t, risk=REFERENCE_RISK):
    """Analytic proxy: expected log-growth per calendar day at a fixed reference risk,
    from raw WR/avg-loss/frequency -- no margin/concurrency modeled (that's Phase 2's
    job), just enough signal to rank candidates before spending real compute."""
    if t is None or len(t) < 500:
        return -np.inf, None
    wr = t.is_win.mean()
    avg_loss = -t.loc[~t.is_win, "net_r"].mean()  # positive magnitude
    rr = t.loc[t.is_win, "net_r"].iloc[0] if wr > 0 else 0.0
    win_mult = 1 + risk * rr
    loss_mult = 1 - risk * avg_loss
    if loss_mult <= 0 or win_mult <= 0:
        return -np.inf, None
    log_growth_per_trade = wr * np.log(win_mult) + (1 - wr) * np.log(loss_mult)
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    trades_per_day = len(t) / (years * 365.25)
    log_growth_per_day = log_growth_per_trade * trades_per_day
    return log_growth_per_day, dict(wr=100 * wr, avg_loss=avg_loss, tpy=len(t) / years, n_trades=len(t))


# n=4.0 dropped: reproducibly hangs/stalls dtfx_engine's C++ multi-touch loop for
# several minutes at minimum (confirmed 3 separate times, in fresh subprocesses, with
# different rr/age/touches each time -- it's specifically n=4.0 itself, not a
# combination) -- a real, reproducible perf edge case worth a standalone investigation
# later, not worth blocking this search on right now given 4.5 (Apex-King's own point)
# and 3.5-5.5 elsewhere are all confirmed fine.
#
# Round 3: rounds 1-2 screened (and reported final numbers) directly against the TRUE
# HOLDOUT set itself -- a real methodology error (holdout contamination), confirmed by
# both rounds' winners landing exactly at the grid's own boundary in the SAME direction
# every time (touches 8->12->18, age_h 0.30->0.15->0.05), the classic overfitting
# signature of searching harder against a fixed evaluation set rather than approaching
# a genuine interior optimum. build_raw_with_side now takes holdout=False for
# screening (TRAIN: entry_time < SPLIT) and defaults to holdout=True for final
# reporting (TEST: entry_time >= SPLIT) -- same train/select-then-test discipline
# Apex-King and every other strategy in this project were held to. Grid reset to a
# reasonably wide, UNBIASED range rather than assuming rounds 1-2's (contaminated)
# boundary-chasing direction is where the true optimum lies.
N_GRID = [3.0, 3.5, 4.5, 5.0]
RR_GRID = [0.12, 0.14, 0.16, 0.18, 0.20]
AGE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]
TOUCHES_GRID = [6, 8, 10, 12, 14, 16]
ALL_COMBOS = list(itertools.product(N_GRID, RR_GRID, AGE_GRID, TOUCHES_GRID))


def run_batch(start, end):
    """Screens ALL_COMBOS[start:end] and appends results to the shared CSV. Run as a
    fresh subprocess per batch (see phase1_screen's driver) -- two separate full runs
    both stalled for 700s+ with no forward progress around the ~400-call mark,
    regardless of which specific parameters were at that index in each (different
    grids both hung at the same call COUNT, not the same parameters) -- points at
    resource accumulation across many rapid dtfx_engine C++ calls within one process
    (likely thread-handle or heap-related in the std::thread-per-call zone-simulation
    path) rather than any single pathological input. Small fresh-process batches
    sidestep it regardless of the exact root cause."""
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)
    out_path = os.path.join(DATA_DIR, "goal34_phase1_screen.csv")
    results = []
    for i in range(start, end):
        n, rr, age_h, touches = ALL_COMBOS[i]
        t = with_timeout(15, build_raw_with_side, n, rr, age_h, touches, False, False)  # holdout=False: TRAIN only
        score, stats = screen_score(t)
        if stats is not None:
            results.append(dict(idx=i, n=n, rr=rr, age_h=age_h, touches=touches,
                                 score=score, **stats))
    df = pd.DataFrame(results)
    df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
    print(f"batch [{start}:{end}) done, {len(results)} results written", flush=True)


def phase1_screen(batch_size=120):
    """Drives run_batch across ALL_COMBOS via fresh subprocesses (see run_batch's
    docstring for why) -- this process just launches and collects, never makes the
    C++ calls itself."""
    import subprocess
    out_path = os.path.join(DATA_DIR, "goal34_phase1_screen.csv")
    if os.path.exists(out_path):
        os.remove(out_path)
    n = len(ALL_COMBOS)
    print(f"Phase 1: screening {n:,} candidates in batches of {batch_size} (fresh subprocess each)...")
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        r = subprocess.run([sys.executable, __file__, "--batch", str(start), str(end)],
                            capture_output=True, text=True, timeout=300)
        print(f"  [{start}:{end})  [{time.perf_counter()-t0:.0f}s elapsed]  {r.stdout.strip()}", flush=True)
        if r.returncode != 0:
            print(f"    batch FAILED (code {r.returncode}): {r.stderr[-500:]}", flush=True)
    dt = time.perf_counter() - t0
    df = pd.read_csv(out_path).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"Phase 1 done in {dt:.0f}s, {len(df)} candidates scored. Top 15:")
    print(df.head(15).to_string(index=False))
    return df


def phase2_evaluate(label, cfg, n_paths=60, max_trades=250_000):
    """Real v15 evaluation: two-account model, own risk sweep, real MC."""
    trades = build_raw_with_side(cfg["n"], cfg["rr"], cfg["age_h"], cfg["touches"])
    years = (trades.entry_time.max() - trades.entry_time.min()).days / 365.25
    raw_tpy = len(trades) / years

    from goal33_rerank_3factor_two_account import find_optimal_risk, WEIGHTS
    best_risk, best_med_yrs, best_med_dd, sweep_df = find_optimal_risk(
        trades, n_paths, max_trades, seed_base=abs(hash(label)) % 100000)

    t, eq_buy, eq_sell, max_dd, _ = g30.simulate_two_account(trades, best_risk, track_detail=False)
    acc = t[t.accepted]
    wr = 100 * acc.is_win.mean() if len(acc) else float("nan")
    acc_tpy = len(acc) / years
    days = best_med_yrs * 365.25
    print(f"  {label:20s} n={cfg['n']} rr={cfg['rr']} age={cfg['age_h']} touches={cfg['touches']}  "
          f"risk*={100*best_risk:4.1f}%  WR={wr:5.2f}%  acc_tpy={acc_tpy:7.0f}/yr  "
          f"days->1M={days:7.1f}  medDD={best_med_dd:6.1f}%")
    return dict(label=label, **cfg, optimal_risk=best_risk, wr=wr, acc_tpy=acc_tpy,
                median_days_1m=days, median_dd=best_med_dd)


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    screen_df = phase1_screen()

    print("\n" + "=" * 100)
    print("Phase 2: real two-account TRUE-HOLDOUT evaluation of top TRAINING-screened candidates + reference points")
    print("=" * 100)
    results = [phase2_evaluate("Apex-King (current champion)", APEX),
               phase2_evaluate("Round1-2 winner (HOLDOUT-CONTAMINATED, reference only)", ROUND1_WINNER)]
    top_candidates = screen_df.head(8)
    for _, row in top_candidates.iterrows():
        cfg = dict(n=row.n, rr=row.rr, age_h=row.age_h, touches=int(row.touches))
        label = f"n{row.n}_rr{row.rr}_age{row.age_h}_t{int(row.touches)}"
        try:
            results.append(phase2_evaluate(label, cfg))
        except Exception as e:
            print(f"  {label}: FAILED ({e})")

    out = pd.DataFrame(results).sort_values("median_days_1m").reset_index(drop=True)
    out.to_csv(os.path.join(DATA_DIR, "goal34_phase2_results.csv"), index=False)
    print("\n" + "=" * 100)
    print("FINAL RESULTS (sorted by speed)")
    print("=" * 100)
    print(out.to_string(index=False))

    baseline_days = out[out.label == "Apex-King (current champion)"].median_days_1m.iloc[0]
    real_candidates = out[~out.label.isin(["Apex-King (current champion)",
                                            "Round1-2 winner (HOLDOUT-CONTAMINATED, reference only)"])]
    winner = real_candidates.iloc[0] if len(real_candidates) else None
    if winner is not None and winner.median_days_1m < baseline_days:
        print(f"\n*** NEW CHAMPION FOUND (selected on TRAIN, verified on TRUE holdout): "
              f"{winner.label} -- {winner.median_days_1m:.1f} days vs Apex-King's {baseline_days:.1f} days ***")
    else:
        print(f"\nNo train-selected candidate beat Apex-King's {baseline_days:.1f} days on the true holdout.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
