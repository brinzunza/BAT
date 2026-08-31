"""Monte-Carlo version of goal20's risk_pct tuning (part4_risk_sweep).

goal20's sweep answered "which risk_pct compounds fastest over the ONE real 16.9yr
historical path" -- useful, but a single realized path is one draw, same concern this
whole project has flagged about small/single-sample conclusions everywhere else. This
block-bootstraps the REAL trade stream many times per risk_pct and reports the
MC-MEDIAN speed to $1M/$10M and drawdown -- the same "median (P50)" ranking criterion
used everywhere else in this project (STRATEGY_RANKINGS.md, goal10-13, etc.), not one
lucky/unlucky path.

Why not just reuse run_mc_capped (this project's standard block-bootstrap MC)? Because
that engine resamples a flat net_r SEQUENCE with no concept of overlapping trades --
margin-based acceptance genuinely depends on which trades overlap in real time, which
a plain net_r stream discards entirely. Instead, each bootstrap block here is a REAL
contiguous slice of the true chronological trade history (entry_time, exit_time,
instrument, entry_price, stop_risk, net_r all preserved together), grafted end-to-end
onto a synthetic clock -- so each block's true internal overlap/concurrency structure
survives resampling, and only the block-to-block sequence is randomized (matching this
project's standard block-bootstrap philosophy, just applied to timed events instead of
a scalar return stream).
"""
import os
import sys
import heapq
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal20_speed_king_margin_report as g20

START = g20.START
MAX_LEVERAGE = g20.MAX_LEVERAGE
CAP = g20.CAP
T1, T10 = g20.T1, g20.T10
RUIN_FLOOR = g20.RUIN_FLOOR
BLOCK = g20.BLOCK
position_size_and_margin = g20.position_size_and_margin

YEAR_NS = 365.25 * 86400 * 1e9


def mc_path(entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, risk_pct, rng,
            block=BLOCK, max_trades=40000, start=START, max_leverage=MAX_LEVERAGE,
            cap=CAP, target1=T1, target2=T10, ruin_floor=RUIN_FLOOR):
    """One Monte Carlo path. Returns (years_to_1m, years_to_10m, max_dd_pct, ruined,
    n_signals_processed, n_accepted). years_to_1m/10m are -1.0 if never reached within
    max_trades. Stops early the moment $10M is crossed or the account is ruined --
    matches run_mc_capped's own early-stop-at-target behavior."""
    n = len(entry_ns)
    equity = start
    peak = start
    dd_worst = 0.0
    margin_used = 0.0
    heap = []  # (exit_time_ns, risk_dollars, margin, net_r)
    clock = 0
    t1 = -1.0
    t10 = -1.0
    ruined = False
    processed = 0
    accepted_n = 0

    while processed < max_trades and t10 < 0 and not ruined:
        start_idx = rng.integers(0, n)
        end_idx = min(start_idx + block, n)
        offset = clock - entry_ns[start_idx]
        block_exit_max = clock

        for idx in range(start_idx, end_idx):
            et = entry_ns[idx] + offset
            xt = exit_ns[idx] + offset
            if xt > block_exit_max:
                block_exit_max = xt

            while heap and heap[0][0] <= et:
                _, r_dollars, margin, r = heapq.heappop(heap)
                equity += r_dollars * r
                margin_used -= margin
                if equity > peak:
                    peak = equity
                dd = 100 * (equity - peak) / peak
                if dd < dd_worst:
                    dd_worst = dd
                if equity <= ruin_floor:
                    ruined = True
            margin_used = max(margin_used, 0.0)
            if ruined:
                break

            _, risk_dollars, margin = position_size_and_margin(
                instrument[idx], equity, entry_price[idx], stop_risk[idx], risk_pct,
                max_leverage, cap)
            avail = min(equity, cap) if cap is not None else equity
            if margin_used + margin <= avail:
                heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx]))
                margin_used += margin
                accepted_n += 1

            processed += 1
            if t1 < 0 and equity >= target1:
                t1 = et / YEAR_NS
            if t1 >= 0 and t10 < 0 and equity >= target2:
                t10 = et / YEAR_NS
                break
            if processed >= max_trades:
                break

        if ruined:
            break
        clock = block_exit_max

    return t1, t10, dd_worst, ruined, processed, accepted_n


def run_mc_for_risk(entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r,
                     risk_pct, n_paths, seed, block=BLOCK, max_trades=40000):
    rng = np.random.default_rng(seed)
    t1s = np.empty(n_paths)
    t10s = np.empty(n_paths)
    dds = np.empty(n_paths)
    ruins = np.zeros(n_paths, dtype=bool)
    accept_rates = np.empty(n_paths)
    for p in range(n_paths):
        t1, t10, dd, ruined, processed, accepted_n = mc_path(
            entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, risk_pct, rng,
            block=block, max_trades=max_trades)
        t1s[p] = t1
        t10s[p] = t10
        dds[p] = dd
        ruins[p] = ruined
        accept_rates[p] = accepted_n / processed if processed else np.nan
    return t1s, t10s, dds, ruins, accept_rates


RISK_GRID = g20.RISK_GRID


def main():
    for inst in g20.TRADEABLE:
        g20.rs.load_instrument(inst)
        g20.rs._prep_zones(inst, g20.pg.MIN_RETRACEMENT)

    full = pd.read_csv(os.path.join(DATA_DIR, "goal20_speed_king_full_true_times.csv"),
                        parse_dates=["entry_time", "exit_time"])
    full = full.sort_values("entry_time").reset_index(drop=True)
    entry_ns = full.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = full.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    instrument = full.instrument.values
    entry_price = full.entry_price.values
    stop_risk = full.stop_risk.values
    net_r = full.net_r.values
    print(f"{len(full):,} real trades loaded as the block-bootstrap source.\n")

    rows = []
    for i, risk_pct in enumerate(RISK_GRID):
        # The <=0.25% region is already decisively slower (confirmed by both goal20's
        # single-path sweep and quick MC probing: 8-20+ years just to reach $1M, vs
        # <1.5yr for 1-5%) -- fewer paths there since we don't need high precision to
        # re-confirm what's already obvious; full precision where the real competition is.
        n_paths = 150 if risk_pct <= 0.0025 else 500
        max_trades = 700_000 if risk_pct <= 0.0025 else 300_000
        t0 = time.perf_counter()
        t1s, t10s, dds, ruins, accept_rates = run_mc_for_risk(
            entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r,
            risk_pct, n_paths, seed=1000 + i, max_trades=max_trades)
        dt = time.perf_counter() - t0
        N_PATHS = n_paths

        t1_hit = t1s[t1s > 0]
        row = dict(
            risk_pct=risk_pct,
            median_yrs_1m=np.median(t1_hit) if len(t1_hit) else np.nan,
            p10_yrs_1m=np.percentile(t1_hit, 10) if len(t1_hit) else np.nan,
            p90_yrs_1m=np.percentile(t1_hit, 90) if len(t1_hit) else np.nan,
            pct_reached_1m=100 * len(t1_hit) / N_PATHS,
            median_dd=np.median(dds),
            p90_dd=np.percentile(dds, 10),  # 10th percentile of dd (most negative = worst 10%)
            pct_ruined=100 * ruins.mean(),
            median_accept_rate=100 * np.nanmedian(accept_rates),
        )
        rows.append(row)
        print(f"risk={100*risk_pct:5.2f}%  ({dt:5.1f}s, {N_PATHS} paths)  "
              f"median yrs->$1M={row['median_yrs_1m']:.3f}  "
              f"reached={row['pct_reached_1m']:.1f}%  "
              f"medianDD={row['median_dd']:.1f}%  "
              f"ruin={row['pct_ruined']:.2f}%  "
              f"accept={row['median_accept_rate']:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal21_speed_king_risk_mc.csv"), index=False)

    best_speed = df.loc[df.median_yrs_1m.idxmin()]
    print(f"\nFastest median MC speed to $1M: risk={100*best_speed.risk_pct:.2f}% "
          f"({best_speed.median_yrs_1m:.3f}yr, medianDD={best_speed.median_dd:.1f}%, "
          f"ruin={best_speed.pct_ruined:.2f}%)")
    print(f"\nSaved: goal21_speed_king_risk_mc.csv")


if __name__ == "__main__":
    main()
