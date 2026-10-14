"""Monte Carlo risk_pct tuning at MAX_LEVERAGE=1.0 (raw account size, no leverage/margin
borrowing at all -- notional can never exceed the cash actually in the account).

This is a structurally different regime from goal20/goal21's 50x-leverage sweeps, not
just a smaller number:

  - At 1:1 leverage, a single LEVERAGE-CAPPED position's notional already equals 100%
    of equity by construction (max_notional = equity * 1). Given Speed-King's typical
    stop is ~0.1-0.2% of price, risk-based sizing wants far more than 1x notional for
    any risk_pct above roughly that same ~0.1-0.2% -- so for essentially the whole
    0.5%-5% grid used in goal20/21, EVERY position hits the leverage ceiling and uses
    ~100% margin, making >1 concurrent position structurally near-impossible regardless
    of risk_pct. The only region where risk_pct actually changes behavior is well UNDER
    1% (confirmed empirically: crossover sits around 0.05-0.3% here, verified against
    goal20's simulate_margin_constrained before building this).

  - Best-case full-history growth at 0 leverage tops out around $30k (vs. $360M at 50x)
    -- so "years to $1M" (goal21's metric) isn't reachable in any realistic horizon for
    most of the grid. Instead this tracks the DISTRIBUTION of outcomes after the same
    16.9yr horizon as the real data (final equity, max drawdown) across many
    block-bootstrapped paths -- directly comparable to the single real-history point,
    just with the same statistical robustness goal21 added for the 50x case.
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
import goal21_speed_king_risk_mc as g21

START = g20.START
CAP = g20.CAP
RUIN_FLOOR = g20.RUIN_FLOOR
BLOCK = g20.BLOCK
position_size_and_margin = g20.position_size_and_margin
YEAR_NS = g21.YEAR_NS

ZERO_LEVERAGE = 1.0
HORIZON_YEARS = 16.9  # matches the real Speed-King history span used everywhere else

RISK_GRID = [0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002,
             0.0025, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02, 0.05]


def mc_path_horizon(entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, risk_pct,
                     rng, horizon_years, trades_per_year, block=BLOCK, start=START,
                     max_leverage=ZERO_LEVERAGE, cap=CAP, ruin_floor=RUIN_FLOOR):
    """Same block-bootstrap/margin-walk as goal21.mc_path, but runs for horizon_years
    worth of trades (processed >= horizon_years * trades_per_year) instead of chasing a
    dollar target -- appropriate here since $1M isn't reachable in any realistic
    horizon at 1:1 leverage. `trades_per_year` is this project's usual fixed rate
    (computed once from the real full history), NOT derived from resampled
    timestamps: sampling blocks uniformly by trade-INDEX over-represents whichever
    period has denser trading (Speed-King's frequency rises in later years), so a
    timestamp-reconstructed clock advances more slowly than trades accumulate and
    needs far more synthetic trades than the real history did to span the same nominal
    years. The real relative entry/exit timing WITHIN each block is still used exactly
    as before -- only the stopping condition changed, not the margin/concurrency logic.
    Returns (final_equity, max_dd_pct, ruined, processed, accepted_n)."""
    target_trades = int(horizon_years * trades_per_year)
    n = len(entry_ns)
    equity = start
    peak = start
    dd_worst = 0.0
    margin_used = 0.0
    heap = []
    clock = 0
    ruined = False
    processed = 0
    accepted_n = 0

    while processed < target_trades and not ruined:
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

        if ruined:
            break
        clock = block_exit_max

    return equity, dd_worst, ruined, processed, accepted_n


def run_mc_for_risk(entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r,
                     risk_pct, n_paths, seed, trades_per_year, horizon_years=HORIZON_YEARS,
                     block=BLOCK):
    rng = np.random.default_rng(seed)
    eqs = np.empty(n_paths)
    dds = np.empty(n_paths)
    ruins = np.zeros(n_paths, dtype=bool)
    accept_rates = np.empty(n_paths)
    for p in range(n_paths):
        eq, dd, ruined, processed, accepted_n = mc_path_horizon(
            entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, risk_pct, rng,
            horizon_years, trades_per_year, block=block)
        eqs[p] = eq
        dds[p] = dd
        ruins[p] = ruined
        accept_rates[p] = accepted_n / processed if processed else np.nan
    return eqs, dds, ruins, accept_rates


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
    years_span = (full.entry_time.max() - full.entry_time.min()).days / 365.25
    trades_per_year = len(full) / years_span
    N_PATHS = 150
    print(f"{len(full):,} real trades loaded ({trades_per_year:,.0f}/yr). "
          f"MAX_LEVERAGE={ZERO_LEVERAGE} (no leverage), "
          f"horizon={HORIZON_YEARS}yr, {N_PATHS} paths/level.\n")

    rows = []
    for i, risk_pct in enumerate(RISK_GRID):
        t0 = time.perf_counter()
        eqs, dds, ruins, accept_rates = run_mc_for_risk(
            entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r,
            risk_pct, N_PATHS, seed=2000 + i, trades_per_year=trades_per_year)
        dt = time.perf_counter() - t0
        row = dict(
            risk_pct=risk_pct,
            median_final_equity=np.median(eqs),
            p10_final_equity=np.percentile(eqs, 10),
            p90_final_equity=np.percentile(eqs, 90),
            median_dd=np.median(dds),
            worst_decile_dd=np.percentile(dds, 10),
            pct_ruined=100 * ruins.mean(),
            median_accept_rate=100 * np.nanmedian(accept_rates),
        )
        rows.append(row)
        print(f"risk={100*risk_pct:7.4f}%  ({dt:5.1f}s)  "
              f"median final=${row['median_final_equity']:>10,.0f}  "
              f"P10-P90=${row['p10_final_equity']:,.0f}-${row['p90_final_equity']:,.0f}  "
              f"medianDD={row['median_dd']:6.1f}%  ruin={row['pct_ruined']:.2f}%  "
              f"accept={row['median_accept_rate']:5.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal22_speed_king_zero_leverage_mc.csv"), index=False)

    best = df.loc[df.median_final_equity.idxmax()]
    print(f"\nMaximizes median {HORIZON_YEARS}yr final equity (0 leverage): "
          f"risk={100*best.risk_pct:.4f}% (${best.median_final_equity:,.0f}, "
          f"medianDD={best.median_dd:.1f}%, accept={best.median_accept_rate:.1f}%)")
    print(f"\nSaved: goal22_speed_king_zero_leverage_mc.csv")


if __name__ == "__main__":
    main()
