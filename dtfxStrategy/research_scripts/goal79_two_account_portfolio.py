"""Direct user request (2026-09-06): backtest the fade-the-pullback dynamic-zigzag
strategy (goal77, invert=True) under this project's ACTUAL two-sub-account live
architecture, not the unconstrained single-pool approximation goal78 used:

  - Two separate $100 sub-accounts: buy-only and sell-only, NEVER pooled (matches
    PRODLIVETRADING's real dual-OANDA-account design -- bot.py's `self.clients` keyed
    by side; see PRODLIVETRADING/config.py's OANDA_ACCOUNT_ID_LONG/SHORT).
  - Position sizing: risk_pct of min(buy_equity, sell_equity) -- matches bot.py's fixed
    sizing formula exactly (goal20_speed_king_margin_report.py::position_size_and_margin,
    a direct port of PRODLIVETRADING/broker/oanda_client.py::position_size).
  - Max 1 CONFIRMED open position per side, across all four pairs combined (one buy
    slot, one sell slot -- not one per symbol). This is the pre-v26 STRATEGY_RANKINGS.md
    concurrency rule: resolve each symbol's zones independently, then walk the merged
    stream in entry-time order, accepting the first candidate to reach each side's one
    slot and discarding every other. (v26 replaced this with a full cross-symbol
    resting-order race simulator for a single symbol, EUR/GBP only -- out of scope here;
    this is the simpler, already-validated-as-reasonable multi-symbol approximation.)
  - 50x max leverage, 1.6-pip fixed spread, $2,000,000 position-sizing cap (this
    project's "usual" cap per goal20's own docstring -- goal78 used the same cap, kept
    for consistency across this session's deliverables rather than the $500K/$1M caps
    a couple of one-off past scripts used for their own specific requests).
  - Risk per trade: 5% (PRODLIVETRADING/config.py's actual live RISK_PCT default --
    other one-off research scripts in this project used 6% for their own specific
    strategies; 5% is used here as the standing live-deployed figure).

Reuses goal77's exact zigzag/zone-building code unchanged; only the resolution step is
extended (in a new function, goal77 itself is left untouched) to also emit entry_price
and stop_risk (price-distance), needed by position_size_and_margin. The two-account
event-driven simulator and Monte Carlo below are adapted from goal30/goal42's own
validated `simulate_two_account` / `run_monte_carlo_two_account` -- same algorithm,
this strategy's trades and this script's constants.
"""
import os
import sys
import time
import heapq

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats as sps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal77_dynamic_zigzag_backtest as g77
import goal20_speed_king_margin_report as g20

rs, pg = g77.rs, g77.pg
position_size_and_margin = g20.position_size_and_margin

INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"]
MAX_WINDOW = 1440
MIN_STOP_PIPS = 3.0

START_PER_SIDE = 100.0
RISK_PCT = 0.05
MAX_LEVERAGE = 50.0
ACCOUNT_CAP = 2_000_000.0
BLOCK = 20
RUIN_FLOOR = 1.0
YEAR_NS = 365.25 * 86400 * 1e9


@njit(cache=True)
def _resolve_zones_ex(z_created, z_top, z_bottom, z_is_buy, high, low, max_window,
                       min_stop_price, spread_price, invert):
    """goal77._resolve_zones, extended to also emit entry_price and stop_risk
    (price-distance of stop from entry) -- everything else identical, byte-for-byte."""
    n_bars = len(high)
    m = len(z_created)
    out_touch = np.full(m, -1, dtype=np.int64)
    out_exit = np.full(m, -1, dtype=np.int64)
    out_outcome = np.zeros(m, dtype=np.int8)
    out_net_r = np.zeros(m, dtype=np.float64)
    out_entry_price = np.zeros(m, dtype=np.float64)
    out_stop_risk = np.zeros(m, dtype=np.float64)

    for zi in range(m):
        top, bottom, is_buy = z_top[zi], z_bottom[zi], z_is_buy[zi]
        if (top - bottom) / 2.0 < min_stop_price:
            continue
        mid = (top + bottom) / 2.0
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars
        touch_idx = -1
        for j in range(c + 1, end):
            touched = (low[j] <= mid) if is_buy else (high[j] >= mid)
            if touched:
                touch_idx = j
                break
        if touch_idx == -1:
            continue
        if is_buy:
            deep = low[touch_idx] <= mid - spread_price
            entry_price = mid if deep else mid + spread_price
        else:
            deep = high[touch_idx] >= mid + spread_price
            entry_price = mid if deep else mid - spread_price

        trade_is_buy = (not is_buy) if invert else is_buy
        stop_price = bottom if trade_is_buy else top
        target_price = top if trade_is_buy else bottom
        eff_risk = abs(entry_price - stop_price)
        if eff_risk <= 0:
            continue
        win_r = abs(target_price - entry_price) / eff_risk

        outcome, exit_idx = 0, -1
        for k in range(touch_idx, end):
            hit_sl = (low[k] <= stop_price) if trade_is_buy else (high[k] >= stop_price)
            hit_tp = (high[k] >= target_price) if trade_is_buy else (low[k] <= target_price)
            if hit_sl:
                outcome, exit_idx = 2, k
                break
            elif hit_tp:
                outcome, exit_idx = 1, k
                break
        if outcome == 0:
            continue
        out_touch[zi] = touch_idx
        out_exit[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = win_r if outcome == 1 else -1.0
        out_entry_price[zi] = entry_price
        out_stop_risk[zi] = eff_risk

    return out_touch, out_exit, out_outcome, out_net_r, out_entry_price, out_stop_risk


def build_fade_trades():
    print("Building fade-the-pullback candidate trades (all 4 symbols, inverted)...", flush=True)
    t0 = time.time()
    frames = []
    for inst in INSTRUMENTS:
        times, lows, highs, df = rs.load_instrument(inst)
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip

        confirm_idx, pivot_idx, is_high, price = g77._zigzag_pivots(high_, low_)
        z_created, z_top, z_bottom, z_is_buy, z_is_flip = g77._build_zones(confirm_idx, is_high, price)
        touch, exit_, outcome, net_r, entry_price, stop_risk = _resolve_zones_ex(
            z_created, z_top, z_bottom, z_is_buy, high_, low_, MAX_WINDOW,
            MIN_STOP_PIPS * pip, spread_price, True)

        resolved = outcome > 0
        entry_idx = touch[resolved]
        exit_idx = exit_[resolved]
        frames.append(pd.DataFrame({
            "instrument": inst,
            "entry_time": times[entry_idx],
            "exit_time": times[exit_idx],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk[resolved],
            "net_r": net_r[resolved],
            "is_buy": ~z_is_buy[resolved],  # traded (inverted) side
            "is_flip": z_is_flip[resolved],
        }))
        print(f"  {inst}: {resolved.sum():,} candidate trades ({time.time()-t0:.0f}s)", flush=True)

    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print(f"Total candidate trades (pre-concurrency-filter): {len(trades):,} ({time.time()-t0:.0f}s)\n", flush=True)
    return trades


def simulate_two_account(trades, risk_pct=RISK_PCT, start_per_side=START_PER_SIDE,
                          max_leverage=MAX_LEVERAGE, cap=ACCOUNT_CAP, track_detail=True):
    """Event-driven walk, two separate sub-account equity/margin pools. Sizing base =
    min(buy_equity, sell_equity); accept/reject uses each side's own real capacity.
    Max 1 open position per side (across all 4 symbols). Adapted from goal30/goal42."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_time = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_time = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    equity = {True: start_per_side, False: start_per_side}
    margin_used = {True: 0.0, False: 0.0}
    open_count = {True: 0, False: 0}
    heap = []

    peak_combined = start_per_side * 2.0
    max_dd_pct = 0.0
    accepted = np.zeros(n, dtype=bool)
    close_times, close_buy, close_sell = [], [], []

    for i in range(n):
        et = entry_time[i]
        while heap and heap[0][0] <= et:
            xt, r_dollars, margin, r, side = heapq.heappop(heap)
            equity[side] += r_dollars * r
            margin_used[side] -= margin
            open_count[side] -= 1
            combined = equity[True] + equity[False]
            if combined > peak_combined:
                peak_combined = combined
            dd = 100.0 * (combined - peak_combined) / peak_combined
            if dd < max_dd_pct:
                max_dd_pct = dd
            if track_detail:
                close_times.append(xt); close_buy.append(equity[True]); close_sell.append(equity[False])
        margin_used[True] = max(margin_used[True], 0.0)
        margin_used[False] = max(margin_used[False], 0.0)

        side = bool(is_buy[i])
        sizing_base = min(equity[True], equity[False])
        if sizing_base <= RUIN_FLOOR:
            continue
        _, risk_dollars, margin = position_size_and_margin(
            instrument[i], sizing_base, entry_price[i], stop_risk[i], risk_pct, max_leverage, cap)
        avail = min(equity[side], cap) - margin_used[side]
        # A position sized to use ALL available margin computes `margin` via a
        # divide-then-multiply-then-divide round trip through max_leverage that lands it
        # within ~1e-14 of `avail`, sometimes a hair above due to float rounding alone --
        # an unguarded `<=` here spuriously rejects a fully affordable trade about half
        # the time this (common, since sizing is often leverage-capped) case occurs.
        tolerance = 1e-9 * max(1.0, abs(avail))

        if margin <= avail + tolerance and open_count[side] < 1:
            accepted[i] = True
            heapq.heappush(heap, (exit_time[i], risk_dollars, margin, net_r[i], side))
            margin_used[side] += margin
            open_count[side] += 1

    while heap:
        xt, r_dollars, margin, r, side = heapq.heappop(heap)
        equity[side] += r_dollars * r
        combined = equity[True] + equity[False]
        if combined > peak_combined:
            peak_combined = combined
        dd = 100.0 * (combined - peak_combined) / peak_combined
        if dd < max_dd_pct:
            max_dd_pct = dd
        if track_detail:
            close_times.append(xt); close_buy.append(equity[True]); close_sell.append(equity[False])

    trace = pd.DataFrame({"close_time": pd.to_datetime(np.array(close_times, dtype="int64")),
                           "equity_buy": close_buy, "equity_sell": close_sell})
    if len(trace):
        trace["equity_combined"] = trace.equity_buy + trace.equity_sell
    t = t.copy()
    t["accepted"] = accepted
    return t, equity[True], equity[False], max_dd_pct, trace


def run_monte_carlo_two_account(trades, n_sims, target, max_trades=100_000, seed=42):
    """Block-bootstrap MC over the (already concurrency-filtered) accepted trade stream,
    replaying the exact two-account mechanics per simulated path. Adapted from goal42's
    run_monte_carlo_two_account, with this script's own constants (5% risk, $2M cap)."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    rng = np.random.default_rng(seed)
    days_to_target, trades_to_target, max_dds = [], [], []
    ruined_count = 0

    for sim in range(n_sims):
        equity = {True: START_PER_SIDE, False: START_PER_SIDE}
        margin_used = {True: 0.0, False: 0.0}
        open_count = {True: 0, False: 0}
        heap = []
        clock = 0
        peak = START_PER_SIDE * 2.0
        worst_dd = 0.0
        ruined = False
        hit_target = False
        t_hit_days = -1.0
        trades_count = 0

        while trades_count < max_trades and not hit_target and not ruined:
            start_idx = rng.integers(0, n)
            end_idx = min(start_idx + BLOCK, n)
            offset = clock - entry_ns[start_idx]
            block_exit_max = clock

            for idx in range(start_idx, end_idx):
                et = entry_ns[idx] + offset
                xt = exit_ns[idx] + offset
                if xt > block_exit_max:
                    block_exit_max = xt

                while heap and heap[0][0] <= et:
                    _, r_dollars, margin, r, side = heapq.heappop(heap)
                    equity[side] += r_dollars * r
                    margin_used[side] -= margin
                    open_count[side] -= 1
                    comb = equity[True] + equity[False]
                    if comb > peak:
                        peak = comb
                    dd = 100.0 * (comb - peak) / peak
                    if dd < worst_dd:
                        worst_dd = dd
                    if comb <= RUIN_FLOOR or equity[True] <= 1.0 or equity[False] <= 1.0:
                        ruined = True
                if ruined:
                    break

                side = bool(is_buy[idx])
                sizing_base = min(equity[True], equity[False])
                _, risk_dollars, margin = position_size_and_margin(
                    instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], RISK_PCT,
                    MAX_LEVERAGE, ACCOUNT_CAP)
                avail = min(equity[side], ACCOUNT_CAP) - margin_used[side]
                tolerance = 1e-9 * max(1.0, abs(avail))  # see simulate_two_account's comment

                if margin <= avail + tolerance and open_count[side] < 1:
                    heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                    margin_used[side] += margin
                    open_count[side] += 1
                    trades_count += 1

                comb = equity[True] + equity[False]
                if comb >= target:
                    hit_target = True
                    t_hit_days = et / 86400e9  # et is a relative clock (starts at 0), not epoch time
                    break

            clock = block_exit_max

        if ruined:
            ruined_count += 1
        if hit_target:
            days_to_target.append(t_hit_days)
            trades_to_target.append(trades_count)
        max_dds.append(worst_dd)

    return np.array(days_to_target), np.array(trades_to_target), np.array(max_dds), ruined_count / n_sims


def period_stats(equity, freq, label):
    e = equity.resample(freq).last().ffill().dropna()
    ret = e.pct_change().dropna()
    n = len(ret)
    n_pos = int((ret > 0).sum())
    pct = 100.0 * n_pos / n if n else float("nan")
    return dict(label=label, n_periods=n, n_profitable=n_pos, pct_profitable=pct)


def main():
    t0 = time.time()
    candidates = build_fade_trades()

    print("=== Two-account event-driven simulation ($100/$100, 1-per-side, 5% risk, 50x lev, $2M cap) ===")
    result, eq_buy, eq_sell, max_dd, trace = simulate_two_account(candidates)
    accepted = result[result["accepted"]].reset_index(drop=True)
    print(f"Candidates: {len(candidates):,}  ->  Accepted: {len(accepted):,} "
          f"({100*len(accepted)/len(candidates):.1f}% of candidates) ({time.time()-t0:.0f}s)")
    print(f"Buy account final: ${eq_buy:,.2f}   Sell account final: ${eq_sell:,.2f}   "
          f"Combined: ${eq_buy+eq_sell:,.2f}")
    print(f"Max drawdown (combined, peak-to-trough): {max_dd:.2f}%")

    n_buy = int(accepted["is_buy"].sum())
    n_sell = len(accepted) - n_buy
    print(f"Accepted by side: buy={n_buy:,}  sell={n_sell:,}")

    net_r = accepted["net_r"].values
    wins = net_r[net_r > 0]
    losses = net_r[net_r <= 0]
    wr = 100.0 * len(wins) / len(net_r)
    avg_win = wins.mean()
    avg_loss = losses.mean()
    rr = avg_win / abs(avg_loss)
    pf = wins.sum() / abs(losses.sum())
    span_start = accepted["entry_time"].min()
    span_end = accepted["exit_time"].max()
    years = (span_end - span_start) / np.timedelta64(365, "D")
    days = (span_end - span_start) / np.timedelta64(1, "D")

    print(f"\n=== Accepted-trade stats ===")
    print(f"  Trades: {len(accepted):,}  |  span: {span_start.date()} to {span_end.date()} ({years:.1f} yrs)")
    print(f"  Win rate: {wr:.2f}%")
    print(f"  Avg win: {avg_win:.3f}R   Avg loss: {avg_loss:.3f}R   R:R = {rr:.3f}")
    print(f"  Profit factor: {pf:.3f}")
    print(f"  Trades/year: {len(accepted)/years:.1f}")
    print(f"  Trades/day:  {len(accepted)/days:.3f}")

    for inst, g in accepted.groupby("instrument"):
        gw = g.loc[g.net_r > 0, "net_r"].sum()
        gl = -g.loc[g.net_r <= 0, "net_r"].sum()
        print(f"  {inst}: n={len(g):,} WR={100*(g.net_r>0).mean():.1f}% "
              f"PF={(gw/gl if gl>0 else float('inf')):.3f}")

    equity = trace.set_index("close_time")["equity_combined"]
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    print(f"\n=== Combined equity curve ===")
    for freq, label in [("D", "Daily"), ("W-FRI", "Weekly"), ("M", "Monthly"), ("A", "Yearly")]:
        ps = period_stats(equity, freq, label)
        print(f"  {label:8s}: {ps['n_profitable']:>4d} / {ps['n_periods']:<4d} profitable "
              f"({ps['pct_profitable']:.1f}%)")

    daily = equity.resample("D").last().ffill().dropna()
    daily_ret = daily.pct_change().dropna().values
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(252) if daily_ret.std(ddof=1) > 0 else float("nan")
    exkurt = sps.kurtosis(daily_ret, fisher=True, bias=False)
    skew = sps.skew(daily_ret, bias=False)
    print(f"\n=== Return distribution (daily, combined equity) ===")
    print(f"  Sharpe (annualized): {sharpe:.3f}")
    print(f"  Excess kurtosis:     {exkurt:.3f}")
    print(f"  Skew:                {skew:.3f}")

    mean_r, std_r = net_r.mean(), net_r.std(ddof=1)
    tpy = len(accepted) / years
    sharpe_pt = mean_r / std_r * np.sqrt(tpy)
    exkurt_pt = sps.kurtosis(net_r, fisher=True, bias=False)
    print(f"\n=== Return distribution (per-trade R-multiple) ===")
    print(f"  Mean R: {mean_r:.4f}  Std R: {std_r:.4f}")
    print(f"  Per-trade Sharpe (annualized by sqrt(trades/yr)): {sharpe_pt:.3f}")
    print(f"  Per-trade excess kurtosis: {exkurt_pt:.3f}")

    N_SIMS = 5000
    print(f"\n=== Monte Carlo: {N_SIMS:,} two-account block-bootstrap sims, race to $1,000,000 ===")
    days, trades_ct, dds, ruin_rate = run_monte_carlo_two_account(accepted, N_SIMS, target=1_000_000.0)
    pct_reached = 100 * len(days) / N_SIMS
    print(f"  Reached $1M: {pct_reached:.2f}% of sims  ({time.time()-t0:.0f}s)")
    if len(days):
        print(f"  Median days to $1M: {np.median(days):.1f}   Mean: {days.mean():.1f}")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"    P{p:2d}: {np.percentile(days, p):.1f} days")
    print(f"  % ruined: {100*ruin_rate:.3f}%")
    dd_pct = dds
    print(f"  Max drawdown -- median: {np.median(dd_pct):.2f}%  worst P5: {np.percentile(dd_pct,5):.2f}%  "
          f"best P95: {np.percentile(dd_pct,95):.2f}%")

    os.makedirs(os.path.join(SCRIPT_DIR, "scratch"), exist_ok=True)
    accepted.to_csv(os.path.join(SCRIPT_DIR, "scratch", "goal79_accepted_trades.csv"), index=False)
    trace.to_csv(os.path.join(SCRIPT_DIR, "scratch", "goal79_equity_trace.csv"), index=False)
    print(f"\nSaved accepted trades + equity trace to scratch/. ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
