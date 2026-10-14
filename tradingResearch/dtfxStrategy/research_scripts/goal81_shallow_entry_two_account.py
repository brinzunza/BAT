"""Direct user request (2026-09-06): keep exploring/backtesting variations of the
fade-the-pullback rule, with robustness checks. This picks up goal80's most promising
lead -- a shallower entry (20%/30% retracement instead of 50%) showed a much higher
profit factor, and held up under a first-half/second-half split -- and takes it the rest
of the way: real entry_price/stop_risk output (so it can run through the SAME two-account
architecture as goal79, not just an R-multiple sample), full profitable-period/Sharpe/
kurtosis reporting, a proper THIRDS robustness check (not just a 50/50 split -- matching
this project's own established bar in the archived STRATEGY_RANKINGS.md: "all three
independent thirds of a strategy's own history individually profitable"), and per-symbol
individual profitability, before anything gets promoted into the rankings file.

Spread model unchanged and confirmed by the user as sufficient for this backtest's
purpose (0.8 pips each way on market-order entries and exits = 1.6 pips round-trip,
matching SPREAD_PIPS_TYPICAL already in use everywhere in this project).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats as sps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal77_dynamic_zigzag_backtest as g77
import goal79_two_account_portfolio as g79

rs, pg = g77.rs, g77.pg
simulate_two_account = g79.simulate_two_account
run_monte_carlo_two_account = g79.run_monte_carlo_two_account
period_stats = g79.period_stats

INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"]
MAX_WINDOW = 1440
MIN_STOP_PIPS = 3.0


@njit(cache=True)
def _resolve_frac_ex(z_created, z_top, z_bottom, z_is_buy, high, low, max_window,
                      min_stop_price, spread_price, frac, invert):
    """goal80._resolve_frac, extended to also emit entry_price and stop_risk (price
    distance) -- needed to run this through goal79's two-account $ simulator, exactly
    as goal79._resolve_zones_ex extended goal77._resolve_zones for the same reason."""
    n_bars = len(high)
    m = len(z_created)
    out_outcome = np.zeros(m, dtype=np.int8)
    out_net_r = np.zeros(m, dtype=np.float64)
    out_entry_price = np.zeros(m, dtype=np.float64)
    out_stop_risk = np.zeros(m, dtype=np.float64)
    out_touch = np.full(m, -1, dtype=np.int64)
    out_exit = np.full(m, -1, dtype=np.int64)

    for zi in range(m):
        top, bottom, is_buy = z_top[zi], z_bottom[zi], z_is_buy[zi]
        height = top - bottom
        entry_level = (top - frac * height) if is_buy else (bottom + frac * height)
        trade_is_buy = (not is_buy) if invert else is_buy
        stop_price = bottom if trade_is_buy else top
        target_price = top if trade_is_buy else bottom
        if abs(entry_level - stop_price) < min_stop_price:
            continue
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars
        touch_idx = -1
        for j in range(c + 1, end):
            touched = (low[j] <= entry_level) if is_buy else (high[j] >= entry_level)
            if touched:
                touch_idx = j
                break
        if touch_idx == -1:
            continue
        if is_buy:
            deep = low[touch_idx] <= entry_level - spread_price
            entry_price = entry_level if deep else entry_level + spread_price
        else:
            deep = high[touch_idx] >= entry_level + spread_price
            entry_price = entry_level if deep else entry_level - spread_price
        eff_risk = abs(entry_price - stop_price)
        if eff_risk <= 0:
            continue
        win_r = abs(target_price - entry_price) / eff_risk

        outcome, exit_idx = 0, -1
        for k in range(touch_idx, end):
            hit_sl = (low[k] <= stop_price) if trade_is_buy else (high[k] >= stop_price)
            hit_tp = (high[k] >= target_price) if trade_is_buy else (low[k] <= target_price)
            if hit_sl:
                outcome, exit_idx = 2, k; break
            elif hit_tp:
                outcome, exit_idx = 1, k; break
        if outcome == 0:
            continue
        out_touch[zi] = touch_idx
        out_exit[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = win_r if outcome == 1 else -1.0
        out_entry_price[zi] = entry_price
        out_stop_risk[zi] = eff_risk

    return out_touch, out_exit, out_outcome, out_net_r, out_entry_price, out_stop_risk


def build_frac_trades(frac):
    frames = []
    for inst in INSTRUMENTS:
        times, lows, highs, df = rs.load_instrument(inst)
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip

        confirm_idx, pivot_idx, is_high, price = g77._zigzag_pivots(high_, low_)
        z_created, z_top, z_bottom, z_is_buy, z_is_flip = g77._build_zones(confirm_idx, is_high, price)
        touch, exit_, outcome, net_r, entry_price, stop_risk = _resolve_frac_ex(
            z_created, z_top, z_bottom, z_is_buy, high_, low_, MAX_WINDOW,
            MIN_STOP_PIPS * pip, spread_price, frac, True)

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
            "is_buy": ~z_is_buy[resolved],
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def report_variant(name, candidates, n_mc_sims=3000):
    print("\n" + "=" * 100)
    print(f"  {name}")
    print("=" * 100)

    result, eq_buy, eq_sell, max_dd, trace = simulate_two_account(candidates)
    accepted = result[result["accepted"]].reset_index(drop=True)
    print(f"Candidates: {len(candidates):,}  ->  Accepted: {len(accepted):,} "
          f"({100*len(accepted)/max(len(candidates),1):.1f}%)")
    if len(accepted) < 30:
        print("  too few accepted trades to report further.")
        return None

    net_r = accepted["net_r"].values
    wins, losses = net_r[net_r > 0], net_r[net_r <= 0]
    wr = 100.0 * len(wins) / len(net_r)
    avg_win, avg_loss = wins.mean(), losses.mean()
    pf = wins.sum() / abs(losses.sum())
    span_start, span_end = accepted["entry_time"].min(), accepted["exit_time"].max()
    years = (span_end - span_start) / np.timedelta64(365, "D")

    print(f"Buy final: ${eq_buy:,.2f}  Sell final: ${eq_sell:,.2f}  Combined: ${eq_buy+eq_sell:,.2f}")
    print(f"Max drawdown: {max_dd:.2f}%")
    print(f"Trades: {len(accepted):,}  span {span_start.date()}..{span_end.date()} ({years:.1f}y)")
    print(f"WR={wr:.2f}%  avg_win={avg_win:.3f}R  avg_loss={avg_loss:.3f}R  R:R={avg_win/abs(avg_loss):.3f}  PF={pf:.3f}")
    print(f"Trades/yr={len(accepted)/years:.1f}  Trades/day={len(accepted)/years/365.25:.3f}")

    per_symbol = {}
    for inst, g in accepted.groupby("instrument"):
        gw, gl = g.loc[g.net_r > 0, "net_r"].sum(), -g.loc[g.net_r <= 0, "net_r"].sum()
        pf_i = gw / gl if gl > 0 else float("inf")
        per_symbol[inst] = pf_i
        print(f"  {inst}: n={len(g):,} WR={100*(g.net_r>0).mean():.1f}% PF={pf_i:.3f}")

    equity = trace.set_index("close_time")["equity_combined"]
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    for freq, label in [("D", "Daily"), ("W-FRI", "Weekly"), ("M", "Monthly"), ("A", "Yearly")]:
        ps = period_stats(equity, freq, label)
        print(f"  {label:8s}: {ps['n_profitable']:>4d}/{ps['n_periods']:<4d} profitable ({ps['pct_profitable']:.1f}%)")

    daily = equity.resample("D").last().ffill().dropna()
    dr = daily.pct_change().dropna().values
    dr = dr[np.isfinite(dr)]
    sharpe = dr.mean() / dr.std(ddof=1) * np.sqrt(252) if dr.std(ddof=1) > 0 else float("nan")
    exkurt = sps.kurtosis(dr, fisher=True, bias=False) if len(dr) > 20 else float("nan")
    print(f"Sharpe(daily)={sharpe:.3f}  ExcessKurtosis(daily)={exkurt:.3f}")

    print(f"Monte Carlo: {n_mc_sims:,} two-account sims, race to $1,000,000...")
    t0 = time.time()
    days, trades_ct, dds, ruin_rate = run_monte_carlo_two_account(accepted, n_mc_sims, target=1_000_000.0)
    pct_reached = 100 * len(days) / n_mc_sims
    print(f"  Reached $1M: {pct_reached:.2f}%  ({time.time()-t0:.0f}s)")
    if len(days):
        print(f"  Median days: {np.median(days):.1f}  P5={np.percentile(days,5):.1f}  P95={np.percentile(days,95):.1f}")
    print(f"  %% ruined: {100*ruin_rate:.3f}%%  DD median={np.median(dds):.2f}%%  DD worst-P5={np.percentile(dds,5):.2f}%%")

    return dict(name=name, accepted=accepted, per_symbol=per_symbol, wr=wr, pf=pf,
                trades_per_year=len(accepted)/years, combined_final=eq_buy+eq_sell,
                max_dd=max_dd, sharpe=sharpe, exkurt=exkurt,
                mc_pct_reached=pct_reached, mc_median_days=np.median(days) if len(days) else None)


def thirds_robustness(accepted, name):
    """This project's own established bar (archived STRATEGY_RANKINGS.md): split into
    THREE equal calendar-time windows and confirm the strategy is individually
    profitable in each -- catches a strategy that passes 2-of-3 periods while hiding a
    regime-dependent losing third that a simple 50/50 split can miss."""
    print(f"\n--- thirds robustness: {name} ---")
    span_start, span_end = accepted["entry_time"].min(), accepted["entry_time"].max()
    edges = pd.date_range(span_start, span_end, periods=4)
    all_ok = True
    for i in range(3):
        mask = (accepted["entry_time"] >= edges[i]) & (accepted["entry_time"] < edges[i + 1] if i < 2 else accepted["entry_time"] <= edges[i + 1])
        chunk = accepted[mask]
        if len(chunk) < 20:
            print(f"  third {i+1} ({edges[i].date()}..{edges[i+1].date()}): n={len(chunk)} (too few)")
            all_ok = False
            continue
        r = chunk["net_r"].values
        gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
        pf = gw / gl if gl > 0 else float("inf")
        wr = 100 * (r > 0).mean()
        ok = pf > 1.0
        all_ok = all_ok and ok
        print(f"  third {i+1} ({edges[i].date()}..{edges[i+1].date()}): n={len(chunk):,} WR={wr:.1f}% PF={pf:.3f} "
              f"{'OK' if ok else 'FAILS (PF<=1)'}")
    print(f"  ALL THREE THIRDS PROFITABLE: {all_ok}")
    return all_ok


if __name__ == "__main__":
    t0 = time.time()

    print("Building candidate trades for frac=0.5 (baseline, = v1 ranking), 0.3, 0.2 ...")
    variants = {}
    for frac, label in [(0.5, "Baseline (50% entry, current v1 ranking)"),
                         (0.3, "Shallow entry (30% retracement)"),
                         (0.2, "Shallow entry (20% retracement)")]:
        cands = build_frac_trades(frac)
        print(f"  frac={frac}: {len(cands):,} candidates ({time.time()-t0:.0f}s)")
        res = report_variant(label, cands, n_mc_sims=3000)
        if res is not None:
            thirds_robustness(res["accepted"], label)
        variants[frac] = res

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for frac, res in variants.items():
        if res is None:
            print(f"  frac={frac}: too few trades")
            continue
        print(f"  frac={frac:.1f}: PF={res['pf']:.3f}  WR={res['wr']:.2f}%  trades/yr={res['trades_per_year']:.0f}  "
              f"combined_final=${res['combined_final']:,.0f}  maxDD={res['max_dd']:.2f}%  "
              f"MC%reach1M={res['mc_pct_reached']:.1f}%  MC_median_days={res['mc_median_days']}")

    print(f"\n(total: {time.time()-t0:.0f}s)")
