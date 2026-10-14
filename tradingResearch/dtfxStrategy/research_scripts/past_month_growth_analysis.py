"""Detailed Past-Month Performance & $200 Account Growth Simulation for the New Setup:
- USDJPY: 10.0 pips
- EURUSD: 5.0 pips
- GBPUSD: 5.0 pips
- EURGBP: 5.0 pips

Metrics analyzed:
1. Average Risk:Reward ratio post-spread (effective net win R vs loss R)
2. Average duration in trade (minutes / bars held)
3. Fresh $200 compounding simulation ($100 buy, $100 sell, 3% risk, 1-per-side max concurrency, 1.6-pip spread)
4. Day-by-day equity progression & milestone timeline
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
import goal79_two_account_portfolio as g79

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
ZONE_PIPS_MAP = {
    "USDJPY": 10.0,
    "EURUSD": 5.0,
    "GBPUSD": 5.0,
    "EURGBP": 5.0,
}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440
RISK_PCT = 0.03


def load_and_build_cache():
    cache = {}
    for inst, (fname, pip) in INSTRUMENTS.items():
        path = os.path.join(DATA_DIR, fname)
        fresh_path = f"/tmp/dtfx_fresh_data/{inst}_merged.csv"
        if os.path.exists(fresh_path):
            path = fresh_path

        df = pd.read_csv(path, parse_dates=["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.drop_duplicates(subset="date").set_index("date").sort_index()

        times = df.index
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        close_ = df["close"].values.astype(np.float64)
        n_bars = len(df)

        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_, low_, close_)
        (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
         ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
        last_high_of_bar, last_low_of_bar = g99.build_last_confirmed(confirm_idx, is_high, price, n_bars)

        cache[inst] = dict(
            pip=pip, times=times, high=high_, low=low_, close=close_,
            z_created=z_created, z_top=z_top, z_bottom=z_bottom, z_is_buy=z_is_buy,
            last_high_of_bar=last_high_of_bar, last_low_of_bar=last_low_of_bar
        )
    return cache


def build_candidate_trades(cache, symmetric_exit_spread=True):
    frames = []
    for inst, d in cache.items():
        min_zone_pips = ZONE_PIPS_MAP[inst]
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price, MAX_WINDOW,
            False, min_zone_height, symmetric_exit_spread
        )

        resolved = outcome > 0
        ei = entry_idx[resolved]
        xi = exit_idx[resolved]
        is_buy = d["z_is_buy"][resolved]
        raw_stop = np.where(is_buy, d["z_bottom"][resolved], d["z_top"][resolved])
        if symmetric_exit_spread:
            half_spread = spread_price / 2.0
            stop_price = np.where(is_buy, raw_stop - half_spread, raw_stop + half_spread)
        else:
            stop_price = raw_stop
        stop_risk = np.abs(entry_price[resolved] - stop_price)
        bars_held = xi - ei
        frames.append(pd.DataFrame({
            "instrument": inst,
            "symbol": inst,
            "entry_time": d["times"][ei],
            "exit_time": d["times"][xi],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk,
            "net_r": net_r[resolved],
            "is_buy": is_buy,
            "bars_held": bars_held,
        }))
    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return trades


def main():
    print("Loading data and precomputing swings...")
    cache = load_and_build_cache()
    all_trades = build_candidate_trades(cache)
    print(f"Total candidate trades across history: {len(all_trades):,}")

    # Determine past month window
    latest_time = all_trades["entry_time"].max()
    month_start = latest_time - pd.Timedelta(days=30)
    print(f"\nPast 30 Days Window: {month_start.strftime('%Y-%m-%d %H:%M')} to {latest_time.strftime('%Y-%m-%d %H:%M')}")

    # Filter trades to past 30 days
    month_trades = all_trades[all_trades["entry_time"] >= month_start].copy().reset_index(drop=True)
    print(f"Candidate trades in past 30 days: {len(month_trades):,}")

    # 1. Simulate fresh $200 account over the past 30 days
    result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
        month_trades, risk_pct=RISK_PCT, start_per_side=100.0,
        max_leverage=50.0, cap=10_000_000.0, track_detail=True
    )
    acc = result[result["accepted"]].copy().reset_index(drop=True)
    n_exec = len(acc)

    # 2. Key Metrics: Win Rate, Profit Factor, Risk:Reward, Time in Trade
    wins = acc[acc["net_r"] > 0]
    losses = acc[acc["net_r"] <= 0]
    wr = 100.0 * len(wins) / n_exec if n_exec else 0.0
    avg_win_r = wins["net_r"].mean() if len(wins) else 0.0
    avg_loss_r = losses["net_r"].mean() if len(losses) else -1.0
    eff_rr = abs(avg_win_r / avg_loss_r) if avg_loss_r != 0 else 0.0
    gw_r = wins["net_r"].sum()
    gl_r = -losses["net_r"].sum()
    pf = gw_r / gl_r if gl_r > 0 else float("inf")

    avg_bars_all = acc["bars_held"].mean()
    median_bars_all = acc["bars_held"].median()
    p90_bars_all = acc["bars_held"].quantile(0.90)
    avg_bars_win = wins["bars_held"].mean() if len(wins) else 0.0
    avg_bars_loss = losses["bars_held"].mean() if len(losses) else 0.0

    final_equity = eq_buy + eq_sell
    total_gain_dollars = final_equity - 200.0
    total_gain_pct = (final_equity / 200.0 - 1.0) * 100.0

    print("\n" + "="*75)
    print("  PAST 30-DAY PERFORMANCE & TRADE METRICS (FRESH $200 ACCOUNT)")
    print("="*75)
    print(f"Starting Capital:          $200.00 ($100 Long Account + $100 Short Account)")
    print(f"Final Capital:             ${final_equity:,.2f}")
    print(f"Net Profit ($):            ${total_gain_dollars:,.2f} (+{total_gain_pct:,.1f}%)")
    print(f"Max Drawdown:              {max_dd*100:.2f}%")
    print(f"Executed Trades:           {n_exec:,} (~{n_exec/30:.1f} trades/day)")
    print(f"Win Rate:                  {wr:.2f}% ({len(wins):,} Wins / {len(losses):,} Losses)")
    print(f"Profit Factor:             {pf:.3f}")

    print("\n" + "-"*75)
    print("  RISK : REWARD & TRADE DURATION METRICS")
    print("-" * 75)
    print(f"Average Win (Post-Spread): +{avg_win_r:.3f} R")
    print(f"Average Loss (Stop Hit):   {avg_loss_r:.3f} R")
    print(f"Effective Post-Spread RR:  1 : {eff_rr:.2f} (Target trimmed slightly by 1.6-pip spread)")
    print(f"Average Time in Trade:     {avg_bars_all:.1f} minutes ({avg_bars_all/60:.2f} hours)")
    print(f"Median Time in Trade:      {median_bars_all:.0f} minutes")
    print(f"90th Percentile Duration:  {p90_bars_all:.0f} minutes")
    print(f"Avg Duration (Winners):    {avg_bars_win:.1f} minutes")
    print(f"Avg Duration (Losses):     {avg_bars_loss:.1f} minutes")

    print("\n" + "-"*75)
    print("  PER-SYMBOL BREAKDOWN (PAST 30 DAYS)")
    print("-" * 75)
    for sym in ["USDJPY", "EURUSD", "GBPUSD", "EURGBP"]:
        s_acc = acc[acc["symbol"] == sym]
        s_wins = s_acc[s_acc["net_r"] > 0]
        s_losses = s_acc[s_acc["net_r"] <= 0]
        s_wr = 100.0 * len(s_wins) / len(s_acc) if len(s_acc) else 0.0
        s_gw = s_wins["net_r"].sum()
        s_gl = -s_losses["net_r"].sum()
        s_pf = s_gw / s_gl if s_gl > 0 else float("inf")
        s_dur = s_acc["bars_held"].mean() if len(s_acc) else 0.0
        s_win_r = s_wins["net_r"].mean() if len(s_wins) else 0.0
        print(f"  {sym:8}: {len(s_acc):>4} trades | WR: {s_wr:>5.1f}% | PF: {s_pf:>5.2f} | Avg Win: +{s_win_r:.2f}R | Avg Dur: {s_dur:>4.1f}m")

    # 3. Day-by-Day Equity Growth Progression
    trace_df = pd.DataFrame(trace)
    trace_df["date"] = pd.to_datetime(trace_df["close_time"]).dt.date
    daily_summary = []
    for day, group in trace_df.groupby("date"):
        d_end_eq = group["equity_combined"].iloc[-1]
        d_trades = len(group)
        daily_summary.append({"date": day, "end_equity": d_end_eq, "trades": d_trades})
    
    daily_df = pd.DataFrame(daily_summary)
    daily_df["prev_equity"] = daily_df["end_equity"].shift(1).fillna(200.0)
    daily_df["daily_gain_pct"] = (daily_df["end_equity"] / daily_df["prev_equity"] - 1.0) * 100.0
    daily_df["cum_gain_pct"] = (daily_df["end_equity"] / 200.0 - 1.0) * 100.0

    print("\n" + "="*75)
    print("  DAY-BY-DAY ACCOUNT GROWTH PROGRESSION ($200 INITIAL)")
    print("="*75)
    print(f"{'Date':<12} | {'Trades':<8} | {'Ending Equity':<15} | {'Daily Gain':<12} | {'Total Return':<12}")
    print("-" * 75)
    for _, row in daily_df.iterrows():
        print(f"{str(row['date']):<12} | {int(row['trades']):<8} | ${row['end_equity']:>13,.2f} | {row['daily_gain_pct']:>+10.2f}% | {row['cum_gain_pct']:>+10.2f}%")
    print("="*75)


if __name__ == "__main__":
    main()
