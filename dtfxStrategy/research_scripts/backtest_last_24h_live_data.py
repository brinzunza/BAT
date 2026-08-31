"""Fetch latest live 1-minute candles from OANDA API and backtest the last 24 hours of strategy signals.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

ROOT = '/Users/brunoinzunza/Documents/github/PRODLIVETRADING'
sys.path.insert(0, ROOT)
import config
from broker.oanda_client import OANDAClient
from strategy.pivots import find_pivots_strict
from strategy.engine import run_structure_fsm
from numba import njit

SESSION_HOUR_FILTER = {
    "EURUSD": None,
    "EURGBP": range(18, 24),
    "GBPUSD": range(20, 23),
    "USDJPY": range(19, 23),
}

MAX_TOUCHES = 4
MIN_FILL_AGE_HOURS = 3.3333333333333335
N_MULTIPLIER = 1.0
MIN_RISK_PCT = 0.0003
MAX_BARS_TO_RESOLVE = 2880

@njit
def sim_multitouch_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                         lows, highs, n_multiplier, min_fill_bars, max_bars, max_touches=4):
    n = len(zone_top)
    n_times = len(lows)

    touch_entry_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_exit_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_status = np.zeros((n, max_touches), dtype=np.int8)  # 0=none/open, 1=loss, 2=win

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        zone_width = top - bottom
        risk = n_multiplier * zone_width / 2.0
        invalidation = bottom if is_buy else top
        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk

        start = zone_start_idx[i]
        if start >= n_times:
            continue

        earliest_valid_entry = start + min_fill_bars
        if earliest_valid_entry >= n_times:
            continue

        # Find first valid midpoint touch
        current_bar = start
        first_touch_bar = -1
        while current_bar < n_times:
            if is_buy and lows[current_bar] <= invalidation:
                break
            elif (not is_buy) and highs[current_bar] >= invalidation:
                break

            mid_hit = (lows[current_bar] <= mid) if is_buy else (highs[current_bar] >= mid)
            if mid_hit:
                if current_bar >= earliest_valid_entry:
                    first_touch_bar = current_bar
                    break
                else:
                    break
            current_bar += 1

        if first_touch_bar == -1:
            continue

        sim_bar = first_touch_bar
        for t in range(max_touches):
            touch_entry_idx[i, t] = sim_bar
            after_start = sim_bar + 1
            after_end = min(after_start + max_bars, n_times)

            sl_bar, tp_bar = -1, -1
            for j in range(after_start, after_end):
                if is_buy:
                    if sl_bar == -1 and lows[j] <= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and highs[j] >= target_level:
                        tp_bar = j
                else:
                    if sl_bar == -1 and highs[j] >= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and lows[j] <= target_level:
                        tp_bar = j
                if sl_bar != -1 and tp_bar != -1:
                    break

            if sl_bar == -1 and tp_bar == -1:
                # Still open
                touch_status[i, t] = 0
                touch_exit_idx[i, t] = n_times - 1
                break

            if sl_bar != -1 and (tp_bar == -1 or sl_bar <= tp_bar):
                touch_status[i, t] = 1  # stopped out
                touch_exit_idx[i, t] = sl_bar
                break
            else:
                touch_status[i, t] = 2  # target hit
                touch_exit_idx[i, t] = tp_bar

            # Re-entry search
            re_search_start = tp_bar + 1
            re_search_end = min(re_search_start + max_bars, n_times)
            next_entry_bar = -1
            search_bar = re_search_start
            while search_bar < re_search_end:
                if is_buy and lows[search_bar] <= invalidation:
                    break
                elif (not is_buy) and highs[search_bar] >= invalidation:
                    break
                re_mid_hit = (lows[search_bar] <= mid) if is_buy else (highs[search_bar] >= mid)
                if re_mid_hit:
                    next_entry_bar = search_bar
                    break
                search_bar += 1

            if next_entry_bar == -1:
                break
            sim_bar = next_entry_bar

    return touch_entry_idx, touch_exit_idx, touch_status

def main():
    print("=" * 115)
    print("  FETCHING LATEST LIVE OANDA 1-MINUTE CANDLES & BACKTESTING LAST 24 HOURS")
    print("=" * 115)

    client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)
    symbols = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]

    now_utc = datetime.now(timezone.utc)
    since_24h = now_utc - timedelta(hours=24)
    print(f"Current Local PDT: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print(f"Current UTC:       {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Audit Window:      {since_24h.strftime('%Y-%m-%d %H:%M:%S UTC')} to Present (Last 24 Hours)\n")

    all_trades = []
    min_fill_bars = int(round(MIN_FILL_AGE_HOURS * 60))

    for sym in symbols:
        print(f"Fetching 4,500 latest 1-min candles for {sym} from OANDA...")
        df = client.fetch_warmup_bars(sym, count=4500)
        times = df.index.values
        lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
        highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)

        zigzag = find_pivots_strict(df, min_retracement=3)
        _, zones = run_structure_fsm(zigzag)

        if not zones:
            print(f"  [{sym}] No zones detected in warmup range.")
            continue

        top = np.array([z["top"] for z in zones], dtype=np.float64)
        bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
        mid = np.array([z["mid"] for z in zones], dtype=np.float64)
        is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
        created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
        start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)

        zone_widths = top - bottom
        risks = N_MULTIPLIER * zone_widths / 2.0
        risk_pcts = risks / mid
        valid_mask = risk_pcts >= MIN_RISK_PCT

        z_top = np.ascontiguousarray(top[valid_mask], dtype=np.float64)
        z_bottom = np.ascontiguousarray(bottom[valid_mask], dtype=np.float64)
        z_mid = np.ascontiguousarray(mid[valid_mask], dtype=np.float64)
        z_buy = np.ascontiguousarray(is_buy[valid_mask], dtype=np.bool_)
        z_start = np.ascontiguousarray(start_idx[valid_mask], dtype=np.int64)

        e_idx, x_idx, status = sim_multitouch_numba(
            z_top, z_bottom, z_mid, z_buy, z_start,
            lows, highs, N_MULTIPLIER, min_fill_bars, MAX_BARS_TO_RESOLVE, max_touches=MAX_TOUCHES
        )

        somega_hours = SESSION_HOUR_FILTER.get(sym)

        for i in range(len(z_top)):
            for t in range(MAX_TOUCHES):
                if e_idx[i, t] != -1:
                    e_time = pd.Timestamp(times[e_idx[i, t]]).tz_localize(None) if pd.Timestamp(times[e_idx[i, t]]).tzinfo is not None else pd.Timestamp(times[e_idx[i, t]])
                    x_time = pd.Timestamp(times[x_idx[i, t]]).tz_localize(None) if pd.Timestamp(times[x_idx[i, t]]).tzinfo is not None else pd.Timestamp(times[x_idx[i, t]])
                    since_ts = pd.Timestamp(since_24h).tz_localize(None)
                    stat = status[i, t]
                    
                    # Filter for trades entering or active in last 24h
                    if e_time >= since_ts or (x_time >= since_ts and stat != 0):
                        if somega_hours is None or e_time.hour in somega_hours:
                            dur_mins = int(round((x_time - e_time).total_seconds() / 60.0))
                            dur_h = dur_mins // 60
                            dur_m = dur_mins % 60
                            
                            z_w = z_top[i] - z_bottom[i]
                            r_amt = N_MULTIPLIER * z_w / 2.0
                            entry_p = z_mid[i]
                            stop_p = entry_p - r_amt if z_buy[i] else entry_p + r_amt
                            target_p = entry_p + r_amt if z_buy[i] else entry_p - r_amt

                            all_trades.append({
                                "symbol": sym,
                                "side": "BUY" if z_buy[i] else "SELL",
                                "touch_count": t + 1,
                                "entry_time_utc": e_time,
                                "exit_time_utc": x_time,
                                "entry_time_pdt": e_time - pd.Timedelta(hours=7),
                                "exit_time_pdt": x_time - pd.Timedelta(hours=7),
                                "duration": f"{dur_h}h {dur_m:02d}m",
                                "entry_price": entry_p,
                                "stop_price": stop_p,
                                "target_price": target_p,
                                "status_code": stat,
                                "outcome": "WIN (+1.0R)" if stat == 2 else ("LOSS (-1.0R)" if stat == 1 else "OPEN (ACTIVE)"),
                                "pnl_r": 1.0 if stat == 2 else (-1.0 if stat == 1 else 0.0),
                            })

    df_trades = pd.DataFrame(all_trades)
    if len(df_trades) == 0:
        print("\n" + "=" * 115)
        print("  NO TRADES TRIGGERED IN THE LAST 24 HOURS ACROSS ALL 4 TICKERS.")
        print("  (Market remained within consolidation bands without triggering midpoint fills).")
        print("=" * 115 + "\n")
        return

    df_trades = df_trades.sort_values("entry_time_utc").reset_index(drop=True)

    print("\n" + "=" * 135)
    print(f"  LAST 24 HOURS EXACT BACKTEST RESULTS ({since_24h.strftime('%Y-%m-%d %H:%M UTC')} to {now_utc.strftime('%Y-%m-%d %H:%M UTC')})")
    print("=" * 135)

    n_tot = len(df_trades)
    n_win = (df_trades["status_code"] == 2).sum()
    n_loss = (df_trades["status_code"] == 1).sum()
    n_open = (df_trades["status_code"] == 0).sum()
    tot_r = df_trades["pnl_r"].sum()
    wr = (n_win / (n_win + n_loss) * 100.0) if (n_win + n_loss) > 0 else 0.0
    pf = (n_win / n_loss) if n_loss > 0 else float("inf")

    print(f"  • Total Trades Triggered: {n_tot}")
    print(f"  • Wins / Losses / Open:   {n_win} Wins / {n_loss} Losses / {n_open} Open")
    print(f"  • Win Rate % (Resolved):  {wr:.1f}%")
    print(f"  • Profit Factor:          {pf:.2f}")
    print(f"  • Net Return:             {tot_r:+.1f} R")
    print("-" * 135)

    START_BALANCE = 100.00
    RISK_PCT = 0.05
    curr_bal = START_BALANCE

    display_rows = []
    for idx, r in df_trades.iterrows():
        risk_dollars = curr_bal * RISK_PCT
        pnl_dollars = r["pnl_r"] * risk_dollars
        bal_after = curr_bal + pnl_dollars if r["status_code"] != 0 else curr_bal

        display_rows.append({
            "Trade #": idx + 1,
            "Ticker": r["symbol"],
            "Side": r["side"],
            "Touch": f"#{r['touch_count']}",
            "Entry Time (PDT)": r["entry_time_pdt"].strftime("%I:%M %p"),
            "Exit Time (PDT)": r["exit_time_pdt"].strftime("%I:%M %p") if r["status_code"] != 0 else "OPEN",
            "Duration": r["duration"] if r["status_code"] != 0 else "ACTIVE",
            "Entry Price": f"{r['entry_price']:.5f}",
            "Stop Price": f"{r['stop_price']:.5f}",
            "Target Price": f"{r['target_price']:.5f}",
            "Bal Before": f"${curr_bal:.2f}",
            "Risk ($)": f"${risk_dollars:.2f}",
            "Outcome": r["outcome"],
            "Profit ($)": f"{pnl_dollars:+.2f}" if r["status_code"] != 0 else "$0.00",
            "Bal After": f"${bal_after:.2f}",
        })
        if r["status_code"] != 0:
            curr_bal = bal_after

    print(pd.DataFrame(display_rows).to_string(index=False))
    print("=" * 135)
    print(f"  • Starting Balance:    ${START_BALANCE:.2f}")
    print(f"  • Ending Total Balance: ${curr_bal:.2f} ({((curr_bal - START_BALANCE)/START_BALANCE)*100:+.2f}%)")
    print("=" * 135 + "\n")

if __name__ == "__main__":
    main()
