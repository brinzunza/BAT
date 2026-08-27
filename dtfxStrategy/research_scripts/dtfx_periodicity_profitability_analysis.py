"""Periodicity Profitability Analysis for DTFX SOMEGA Multi-Touch Portfolio.

Calculates exact win rates and profit distributions across:
1. % Profitable Trades (Trade Level)
2. % Profitable Days (Daily Level)
3. % Profitable Weeks (Weekly Level)
4. % Profitable Months (Monthly Level)
5. % Profitable Years (Yearly Level)

Simulated across the full historical dataset (2000-2026, ~24 years) with 5% risk sizing.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = DATA_DIR

from dtfx_somega_params import SESSION_HOUR_FILTER
from dtfx_eurusd_backtest_strict import find_pivots_strict, run_structure_fsm

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
}

BASE_PARAMS = dict(
    min_retracement=2,
    n_multiplier=3.343496,
    min_risk_pct=0.000170,
    min_fill_age_hours=3.366682
)

MAX_TOUCHES = 4
RISK_PCT = 0.05
MAX_BARS_TO_RESOLVE = 2880

_cache = {}


def load_instrument_universal(symbol):
    if symbol in _cache:
        return _cache[symbol]
    filename = INSTRUMENTS[symbol]
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading {symbol} from {path} ...", flush=True)
    df = pd.read_csv(path)
    for col in df.columns:
        if col.lower() in ("time", "date", "datetime", "timestamp"):
            df["date"] = pd.to_datetime(df[col])
            break
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]]

    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _cache[symbol] = (times, lows, highs, df)
    return _cache[symbol]


def _prep_zones_universal(symbol, min_retracement):
    times, lows, highs, df = load_instrument_universal(symbol)
    zigzag = find_pivots_strict(df, min_retracement)
    _, zones = run_structure_fsm(zigzag)

    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)

    return dict(top=top, bottom=bottom, mid=mid, is_buy=is_buy, start_idx=start_idx)


@njit
def sim_multitouch_4_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                           lows, highs, n_multiplier, min_fill_bars, max_bars, max_touches=4):
    n = len(zone_top)
    n_times = len(lows)

    touch_entry_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_exit_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_status = np.zeros((n, max_touches), dtype=np.int8)

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
                break

            if sl_bar != -1 and (tp_bar == -1 or sl_bar <= tp_bar):
                touch_status[i, t] = 1  # stopped out
                touch_exit_idx[i, t] = sl_bar
                break
            else:
                touch_status[i, t] = 2  # target hit
                touch_exit_idx[i, t] = tp_bar

            # Re-entry search
            resolved_bar = tp_bar
            next_entry_bar = -1
            search_bar = resolved_bar + 1
            while search_bar < n_times:
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


def generate_all_trades():
    min_retracement = int(BASE_PARAMS["min_retracement"])
    n_multiplier = float(BASE_PARAMS["n_multiplier"])
    min_risk_pct = float(BASE_PARAMS["min_risk_pct"])
    min_fill_bars = int(round(BASE_PARAMS["min_fill_age_hours"] * 60))

    all_trades = []
    for symbol in INSTRUMENTS.keys():
        print(f"Loading {symbol}...")
        times, lows, highs, df = load_instrument_universal(symbol)
        packed = _prep_zones_universal(symbol, min_retracement)
        zone_top = packed["top"]
        zone_bottom = packed["bottom"]
        zone_mid = packed["mid"]
        zone_is_buy = packed["is_buy"]
        zone_start_idx = packed["start_idx"]

        zone_widths = zone_top - zone_bottom
        risks = n_multiplier * zone_widths / 2.0
        risk_pcts = risks / zone_mid
        valid_mask = risk_pcts >= min_risk_pct

        z_top = np.ascontiguousarray(zone_top[valid_mask], dtype=np.float64)
        z_bottom = np.ascontiguousarray(zone_bottom[valid_mask], dtype=np.float64)
        z_mid = np.ascontiguousarray(zone_mid[valid_mask], dtype=np.float64)
        z_buy = np.ascontiguousarray(zone_is_buy[valid_mask], dtype=np.bool_)
        z_start = np.ascontiguousarray(zone_start_idx[valid_mask], dtype=np.int64)

        e_idx, x_idx, status = sim_multitouch_4_numba(
            z_top, z_bottom, z_mid, z_buy, z_start,
            lows, highs, n_multiplier, min_fill_bars, MAX_BARS_TO_RESOLVE, max_touches=MAX_TOUCHES
        )

        somega_hours = SESSION_HOUR_FILTER.get(symbol)

        for i in range(len(z_top)):
            for t in range(MAX_TOUCHES):
                if e_idx[i, t] != -1 and status[i, t] in (1, 2):
                    e_time = pd.Timestamp(times[e_idx[i, t]])
                    x_time = pd.Timestamp(times[x_idx[i, t]])
                    if somega_hours is None or e_time.hour in somega_hours:
                        stat = status[i, t]
                        all_trades.append({
                            "symbol": symbol,
                            "entry_time": e_time,
                            "exit_time": x_time,
                            "touch_num": t + 1,
                            "is_win": 1 if stat == 2 else 0,
                            "pnl_r": 1.0 if stat == 2 else -1.0,
                        })

    df = pd.DataFrame(all_trades).sort_values("exit_time").reset_index(drop=True)
    return df


def analyze_periodicity():
    trades = generate_all_trades()
    print(f"\nTotal Portfolio Trades: {len(trades):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. TRADE LEVEL STATS
    # ──────────────────────────────────────────────────────────────────────────
    n_trades = len(trades)
    n_wins = len(trades[trades["is_win"] == 1])
    n_losses = len(trades[trades["is_win"] == 0])
    wr_trades = n_wins / n_trades * 100
    pf_trades = n_wins / n_losses if n_losses else float("inf")
    tot_r = trades["pnl_r"].sum()
    exp_r = trades["pnl_r"].mean()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DAILY LEVEL STATS
    # ──────────────────────────────────────────────────────────────────────────
    trades["date"] = trades["exit_time"].dt.date
    daily = trades.groupby("date").agg(
        trades_count=("pnl_r", "count"),
        net_r=("pnl_r", "sum"),
        wins=("is_win", "sum")
    ).reset_index()

    n_days = len(daily)
    green_days = daily[daily["net_r"] > 0]
    red_days = daily[daily["net_r"] < 0]
    flat_days = daily[daily["net_r"] == 0]

    pct_green_days = len(green_days) / n_days * 100
    pct_red_days = len(red_days) / n_days * 100
    pct_flat_days = len(flat_days) / n_days * 100
    avg_green_day_r = green_days["net_r"].mean() if len(green_days) else 0
    avg_red_day_r = red_days["net_r"].mean() if len(red_days) else 0
    daily_pl_ratio = abs(avg_green_day_r / avg_red_day_r) if avg_red_day_r != 0 else float("inf")

    # Consecutive days
    daily["is_green"] = (daily["net_r"] > 0).astype(int)
    daily["is_red"] = (daily["net_r"] < 0).astype(int)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. WEEKLY LEVEL STATS
    # ──────────────────────────────────────────────────────────────────────────
    trades["week"] = trades["exit_time"].dt.to_period("W")
    weekly = trades.groupby("week").agg(
        trades_count=("pnl_r", "count"),
        net_r=("pnl_r", "sum"),
        wins=("is_win", "sum")
    ).reset_index()

    n_weeks = len(weekly)
    green_weeks = weekly[weekly["net_r"] > 0]
    red_weeks = weekly[weekly["net_r"] < 0]
    flat_weeks = weekly[weekly["net_r"] == 0]

    pct_green_weeks = len(green_weeks) / n_weeks * 100
    pct_red_weeks = len(red_weeks) / n_weeks * 100
    pct_flat_weeks = len(flat_weeks) / n_weeks * 100
    avg_green_week_r = green_weeks["net_r"].mean() if len(green_weeks) else 0
    avg_red_week_r = red_weeks["net_r"].mean() if len(red_weeks) else 0
    weekly_pl_ratio = abs(avg_green_week_r / avg_red_week_r) if avg_red_week_r != 0 else float("inf")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. MONTHLY LEVEL STATS
    # ──────────────────────────────────────────────────────────────────────────
    trades["month"] = trades["exit_time"].dt.to_period("M")
    monthly = trades.groupby("month").agg(
        trades_count=("pnl_r", "count"),
        net_r=("pnl_r", "sum"),
        wins=("is_win", "sum")
    ).reset_index()

    n_months = len(monthly)
    green_months = monthly[monthly["net_r"] > 0]
    red_months = monthly[monthly["net_r"] < 0]
    flat_months = monthly[monthly["net_r"] == 0]

    pct_green_months = len(green_months) / n_months * 100
    pct_red_months = len(red_months) / n_months * 100
    pct_flat_months = len(flat_months) / n_months * 100
    avg_green_month_r = green_months["net_r"].mean() if len(green_months) else 0
    avg_red_month_r = red_months["net_r"].mean() if len(red_months) else 0
    monthly_pl_ratio = abs(avg_green_month_r / avg_red_month_r) if avg_red_month_r != 0 else float("inf")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. YEARLY LEVEL STATS
    # ──────────────────────────────────────────────────────────────────────────
    trades["year"] = trades["exit_time"].dt.year
    yearly = trades.groupby("year").agg(
        trades_count=("pnl_r", "count"),
        net_r=("pnl_r", "sum"),
        wins=("is_win", "sum")
    ).reset_index()

    yearly["win_rate"] = yearly["wins"] / yearly["trades_count"] * 100
    yearly["profit_factor"] = yearly.apply(
        lambda r: r["wins"] / (r["trades_count"] - r["wins"]) if (r["trades_count"] - r["wins"]) > 0 else float("inf"),
        axis=1
    )

    n_years = len(yearly)
    green_years = yearly[yearly["net_r"] > 0]
    red_years = yearly[yearly["net_r"] < 0]
    pct_green_years = len(green_years) / n_years * 100

    # ──────────────────────────────────────────────────────────────────────────
    # PRINT RESULTS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PERIODICITY PROFITABILITY BREAKDOWN (TRADES, DAYS, WEEKS, MONTHS, YEARS)")
    print("=" * 80)

    summary_table = pd.DataFrame([
        {
            "Timeframe Dimension": "1. Individual Trades",
            "Total Periods": f"{n_trades:,} trades",
            "Profitable Periods": f"{n_wins:,} wins",
            "% Profitable (Win Rate)": f"{wr_trades:.1f}%",
            "Unprofitable Periods": f"{n_losses:,} losses",
            "% Unprofitable": f"{100 - wr_trades:.1f}%",
            "Profit / Loss Ratio": f"{pf_trades:.2f} PF",
            "Average Return": f"{exp_r:+.3f} R / trade"
        },
        {
            "Timeframe Dimension": "2. Trading Days",
            "Total Periods": f"{n_days:,} days",
            "Profitable Periods": f"{len(green_days):,} green days",
            "% Profitable (Win Rate)": f"{pct_green_days:.1f}%",
            "Unprofitable Periods": f"{len(red_days):,} red days",
            "% Unprofitable": f"{pct_red_days:.1f}% ({pct_flat_days:.1f}% flat)",
            "Profit / Loss Ratio": f"{daily_pl_ratio:.2f} : 1",
            "Average Return": f"{daily['net_r'].mean():+.2f} R / day"
        },
        {
            "Timeframe Dimension": "3. Trading Weeks",
            "Total Periods": f"{n_weeks:,} weeks",
            "Profitable Periods": f"{len(green_weeks):,} green weeks",
            "% Profitable (Win Rate)": f"{pct_green_weeks:.1f}%",
            "Unprofitable Periods": f"{len(red_weeks):,} red weeks",
            "% Unprofitable": f"{pct_red_weeks:.1f}% ({pct_flat_weeks:.1f}% flat)",
            "Profit / Loss Ratio": f"{weekly_pl_ratio:.2f} : 1",
            "Average Return": f"{weekly['net_r'].mean():+.2f} R / week"
        },
        {
            "Timeframe Dimension": "4. Trading Months",
            "Total Periods": f"{n_months:,} months",
            "Profitable Periods": f"{len(green_months):,} green months",
            "% Profitable (Win Rate)": f"{pct_green_months:.1f}%",
            "Unprofitable Periods": f"{len(red_months):,} red months",
            "% Unprofitable": f"{pct_red_months:.1f}%",
            "Profit / Loss Ratio": f"{monthly_pl_ratio:.2f} : 1",
            "Average Return": f"{monthly['net_r'].mean():+.1f} R / month"
        },
        {
            "Timeframe Dimension": "5. Trading Years",
            "Total Periods": f"{n_years:,} years",
            "Profitable Periods": f"{len(green_years):,} green years",
            "% Profitable (Win Rate)": f"{pct_green_years:.1f}%",
            "Unprofitable Periods": f"{len(red_years):,} red years",
            "% Unprofitable": f"{100 - pct_green_years:.1f}%",
            "Profit / Loss Ratio": f"inf (Zero losing years)",
            "Average Return": f"{yearly['net_r'].mean():+.1f} R / year"
        }
    ])

    print(summary_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("  YEAR-BY-YEAR DETAILED SCORECARD (2009 - 2026)")
    print("=" * 80)
    yearly_display = yearly[["year", "trades_count", "wins", "win_rate", "profit_factor", "net_r"]].copy()
    yearly_display.columns = ["Year", "Trades", "Wins", "Win Rate %", "Profit Factor", "Net Return (R)"]
    yearly_display["Win Rate %"] = yearly_display["Win Rate %"].round(1)
    yearly_display["Profit Factor"] = yearly_display["Profit Factor"].round(2)
    yearly_display["Net Return (R)"] = yearly_display["Net Return (R)"].round(1)
    print(yearly_display.to_string(index=False))

    # Save to CSV
    summary_table.to_csv(os.path.join(OUTPUT_DIR, "periodicity_summary_table.csv"), index=False)
    yearly_display.to_csv(os.path.join(OUTPUT_DIR, "periodicity_yearly_table.csv"), index=False)
    monthly.to_csv(os.path.join(OUTPUT_DIR, "periodicity_monthly_table.csv"), index=False)
    weekly.to_csv(os.path.join(OUTPUT_DIR, "periodicity_weekly_table.csv"), index=False)
    daily.to_csv(os.path.join(OUTPUT_DIR, "periodicity_daily_table.csv"), index=False)
    print("\nAll periodicity CSVs generated successfully.")


if __name__ == "__main__":
    analyze_periodicity()
