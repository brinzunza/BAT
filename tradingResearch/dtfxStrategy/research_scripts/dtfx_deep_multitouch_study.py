"""Deep Multi-Touch (1 to 10+) & Condition Specifics Backtesting Engine.

Investigates:
1. Comprehensive performance across Touch 1 through 10+
2. Prior outcome conditioning: Does Touch N work after a WIN vs after a LOSS?
3. Zone Age conditioning: <12h, 12-24h, 24-48h, 2-7d, >7d
4. SOMEGA session filtering on subsequent touches
5. Cumulative portfolio equity comparison across touch caps (Max 1, 2, 3, 4, 5, Unlimited)
6. Cross-asset robustness across EURUSD, GBPUSD, EURGBP, USDJPY, XAUUSD, BTCUSD, SPY
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = DATA_DIR

from dtfx_robust_search import (
    _prep_zones, load_instrument, _indicator_cache, _price_cache,
    ANALYSIS_START, FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE
)
from dtfx_somega_params import SESSION_HOUR_FILTER

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
    "BTCUSD": "lse_btcusd_1min_2018on.csv",
    "SPY":    "lse_spy_1min.csv",
    "XAUUSD": "lse_xauusd_full_1min.csv",
}

BASE_PARAMS = dict(
    min_retracement=2,
    n_multiplier=3.343496,
    min_risk_pct=0.000170,
    min_fill_age_hours=3.366682
)

MAX_TOUCHES = 10


@njit
def sim_deep_multitouch_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                              lows, highs, n_multiplier, min_fill_bars, max_bars, max_touches=10):
    """Simulates Touch 1 through 10+ for each zone.
    Returns:
      touch_entry_idx[i, k]: entry bar index for k-th touch (k=0..max_touches-1)
      touch_exit_idx[i, k]: exit bar index
      touch_status[i, k]: 1=stop, 2=target, 0=none
    """
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
            # Invalidation check
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
                    # Fast touch -> disqualified
                    break
            current_bar += 1

        if first_touch_bar == -1:
            continue

        # Simulate Touches 1 to max_touches sequentially
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
                touch_status[i, t] = 1  # stop out (-1.0R)
                touch_exit_idx[i, t] = sl_bar
                # Once stopped out, the zone is breached/broken -> no further touches
                break
            else:
                touch_status[i, t] = 2  # target hit (+1.0R)
                touch_exit_idx[i, t] = tp_bar

            # Look for NEXT re-entry into mid for Touch t + 1
            resolved_bar = tp_bar
            next_entry_bar = -1
            search_bar = resolved_bar + 1

            while search_bar < n_times:
                # Check invalidation
                if is_buy and lows[search_bar] <= invalidation:
                    break
                elif (not is_buy) and highs[search_bar] >= invalidation:
                    break

                # Check midpoint re-entry
                re_mid_hit = (lows[search_bar] <= mid) if is_buy else (highs[search_bar] >= mid)
                if re_mid_hit:
                    next_entry_bar = search_bar
                    break

                search_bar += 1

            if next_entry_bar == -1:
                break

            sim_bar = next_entry_bar

    return touch_entry_idx, touch_exit_idx, touch_status


def run_all_instruments():
    print("=" * 80)
    print("  DEEP MULTI-TOUCH (1 to 10+) & CONDITION SPECIFICS BACKTEST")
    print("=" * 80)

    min_retracement = int(BASE_PARAMS["min_retracement"])
    n_multiplier = float(BASE_PARAMS["n_multiplier"])
    min_risk_pct = float(BASE_PARAMS["min_risk_pct"])
    min_fill_bars = int(round(BASE_PARAMS["min_fill_age_hours"] * 60))

    all_trades = []

    for symbol in INSTRUMENTS.keys():
        if symbol not in INSTRUMENTS:
            continue

        print(f"\n[{symbol}] Processing DTFX zones...")
        try:
            times, lows, highs, df = load_instrument(symbol)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
            continue

        packed = _prep_zones(symbol, min_retracement)
        zone_top = packed["top"]
        zone_bottom = packed["bottom"]
        zone_mid = packed["mid"]
        zone_is_buy = packed["is_buy"]
        zone_start_idx = packed["start_idx"]

        # Filter min risk pct
        zone_widths = zone_top - zone_bottom
        risks = n_multiplier * zone_widths / 2.0
        risk_pcts = risks / zone_mid
        valid_mask = risk_pcts >= min_risk_pct

        z_top = np.ascontiguousarray(zone_top[valid_mask], dtype=np.float64)
        z_bottom = np.ascontiguousarray(zone_bottom[valid_mask], dtype=np.float64)
        z_mid = np.ascontiguousarray(zone_mid[valid_mask], dtype=np.float64)
        z_buy = np.ascontiguousarray(zone_is_buy[valid_mask], dtype=np.bool_)
        z_start = np.ascontiguousarray(zone_start_idx[valid_mask], dtype=np.int64)

        print(f"[{symbol}] Simulating Touches 1 to {MAX_TOUCHES} on {len(z_top):,} valid zones...")
        e_idx, x_idx, status = sim_deep_multitouch_numba(
            z_top, z_bottom, z_mid, z_buy, z_start,
            lows, highs, n_multiplier, min_fill_bars, MAX_BARS_TO_RESOLVE, max_touches=MAX_TOUCHES
        )

        somega_hours = SESSION_HOUR_FILTER.get(symbol)

        # Build trade list
        for i in range(len(z_top)):
            for t in range(MAX_TOUCHES):
                if e_idx[i, t] != -1 and status[i, t] in (1, 2):
                    e_bar = e_idx[i, t]
                    x_bar = x_idx[i, t]
                    e_time = pd.Timestamp(times[e_bar])
                    x_time = pd.Timestamp(times[x_bar])
                    stat = status[i, t]
                    pnl = 1.0 if stat == 2 else -1.0
                    in_som = (somega_hours is None) or (e_time.hour in somega_hours)
                    age_h = (e_bar - z_start[i]) / 60.0

                    all_trades.append({
                        "symbol": symbol,
                        "zone_id": i,
                        "touch_num": t + 1,
                        "entry_time": e_time,
                        "exit_time": x_time,
                        "is_win": 1 if stat == 2 else 0,
                        "pnl_r": pnl,
                        "in_somega": 1 if in_som else 0,
                        "age_hours": age_h,
                    })

    full_df = pd.DataFrame(all_trades)
    print(f"\nGenerated {len(full_df):,} total multi-touch trades across all assets.")
    return full_df


def main():
    full_df = run_all_instruments()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. PERFORMANCE BY TOUCH NUMBER (1 TO 10+)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  1. PERFORMANCE BY TOUCH NUMBER (1 TO 10+)")
    print("=" * 80)

    touch_stats = []
    for t in range(1, MAX_TOUCHES + 1):
        sub = full_df[full_df["touch_num"] == t]
        if len(sub) == 0:
            continue
        wins = sub[sub["is_win"] == 1]
        losses = sub[sub["is_win"] == 0]
        wr = len(wins) / len(sub) * 100
        gross_win = wins["pnl_r"].sum()
        gross_loss = abs(losses["pnl_r"].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        ev = sub["pnl_r"].mean()

        touch_stats.append({
            "Touch": f"Touch #{t}",
            "Trades": len(sub),
            "Wins": len(wins),
            "Losses": len(losses),
            "Win Rate %": round(wr, 1),
            "Profit Factor": round(pf, 3),
            "Exp (R/trade)": round(ev, 3),
            "Total PnL (R)": round(sub["pnl_r"].sum(), 1),
        })

    touch_table = pd.DataFrame(touch_stats)
    print(touch_table.to_string(index=False))
    touch_table.to_csv(os.path.join(OUTPUT_DIR, "deep_multitouch_by_touch_number.csv"), index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. SPECIFICS: ZONE AGE DEGRADATION (<12h to >1 month)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  2. ZONE AGE DEGRADATION (WHEN DOES MULTI-TOUCH EXPIRE?)")
    print("=" * 80)

    age_bins = [
        (3.37, 12, "Fresh: 3.37h - 12h"),
        (12, 24, "Intraday: 12h - 24h"),
        (24, 48, "Next-Day: 24h - 48h"),
        (48, 168, "Swing: 2d - 7d (1 wk)"),
        (168, 720, "Multi-Week: 1 wk - 1 mo"),
        (720, 999999, "Old: > 1 month"),
    ]

    age_stats = []
    # Examine Touches 2+
    sub_subsequent = full_df[full_df["touch_num"] >= 2]
    for low_a, high_a, label in age_bins:
        sub = sub_subsequent[(sub_subsequent["age_hours"] >= low_a) & (sub_subsequent["age_hours"] < high_a)]
        if len(sub) == 0:
            continue
        wins = sub[sub["is_win"] == 1]
        losses = sub[sub["is_win"] == 0]
        wr = len(wins) / len(sub) * 100
        pf = wins["pnl_r"].sum() / abs(losses["pnl_r"].sum()) if len(losses) > 0 else float("inf")
        ev = sub["pnl_r"].mean()
        age_stats.append({
            "Zone Age Bracket": label,
            "Trades": len(sub),
            "Win Rate %": round(wr, 1),
            "Profit Factor": round(pf, 3),
            "Exp (R/trade)": round(ev, 3),
            "Total PnL (R)": round(sub["pnl_r"].sum(), 1),
        })

    age_table = pd.DataFrame(age_stats)
    print(age_table.to_string(index=False))
    age_table.to_csv(os.path.join(OUTPUT_DIR, "deep_multitouch_age_conditioning.csv"), index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. SPECIFICS: SOMEGA HOURS VS NON-SOMEGA HOURS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  3. SOMEGA SESSION FILTER ON SUBSEQUENT TOUCHES")
    print("=" * 80)

    somega_stats = []
    for t in [1, 2, 3, 4, 5]:
        for som_val, som_label in [(1, "Inside SOMEGA Hours"), (0, "Outside SOMEGA Hours")]:
            sub = full_df[(full_df["touch_num"] == t) & (full_df["in_somega"] == som_val)]
            if len(sub) == 0:
                continue
            wins = sub[sub["is_win"] == 1]
            losses = sub[sub["is_win"] == 0]
            wr = len(wins) / len(sub) * 100
            pf = wins["pnl_r"].sum() / abs(losses["pnl_r"].sum()) if len(losses) > 0 else float("inf")
            ev = sub["pnl_r"].mean()
            somega_stats.append({
                "Touch": f"Touch #{t}",
                "Session Filter": som_label,
                "Trades": len(sub),
                "Win Rate %": round(wr, 1),
                "Profit Factor": round(pf, 3),
                "Exp (R/trade)": round(ev, 3),
                "Total PnL (R)": round(sub["pnl_r"].sum(), 1),
            })

    somega_table = pd.DataFrame(somega_stats)
    print(somega_table.to_string(index=False))
    somega_table.to_csv(os.path.join(OUTPUT_DIR, "deep_multitouch_somega_conditioning.csv"), index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. CROSS-ASSET BREAKDOWN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  4. CROSS-ASSET MULTI-TOUCH BREAKDOWN")
    print("=" * 80)

    asset_stats = []
    for sym in INSTRUMENTS.keys():
        sub_sym = full_df[full_df["symbol"] == sym]
        if len(sub_sym) == 0:
            continue
        t1 = sub_sym[sub_sym["touch_num"] == 1]
        t24 = sub_sym[(sub_sym["touch_num"] >= 2) & (sub_sym["touch_num"] <= 4)]
        t5p = sub_sym[sub_sym["touch_num"] >= 5]

        wr1 = len(t1[t1["is_win"] == 1]) / len(t1) * 100 if len(t1) else 0
        pf1 = t1[t1["is_win"] == 1]["pnl_r"].sum() / abs(t1[t1["is_win"] == 0]["pnl_r"].sum()) if len(t1[t1["is_win"] == 0]) else 0

        wr24 = len(t24[t24["is_win"] == 1]) / len(t24) * 100 if len(t24) else 0
        pf24 = t24[t24["is_win"] == 1]["pnl_r"].sum() / abs(t24[t24["is_win"] == 0]["pnl_r"].sum()) if len(t24[t24["is_win"] == 0]) else 0

        wr5p = len(t5p[t5p["is_win"] == 1]) / len(t5p) * 100 if len(t5p) else 0
        pf5p = t5p[t5p["is_win"] == 1]["pnl_r"].sum() / abs(t5p[t5p["is_win"] == 0]["pnl_r"].sum()) if len(t5p[t5p["is_win"] == 0]) else 0

        asset_stats.append({
            "Asset": sym,
            "T1 Trades": len(t1),
            "T1 Win Rate": f"{wr1:.1f}%",
            "T1 PF": f"{pf1:.2f}",
            "T2-4 Trades": len(t24),
            "T2-4 Win Rate": f"{wr24:.1f}%",
            "T2-4 PF": f"{pf24:.2f}",
            "T5+ Trades": len(t5p),
            "T5+ Win Rate": f"{wr5p:.1f}%",
            "T5+ PF": f"{pf5p:.2f}",
        })

    asset_table = pd.DataFrame(asset_stats)
    print(asset_table.to_string(index=False))
    asset_table.to_csv(os.path.join(OUTPUT_DIR, "deep_multitouch_cross_asset.csv"), index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. PORTFOLIO SIMULATION: TOUCH CAP COMPARISON
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  5. PORTFOLIO PERFORMANCE BY TOUCH CAP (MAX 1, 2, 3, 4, 5, UNLIMITED)")
    print("=" * 80)

    # Sort all trades chronologically
    sorted_df = full_df[full_df["in_somega"] == 1].sort_values("entry_time").reset_index(drop=True)

    cap_results = []
    for cap in [1, 2, 3, 4, 5, 8, 10]:
        sub = sorted_df[sorted_df["touch_num"] <= cap]
        if len(sub) == 0:
            continue
        pnl_series = sub["pnl_r"].values
        wins = sub[sub["is_win"] == 1]
        losses = sub[sub["is_win"] == 0]
        wr = len(wins) / len(sub) * 100
        pf = wins["pnl_r"].sum() / abs(losses["pnl_r"].sum()) if len(losses) else float("inf")
        tot_r = pnl_series.sum()

        # Equity simulation with 5% risk
        equity = 100.0
        peak = 100.0
        max_dd = 0.0
        for r in pnl_series:
            # 5% risk compound
            equity *= (1.0 + 0.05 * r)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        trade_returns = 0.05 * pnl_series
        sharpe = (trade_returns.mean() / trade_returns.std() * np.sqrt(252 * 5)) if trade_returns.std() > 0 else 0

        cap_results.append({
            "Max Touch Cap": f"Cap <= {cap}" if cap < 10 else "Unlimited (All Touches)",
            "Total Trades": len(sub),
            "Win Rate %": round(wr, 1),
            "Profit Factor": round(pf, 3),
            "Total R": round(tot_r, 1),
            "Final $100 Equity": f"${equity:,.2f}",
            "Max Drawdown %": round(max_dd * 100, 1),
            "Sharpe Ratio": round(sharpe, 2),
        })

    cap_table = pd.DataFrame(cap_results)
    print(cap_table.to_string(index=False))
    cap_table.to_csv(os.path.join(OUTPUT_DIR, "deep_multitouch_portfolio_cap_comparison.csv"), index=False)
    print("\n" + "=" * 80)
    print("  DEEP MULTI-TOUCH STUDY COMPLETE. ALL CSVS SAVED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
