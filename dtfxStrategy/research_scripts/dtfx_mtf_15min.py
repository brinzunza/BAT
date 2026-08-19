import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
from dtfx_2025_backtest import find_pivots, build_zigzag, run_structure_fsm
from dtfx_timeframe_backtest import resample_ohlc, simulate_zone_original, r_multiple as r_multiple_base, simulate_equity
from dtfx_mtf_backtest import build_htf_trend_timeline, trend_at

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
TICKERS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XAUT-USDT"]
ENTRY_RULE = "15min"
HTF_RULE = "4h"
PIVOT_WINDOW = 3


def summarize(trades, ticker, label, n_days):
    completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
    for tr in completed:
        tr["r"] = r_multiple_base(tr, "original")
        is_buy_zone = tr["side"] == "buy"
        risk_price = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
        tr["risk_pct_of_price"] = abs(risk_price) / tr["mid"]
    completed = [tr for tr in completed if tr["r"] is not None]
    completed.sort(key=lambda tr: tr["entry_time"])

    wins = [tr for tr in completed if tr["status"] == "take_profit"]
    losses = [tr for tr in completed if tr["status"] == "stopped_out"]
    win_rate = len(wins) / len(completed) if completed else np.nan
    avg_win_r = np.mean([tr["r"] for tr in wins]) if wins else np.nan
    avg_loss_r = np.mean([tr["r"] for tr in losses]) if losses else np.nan
    sum_r = sum(tr["r"] for tr in completed)
    final_gross = simulate_equity(completed, with_fees=False)
    return dict(
        ticker=ticker, label=label, trades=len(completed),
        wins=len(wins), losses=len(losses), win_rate=win_rate,
        avg_win_r=avg_win_r, avg_loss_r=avg_loss_r, sum_r=sum_r,
        trades_per_day=len(completed) / n_days, final_gross_10k=final_gross,
    )


results = []
for ticker in TICKERS:
    print(f"=== {ticker} ===", flush=True)
    df1m = pd.read_csv(f"{CACHE_DIR}/polygon_{ticker.replace('-','_')}_2025.csv", parse_dates=["date"]).set_index("date").sort_index()
    df1m = df1m[["open", "high", "low", "close"]]
    n_days = (df1m.index.max() - df1m.index.min()).days + 1

    df15 = resample_ohlc(df1m, ENTRY_RULE)
    df4h = resample_ohlc(df1m, HTF_RULE)
    htf_timeline = build_htf_trend_timeline(df4h, window=PIVOT_WINDOW)

    swing_highs, swing_lows = find_pivots(df15["high"], df15["low"], window=PIVOT_WINDOW)
    zigzag = build_zigzag(swing_highs, swing_lows)
    _, zones = run_structure_fsm(zigzag)

    times, lows, highs = df15.index.values, df15["low"].values, df15["high"].values
    all_trades = [simulate_zone_original(z, times, lows, highs) for z in zones]

    aligned_trades = []
    for z, tr in zip(zones, all_trades):
        htf_trend = trend_at(htf_timeline, z["created_at"])
        wanted_trend = "up" if z["side"] == "buy" else "down"
        if htf_trend == wanted_trend:
            aligned_trades.append(tr)

    print(f"  zones={len(zones)}  baseline_completed={len([t for t in all_trades if t['status'] in ('take_profit','stopped_out')])}  "
          f"aligned_completed={len([t for t in aligned_trades if t['status'] in ('take_profit','stopped_out')])}", flush=True)

    results.append(summarize(all_trades, ticker, "15min_baseline", n_days))
    results.append(summarize(aligned_trades, ticker, "15min_mtf_4h_aligned", n_days))

summary = pd.DataFrame(results)
summary.to_csv(f"{CACHE_DIR}/dtfx_mtf_15min_summary.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
pd.set_option("display.width", 200)
print("\n" + summary.to_string(index=False))
