"""Implementation & Validation of Clean Fractal Swing Zones + Strict Invalidation Engine.

Fixes:
1. Strict Pre-Overshoot Invalidation:
   - BUY Zone: Killed if price dips <= bottom before overshooting > top.
   - SELL Zone: Killed if price spikes >= top before overshooting < bottom.
2. Genuine Fractal Swing Pivots:
   - Swings identified using genuine N-bar local extrema (k=2: 5-bar fractal; k=3: 7-bar fractal).
3. Evaluates full 23.7-year backtest and generates updated interactive chart.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
ARTIFACTS_DIR = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05"

import goal79_two_account_portfolio as g79
import audit_and_retest_full_spread_and_fixed_mc as audited

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440


@njit(cache=True)
def find_fractal_swings(high, low, k=2):
    """Finds genuine local swing highs and lows using a k-bar fractal filter
    (a high must be >= k bars on left and k bars on right)."""
    n = len(high)
    out_confirm = np.empty(n, dtype=np.int64)
    out_pivot = np.empty(n, dtype=np.int64)
    out_is_high = np.empty(n, dtype=np.bool_)
    out_price = np.empty(n, dtype=np.float64)
    count = 0

    last_type = 0  # 1: high, -1: low
    last_p = 0.0

    for i in range(k, n - k):
        # Check Swing High
        is_swing_high = True
        for j in range(1, k + 1):
            if high[i] < high[i - j] or high[i] < high[i + j]:
                is_swing_high = False
                break

        # Check Swing Low
        is_swing_low = True
        for j in range(1, k + 1):
            if low[i] > low[i - j] or low[i] > low[i + j]:
                is_swing_low = False
                break

        conf_bar = i + k

        if is_swing_high and not is_swing_low:
            if last_type != 1 or high[i] > last_p:
                out_confirm[count] = conf_bar
                out_pivot[count] = i
                out_is_high[count] = True
                out_price[count] = high[i]
                count += 1
                last_type = 1
                last_p = high[i]
        elif is_swing_low and not is_swing_high:
            if last_type != -1 or low[i] < last_p:
                out_confirm[count] = conf_bar
                out_pivot[count] = i
                out_is_high[count] = False
                out_price[count] = low[i]
                count += 1
                last_type = -1
                last_p = low[i]

    return out_confirm[:count], out_pivot[:count], out_is_high[:count], out_price[:count]


@njit(cache=True)
def build_clean_zones(confirm_idx, pivot_idx, is_high, price):
    """Builds clean swing zones between consecutive opposite confirmed pivots."""
    m = len(confirm_idx)
    z_created = np.empty(m, dtype=np.int64)
    z_top = np.empty(m, dtype=np.float64)
    z_bottom = np.empty(m, dtype=np.float64)
    z_top_idx = np.empty(m, dtype=np.int64)
    z_bottom_idx = np.empty(m, dtype=np.int64)
    z_is_buy = np.empty(m, dtype=np.bool_)
    zc = 0

    has_high = False
    has_low = False
    last_high_p = 0.0
    last_low_p = 0.0
    last_high_i = -1
    last_low_i = -1

    for k in range(m):
        ci = confirm_idx[k]
        pi = pivot_idx[k]
        p = price[k]

        if is_high[k]:
            last_high_p = p
            last_high_i = pi
            has_high = True
            if has_low and last_high_p > last_low_p:
                # Up-swing formed -> creates a BUY zone
                z_created[zc] = ci
                z_top[zc] = last_high_p
                z_bottom[zc] = last_low_p
                z_top_idx[zc] = last_high_i
                z_bottom_idx[zc] = last_low_i
                z_is_buy[zc] = True
                zc += 1
        else:
            last_low_p = p
            last_low_i = pi
            has_low = True
            if has_high and last_high_p > last_low_p:
                # Down-swing formed -> creates a SELL zone
                z_created[zc] = ci
                z_top[zc] = last_high_p
                z_bottom[zc] = last_low_p
                z_top_idx[zc] = last_high_i
                z_bottom_idx[zc] = last_low_i
                z_is_buy[zc] = False
                zc += 1

    return z_created[:zc], z_top[:zc], z_bottom[:zc], z_top_idx[:zc], z_bottom_idx[:zc], z_is_buy[:zc]


@njit(cache=True)
def resolve_clean_trades(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height,
    sl_mult, tp_mult
):
    n_zones = len(z_created)
    n_bars = len(high)

    entry_idx = np.full(n_zones, -1, dtype=np.int64)
    exit_idx = np.full(n_zones, -1, dtype=np.int64)
    outcome = np.zeros(n_zones, dtype=np.int8)  # 0: unres, 1: win, 2: loss
    net_r = np.zeros(n_zones, dtype=np.float64)
    entry_price_out = np.zeros(n_zones, dtype=np.float64)
    stop_risk_out = np.zeros(n_zones, dtype=np.float64)

    half_spread = spread_price / 2.0

    for zi in range(n_zones):
        top = z_top[zi]
        bottom = z_bottom[zi]
        height = top - bottom
        if height < min_zone_height:
            continue

        h = height / 2.0
        mid = (top + bottom) / 2.0
        c_bar = z_created[zi]
        is_buy = z_is_buy[zi]

        # 1. Strict Pre-Overshoot Search: KILL zone if price breaches opposite edge!
        overshot = False
        overshoot_bar = -1
        w_end = min(c_bar + max_window, n_bars)

        for bi in range(c_bar, w_end):
            if is_buy:
                if low[bi] <= bottom:
                    # Broken support -> Kill zone
                    break
                if high[bi] > top:
                    overshot = True
                    overshoot_bar = bi
                    break
            else:
                if high[bi] >= top:
                    # Broken resistance -> Kill zone
                    break
                if low[bi] < bottom:
                    overshot = True
                    overshoot_bar = bi
                    break

        if not overshot:
            continue

        # 2. Strict Midpoint Retrace Search: KILL zone if price breaches opposite edge before touching mid!
        touched = False
        touch_bar = -1
        w_end2 = min(overshoot_bar + max_window, n_bars)

        for bi in range(overshoot_bar, w_end2):
            if is_buy:
                if low[bi] <= bottom:
                    # Sliced through before mid touch -> Kill zone
                    break
                if low[bi] <= mid:
                    touched = True
                    touch_bar = bi
                    break
            else:
                if high[bi] >= top:
                    # Sliced through before mid touch -> Kill zone
                    break
                if high[bi] >= mid:
                    touched = True
                    touch_bar = bi
                    break

        if not touched:
            continue

        # 3. Setup Price Levels with FULL SYMMETRIC SPREAD
        if is_buy:
            entry_p = mid + half_spread
            raw_stop = mid - sl_mult * h
            stop_p = raw_stop - half_spread
            raw_target = mid + tp_mult * h
            target_p = raw_target - half_spread
            risk = entry_p - stop_p
            if risk < min_stop_price:
                continue
            win_r_val = (target_p - entry_p) / risk if risk > 0 else 0.0
        else:
            entry_p = mid - half_spread
            raw_stop = mid + sl_mult * h
            stop_p = raw_stop + half_spread
            raw_target = mid - tp_mult * h
            target_p = raw_target + half_spread
            risk = stop_p - entry_p
            if risk < min_stop_price:
                continue
            win_r_val = (entry_p - target_p) / risk if risk > 0 else 0.0

        # 4. Resolve Exit
        w_end3 = min(touch_bar + max_window, n_bars)
        for bi in range(touch_bar, w_end3):
            if is_buy:
                if low[bi] <= stop_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
                elif high[bi] >= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
            else:
                if high[bi] >= stop_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
                elif low[bi] <= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break

    return entry_idx, exit_idx, outcome, net_r, entry_price_out, stop_risk_out


def run_clean_backtest():
    print("==========================================================================")
    print("  EVALUATING FRACTAL SWING ZONES + STRICT PRE-OVERSHOOT INVALIDATION")
    print("==========================================================================")
    cache = audited.load_cache()
    all_times = [d["times"].min() for d in cache.values()] + [d["times"].max() for d in cache.values()]
    total_days = (max(all_times) - min(all_times)).total_seconds() / 86400.0
    trading_days = total_days * (5.0 / 7.0)

    min_pips_dict = {"USDJPY": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}

    # Test different fractal sizes k=1, k=2, k=3
    for k in [1, 2, 3]:
        frames = []
        for inst, d in cache.items():
            pip = d["pip"]
            spread_p = 1.6 * pip
            min_stop_p = 3.0 * pip
            min_zone_h = min_pips_dict[inst] * pip

            # 1. Detect clean fractal swings
            conf_i, piv_i, is_h, p_val = find_fractal_swings(d["high"], d["low"], k=k)
            z_c, z_t, z_b, z_ti, z_bi, z_buy = build_clean_zones(conf_i, piv_i, is_h, p_val)

            # 2. Resolve with strict invalidation
            ei, xi, outcome, net_r, entry_p, stop_r = resolve_clean_trades(
                z_c, z_t, z_b, z_buy, d["high"], d["low"],
                spread_p, min_stop_p, MAX_WINDOW, min_zone_h, 1.0, 0.8
            )

            resolved = outcome > 0
            frames.append(pd.DataFrame({
                "instrument": inst, "symbol": inst,
                "entry_time": d["times"][ei[resolved]],
                "exit_time": d["times"][xi[resolved]],
                "entry_price": entry_p[resolved],
                "stop_risk": stop_r[resolved],
                "net_r": net_r[resolved],
                "is_buy": z_buy[resolved]
            }))

        df_all = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        res, eq_b, eq_s, max_dd, _ = g79.simulate_two_account(
            df_all, risk_pct=0.03, start_per_side=100.0, max_leverage=50.0, cap=10_000_000.0
        )
        acc = res[res["accepted"]]
        wr = 100.0 * (acc["net_r"] > 0).mean()
        gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
        gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
        pf = gw / gl if gl > 0 else 0.0
        med_days, med_dd = audited.run_mc_fixed(df_all, n_sims=300)

        desc = f"Fractal k={k} ({2*k+1}-bar swing window)"
        print(f"\n>>> {desc:<30} | Execs: {len(acc):<6} | T/Day: {len(acc)/trading_days:>4.1f} | WR: {wr:>5.1f}% | PF: {pf:>4.2f} | MC $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")

    # Generate Updated Interactive Trades Inspector with Clean Swings & Strict Invalidation
    print("\nGenerating updated Clean Trades Inspector HTML...")
    # Using k=2 as the gold-standard 5-bar fractal
    k = 2
    detailed_trades = []
    trade_candles_cache = {}

    for inst, d in cache.items():
        pip = d["pip"]
        spread_p = 1.6 * pip
        min_stop_p = 3.0 * pip
        min_zone_h = min_pips_dict[inst] * pip
        half_spread = spread_p / 2.0

        conf_i, piv_i, is_h, p_val = find_fractal_swings(d["high"], d["low"], k=k)
        z_c, z_t, z_b, z_ti, z_bi, z_buy = build_clean_zones(conf_i, piv_i, is_h, p_val)

        ei, xi, outcome, net_r, entry_p, stop_r = resolve_clean_trades(
            z_c, z_t, z_b, z_buy, d["high"], d["low"],
            spread_p, min_stop_p, MAX_WINDOW, min_zone_h, 1.0, 0.8
        )

        resolved = outcome > 0
        zc_res = z_c[resolved]
        zt_res = z_t[resolved]
        zb_res = z_b[resolved]
        zbuy_res = z_buy[resolved]
        ei_res = ei[resolved]
        xi_res = xi[resolved]
        out_res = outcome[resolved]
        netr_res = net_r[resolved]

        for i in range(len(ei_res)):
            is_buy = bool(zbuy_res[i])
            top = float(zt_res[i])
            bottom = float(zb_res[i])
            mid = (top + bottom) / 2.0
            h = (top - bottom) / 2.0

            raw_entry = mid
            raw_tp = mid + 0.8 * h if is_buy else mid - 0.8 * h
            raw_sl = mid - 1.0 * h if is_buy else mid + 1.0 * h

            act_entry = mid + half_spread if is_buy else mid - half_spread
            act_tp = raw_tp - half_spread if is_buy else raw_tp + half_spread
            act_sl = raw_sl - half_spread if is_buy else raw_sl + half_spread
            act_exit = act_tp if out_res[i] == 1 else act_sl

            detailed_trades.append({
                "instrument": inst, "symbol": inst,
                "pip": pip, "is_buy": is_buy,
                "side": "BUY" if is_buy else "SELL",
                "outcome": "WIN" if out_res[i] == 1 else "LOSS",
                "net_r": float(netr_res[i]),
                "c_bar": int(zc_res[i]),
                "entry_bar": int(ei_res[i]),
                "exit_bar": int(xi_res[i]),
                "entry_time": d["times"][ei_res[i]],
                "exit_time": d["times"][xi_res[i]],
                "created_time": d["times"][zc_res[i]],
                "top": top, "bottom": bottom, "mid": mid,
                "zone_pips": round((top - bottom) / pip, 2),
                "raw_entry": raw_entry, "raw_tp": raw_tp, "raw_sl": raw_sl,
                "act_entry": act_entry, "act_tp": act_tp, "act_sl": act_sl,
                "act_exit": act_exit, "bars_held": int(xi_res[i] - ei_res[i]),
            })

    df_dt = pd.DataFrame(detailed_trades).sort_values("entry_time").reset_index(drop=True)
    last_10 = df_dt.tail(10).to_dict(orient="records")

    chart_trades_payload = []
    for t_idx, tr in enumerate(last_10):
        inst = tr["instrument"]
        d = cache[inst]
        times = d["times"]
        high = d["high"]
        low = d["low"]
        close = d["close"]

        c_bar = tr["c_bar"]
        ei_bar = tr["entry_bar"]
        xi_bar = tr["exit_bar"]

        w_start = max(0, c_bar - 25)
        w_end = min(len(times), xi_bar + 25)

        candles = []
        for bi in range(w_start, w_end):
            candles.append({
                "time": str(times[bi]),
                "open": float(close[bi - 1]) if bi > 0 else float(close[bi]),
                "high": float(high[bi]), "low": float(low[bi]), "close": float(close[bi]),
            })

        chart_trades_payload.append({
            "trade_id": t_idx + 1,
            "instrument": tr["instrument"],
            "side": tr["side"],
            "outcome": tr["outcome"],
            "net_r": round(tr["net_r"], 2),
            "bars_held": tr["bars_held"],
            "zone_pips": tr["zone_pips"],
            "created_time": str(tr["created_time"]),
            "entry_time": str(tr["entry_time"]),
            "exit_time": str(tr["exit_time"]),
            "top": round(tr["top"], 5),
            "bottom": round(tr["bottom"], 5),
            "mid": round(tr["mid"], 5),
            "raw_entry": round(tr["raw_entry"], 5),
            "raw_tp": round(tr["raw_tp"], 5),
            "raw_sl": round(tr["raw_sl"], 5),
            "act_entry": round(tr["act_entry"], 5),
            "act_tp": round(tr["act_tp"], 5),
            "act_sl": round(tr["act_sl"], 5),
            "act_exit": round(tr["act_exit"], 5),
            "candles": candles,
        })

    # Render HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clean Fractal Swing Zones + Strict Invalidation Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg-main: #0B0E14;
    --bg-card: #151A23;
    --bg-card-hover: #1C2331;
    --border-color: #262E3D;
    --text-primary: #F0F3F8;
    --text-secondary: #8B98A5;
    --accent-blue: #388BFD;
    --accent-green: #238636;
    --accent-red: #DA3633;
    --accent-gold: #F1E05A;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
  
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }}
  .title {{ font-size: 22px; font-weight: 700; color: #fff; }}
  .badge {{ padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase; background: #1f293d; color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
  
  .trade-selector {{ display: flex; gap: 8px; overflow-x: auto; padding-bottom: 12px; margin-bottom: 20px; border-bottom: 1px solid var(--border-color); }}
  .trade-btn {{ background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 10px 16px; border-radius: 8px; cursor: pointer; white-space: nowrap; font-size: 13px; font-weight: 600; transition: all 0.2s; }}
  .trade-btn:hover {{ background: var(--bg-card-hover); color: #fff; }}
  .trade-btn.active {{ background: #1C2B44; color: var(--accent-blue); border-color: var(--accent-blue); box-shadow: 0 0 10px rgba(56, 139, 253, 0.3); }}
  .trade-btn.win.active {{ color: #4EBE71; border-color: #238636; background: #12281D; box-shadow: 0 0 10px rgba(35, 134, 54, 0.3); }}
  .trade-btn.loss.active {{ color: #F87171; border-color: #DA3633; background: #2A1719; box-shadow: 0 0 10px rgba(218, 54, 51, 0.3); }}
  
  .chart-container {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 25px; }}
  .trade-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }}
  .trade-title {{ font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 10px; }}
  .tag-win {{ background: rgba(35, 134, 54, 0.2); color: #4EBE71; border: 1px solid #238636; padding: 3px 8px; border-radius: 4px; font-size: 12px; }}
  .tag-loss {{ background: rgba(218, 54, 51, 0.2); color: #F87171; border: 1px solid #DA3633; padding: 3px 8px; border-radius: 4px; font-size: 12px; }}
  .tag-side-buy {{ background: rgba(56, 139, 253, 0.2); color: #58A6FF; border: 1px solid #388BFD; padding: 3px 8px; border-radius: 4px; font-size: 12px; }}
  .tag-side-sell {{ background: rgba(218, 54, 51, 0.2); color: #F87171; border: 1px solid #DA3633; padding: 3px 8px; border-radius: 4px; font-size: 12px; }}
  
  .legend-box {{ display: flex; flex-wrap: wrap; gap: 15px; background: #0E121A; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 16px; margin-top: 15px; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  .line-sample {{ width: 24px; height: 3px; border-radius: 2px; }}
  .line-sample.dotted {{ border-top: 2px dotted; height: 0; background: transparent; }}
  
  #chartDiv {{ width: 100%; height: 580px; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="title">🛡️ Clean Fractal Swings + Strict Pre-Overshoot Invalidation</div>
    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
      All zones anchored to <b>true multi-bar fractal swing pivots</b> | Broken levels automatically discarded before entry
    </div>
  </div>
  <div class="badge">CLEAN SWING ENGINE</div>
</div>

<div class="trade-selector" id="tradeSelector"></div>

<div class="chart-container">
  <div class="trade-header">
    <div class="trade-title" id="tradeTitle"></div>
    <div id="tradeSummaryText" style="font-size: 13px; color: var(--text-secondary);"></div>
  </div>
  
  <div id="chartDiv"></div>

  <div class="legend-box">
    <div class="legend-item"><div class="line-sample" style="background: rgba(56, 139, 253, 0.4); height: 12px; width: 18px; border: 1px solid #388BFD;"></div> <b>Clean Fractal Zone (Swing Top/Bottom)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #388BFD;"></div> <b>Theoretical Mid Entry</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #58A6FF;"></div> <b>Actual Entry Given Spread</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #238636;"></div> <b>Theoretical TP</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #4EBE71;"></div> <b>Actual TP Exit Given Spread</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #DA3633;"></div> <b>Theoretical SL</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #F87171;"></div> <b>Actual SL Exit Given Spread</b></div>
  </div>
</div>

<script>
const tradesData = {json.dumps(chart_trades_payload)};
let currentTradeIndex = tradesData.length - 1;

function renderTradeSelector() {{
  const container = document.getElementById("tradeSelector");
  container.innerHTML = "";
  tradesData.forEach((tr, idx) => {{
    const btn = document.createElement("button");
    const isWin = tr.outcome === "WIN";
    btn.className = `trade-btn ${{isWin ? "win" : "loss"}} ${{idx === currentTradeIndex ? "active" : ""}}`;
    btn.innerHTML = `Trade #${{tr.trade_id}}: ${{tr.instrument}} ${{tr.side}} (${{tr.outcome}} ${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}}R)`;
    btn.onclick = () => {{
      currentTradeIndex = idx;
      renderTradeSelector();
      renderChart(idx);
    }};
    container.appendChild(btn);
  }});
}}

function renderChart(idx) {{
  const tr = tradesData[idx];
  const isWin = tr.outcome === "WIN";
  const isBuy = tr.side === "BUY";
  
  const titleEl = document.getElementById("tradeTitle");
  titleEl.innerHTML = `
    <span>Trade #${{tr.trade_id}}: ${{tr.instrument}}</span>
    <span class="${{isBuy ? "tag-side-buy" : "tag-side-sell"}}">${{tr.side}}</span>
    <span class="${{isWin ? "tag-win" : "tag-loss"}}">${{tr.outcome}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}} R)</span>
    <span style="font-size: 13px; color: var(--text-secondary);">Zone: <b>${{tr.zone_pips}} pips</b> | Held: <b>${{tr.bars_held}} min</b></span>
  `;

  document.getElementById("tradeSummaryText").innerHTML = `
    Formed: <b>${{tr.created_time}}</b> | Entered: <b>${{tr.entry_time}}</b> | Exited: <b>${{tr.exit_time}}</b>
  `;

  const times = tr.candles.map(c => c.time);
  const opens = tr.candles.map(c => c.open);
  const highs = tr.candles.map(c => c.high);
  const lows = tr.candles.map(c => c.low);
  const closes = tr.candles.map(c => c.close);

  const candlestickTrace = {{
    x: times,
    open: opens,
    high: highs,
    low: lows,
    close: closes,
    type: 'candlestick',
    name: tr.instrument,
    increasing: {{ line: {{ color: '#26A69A' }} }},
    decreasing: {{ line: {{ color: '#EF5350' }} }},
    showlegend: false
  }};

  const zoneBoxX = [tr.created_time, tr.exit_time, tr.exit_time, tr.created_time, tr.created_time];
  const zoneBoxY = [tr.top, tr.top, tr.bottom, tr.bottom, tr.top];
  const zoneTrace = {{
    x: zoneBoxX,
    y: zoneBoxY,
    fill: 'toself',
    fillcolor: isBuy ? 'rgba(56, 139, 253, 0.15)' : 'rgba(218, 54, 51, 0.15)',
    line: {{ color: isBuy ? '#388BFD' : '#DA3633', width: 1.5 }},
    name: 'Fractal Zone Range',
    mode: 'lines'
  }};

  const lineTimes = [tr.created_time, tr.exit_time];

  const midTrace = {{
    x: lineTimes,
    y: [tr.mid, tr.mid],
    mode: 'lines',
    line: {{ color: '#388BFD', width: 2 }},
    name: `Theoretical Mid (${{tr.mid}})`
  }};

  const actEntryTrace = {{
    x: [tr.entry_time, tr.exit_time],
    y: [tr.act_entry, tr.act_entry],
    mode: 'lines',
    line: {{ color: '#58A6FF', width: 2, dash: 'dot' }},
    name: `Actual Entry (Spread: ${{tr.act_entry}})`
  }};

  const rawTpTrace = {{
    x: [tr.entry_time, tr.exit_time],
    y: [tr.raw_tp, tr.raw_tp],
    mode: 'lines',
    line: {{ color: '#238636', width: 1.5 }},
    name: `Theoretical TP (${{tr.raw_tp}})`
  }};

  const actTpTrace = {{
    x: [tr.entry_time, tr.exit_time],
    y: [tr.act_tp, tr.act_tp],
    mode: 'lines',
    line: {{ color: '#4EBE71', width: 2, dash: 'dot' }},
    name: `Actual TP (Spread: ${{tr.act_tp}})`
  }};

  const rawSlTrace = {{
    x: [tr.entry_time, tr.exit_time],
    y: [tr.raw_sl, tr.raw_sl],
    mode: 'lines',
    line: {{ color: '#DA3633', width: 1.5 }},
    name: `Theoretical SL (${{tr.raw_sl}})`
  }};

  const actSlTrace = {{
    x: [tr.entry_time, tr.exit_time],
    y: [tr.act_sl, tr.act_sl],
    mode: 'lines',
    line: {{ color: '#F87171', width: 2, dash: 'dot' }},
    name: `Actual SL (Spread: ${{tr.act_sl}})`
  }};

  const annotations = [
    {{
      x: tr.created_time,
      y: isBuy ? tr.top : tr.bottom,
      text: '🌱 Genuine Swing Zone',
      showarrow: true,
      arrowhead: 2,
      ax: 0, ay: isBuy ? -35 : 35,
      bgcolor: '#1C2B44',
      bordercolor: '#388BFD',
      font: {{ color: '#fff', size: 11 }}
    }},
    {{
      x: tr.entry_time,
      y: tr.act_entry,
      text: `🚀 ${{tr.side}} Entry @ ${{tr.act_entry}}`,
      showarrow: true,
      arrowhead: 2,
      ax: -30, ay: isBuy ? 35 : -35,
      bgcolor: isBuy ? '#12281D' : '#2A1719',
      bordercolor: isBuy ? '#238636' : '#DA3633',
      font: {{ color: '#fff', size: 11 }}
    }},
    {{
      x: tr.exit_time,
      y: tr.act_exit,
      text: `🏁 ${{tr.outcome}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}}R)`,
      showarrow: true,
      arrowhead: 2,
      ax: 30, ay: isWin ? (isBuy ? -35 : 35) : (isBuy ? 35 : -35),
      bgcolor: isWin ? '#12281D' : '#2A1719',
      bordercolor: isWin ? '#238636' : '#DA3633',
      font: {{ color: '#fff', size: 11, weight: 'bold' }}
    }}
  ];

  const layout = {{
    margin: {{ l: 50, r: 50, t: 30, b: 40 }},
    paper_bgcolor: '#151A23',
    plot_bgcolor: '#151A23',
    xaxis: {{
      type: 'date',
      rangeslider: {{ visible: false }},
      gridcolor: '#262E3D',
      color: '#8B98A5',
      tickformat: '%Y-%m-%d %H:%M'
    }},
    yaxis: {{
      gridcolor: '#262E3D',
      color: '#8B98A5',
      autorange: true
    }},
    annotations: annotations,
    showlegend: true,
    legend: {{
      orientation: 'h',
      x: 0, y: 1.12,
      font: {{ color: '#8B98A5', size: 10 }}
    }}
  }};

  Plotly.newPlot('chartDiv', [
    candlestickTrace,
    zoneTrace,
    midTrace,
    actEntryTrace,
    rawTpTrace,
    actTpTrace,
    rawSlTrace,
    actSlTrace
  ], layout, {{ responsive: true, displayModeBar: true }});
}}

renderTradeSelector();
renderChart(currentTradeIndex);
</script>
</body>
</html>
"""

    out_file = os.path.join(ARTIFACTS_DIR, "clean_fractal_swing_trades_inspector.html")
    with open(out_file, "w") as f:
        f.write(html_content)
    print(f"\nSaved Clean Fractal Swing Visualizer to: {out_file}")


if __name__ == "__main__":
    run_clean_backtest()
