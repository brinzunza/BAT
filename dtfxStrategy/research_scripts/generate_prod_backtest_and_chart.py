"""Backtest the Current PROD Strategy & Generate Interactive Last-10-Trades Chart.

PROD Configuration:
- Symbols: EURUSD, GBPUSD, EURGBP, USDJPY
- MIN_ZONE_PIPS: 10.0
- SL_MULT: 1.0
- TP_MULT: 0.8
- Concurrency: 1-per-side (Two-Account Portfolio)
- Spread: 1.6 pips (with Full Symmetric Round-Trip Spread)

Generates:
- Comprehensive backtest statistics (WR, PF, Trades/Day, Net R, MC Race to $1M).
- Detailed trade records + 1-min candlestick OHLC series for the last 10 trades.
- Standalone interactive HTML visualizer with zone box, mid entry, actual entry given spread, theoretical TP/SL, and actual exit given spread.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
ARTIFACTS_DIR = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05"

import goal95_every_swing_zones as g95
import goal79_two_account_portfolio as g79
import audit_and_retest_full_spread_and_fixed_mc as audited

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}

MIN_ZONE_PIPS = {"EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0, "USDJPY": 10.0}
SL_MULT = 1.0
TP_MULT = 0.8
SPREAD_PIPS = 1.6
RISK_PCT = 0.03


def main():
    print("Loading cache...")
    cache = audited.load_cache()

    print(f"Building trades for PROD (MIN_ZONE_PIPS=10.0, SL_MULT={SL_MULT}, TP_MULT={TP_MULT}, SPREAD={SPREAD_PIPS}p)...")
    
    trade_records = []

    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = 3.0 * pip
        min_zone_height = MIN_ZONE_PIPS[inst] * pip
        half_spread = spread_price / 2.0

        (entry_idx, exit_idx, outcome, net_r, entry_price_arr, stop_risk_arr
         ) = audited.resolve_trades_full_spread(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, 1440, min_zone_height, SL_MULT, TP_MULT
        )

        resolved = outcome > 0
        z_created = d["z_created"][resolved]
        z_top = d["z_top"][resolved]
        z_bottom = d["z_bottom"][resolved]
        z_is_buy = d["z_is_buy"][resolved]
        ei = entry_idx[resolved]
        xi = exit_idx[resolved]
        outcomes = outcome[resolved]
        net_rs = net_r[resolved]
        ep_res = entry_price_arr[resolved]
        sr_res = stop_risk_arr[resolved]

        for i in range(len(ei)):
            is_buy = bool(z_is_buy[i])
            top = float(z_top[i])
            bottom = float(z_bottom[i])
            mid = (top + bottom) / 2.0
            h = (top - bottom) / 2.0
            
            # Theoretical prices
            raw_entry = mid
            raw_tp = mid + TP_MULT * h if is_buy else mid - TP_MULT * h
            raw_sl = mid - SL_MULT * h if is_buy else mid + SL_MULT * h
            
            # Actual prices given spread
            act_entry = float(ep_res[i])
            act_tp = raw_tp - half_spread if is_buy else raw_tp + half_spread
            act_sl = raw_sl - half_spread if is_buy else raw_sl + half_spread
            act_exit = act_tp if outcomes[i] == 1 else act_sl

            trade_records.append({
                "instrument": inst,
                "symbol": inst,
                "pip": pip,
                "is_buy": is_buy,
                "side": "BUY" if is_buy else "SELL",
                "outcome": "WIN" if outcomes[i] == 1 else "LOSS",
                "net_r": float(net_rs[i]),
                "entry_price": float(ep_res[i]),
                "stop_risk": float(sr_res[i]),
                "c_bar": int(z_created[i]),
                "entry_bar": int(ei[i]),
                "exit_bar": int(xi[i]),
                "entry_time": d["times"][ei[i]],
                "exit_time": d["times"][xi[i]],
                "created_time": d["times"][z_created[i]],
                "top": top,
                "bottom": bottom,
                "mid": mid,
                "zone_pips": round((top - bottom) / pip, 2),
                "raw_entry": raw_entry,
                "raw_tp": raw_tp,
                "raw_sl": raw_sl,
                "act_entry": act_entry,
                "act_tp": act_tp,
                "act_sl": act_sl,
                "act_exit": act_exit,
                "bars_held": int(xi[i] - ei[i]),
            })

    df_trades = pd.DataFrame(trade_records).sort_values("entry_time").reset_index(drop=True)
    print(f"Total resolved trades: {len(df_trades):,}")

    # 1. Run portfolio backtest simulation
    res_std, eq_b, eq_s, max_dd, _ = g79.simulate_two_account(
        df_trades, risk_pct=RISK_PCT, start_per_side=100.0,
        max_leverage=50.0, cap=10_000_000.0, track_detail=False
    )
    acc = res_std[res_std["accepted"]]
    n_exec = len(acc)
    all_times = [d["times"].min() for d in cache.values()] + [d["times"].max() for d in cache.values()]
    total_days = (max(all_times) - min(all_times)).total_seconds() / 86400.0
    trading_days = total_days * (5.0 / 7.0)

    wr = 100.0 * (acc["net_r"] > 0).mean()
    avg_win = acc.loc[acc["net_r"] > 0, "net_r"].mean()
    gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
    gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else 0.0

    med_days, med_dd = audited.run_mc_fixed(df_trades, n_sims=300)

    print("\n" + "=" * 80)
    print("  CURRENT PROD LIVE STRATEGY BACKTEST RESULTS (23.7 YEARS)")
    print("=" * 80)
    print(f"  Total Executions:       {n_exec:,}")
    print(f"  Trades per Day:         {n_exec / trading_days:.1f}")
    print(f"  Win Rate:               {wr:.2f}%")
    print(f"  Avg Win Payout:         +{avg_win:.2f} R")
    print(f"  Profit Factor (1.6p):   {pf:.2f}")
    print(f"  Monte Carlo Days to $1M:{med_days:.1f} days")
    print(f"  Monte Carlo Max DD:     {med_dd:.1f}%")
    print("=" * 80)

    # 2. Extract Last 10 Trades & Candlestick Windows
    last_10_trades = df_trades.tail(10).to_dict(orient="records")

    trades_payload = []

    for t_idx, tr in enumerate(last_10_trades):
        inst = tr["instrument"]
        d = cache[inst]
        times = d["times"]
        high = d["high"]
        low = d["low"]
        close = d["close"]

        c_bar = tr["c_bar"]
        ei_bar = tr["entry_bar"]
        xi_bar = tr["exit_bar"]

        # Window: 30 bars before creation to 30 bars after exit
        w_start = max(0, c_bar - 30)
        w_end = min(len(times), xi_bar + 35)

        # Candlesticks
        candles = []
        for bi in range(w_start, w_end):
            candles.append({
                "time": str(times[bi]),
                "open": float(close[bi - 1]) if bi > 0 else float(close[bi]),
                "high": float(high[bi]),
                "low": float(low[bi]),
                "close": float(close[bi]),
                "is_created": (bi == c_bar),
                "is_entry": (bi == ei_bar),
                "is_exit": (bi == xi_bar),
            })

        trades_payload.append({
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
            "c_time_str": str(tr["created_time"]),
            "entry_time_str": str(tr["entry_time"]),
            "exit_time_str": str(tr["exit_time"]),
        })

    # 3. Build Interactive HTML Dashboard
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PROD Strategy Live Backtest & Last 10 Trades Visualizer</title>
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
    --accent-purple: #A371F7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
  
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }}
  .title {{ font-size: 22px; font-weight: 700; color: #fff; }}
  .badge {{ padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase; background: #1f293d; color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 15px; text-align: center; }}
  .metric-val {{ font-size: 24px; font-weight: 800; color: #fff; margin-top: 5px; }}
  .metric-lbl {{ font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }}
  
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
  .line-sample.dashed {{ border-top: 2px dashed; height: 0; background: transparent; }}
  .line-sample.dotted {{ border-top: 2px dotted; height: 0; background: transparent; }}
  
  #chartDiv {{ width: 100%; height: 580px; }}
  
  .details-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
  .details-table th, .details-table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border-color); }}
  .details-table th {{ color: var(--text-secondary); font-weight: 600; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="title">⚡ PROD Live Strategy Backtest & Last 10 Trades Inspector</div>
    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
      Config: <code>MIN_ZONE_PIPS = 10.0</code> | <code>SL_MULT = 1.0</code> | <code>TP_MULT = 0.8</code> | Full Symmetric Spread ($1.6\text{{p}}$)
    </div>
  </div>
  <div class="badge">PRODLIVETRADING ENGINE</div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-lbl">Win Rate (23.7 Yrs)</div>
    <div class="metric-val" style="color: #4EBE71;">{wr:.2f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-lbl">Profit Factor (1.6p)</div>
    <div class="metric-val" style="color: #58A6FF;">{pf:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-lbl">Avg Win Payout</div>
    <div class="metric-val" style="color: #F1E05A;">+{avg_win:.2f} R</div>
  </div>
  <div class="metric-card">
    <div class="metric-lbl">Trades / Day</div>
    <div class="metric-val">{n_exec / trading_days:.1f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-lbl">Days to $1M ($200 Start)</div>
    <div class="metric-val" style="color: #A371F7;">{med_days:.1f}d</div>
  </div>
  <div class="metric-card">
    <div class="metric-lbl">Monte Carlo Max DD</div>
    <div class="metric-val" style="color: #F87171;">{med_dd:.1f}%</div>
  </div>
</div>

<div class="trade-selector" id="tradeSelector"></div>

<div class="chart-container">
  <div class="trade-header">
    <div class="trade-title" id="tradeTitle"></div>
    <div id="tradeSummaryText" style="font-size: 13px; color: var(--text-secondary);"></div>
  </div>
  
  <div id="chartDiv"></div>

  <div class="legend-box">
    <div class="legend-item"><div class="line-sample" style="background: rgba(56, 139, 253, 0.4); height: 12px; width: 18px; border: 1px solid #388BFD;"></div> <b>Zone Box (Top/Bottom)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #388BFD;"></div> <b>Zone Midpoint (Theoretical Entry)</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #58A6FF;"></div> <b>Actual Entry Given Spread (Bid/Ask)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #238636;"></div> <b>Theoretical TP (+0.8x Half-Zone)</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #4EBE71;"></div> <b>Actual TP Exit Given Spread</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #DA3633;"></div> <b>Theoretical SL (-1.0x Half-Zone)</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #F87171;"></div> <b>Actual SL Exit Given Spread</b></div>
  </div>

  <table class="details-table" id="detailsTable">
    <thead>
      <tr>
        <th>Zone Formed</th>
        <th>Entry Time</th>
        <th>Exit Time</th>
        <th>Zone Height</th>
        <th>Theoretical Mid</th>
        <th>Actual Entry (Spread)</th>
        <th>Actual Exit (Spread)</th>
        <th>Bars Held</th>
        <th>Net Payout</th>
      </tr>
    </thead>
    <tbody>
      <tr id="detailsRow"></tr>
    </tbody>
  </table>
</div>

<script>
const tradesData = {json.dumps(trades_payload)};
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
  
  // Update header
  const titleEl = document.getElementById("tradeTitle");
  titleEl.innerHTML = `
    <span>Trade #${{tr.trade_id}}: ${{tr.instrument}}</span>
    <span class="${{isBuy ? "tag-side-buy" : "tag-side-sell"}}">${{tr.side}}</span>
    <span class="${{isWin ? "tag-win" : "tag-loss"}}">${{tr.outcome}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}} R)</span>
  `;

  document.getElementById("tradeSummaryText").innerHTML = `
    Formed: <b>${{tr.created_time}}</b> | Entered: <b>${{tr.entry_time}}</b> | Exited: <b>${{tr.exit_time}}</b>
  `;

  // Update table row
  document.getElementById("detailsRow").innerHTML = `
    <td>${{tr.created_time}}</td>
    <td>${{tr.entry_time}}</td>
    <td>${{tr.exit_time}}</td>
    <td><b>${{tr.zone_pips}} pips</b></td>
    <td>${{tr.mid}}</td>
    <td style="color: #58A6FF;"><b>${{tr.act_entry}}</b></td>
    <td style="color: ${{isWin ? "#4EBE71" : "#F87171"}};"><b>${{tr.act_exit}}</b></td>
    <td>${{tr.bars_held}} min</td>
    <td style="font-weight: 800; color: ${{isWin ? "#4EBE71" : "#F87171"}};">${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}} R</td>
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

  // Shaded Zone Box
  const zoneBoxX = [tr.created_time, tr.exit_time, tr.exit_time, tr.created_time, tr.created_time];
  const zoneBoxY = [tr.top, tr.top, tr.bottom, tr.bottom, tr.top];
  const zoneTrace = {{
    x: zoneBoxX,
    y: zoneBoxY,
    fill: 'toself',
    fillcolor: isBuy ? 'rgba(56, 139, 253, 0.15)' : 'rgba(218, 54, 51, 0.15)',
    line: {{ color: isBuy ? '#388BFD' : '#DA3633', width: 1.5 }},
    name: 'Zone Range',
    mode: 'lines',
    hoverinfo: 'name'
  }};

  // Level lines (from created_time to exit_time)
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

  // Marker annotations
  const annotations = [
    {{
      x: tr.created_time,
      y: isBuy ? tr.top : tr.bottom,
      xref: 'x', yref: 'y',
      text: '🌱 Zone Created',
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
      xref: 'x', yref: 'y',
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
      xref: 'x', yref: 'y',
      text: `🏁 ${{tr.outcome}} @ ${{tr.act_exit}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}}R)`,
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

// Initialize
renderTradeSelector();
renderChart(currentTradeIndex);
</script>
</body>
</html>
"""

    out_html = os.path.join(ARTIFACTS_DIR, "prod_live_strategy_trades_inspector.html")
    with open(out_html, "w") as f:
        f.write(html_content)
    print(f"\nSaved interactive chart visualizer to {out_html}")


if __name__ == "__main__":
    main()
