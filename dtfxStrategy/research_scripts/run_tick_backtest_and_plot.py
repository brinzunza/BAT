"""High-Resolution Tick & S5 Bid/Ask True Spread Backtest & Interactive Tick Plotter.

Pulls 40,000 S5 Bid/Ask tick candles (~3.5 days of continuous live tick data) from OANDA v20 API for:
- USDJPY (MIN_ZONE_PIPS = 20.0 and 10.0)
- EURUSD, GBPUSD, EURGBP (MIN_ZONE_PIPS = 10.0 and 5.0)
- SL_MULT = 1.0, TP_MULT = 0.8

Generates:
1. High-resolution tick execution reconciliation.
2. Interactive tick-level chart showing exact Ask fill and Bid exit on the 5-second tick series.
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PROD_DIR = "/Users/brunoinzunza/Documents/github/PRODLIVETRADING"
sys.path.insert(0, PROD_DIR)
ARTIFACTS_DIR = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05"

import config
from broker.oanda_client import OANDAClient, SYMBOL_MAP
import oandapyV20.endpoints.instruments as instruments
import goal95_every_swing_zones as g95

SYMBOLS = ["USDJPY", "EURUSD", "GBPUSD", "EURGBP"]
MIN_ZONE_PIPS = {
    "USDJPY": 10.0,
    "EURUSD": 5.0,
    "GBPUSD": 5.0,
    "EURGBP": 5.0,
}
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
SL_MULT = 1.0
TP_MULT = 0.8
MIN_STOP_PIPS = 3.0


def fetch_s5_candles(client, symbol, total_count=35000):
    instrument = SYMBOL_MAP[symbol]
    all_rows = []
    to_time = None
    print(f"  [{symbol}] Fetching {total_count:,} S5 tick candles...", end=" ", flush=True)
    t0 = time.time()
    
    while len(all_rows) < total_count:
        fetch_size = min(total_count - len(all_rows), 5000)
        params = {"count": fetch_size, "granularity": "S5", "price": "BA"}
        if to_time is not None:
            params["to"] = to_time
            
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.api.request(r)
        raw_candles = r.response.get("candles", [])
        complete = [c for c in raw_candles if c.get("complete")]
        if not complete:
            break
            
        for c in complete:
            t_str = c["time"]
            t_dt = pd.Timestamp(t_str).tz_localize(None)
            bid = c["bid"]
            ask = c["ask"]
            all_rows.append({
                "date": t_dt,
                "bid_o": float(bid["o"]), "bid_h": float(bid["h"]), "bid_l": float(bid["l"]), "bid_c": float(bid["c"]),
                "ask_o": float(ask["o"]), "ask_h": float(ask["h"]), "ask_l": float(ask["l"]), "ask_c": float(ask["c"]),
                "mid_o": (float(bid["o"]) + float(ask["o"])) / 2.0,
                "mid_h": (float(bid["h"]) + float(ask["h"])) / 2.0,
                "mid_l": (float(bid["l"]) + float(ask["l"])) / 2.0,
                "mid_c": (float(bid["c"]) + float(ask["c"])) / 2.0,
            })
            
        next_time = complete[0]["time"]
        if to_time is not None and pd.Timestamp(next_time) >= pd.Timestamp(to_time):
            break
        to_time = next_time
        if len(raw_candles) < fetch_size:
            break
            
    df = pd.DataFrame(all_rows).drop_duplicates(subset="date").set_index("date").sort_index()
    print(f"Done! {len(df):,} S5 slices loaded ({df.index[0]} to {df.index[-1]}) in {time.time()-t0:.1f}s")
    return df


def run_tick_analysis():
    print("=" * 90)
    print("  HIGH-RESOLUTION S5 TICK & BID/ASK TRUE SPREAD LIVE BACKTEST")
    print("=" * 90)

    client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)

    all_tick_trades = []
    symbol_tick_dfs = {}

    for sym in SYMBOLS:
        df_s5 = fetch_s5_candles(client, sym, total_count=35000)
        symbol_tick_dfs[sym] = df_s5
        pip = PIP[sym]

        # Resample to 1-Minute M1 bars to build zones exactly as the live engine does
        df_m1 = df_s5.resample("1min").agg({
            "mid_o": "first", "mid_h": "max", "mid_l": "min", "mid_c": "last"
        }).dropna()
        df_m1.rename(columns={"mid_o": "open", "mid_h": "high", "mid_l": "low", "mid_c": "close"}, inplace=True)

        high_m1 = df_m1["high"].values.astype(np.float64)
        low_m1 = df_m1["low"].values.astype(np.float64)
        close_m1 = df_m1["close"].values.astype(np.float64)
        m1_times = df_m1.index

        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_m1, low_m1, close_m1)
        (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
         ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)

        min_p = MIN_ZONE_PIPS.get(sym, 5.0)

        for zi in range(len(z_created)):
            top = float(z_top[zi])
            bottom = float(z_bottom[zi])
            height = top - bottom
            if height < min_p * pip:
                continue

            h = height / 2.0
            mid = (top + bottom) / 2.0
            is_buy = bool(z_is_buy[zi])
            c_m1_idx = z_created[zi]
            c_time = m1_times[c_m1_idx]

            raw_sl = mid - SL_MULT * h if is_buy else mid + SL_MULT * h
            raw_tp = mid + TP_MULT * h if is_buy else mid - TP_MULT * h
            eff_risk = abs(mid - raw_sl)

            if eff_risk < MIN_STOP_PIPS * pip:
                continue

            # Check ticks forward from zone creation time
            s5_forward = df_s5[df_s5.index >= c_time]
            if len(s5_forward) < 5:
                continue

            # 1. Overshoot check on Mid
            overshoot_time = None
            for t_idx, row in s5_forward.iterrows():
                if is_buy and row["mid_h"] > top:
                    overshoot_time = t_idx
                    break
                elif (not is_buy) and row["mid_l"] < bottom:
                    overshoot_time = t_idx
                    break

            if overshoot_time is None:
                continue

            # 2. Limit Fill Check on TRUE BID/ASK
            # Buy Limit rests at Mid -> Fills when ASK drops to <= mid
            # Sell Limit rests at Mid -> Fills when BID rises to >= mid
            s5_after_overshoot = s5_forward[s5_forward.index >= overshoot_time]
            fill_time = None
            fill_price = None

            for t_idx, row in s5_after_overshoot.iterrows():
                # Invalidation breach
                if is_buy and row["bid_l"] <= bottom:
                    break
                if (not is_buy) and row["ask_h"] >= top:
                    break

                if is_buy and row["ask_l"] <= mid:
                    fill_time = t_idx
                    fill_price = mid
                    break
                elif (not is_buy) and row["bid_h"] >= mid:
                    fill_time = t_idx
                    fill_price = mid
                    break

            if fill_time is None:
                continue

            # 3. Exit Resolution on TRUE BID/ASK
            # BUY: TP triggers when BID >= raw_tp; SL triggers when BID <= raw_sl
            # SELL: TP triggers when ASK <= raw_tp; SL triggers when ASK >= raw_sl
            s5_after_fill = s5_after_overshoot[s5_after_overshoot.index >= fill_time]
            outcome = None
            exit_time = None
            exit_price = None

            for t_idx, row in s5_after_fill.iterrows():
                if is_buy:
                    if row["bid_l"] <= raw_sl:
                        outcome = "LOSS"
                        exit_time = t_idx
                        exit_price = raw_sl
                        break
                    elif row["bid_h"] >= raw_tp:
                        outcome = "WIN"
                        exit_time = t_idx
                        exit_price = raw_tp
                        break
                else:
                    if row["ask_h"] >= raw_sl:
                        outcome = "LOSS"
                        exit_time = t_idx
                        exit_price = raw_sl
                        break
                    elif row["ask_l"] <= raw_tp:
                        outcome = "WIN"
                        exit_time = t_idx
                        exit_price = raw_tp
                        break

            if outcome is not None:
                net_r = TP_MULT if outcome == "WIN" else -1.0
                all_tick_trades.append({
                    "symbol": sym,
                    "side": "BUY" if is_buy else "SELL",
                    "outcome": outcome,
                    "net_r": round(net_r, 2),
                    "zone_pips": round(height / pip, 2),
                    "created_time": str(c_time),
                    "overshoot_time": str(overshoot_time),
                    "fill_time": str(fill_time),
                    "exit_time": str(exit_time),
                    "top": top,
                    "bottom": bottom,
                    "mid": mid,
                    "raw_tp": raw_tp,
                    "raw_sl": raw_sl,
                    "fill_price": fill_price,
                    "exit_price": exit_price,
                    "duration_sec": int((exit_time - fill_time).total_seconds()),
                })

    df_results = pd.DataFrame(all_tick_trades).sort_values("fill_time").reset_index(drop=True)
    print(f"\nTotal High-Resolution Tick Trades Found: {len(df_results)}")

    if len(df_results) > 0:
        wr = 100.0 * (df_results["outcome"] == "WIN").mean()
        gw = df_results.loc[df_results["outcome"] == "WIN", "net_r"].sum()
        gl = -df_results.loc[df_results["outcome"] == "LOSS", "net_r"].sum()
        pf = gw / gl if gl > 0 else 0.0
        print(f"  Live Tick Win Rate:     {wr:.1f}%")
        print(f"  Live Tick Profit Factor: {pf:.2f}")

    # Build Interactive S5 Tick Candlestick Chart for Recent 10 Trades
    chart_payload = []
    trades_to_plot = df_results.tail(10).to_dict(orient="records")

    for t_idx, tr in enumerate(trades_to_plot):
        sym = tr["symbol"]
        df_s5 = symbol_tick_dfs[sym]
        f_time = pd.Timestamp(tr["fill_time"])
        x_time = pd.Timestamp(tr["exit_time"])
        c_time = pd.Timestamp(tr["created_time"])

        # Slice 120 S5 bars around trade
        sub_s5 = df_s5[(df_s5.index >= c_time - pd.Timedelta(minutes=5)) & (df_s5.index <= x_time + pd.Timedelta(minutes=5))]
        
        tick_candles = []
        for ts, r in sub_s5.iterrows():
            tick_candles.append({
                "time": str(ts),
                "bid_o": r["bid_o"], "bid_h": r["bid_h"], "bid_l": r["bid_l"], "bid_c": r["bid_c"],
                "ask_o": r["ask_o"], "ask_h": r["ask_h"], "ask_l": r["ask_l"], "ask_c": r["ask_c"],
                "mid_o": r["mid_o"], "mid_h": r["mid_h"], "mid_l": r["mid_l"], "mid_c": r["mid_c"],
            })

        chart_payload.append({
            "trade_id": t_idx + 1,
            "symbol": tr["symbol"],
            "side": tr["side"],
            "outcome": tr["outcome"],
            "net_r": tr["net_r"],
            "zone_pips": tr["zone_pips"],
            "created_time": tr["created_time"],
            "fill_time": tr["fill_time"],
            "exit_time": tr["exit_time"],
            "duration_sec": tr["duration_sec"],
            "top": round(tr["top"], 5),
            "bottom": round(tr["bottom"], 5),
            "mid": round(tr["mid"], 5),
            "raw_tp": round(tr["raw_tp"], 5),
            "raw_sl": round(tr["raw_sl"], 5),
            "ticks": tick_candles
        })

    # Generate Tick Visualizer HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S5 High-Resolution Tick & Bid/Ask Chart Inspector</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg-main: #080B10;
    --bg-card: #111622;
    --bg-card-hover: #182030;
    --border-color: #20293A;
    --text-primary: #F0F4F8;
    --text-secondary: #7E8C9F;
    --accent-blue: #388BFD;
    --accent-green: #238636;
    --accent-red: #DA3633;
    --accent-gold: #F1E05A;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
  
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }}
  .title {{ font-size: 22px; font-weight: 700; color: #fff; }}
  .badge {{ padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase; background: #1C2B44; color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
  
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
  
  #chartDiv {{ width: 100%; height: 600px; }}
  
  .legend-box {{ display: flex; flex-wrap: wrap; gap: 15px; background: #0A0E17; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 16px; margin-top: 15px; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  .line-sample {{ width: 24px; height: 3px; border-radius: 2px; }}
  .line-sample.dotted {{ border-top: 2px dotted; height: 0; background: transparent; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="title">⚡ Sub-Minute S5 Tick & True Bid/Ask Order Book Visualizer</div>
    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
      Resolution: <b>5-Second (S5) True Bid & Ask Quotes</b> | Limit Fills & Stop/Target Triggered against Real Spread
    </div>
  </div>
  <div class="badge">OANDA V20 TICK ENGINE</div>
</div>

<div class="trade-selector" id="tradeSelector"></div>

<div class="chart-container">
  <div class="trade-header">
    <div class="trade-title" id="tradeTitle"></div>
    <div id="tradeSummaryText" style="font-size: 13px; color: var(--text-secondary);"></div>
  </div>
  
  <div id="chartDiv"></div>

  <div class="legend-box">
    <div class="legend-item"><div class="line-sample" style="background: rgba(56, 139, 253, 0.35); height: 12px; width: 18px; border: 1px solid #388BFD;"></div> <b>Zone Box (Top/Bottom)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #388BFD; height: 2px;"></div> <b>Zone Midpoint (Resting Limit Order)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: rgba(88, 166, 255, 0.8); height: 1.5px;"></div> <b>Ask Tick Stream (High)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: rgba(239, 83, 80, 0.8); height: 1.5px;"></div> <b>Bid Tick Stream (Low)</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #4EBE71;"></div> <b>Target (+0.8x Half-Zone)</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #F87171;"></div> <b>Stop Loss (-1.0x Half-Zone)</b></div>
  </div>
</div>

<script>
const chartData = {json.dumps(chart_payload)};
let currentTradeIndex = 0;

function renderTradeSelector() {{
  const container = document.getElementById("tradeSelector");
  container.innerHTML = "";
  if (!chartData.length) {{
    container.innerHTML = "<div style='color: var(--text-secondary);'>No recent S5 trades in sample window.</div>";
    return;
  }}
  chartData.forEach((tr, idx) => {{
    const btn = document.createElement("button");
    const isWin = tr.outcome === "WIN";
    btn.className = `trade-btn ${{isWin ? "win" : "loss"}} ${{idx === currentTradeIndex ? "active" : ""}}`;
    btn.innerHTML = `Trade #${{tr.trade_id}}: ${{tr.symbol}} ${{tr.side}} (${{tr.outcome}} ${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}}R)`;
    btn.onclick = () => {{
      currentTradeIndex = idx;
      renderTradeSelector();
      renderChart(idx);
    }};
    container.appendChild(btn);
  }});
}}

function renderChart(idx) {{
  if (!chartData.length) return;
  const tr = chartData[idx];
  const isWin = tr.outcome === "WIN";
  const isBuy = tr.side === "BUY";
  
  const titleEl = document.getElementById("tradeTitle");
  titleEl.innerHTML = `
    <span>Trade #${{tr.trade_id}}: ${{tr.symbol}}</span>
    <span class="${{isBuy ? "tag-side-buy" : "tag-side-sell"}}">${{tr.side}}</span>
    <span class="${{isWin ? "tag-win" : "tag-loss"}}">${{tr.outcome}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}} R)</span>
    <span style="font-size: 13px; color: var(--text-secondary);">Zone: <b>${{tr.zone_pips}} pips</b> | Held: <b>${{tr.duration_sec}}s (${{(tr.duration_sec/60).toFixed(1)}}m)</b></span>
  `;

  document.getElementById("tradeSummaryText").innerHTML = `
    Formed: <b>${{tr.created_time}}</b> | Filled: <b>${{tr.fill_time}}</b> | Exited: <b>${{tr.exit_time}}</b>
  `;

  const times = tr.ticks.map(t => t.time);
  const midO = tr.ticks.map(t => t.mid_o);
  const midH = tr.ticks.map(t => t.mid_h);
  const midL = tr.ticks.map(t => t.mid_l);
  const midC = tr.ticks.map(t => t.mid_c);

  const askH = tr.ticks.map(t => t.ask_h);
  const bidL = tr.ticks.map(t => t.bid_l);

  // S5 Mid Candlesticks
  const candleTrace = {{
    x: times,
    open: midO,
    high: midH,
    low: midL,
    close: midC,
    type: 'candlestick',
    name: 'S5 Mid Candlestick',
    increasing: {{ line: {{ color: '#26A69A' }} }},
    decreasing: {{ line: {{ color: '#EF5350' }} }},
    showlegend: false
  }};

  // S5 Ask Line (Top of Spread Ribbon)
  const askTrace = {{
    x: times,
    y: askH,
    mode: 'lines',
    line: {{ color: 'rgba(56, 139, 253, 0.4)', width: 1 }},
    name: 'Ask Price (High)',
    hoverinfo: 'y+name'
  }};

  // S5 Bid Line (Bottom of Spread Ribbon)
  const bidTrace = {{
    x: times,
    y: bidL,
    mode: 'lines',
    fill: 'tonexty',
    fillcolor: 'rgba(56, 139, 253, 0.08)',
    line: {{ color: 'rgba(239, 83, 80, 0.4)', width: 1 }},
    name: 'Bid Price (Low)',
    hoverinfo: 'y+name'
  }};

  // Zone Range Box
  const zoneBoxX = [tr.created_time, tr.exit_time, tr.exit_time, tr.created_time, tr.created_time];
  const zoneBoxY = [tr.top, tr.top, tr.bottom, tr.bottom, tr.top];
  const zoneTrace = {{
    x: zoneBoxX,
    y: zoneBoxY,
    fill: 'toself',
    fillcolor: isBuy ? 'rgba(56, 139, 253, 0.15)' : 'rgba(218, 54, 51, 0.15)',
    line: {{ color: isBuy ? '#388BFD' : '#DA3633', width: 1.5 }},
    name: 'Zone Range',
    mode: 'lines'
  }};

  // Midpoint Line
  const midTrace = {{
    x: [tr.created_time, tr.exit_time],
    y: [tr.mid, tr.mid],
    mode: 'lines',
    line: {{ color: '#388BFD', width: 2 }},
    name: `Mid Limit Price (${{tr.mid}})`
  }};

  // TP Line
  const tpTrace = {{
    x: [tr.fill_time, tr.exit_time],
    y: [tr.raw_tp, tr.raw_tp],
    mode: 'lines',
    line: {{ color: '#4EBE71', width: 2, dash: 'dot' }},
    name: `TP Level (${{tr.raw_tp}})`
  }};

  // SL Line
  const slTrace = {{
    x: [tr.fill_time, tr.exit_time],
    y: [tr.raw_sl, tr.raw_sl],
    mode: 'lines',
    line: {{ color: '#F87171', width: 2, dash: 'dot' }},
    name: `SL Level (${{tr.raw_sl}})`
  }};

  const annotations = [
    {{
      x: tr.created_time,
      y: isBuy ? tr.top : tr.bottom,
      text: '🌱 Zone Created',
      showarrow: true,
      arrowhead: 2,
      ax: 0, ay: isBuy ? -30 : 30,
      bgcolor: '#1C2B44',
      bordercolor: '#388BFD',
      font: {{ color: '#fff', size: 10 }}
    }},
    {{
      x: tr.fill_time,
      y: tr.mid,
      text: `🚀 ${{tr.side}} Limit Fill @ ${{tr.mid}}`,
      showarrow: true,
      arrowhead: 2,
      ax: -30, ay: isBuy ? 35 : -35,
      bgcolor: isBuy ? '#12281D' : '#2A1719',
      bordercolor: isBuy ? '#238636' : '#DA3633',
      font: {{ color: '#fff', size: 10 }}
    }},
    {{
      x: tr.exit_time,
      y: isWin ? tr.raw_tp : tr.raw_sl,
      text: `🏁 ${{tr.outcome}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}}R)`,
      showarrow: true,
      arrowhead: 2,
      ax: 30, ay: isWin ? (isBuy ? -35 : 35) : (isBuy ? 35 : -35),
      bgcolor: isWin ? '#12281D' : '#2A1719',
      bordercolor: isWin ? '#238636' : '#DA3633',
      font: {{ color: '#fff', size: 10, weight: 'bold' }}
    }}
  ];

  const layout = {{
    margin: {{ l: 50, r: 50, t: 30, b: 40 }},
    paper_bgcolor: '#111622',
    plot_bgcolor: '#111622',
    xaxis: {{
      type: 'date',
      rangeslider: {{ visible: false }},
      gridcolor: '#20293A',
      color: '#7E8C9F',
      tickformat: '%H:%M:%S'
    }},
    yaxis: {{
      gridcolor: '#20293A',
      color: '#7E8C9F',
      autorange: true
    }},
    annotations: annotations,
    showlegend: true,
    legend: {{
      orientation: 'h',
      x: 0, y: 1.12,
      font: {{ color: '#7E8C9F', size: 10 }}
    }}
  }};

  Plotly.newPlot('chartDiv', [
    candleTrace,
    askTrace,
    bidTrace,
    zoneTrace,
    midTrace,
    tpTrace,
    slTrace
  ], layout, {{ responsive: true, displayModeBar: true }});
}}

renderTradeSelector();
renderChart(currentTradeIndex);
</script>
</body>
</html>
"""

    out_file = os.path.join(ARTIFACTS_DIR, "s5_tick_strategy_visualizer.html")
    with open(out_file, "w") as f:
        f.write(html)
    print(f"\nSaved S5 Tick Chart Visualizer to: {out_file}")


if __name__ == "__main__":
    run_tick_analysis()
