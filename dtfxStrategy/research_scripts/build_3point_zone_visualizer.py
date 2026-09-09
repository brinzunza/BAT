"""3-Point Zone Validation Visualizer:
For each zone, shows:
1. Point 1: The First Extreme (Higher High for BUY / Lower Low for SELL) with exact Timestamp & Price.
2. Point 2: The Pullback Extreme (Higher Low for BUY / Lower High for SELL) with exact Timestamp & Price.
3. Point 3: The Validation Moment (The exact moment price breaks beyond Point 1 to validate the zone) with exact Timestamp & Price.
4. The 50% Midpoint Touch Entry & Outcome Exit.
"""
import os
import sys
import json
import time
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

SYMBOLS = ["USDJPY", "EURUSD", "GBPUSD", "EURGBP"]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
MIN_ZONE_PIPS = {"USDJPY": 10.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}
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


def find_fractal_swings_df(df, k=3):
    highs = df["mid_h"].values
    lows = df["mid_l"].values
    times = df.index
    n = len(highs)

    swings = []
    last_type = None

    for i in range(k, n - k):
        # Swing high
        if all(highs[i] >= highs[i - j] and highs[i] >= highs[i + j] for j in range(1, k + 1)):
            if last_type != "HIGH":
                swings.append({"type": "HIGH", "idx": i, "time": times[i], "price": highs[i], "conf_time": times[i + k]})
                last_type = "HIGH"
            elif highs[i] > swings[-1]["price"]:
                swings[-1] = {"type": "HIGH", "idx": i, "time": times[i], "price": highs[i], "conf_time": times[i + k]}

        # Swing low
        elif all(lows[i] <= lows[i - j] and lows[i] <= lows[i + j] for j in range(1, k + 1)):
            if last_type != "LOW":
                swings.append({"type": "LOW", "idx": i, "time": times[i], "price": lows[i], "conf_time": times[i + k]})
                last_type = "LOW"
            elif lows[i] < swings[-1]["price"]:
                swings[-1] = {"type": "LOW", "idx": i, "time": times[i], "price": lows[i], "conf_time": times[i + k]}

    return swings


def extract_3point_zones(df_s5, sym):
    pip = PIP[sym]
    min_zone_height = MIN_ZONE_PIPS.get(sym, 5.0) * pip
    min_stop_price = MIN_STOP_PIPS * pip

    swings = find_fractal_swings_df(df_s5, k=3)
    trades = []

    for i in range(len(swings) - 1):
        s1 = swings[i]
        s2 = swings[i + 1]

        if s1["type"] == "LOW" and s2["type"] == "HIGH":
            # BUY Zone: Point 1 = High (HH/Top), Point 2 = Low (HL/Bottom)
            # Actually: in an uptrend, price was at s1 (Low), pushed up to s2 (High), then pulls back
            # Point 1 = Swing High (s2), Point 2 = Swing Low (s1)
            is_buy = True
            p1_name = "Higher High (Peak)"
            p1_time = str(s2["time"])
            p1_price = float(s2["price"])
            
            p2_name = "Higher Low (Swing Low)"
            p2_time = str(s1["time"])
            p2_price = float(s1["price"])

            top = p1_price
            bottom = p2_price
            c_time = s2["conf_time"]

        elif s1["type"] == "HIGH" and s2["type"] == "LOW":
            # SELL Zone: Point 1 = Low (LL/Bottom), Point 2 = High (LH/Top)
            is_buy = False
            p1_name = "Lower Low (Valley)"
            p1_time = str(s2["time"])
            p1_price = float(s2["price"])
            
            p2_name = "Lower High (Swing High)"
            p2_time = str(s1["time"])
            p2_price = float(s1["price"])

            top = p2_price
            bottom = p1_price
            c_time = s2["conf_time"]
        else:
            continue

        height = top - bottom
        if height < min_zone_height:
            continue

        h = height / 2.0
        mid = (top + bottom) / 2.0
        eff_risk = h
        if eff_risk < min_stop_price:
            continue

        raw_tp = mid + TP_MULT * h if is_buy else mid - TP_MULT * h
        raw_sl = mid - SL_MULT * h if is_buy else mid + SL_MULT * h

        s5_after_creation = df_s5[df_s5.index >= c_time]
        if len(s5_after_creation) < 5:
            continue

        # Point 3: Validation Moment (Overshoot)
        validation_time = None
        validation_price = None
        killed_in_stage1 = False

        for t_idx, row in s5_after_creation.iterrows():
            if is_buy:
                if row["bid_l"] <= bottom:
                    # Invalidation before validation
                    killed_in_stage1 = True
                    break
                if row["mid_h"] > top:
                    validation_time = t_idx
                    validation_price = float(row["mid_h"])
                    break
            else:
                if row["ask_h"] >= top:
                    # Invalidation before validation
                    killed_in_stage1 = True
                    break
                if row["mid_l"] < bottom:
                    validation_time = t_idx
                    validation_price = float(row["mid_l"])
                    break

        if killed_in_stage1 or validation_time is None:
            continue

        # 4. Limit Fill at Midpoint
        s5_after_validation = s5_after_creation[s5_after_creation.index >= validation_time]
        fill_time = None
        killed_in_stage2 = False

        for t_idx, row in s5_after_validation.iterrows():
            if is_buy:
                if row["bid_l"] <= bottom:
                    killed_in_stage2 = True
                    break
                if row["ask_l"] <= mid:
                    fill_time = t_idx
                    break
            else:
                if row["ask_h"] >= top:
                    killed_in_stage2 = True
                    break
                if row["bid_h"] >= mid:
                    fill_time = t_idx
                    break

        if killed_in_stage2 or fill_time is None:
            continue

        # 5. Exit Resolution
        s5_after_fill = s5_after_validation[s5_after_validation.index >= fill_time]
        outcome = None
        exit_time = None

        for t_idx, row in s5_after_fill.iterrows():
            if is_buy:
                if row["bid_l"] <= raw_sl:
                    outcome = "LOSS"
                    exit_time = t_idx
                    break
                elif row["bid_h"] >= raw_tp:
                    outcome = "WIN"
                    exit_time = t_idx
                    break
            else:
                if row["ask_h"] >= raw_sl:
                    outcome = "LOSS"
                    exit_time = t_idx
                    break
                elif row["ask_l"] <= raw_tp:
                    outcome = "WIN"
                    exit_time = t_idx
                    break

        if outcome is not None:
            net_r = TP_MULT if outcome == "WIN" else -1.0
            trades.append({
                "symbol": sym,
                "side": "BUY" if is_buy else "SELL",
                "outcome": outcome,
                "net_r": round(net_r, 2),
                "zone_pips": round(height / pip, 2),
                # The 3 Points:
                "p1_name": p1_name,
                "p1_time": p1_time,
                "p1_price": round(p1_price, 5),
                "p2_name": p2_name,
                "p2_time": p2_time,
                "p2_price": round(p2_price, 5),
                "p3_name": "Point 3: Zone Validated (Price Breaks Point 1)",
                "p3_time": str(validation_time),
                "p3_price": round(validation_price, 5),
                # Trade Execution:
                "fill_time": str(fill_time),
                "fill_price": round(mid, 5),
                "exit_time": str(exit_time),
                "exit_price": round(raw_tp if outcome == "WIN" else raw_sl, 5),
                "top": round(top, 5),
                "bottom": round(bottom, 5),
                "mid": round(mid, 5),
                "raw_tp": round(raw_tp, 5),
                "raw_sl": round(raw_sl, 5),
                "duration_sec": int((exit_time - fill_time).total_seconds()),
            })

    return trades


def main():
    print("=" * 90)
    print("  BUILDING 3-POINT ZONE VALIDATION VISUALIZER")
    print("=" * 90)

    client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)

    all_trades = []
    symbol_s5_dfs = {}

    for sym in SYMBOLS:
        df_s5 = fetch_s5_candles(client, sym, total_count=35000)
        symbol_s5_dfs[sym] = df_s5
        t_list = extract_3point_zones(df_s5, sym)
        all_trades.extend(t_list)

    df_trades = pd.DataFrame(all_trades).sort_values("fill_time").reset_index(drop=True)
    print(f"\nTotal Validated 3-Point Trades: {len(df_trades)}")

    # Prepare Last 10 Trades Payload
    trades_to_plot = df_trades.tail(10).to_dict(orient="records")
    chart_payload = []

    for t_idx, tr in enumerate(trades_to_plot):
        sym = tr["symbol"]
        df_s5 = symbol_s5_dfs[sym]
        
        # Start slice from Point 2 (or Point 1, whichever is earlier) minus 2 minutes
        t_early = min(pd.Timestamp(tr["p1_time"]), pd.Timestamp(tr["p2_time"]))
        t_late = pd.Timestamp(tr["exit_time"])

        sub_s5 = df_s5[(df_s5.index >= t_early - pd.Timedelta(minutes=3)) & (df_s5.index <= t_late + pd.Timedelta(minutes=3))]

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
            "trade_info": tr,
            "ticks": tick_candles,
        })

    # Render HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3-Point Zone Validation & Microstructure Visualizer</title>
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
    --accent-purple: #A371F7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
  
  .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }}
  .title {{ font-size: 22px; font-weight: 700; color: #fff; }}
  .badge {{ padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; text-transform: uppercase; background: #1C2B44; color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
  
  .three-points-card {{ background: #0E131E; border: 1px solid var(--border-color); border-radius: 10px; padding: 15px 20px; margin-bottom: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 15px; }}
  .point-box {{ background: #151C2C; border: 1px solid #253147; border-radius: 8px; padding: 12px 15px; }}
  .point-tag {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
  .point-time {{ font-size: 12px; color: var(--text-secondary); margin-top: 2px; }}
  .point-price {{ font-size: 16px; font-weight: 800; color: #fff; }}
  
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
    <div class="title">📐 3-Point Structural Zone Validation & Tick Execution Visualizer</div>
    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
      Explicitly displays <b>Point 1 (Peak/Valley)</b>, <b>Point 2 (Pullback Swing)</b>, and <b>Point 3 (Zone Validation Breakout)</b> with exact timestamps.
    </div>
  </div>
  <div class="badge">3-POINT VALIDATION ENGINE</div>
</div>

<div class="trade-selector" id="tradeSelector"></div>

<!-- 3 Points Detailed Cards -->
<div class="three-points-card" id="threePointsContainer"></div>

<div class="chart-container">
  <div class="trade-header">
    <div class="trade-title" id="tradeTitle"></div>
    <div id="tradeSummaryText" style="font-size: 13px; color: var(--text-secondary);"></div>
  </div>
  
  <div id="chartDiv"></div>

  <div class="legend-box">
    <div class="legend-item"><div class="line-sample" style="background: rgba(56, 139, 253, 0.35); height: 12px; width: 18px; border: 1px solid #388BFD;"></div> <b>Zone Box (Top/Bottom)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #388BFD; height: 2px;"></div> <b>50% Midpoint Limit Order</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #4EBE71;"></div> <b>Target (+0.8x Half-Zone)</b></div>
    <div class="legend-item"><div class="line-sample dotted" style="border-color: #F87171;"></div> <b>Stop Loss (-1.0x Half-Zone)</b></div>
    <div class="legend-item"><div class="line-sample" style="background: #F1E05A; height: 2px;"></div> <b>Point 3 Validation Trigger Price</b></div>
  </div>
</div>

<script>
const chartData = {json.dumps(chart_payload)};
let currentTradeIndex = chartData.length - 1;

function renderTradeSelector() {{
  const container = document.getElementById("tradeSelector");
  container.innerHTML = "";
  if (!chartData.length) {{
    container.innerHTML = "<div style='color: var(--text-secondary);'>No recent validated 3-point trades found.</div>";
    return;
  }}
  chartData.forEach((item, idx) => {{
    const tr = item.trade_info;
    const btn = document.createElement("button");
    const isWin = tr.outcome === "WIN";
    btn.className = `trade-btn ${{isWin ? "win" : "loss"}} ${{idx === currentTradeIndex ? "active" : ""}}`;
    btn.innerHTML = `Trade #${{item.trade_id}}: ${{tr.symbol}} ${{tr.side}} (${{tr.outcome}} ${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}}R)`;
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
  const item = chartData[idx];
  const tr = item.trade_info;
  const isWin = tr.outcome === "WIN";
  const isBuy = tr.side === "BUY";
  
  // Render 3 Points Cards
  document.getElementById("threePointsContainer").innerHTML = `
    <div class="point-box" style="border-color: ${{isBuy ? "#388BFD" : "#DA3633"}};">
      <div class="point-tag" style="color: ${{isBuy ? "#58A6FF" : "#F87171"}};">📍 Point 1: ${{tr.p1_name}}</div>
      <div class="point-price">${{tr.p1_price}}</div>
      <div class="point-time">Timestamp: <b>${{tr.p1_time}}</b></div>
    </div>
    <div class="point-box" style="border-color: #A371F7;">
      <div class="point-tag" style="color: #D2A8FF;">📍 Point 2: ${{tr.p2_name}}</div>
      <div class="point-price">${{tr.p2_price}}</div>
      <div class="point-time">Timestamp: <b>${{tr.p2_time}}</b></div>
    </div>
    <div class="point-box" style="border-color: #F1E05A;">
      <div class="point-tag" style="color: #F1E05A;">⚡ Point 3: Zone Validated (Breakout)</div>
      <div class="point-price">${{tr.p3_price}}</div>
      <div class="point-time">Timestamp: <b>${{tr.p3_time}}</b></div>
    </div>
    <div class="point-box" style="border-color: #238636;">
      <div class="point-tag" style="color: #4EBE71;">🎯 Retest Entry & Exit</div>
      <div class="point-price">${{tr.fill_price}} ➔ ${{tr.exit_price}}</div>
      <div class="point-time">Filled: <b>${{tr.fill_time}}</b> | Exited: <b>${{tr.exit_time}}</b></div>
    </div>
  `;

  const titleEl = document.getElementById("tradeTitle");
  titleEl.innerHTML = `
    <span>Trade #${{item.trade_id}}: ${{tr.symbol}}</span>
    <span class="${{isBuy ? "tag-side-buy" : "tag-side-sell"}}">${{tr.side}}</span>
    <span class="${{isWin ? "tag-win" : "tag-loss"}}">${{tr.outcome}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}} R)</span>
    <span style="font-size: 13px; color: var(--text-secondary);">Zone: <b>${{tr.zone_pips}} pips</b> | Duration: <b>${{tr.duration_sec}}s</b></span>
  `;

  document.getElementById("tradeSummaryText").innerHTML = `
    P1: <b>${{tr.p1_time}}</b> ➔ P2: <b>${{tr.p2_time}}</b> ➔ P3 Validated: <b>${{tr.p3_time}}</b> ➔ Fill: <b>${{tr.fill_time}}</b>
  `;

  const times = item.ticks.map(t => t.time);
  const midO = item.ticks.map(t => t.mid_o);
  const midH = item.ticks.map(t => t.mid_h);
  const midL = item.ticks.map(t => t.mid_l);
  const midC = item.ticks.map(t => t.mid_c);

  const candleTrace = {{
    x: times,
    open: midO,
    high: midH,
    low: midL,
    close: midC,
    type: 'candlestick',
    name: 'S5 Candlestick',
    increasing: {{ line: {{ color: '#26A69A' }} }},
    decreasing: {{ line: {{ color: '#EF5350' }} }},
    showlegend: false
  }};

  // Zone Box (From Point 2 to Exit Time)
  const zoneBoxX = [tr.p2_time, tr.exit_time, tr.exit_time, tr.p2_time, tr.p2_time];
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
    x: [tr.p2_time, tr.exit_time],
    y: [tr.mid, tr.mid],
    mode: 'lines',
    line: {{ color: '#388BFD', width: 2 }},
    name: `50% Mid Limit Price (${{tr.mid}})`
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
      x: tr.p1_time,
      y: tr.p1_price,
      text: `📍 1. ${{isBuy ? "Higher High" : "Lower Low"}} (${{tr.p1_price}})`,
      showarrow: true,
      arrowhead: 2,
      ax: 0, ay: isBuy ? -35 : 35,
      bgcolor: isBuy ? '#1C2B44' : '#2A1719',
      bordercolor: isBuy ? '#388BFD' : '#DA3633',
      font: {{ color: '#fff', size: 10, weight: 'bold' }}
    }},
    {{
      x: tr.p2_time,
      y: tr.p2_price,
      text: `📍 2. ${{isBuy ? "Higher Low" : "Lower High"}} (${{tr.p2_price}})`,
      showarrow: true,
      arrowhead: 2,
      ax: 0, ay: isBuy ? 35 : -35,
      bgcolor: '#201830',
      bordercolor: '#A371F7',
      font: {{ color: '#fff', size: 10, weight: 'bold' }}
    }},
    {{
      x: tr.p3_time,
      y: tr.p3_price,
      text: `⚡ 3. Zone Validated (${{tr.p3_price}})`,
      showarrow: true,
      arrowhead: 2,
      ax: 0, ay: isBuy ? -35 : 35,
      bgcolor: '#2A2612',
      bordercolor: '#F1E05A',
      font: {{ color: '#F1E05A', size: 10, weight: 'bold' }}
    }},
    {{
      x: tr.fill_time,
      y: tr.fill_price,
      text: `🎯 Limit Fill (${{tr.fill_price}})`,
      showarrow: true,
      arrowhead: 2,
      ax: -25, ay: isBuy ? 35 : -35,
      bgcolor: isBuy ? '#12281D' : '#2A1719',
      bordercolor: isBuy ? '#238636' : '#DA3633',
      font: {{ color: '#fff', size: 10 }}
    }},
    {{
      x: tr.exit_time,
      y: tr.exit_price,
      text: `🏁 ${{tr.outcome}} (${{tr.net_r >= 0 ? "+" + tr.net_r : tr.net_r}}R)`,
      showarrow: true,
      arrowhead: 2,
      ax: 25, ay: isWin ? (isBuy ? -35 : 35) : (isBuy ? 35 : -35),
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

    out_file = os.path.join(ARTIFACTS_DIR, "three_point_zone_validation_inspector.html")
    with open(out_file, "w") as f:
        f.write(html_content)
    print(f"\nSaved 3-Point Zone Validation Inspector to: {out_file}")


if __name__ == "__main__":
    main()
