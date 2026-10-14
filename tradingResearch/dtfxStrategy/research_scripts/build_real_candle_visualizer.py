import json
import os
import pandas as pd
from test_real_dtfx_data import process_real_market_data, run_real_dtfx_state_machine
from test_strict_50pct_lifecycles import track_strict_50pct_zone_lifecycles

def generate_real_market_visualizer():
    csv_file = "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data/EURUSD_15m.csv"
    candles = process_real_market_data(csv_file, max_candles=600)
    zones, pivots, events = run_real_dtfx_state_machine(candles)
    
    # Calculate exact zone termination: VALID UNTIL PRICE TOUCHES 50% AND LEAVES THE ZONE
    zones = track_strict_50pct_zone_lifecycles(candles, zones)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DTFX Real Market Candlestick Zones Visualizer (Left-Anchored & Mitigated)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg-main: #080B10;
    --bg-card: #111622;
    --border-color: #20293A;
    --text-primary: #F0F4F8;
    --text-secondary: #7E8C9F;
    --accent-blue: #388BFD;
    --accent-green: #238636;
    --accent-red: #DA3633;
    --accent-gold: #F1E05A;
    --accent-cyan: #39C5BB;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
  
  .header {{ border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }}
  .title {{ font-size: 22px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }}
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 15px; margin-bottom: 20px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 15px; }}
  .metric-label {{ font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
  .metric-val {{ font-size: 19px; font-weight: 700; }}

  .tab-bar {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
  .tab-btn {{ background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 9px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 700; transition: all 0.2s; }}
  .tab-btn:hover {{ color: #fff; background: #1C2B44; }}
  .tab-btn.active {{ background: #1C2B44; color: var(--accent-blue); border-color: var(--accent-blue); box-shadow: 0 0 12px rgba(56, 139, 253, 0.35); }}
  
  .chart-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 25px; }}
  .chart-title {{ font-size: 17px; font-weight: 700; margin-bottom: 12px; color: #fff; }}
  
  .table-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-top: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
  th {{ background: #0E131E; color: var(--text-secondary); padding: 10px 14px; border-bottom: 1px solid var(--border-color); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #1A2234; }}
  tr:hover {{ background: #151C2C; }}

  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }}
  .badge-buy {{ background: rgba(35, 134, 54, 0.2); color: #4ADE80; border: 1px solid #238636; }}
  .badge-sell {{ background: rgba(218, 54, 51, 0.2); color: #F87171; border: 1px solid #DA3633; }}
  .badge-flip-sell {{ background: rgba(218, 54, 51, 0.35); color: #FF8080; border: 1.5px solid #FF4444; }}
  .badge-flip-buy {{ background: rgba(35, 134, 54, 0.35); color: #80FFAA; border: 1.5px solid #00FF66; }}
  .badge-won {{ background: rgba(35, 134, 54, 0.25); color: #4ADE80; border: 1px solid #238636; }}
  .badge-inv {{ background: rgba(218, 54, 51, 0.25); color: #F87171; border: 1px solid #DA3633; }}
  
  #chartDiv {{ width: 100%; height: 750px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">📊 DTFX Real Market Candlestick Zones (Left-Anchored & Mitigated Lifecycle)</div>
  <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 6px;">
    Real <b>EUR/USD 15-Minute</b> Market Data: Zones start at the <b>left-most origin candle</b> and <b>end immediately once the retracement trade resolves / leaves the zone (Win / Mitigated)</b>.
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Asset & Timeframe</div>
    <div class="metric-val" style="color: #60A5FA;">EUR/USD 15M</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Real Market Candles</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Total Real Candles</div>
    <div class="metric-val" style="color: #fff;">{len(candles)} Candles</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">{candles[0]['t'].split(' ')[0]} to {candles[-1]['t'].split(' ')[0]}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Detected Structural Zones</div>
    <div class="metric-val" style="color: #4ADE80;">{len(zones)} Confirmed Zones</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Strict Lifecycle Mitigation</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Zone Bounds Rule</div>
    <div class="metric-val" style="color: #FDE047;">Left Origin ➔ Trade Exit</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Zones Terminate Upon Resolution</div>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" id="tabAll" onclick="setFilter('all')">🌐 Entire Multi-Day History ({len(zones)} Zones)</button>
  <button class="tab-btn" id="tabCycle1" onclick="setFilter('c1')">🟢 Cycle 1: Uptrend Wave (Z1 - Z5)</button>
  <button class="tab-btn" id="tabCycle2" onclick="setFilter('c2')">🔴 Cycle 2: Sell Flip ➔ Downtrend (Z6 - Z10)</button>
  <button class="tab-btn" id="tabCycle3" onclick="setFilter('c3')">🟢 Cycle 3: Buy Flip ➔ Uptrend (Z11 - Z15)</button>
</div>

<div class="chart-card">
  <div class="chart-title" id="chartHeaderTitle">📊 Real Market Candlesticks with Left-Anchored & Mitigated Zones</div>
  <div id="chartDiv"></div>
</div>

<div class="table-card">
  <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📋 Confirmed Zone Lifecycle Ledger (Origin ➔ Validation ➔ Resolution End)</div>
  <table>
    <thead>
      <tr>
        <th>Zone ID</th>
        <th>Type</th>
        <th>Side</th>
        <th>Left-Most Origin Time</th>
        <th>Validated Time</th>
        <th>Resolution / End Time</th>
        <th>Zone Top</th>
        <th>Zone Bottom</th>
        <th>50% Midpoint</th>
        <th>Lifecycle Status & Reason</th>
      </tr>
    </thead>
    <tbody id="zonesTableBody"></tbody>
  </table>
</div>

<script>
const candles = {json.dumps(candles)};
const zones = {json.dumps(zones)};
const pivots = {json.dumps(pivots)};

let activeFilter = 'all';

function setFilter(filter) {{
  activeFilter = filter;
  document.getElementById("tabAll").className = `tab-btn ${{filter === 'all' ? 'active' : ''}}`;
  document.getElementById("tabCycle1").className = `tab-btn ${{filter === 'c1' ? 'active' : ''}}`;
  document.getElementById("tabCycle2").className = `tab-btn ${{filter === 'c2' ? 'active' : ''}}`;
  document.getElementById("tabCycle3").className = `tab-btn ${{filter === 'c3' ? 'active' : ''}}`;
  renderChart();
}}

function renderChart() {{
  let filteredCandles = candles;
  let filteredZones = zones;
  let title = "🌐 Full Real Market History: All Left-Anchored & Mitigated Zones";

  if (activeFilter === 'c1') {{
    filteredCandles = candles.slice(0, 100);
    filteredZones = zones.slice(0, 5);
    title = "🟢 Cycle 1: Initial Uptrend Wave (Buy Zones Z1 - Z5)";
  }} else if (activeFilter === 'c2') {{
    filteredCandles = candles.slice(75, 260);
    filteredZones = zones.slice(5, 10);
    title = "🔴 Cycle 2: Sell Flip ➔ Downtrend Continuation (Z6 - Z10)";
  }} else if (activeFilter === 'c3') {{
    filteredCandles = candles.slice(230, 450);
    filteredZones = zones.slice(10, 15);
    title = "🟢 Cycle 3: Buy Flip ➔ Strong Uptrend Surge (Z11 - Z15)";
  }}

  document.getElementById("chartHeaderTitle").innerText = title;

  const times = filteredCandles.map(c => c.t);
  const opens = filteredCandles.map(c => c.o);
  const highs = filteredCandles.map(c => c.h);
  const lows = filteredCandles.map(c => c.l);
  const closes = filteredCandles.map(c => c.c);

  const candleTrace = {{
    x: times, open: opens, high: highs, low: lows, close: closes,
    type: 'candlestick', name: 'EUR/USD 15M',
    increasing: {{ line: {{ color: '#26A69A' }} }},
    decreasing: {{ line: {{ color: '#EF5350' }} }},
    showlegend: false
  }};

  const traces = [candleTrace];
  const shapes = [];
  const annotations = [];

  // Add Left-Anchored & Mitigated Zone boxes
  filteredZones.forEach((z) => {{
    const isBuy = z.side === 'BUY';
    const isFlip = z.type.includes("FLIP");
    const isWon = z.end_reason.includes("Win") || z.end_reason.includes("Completed");
    
    let fillColor = isBuy ? 'rgba(35, 134, 54, 0.18)' : 'rgba(218, 54, 51, 0.18)';
    let borderColor = isBuy ? '#238636' : '#DA3633';
    let borderWidth = 1.5;

    if (isFlip) {{
      fillColor = isBuy ? 'rgba(0, 255, 102, 0.25)' : 'rgba(255, 68, 68, 0.25)';
      borderColor = isBuy ? '#00FF66' : '#FF4444';
      borderWidth = 2.5;
    }}

    // Left-most start time to exact resolution end time!
    let startTime = z.left_time;
    let endTime = z.end_time;

    // Check bounds in visible view
    if (times.indexOf(startTime) === -1) {{
      startTime = times[0];
    }}
    if (times.indexOf(endTime) === -1) {{
      endTime = times[times.length - 1];
    }}

    shapes.push({{
      type: 'rect',
      x0: startTime, x1: endTime,
      y0: z.bottom, y1: z.top,
      fillcolor: fillColor,
      line: {{ color: borderColor, width: borderWidth }},
      layer: 'below'
    }});

    shapes.push({{
      type: 'line',
      x0: startTime, x1: endTime,
      y0: z.mid, y1: z.mid,
      line: {{ color: '#388BFD', width: 1.5, dash: 'dot' }}
    }});

    annotations.push({{
      x: startTime, y: z.top,
      text: `${{isFlip ? '⚡ ' : ''}}${{z.name}}<br>50%: ${{z.mid.toFixed(5)}}`,
      showarrow: false,
      xanchor: 'left', yanchor: 'bottom',
      bgcolor: '#0E131E', bordercolor: borderColor,
      font: {{ color: isBuy ? '#4ADE80' : '#F87171', size: 9, weight: 'bold' }}
    }});

    // Exit annotation at end of zone
    // Exit annotation at end of zone (only if zone actually completed or invalidated)
    const isEnded = z.end_reason.includes("Win") || z.end_reason.includes("Completed") || z.end_reason.includes("Invalidated");
    if (isEnded && times.indexOf(z.end_time) !== -1) {{
      annotations.push({{
        x: z.end_time, y: isBuy ? z.top : z.bottom,
        text: isWon ? '🏆 50% Touched & Left (Ended)' : '❌ Invalidated',
        showarrow: true, arrowhead: 2,
        ax: 0, ay: isBuy ? -25 : 25,
        bgcolor: isWon ? '#142C1D' : '#2A1719',
        bordercolor: isWon ? '#238636' : '#DA3633',
        font: {{ color: isWon ? '#4ADE80' : '#F87171', size: 8, weight: 'bold' }}
      }});
    }}
  }});

  const layout = {{
    paper_bgcolor: '#111622',
    plot_bgcolor: '#080B10',
    xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F', rangeslider: {{ visible: false }} }},
    yaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
    margin: {{ l: 60, r: 50, t: 30, b: 40 }},
    shapes: shapes,
    annotations: annotations
  }};

  Plotly.react('chartDiv', traces, layout, {{ responsive: true }});
}}

function renderTable() {{
  const tbody = document.getElementById("zonesTableBody");
  tbody.innerHTML = zones.map(z => {{
    let badgeClass = z.side === 'BUY' ? 'badge-buy' : 'badge-sell';
    if (z.type === 'BUY_FLIP') badgeClass = 'badge-flip-buy';
    if (z.type === 'SELL_FLIP') badgeClass = 'badge-flip-sell';
    
    let statusBadge = '<span class="badge badge-won">🏆 Touched 50% & Left</span>';
    if (z.end_reason.includes("Invalidated")) {{
      statusBadge = '<span class="badge badge-inv">❌ Invalidated</span>';
    }} else if (z.end_reason.includes("Active") || z.end_reason.includes("Unmitigated")) {{
      statusBadge = '<span class="badge" style="background: rgba(56, 139, 253, 0.25); color: #60A5FA; border: 1px solid #388BFD;">⏳ Active / 50% Untouched</span>';
    }}

    return `
      <tr>
        <td style="font-weight: 700; color: #fff;">${{z.id}}</td>
        <td><span class="badge ${{badgeClass}}">${{z.name}}</span></td>
        <td style="font-weight: 600; color: ${{z.side === 'BUY' ? '#4ADE80' : '#F87171'}};">${{z.side}}</td>
        <td style="color: #FDE047; font-weight: 600;">${{z.left_time}}</td>
        <td style="color: var(--text-secondary);">${{z.created_time}}</td>
        <td style="color: #60A5FA; font-weight: 600;">${{z.end_time}}</td>
        <td style="font-weight: 600; color: #fff;">${{z.top.toFixed(5)}}</td>
        <td style="font-weight: 600; color: #fff;">${{z.bottom.toFixed(5)}}</td>
        <td style="font-weight: 700; color: #388BFD;">${{z.mid.toFixed(5)}}</td>
        <td>${{statusBadge}} <span style="font-size: 11px; color: #7E8C9F; margin-left: 5px;">${{z.end_reason}}</span></td>
      </tr>
    `;
  }}).join('');
}}

renderTable();
renderChart();
</script>
</body>
</html>
"""
    output_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/dtfx_real_candle_zones_visualizer.html"
    with open(output_path, "w") as f:
        f.write(html)
    print("Real market visualizer with strict zone mitigation termination generated successfully.")

if __name__ == "__main__":
    generate_real_market_visualizer()
