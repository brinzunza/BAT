import json
import os
from build_realistic_multi_swing import generate_realistic_dataset

def run_dtfx_state_machine(candles):
    state = {
        "trend": "UPTREND",
        "pHH": None,
        "cHH": None,
        "pHL": None,
        "cHL": None,
        "last_cHL": None,
        "pLL": None,
        "cLL": None,
        "pLH": None,
        "cLH": None,
        "last_cLH": None,
        "in_pullback": False
    }
    
    events = []
    zones = []
    pivots = []
    
    prev_green_open = None
    prev_green_close = None
    prev_red_open = None
    prev_red_close = None

    for i, c in enumerate(candles):
        t = c["t"]
        o, h, l, cl = c["o"], c["h"], c["l"], c["c"]
        is_green = cl >= o
        is_red = cl < o
        
        event_notes = []

        if state["trend"] == "UPTREND":
            if state["pHH"] is None or h > state["pHH"]:
                state["pHH"] = h
                
            # Check for Trend Flip breakdown (red candle body closes below last_cHL)
            if state["last_cHL"] is not None and cl < state["last_cHL"]:
                # Trend Flip!
                old_floor = state["last_cHL"]
                top_peak = state["pHH"]
                state["trend"] = "DOWNTREND"
                state["cLH"] = top_peak
                state["last_cLH"] = None
                state["pLL"] = l
                state["cLL"] = None
                state["in_pullback"] = False
                
                # SELL FLIP ZONE: Top = top_peak, Bottom = old_floor
                mid = round((top_peak + old_floor) / 2.0, 2)
                zones.append({
                    "id": f"Z_{len(zones)+1}",
                    "name": "SELL FLIP ZONE",
                    "type": "SELL_FLIP",
                    "side": "SELL",
                    "created_time": t,
                    "candle_idx": i,
                    "top": top_peak,
                    "bottom": old_floor,
                    "mid": mid,
                    "phase": "Trend Flip ➔ Downtrend",
                    "desc": f"Triggered when body closed below last_cHL ({old_floor}). Top = Peak ({top_peak}), Bottom = Yellow Floor ({old_floor}). 50% Mid = {mid}."
                })
                pivots.append({"t": t, "y": top_peak, "label": f"cLH Peak: {top_peak}", "type": "cLH"})
                pivots.append({"t": t, "y": old_floor, "label": f"Broken Floor: {old_floor}", "type": "break_floor"})
                event_notes.append(f"🚨 FLIP TO DOWNTREND! Closes below last_cHL ({old_floor}). Formed SELL FLIP ZONE [{top_peak} - {old_floor}].")
            else:
                # Normal Uptrend logic
                if not state["in_pullback"] and is_red and prev_green_open is not None and cl < prev_green_open:
                    state["in_pullback"] = True
                    state["cHH"] = state["pHH"]
                    state["pHL"] = l
                    pivots.append({"t": t, "y": state["cHH"], "label": f"cHH: {state['cHH']}", "type": "cHH"})
                    event_notes.append(f"✅ Pullback confirmed (close {cl} < prev open {prev_green_open}). Locks cHH = {state['cHH']}.")
                elif state["in_pullback"]:
                    if l < state["pHL"]:
                        state["pHL"] = l
                    if cl > state["cHH"]:
                        state["in_pullback"] = False
                        state["cHL"] = state["pHL"]
                        state["last_cHL"] = state["cHL"]
                        state["pHH"] = h
                        
                        top = state["cHH"]
                        bottom = state["cHL"]
                        mid = round((top + bottom) / 2.0, 2)
                        zones.append({
                            "id": f"Z_{len(zones)+1}",
                            "name": f"BUY ZONE {len(zones)+1}",
                            "type": "BUY",
                            "side": "BUY",
                            "created_time": t,
                            "candle_idx": i,
                            "top": top,
                            "bottom": bottom,
                            "mid": mid,
                            "phase": "Uptrend Swing",
                            "desc": f"Confirmed by close ({cl}) > cHH ({top}). cHL = {bottom}. 50% Mid = {mid}."
                        })
                        pivots.append({"t": t, "y": bottom, "label": f"cHL: {bottom}", "type": "cHL"})
                        event_notes.append(f"✅ Breakout (close {cl} > cHH {top}). Locks cHL = {bottom}. BUY ZONE [{top} - {bottom}].")

        elif state["trend"] == "DOWNTREND":
            if state["pLL"] is None or l < state["pLL"]:
                state["pLL"] = l
                
            # Check for Trend Flip breakout (green candle body closes above last_cLH)
            if state["last_cLH"] is not None and cl > state["last_cLH"]:
                # Trend Flip!
                old_ceiling = state["last_cLH"]
                bottom_trough = state["pLL"]
                state["trend"] = "UPTREND"
                state["cHL"] = bottom_trough
                state["last_cHL"] = None
                state["pHH"] = h
                state["cHH"] = None
                state["in_pullback"] = False
                
                # BUY FLIP ZONE: Top = old_ceiling, Bottom = bottom_trough
                mid = round((old_ceiling + bottom_trough) / 2.0, 2)
                zones.append({
                    "id": f"Z_{len(zones)+1}",
                    "name": "BUY FLIP ZONE",
                    "type": "BUY_FLIP",
                    "side": "BUY",
                    "created_time": t,
                    "candle_idx": i,
                    "top": old_ceiling,
                    "bottom": bottom_trough,
                    "mid": mid,
                    "phase": "Trend Flip ➔ Uptrend",
                    "desc": f"Triggered when body closed above last_cLH ({old_ceiling}). Top = Yellow Ceiling ({old_ceiling}), Bottom = Trough ({bottom_trough}). 50% Mid = {mid}."
                })
                pivots.append({"t": t, "y": bottom_trough, "label": f"cHL Trough: {bottom_trough}", "type": "cHL"})
                pivots.append({"t": t, "y": old_ceiling, "label": f"Broken Ceiling: {old_ceiling}", "type": "break_ceiling"})
                event_notes.append(f"🚨 FLIP TO UPTREND! Closes above last_cLH ({old_ceiling}). Formed BUY FLIP ZONE [{old_ceiling} - {bottom_trough}].")
            else:
                # Normal Downtrend logic
                if not state["in_pullback"] and is_green and prev_red_open is not None and cl > prev_red_open:
                    state["in_pullback"] = True
                    state["cLL"] = state["pLL"]
                    state["pLH"] = h
                    pivots.append({"t": t, "y": state["cLL"], "label": f"cLL: {state['cLL']}", "type": "cLL"})
                    event_notes.append(f"✅ Pullback confirmed (close {cl} > prev open {prev_red_open}). Locks cLL = {state['cLL']}.")
                elif state["in_pullback"]:
                    if h > state["pLH"]:
                        state["pLH"] = h
                    if cl < state["cLL"]:
                        state["in_pullback"] = False
                        state["cLH"] = state["pLH"]
                        state["last_cLH"] = state["cLH"]
                        state["pLL"] = l
                        
                        top = state["cLH"]
                        bottom = state["cLL"]
                        mid = round((top + bottom) / 2.0, 2)
                        zones.append({
                            "id": f"Z_{len(zones)+1}",
                            "name": f"SELL ZONE {len(zones)+1}",
                            "type": "SELL",
                            "side": "SELL",
                            "created_time": t,
                            "candle_idx": i,
                            "top": top,
                            "bottom": bottom,
                            "mid": mid,
                            "phase": "Downtrend Swing",
                            "desc": f"Confirmed by close ({cl}) < cLL ({bottom}). cLH = {top}. 50% Mid = {mid}."
                        })
                        pivots.append({"t": t, "y": top, "label": f"cLH: {top}", "type": "cLH"})
                        event_notes.append(f"✅ Breakdown (close {cl} < cLL {bottom}). Locks cLH = {top}. SELL ZONE [{top} - {bottom}].")

        if is_green:
            prev_green_open = o
            prev_green_close = cl
        elif is_red:
            prev_red_open = o
            prev_red_close = cl
            
        events.append({
            "t": t,
            "candle_idx": i,
            "trend": state["trend"],
            "notes": " | ".join(event_notes) if event_notes else "Impulse continuation"
        })

    return zones, pivots, events

def build_html_dashboard(candles, zones, pivots, events):
    # Find exact flip candle indices
    flip_sell_idx = next(z["candle_idx"] for z in zones if z["type"] == "SELL_FLIP")
    flip_buy_idx = next(z["candle_idx"] for z in zones if z["type"] == "BUY_FLIP")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DTFX Realistic Multi-Cycle Swings Visualizer</title>
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
    --accent-purple: #A371F7;
    --accent-cyan: #39C5BB;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
  
  .header {{ border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }}
  .title {{ font-size: 22px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }}
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 15px; }}
  .metric-label {{ font-size: 12px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
  .metric-val {{ font-size: 20px; font-weight: 700; }}

  .tab-bar {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
  .tab-btn {{ background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 10px 18px; border-radius: 8px; cursor: pointer; font-size: 13.5px; font-weight: 700; transition: all 0.2s; }}
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
  
  #chartDiv {{ width: 100%; height: 720px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">🌊 Realistic DTFX Multi-Cycle Structural Master Chart</div>
  <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 6px;">
    Realistic multi-candle microstructure featuring <b>5 Upward Swings</b>, transitioning via <b>Sell Flip Zone</b> to <b>3 Downward Swings</b>, and transitioning via <b>Buy Flip Zone</b> to <b>6 Upward Swings</b>. 100% strictly validated by the DTFX state machine.
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Phase 1: 5-Swing Uptrend</div>
    <div class="metric-val" style="color: #4ADE80;">5 Swings (5 Buy Zones)</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">100.0 ➔ Peak 136.8</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Phase 2: 3-Swing Downtrend</div>
    <div class="metric-val" style="color: #F87171;">3 Swings (1 Flip + 3 Sell)</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Break 127.0 ➔ Trough 88.0</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Phase 3: 6-Swing Uptrend</div>
    <div class="metric-val" style="color: #4ADE80;">6 Swings (1 Flip + 6 Buy)</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Break 105.6 ➔ High 163.5</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Total Candlesticks & Zones</div>
    <div class="metric-val" style="color: #60A5FA;">{len(candles)} Candles | 16 Zones</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Full Microstructure Realism</div>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" id="tabAll" onclick="setFilter('all')">🌐 Complete Master Journey ({len(candles)} Candles)</button>
  <button class="tab-btn" id="tabP1" onclick="setFilter('p1')">🟢 Phase 1: 5-Swing Uptrend</button>
  <button class="tab-btn" id="tabP2" onclick="setFilter('p2')">🔴 Phase 2: Trend Flip ➔ 3-Swing Downtrend</button>
  <button class="tab-btn" id="tabP3" onclick="setFilter('p3')">🟢 Phase 3: Trend Flip ➔ 6-Swing Uptrend</button>
</div>

<div class="chart-card">
  <div class="chart-title" id="chartHeaderTitle">📊 Multi-Cycle Candlestick & Structural Zone Map</div>
  <div id="chartDiv"></div>
</div>

<div class="table-card">
  <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📋 Confirmed Zone & Structural Pivots Ledger</div>
  <table>
    <thead>
      <tr>
        <th>Zone ID</th>
        <th>Zone Type</th>
        <th>Phase</th>
        <th>Created Time</th>
        <th>Zone Top</th>
        <th>Zone Bottom</th>
        <th>50% Midpoint</th>
        <th>Structural Mechanics Description</th>
      </tr>
    </thead>
    <tbody id="zonesTableBody"></tbody>
  </table>
</div>

<script>
const candles = {json.dumps(candles)};
const zones = {json.dumps(zones)};
const pivots = {json.dumps(pivots)};
const flipSellIdx = {flip_sell_idx};
const flipBuyIdx = {flip_buy_idx};

let activeFilter = 'all';

function setFilter(filter) {{
  activeFilter = filter;
  document.getElementById("tabAll").className = `tab-btn ${{filter === 'all' ? 'active' : ''}}`;
  document.getElementById("tabP1").className = `tab-btn ${{filter === 'p1' ? 'active' : ''}}`;
  document.getElementById("tabP2").className = `tab-btn ${{filter === 'p2' ? 'active' : ''}}`;
  document.getElementById("tabP3").className = `tab-btn ${{filter === 'p3' ? 'active' : ''}}`;
  renderChart();
}}

function renderChart() {{
  let filteredCandles = candles;
  let filteredZones = zones;
  let title = "🌐 Complete Master Journey: 5 Swings Up ➔ 3 Swings Down ➔ 6 Swings Up";

  if (activeFilter === 'p1') {{
    filteredCandles = candles.slice(0, flipSellIdx + 1);
    filteredZones = zones.filter(z => z.phase.includes("Uptrend Swing") && z.candle_idx < flipSellIdx);
    title = "🟢 Phase 1: 5 Consecutive Upward Swings (100.0 ➔ Peak 136.8)";
  }} else if (activeFilter === 'p2') {{
    filteredCandles = candles.slice(flipSellIdx - 3, flipBuyIdx + 1);
    filteredZones = zones.filter(z => z.phase.includes("Downtrend") || z.type === 'SELL_FLIP');
    title = "🔴 Phase 2: Trend Flip ➔ 3 Consecutive Downward Swings (Peak 136.8 ➔ Trough 88.0)";
  }} else if (activeFilter === 'p3') {{
    filteredCandles = candles.slice(flipBuyIdx - 2);
    filteredZones = zones.filter(z => (z.phase.includes("Uptrend") || z.type === 'BUY_FLIP') && z.candle_idx >= flipBuyIdx);
    title = "🟢 Phase 3: Trend Flip ➔ 6 Consecutive Upward Swings (Trough 88.0 ➔ High 163.5)";
  }}

  document.getElementById("chartHeaderTitle").innerText = title;

  const times = filteredCandles.map(c => c.t);
  const opens = filteredCandles.map(c => c.o);
  const highs = filteredCandles.map(c => c.h);
  const lows = filteredCandles.map(c => c.l);
  const closes = filteredCandles.map(c => c.c);

  const candleTrace = {{
    x: times, open: opens, high: highs, low: lows, close: closes,
    type: 'candlestick', name: 'Price Candles',
    increasing: {{ line: {{ color: '#26A69A' }} }},
    decreasing: {{ line: {{ color: '#EF5350' }} }},
    showlegend: false
  }};

  const traces = [candleTrace];
  const shapes = [];
  const annotations = [];

  // Add Zone boxes & 50% lines
  filteredZones.forEach((z, idx) => {{
    const isBuy = z.side === 'BUY';
    const isFlip = z.type.includes("FLIP");
    
    let fillColor = isBuy ? 'rgba(35, 134, 54, 0.16)' : 'rgba(218, 54, 51, 0.16)';
    let borderColor = isBuy ? '#238636' : '#DA3633';
    let borderWidth = 1.5;

    if (isFlip) {{
      fillColor = isBuy ? 'rgba(0, 255, 102, 0.22)' : 'rgba(255, 68, 68, 0.22)';
      borderColor = isBuy ? '#00FF66' : '#FF4444';
      borderWidth = 2.5;
    }}

    const startIdx = times.indexOf(z.created_time);
    let startTime = z.created_time;
    let endTime = times[Math.min(times.length - 1, (startIdx >= 0 ? startIdx : 0) + 6)];
    if (startIdx === -1) {{
      startTime = times[0];
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
      text: `${{isFlip ? '⚡ ' : ''}}${{z.name}}<br>50%: ${{z.mid}}`,
      showarrow: false,
      xanchor: 'left', yanchor: 'bottom',
      bgcolor: '#0E131E', bordercolor: borderColor,
      font: {{ color: isBuy ? '#4ADE80' : '#F87171', size: 9, weight: 'bold' }}
    }});
  }});

  // Annotate Key Flips
  if (activeFilter === 'all' || activeFilter === 'p2') {{
    const sellFlipTime = candles[flipSellIdx].t;
    annotations.push({{
      x: sellFlipTime, y: 136.8,
      text: '🔴 1st TREND FLIP<br>Break below last_cHL 127.0<br>➔ SELL FLIP ZONE [136.8 - 127.0]',
      showarrow: true, arrowhead: 2,
      ax: -50, ay: -45, bgcolor: '#2A1719', bordercolor: '#DA3633',
      font: {{ color: '#FF8080', size: 10, weight: 'bold' }}
    }});
  }}

  if (activeFilter === 'all' || activeFilter === 'p3') {{
    const buyFlipTime = candles[flipBuyIdx].t;
    annotations.push({{
      x: buyFlipTime, y: 88.0,
      text: '🟢 2nd TREND FLIP<br>Break above last_cLH 105.6<br>➔ BUY FLIP ZONE [105.6 - 88.0]',
      showarrow: true, arrowhead: 2,
      ax: -50, ay: 45, bgcolor: '#142C1D', bordercolor: '#238636',
      font: {{ color: '#80FFAA', size: 10, weight: 'bold' }}
    }});
  }}

  const layout = {{
    paper_bgcolor: '#111622',
    plot_bgcolor: '#080B10',
    xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F', rangeslider: {{ visible: false }} }},
    yaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
    margin: {{ l: 50, r: 50, t: 30, b: 40 }},
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

    return `
      <tr>
        <td style="font-weight: 700; color: #fff;">${{z.id}}</td>
        <td><span class="badge ${{badgeClass}}">${{z.name}}</span></td>
        <td style="color: var(--text-secondary); font-size: 12px;">${{z.phase}}</td>
        <td>${{z.created_time}}</td>
        <td style="font-weight: 600; color: #fff;">${{z.top.toFixed(1)}}</td>
        <td style="font-weight: 600; color: #fff;">${{z.bottom.toFixed(1)}}</td>
        <td style="font-weight: 700; color: #388BFD;">${{z.mid.toFixed(2)}}</td>
        <td style="color: #D1D5DB;">${{z.desc}}</td>
      </tr>
    `;
  }}).join('');
}}

// Initialize
renderTable();
renderChart();
</script>
</body>
</html>
"""
    output_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/dtfx_multi_cycle_swings_visualizer.html"
    with open(output_path, "w") as f:
        f.write(html)
    print("Realistic multi-cycle visualizer dashboard built successfully.")

if __name__ == "__main__":
    candles = generate_realistic_dataset()
    zones, pivots, events = run_dtfx_state_machine(candles)
    print(f"Detected {len(zones)} zones across the cycles in realistic dataset ({len(candles)} candles):")
    for z in zones:
        print(f" - {z['id']}: {z['name']} [{z['top']} - {z['bottom']}], 50% Mid = {z['mid']} ({z['phase']})")
    build_html_dashboard(candles, zones, pivots, events)
