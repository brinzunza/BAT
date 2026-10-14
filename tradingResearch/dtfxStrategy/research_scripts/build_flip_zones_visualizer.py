"""Trend Flip Zones Visualizer:
Demonstrating how trend flips (Uptrend -> Downtrend and Downtrend -> Uptrend) create Flip Zones:
1. Uptrend -> Downtrend Flip:
   - Uptrend has last_cHL (floor) and cHH (peak).
   - Price dumps: candle body closes below last_cHL.
   - TREND FLIPS TO DOWNTREND!
   - cHH becomes the new Confirmed Lower High (cLH, Zone Top).
   - The breakdown low becomes the Confirmed Lower Low (cLL, Zone Bottom).
   - A SELL FLIP ZONE is immediately created [Top: cLH, Bottom: cLL].
   - Price pulls back to 50% midpoint -> SELL Limit Fill -> TP Hit.

2. Downtrend -> Uptrend Flip (Inverted):
   - Downtrend has last_cLH (ceiling) and cLL (valley).
   - Price explodes: candle body closes above last_cLH.
   - TREND FLIPS TO UPTREND!
   - cLL becomes the new Confirmed Higher Low (cHL, Zone Bottom).
   - The breakout high becomes the Confirmed Higher High (cHH, Zone Top).
   - A BUY FLIP ZONE is immediately created [Top: cHH, Bottom: cHL].
   - Price pulls back to 50% midpoint -> BUY Limit Fill -> TP Hit.
"""
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
ARTIFACTS_DIR = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05"


def build_flip_scenarios():
    # --- Scenario 1: UPTREND TO DOWNTREND FLIP ZONE ---
    candles_flip_sell = [
        # Uptrend establish
        {"t": "10:00", "o": 100.0, "h": 102.0, "l": 99.8, "c": 101.8},
        {"t": "10:01", "o": 101.8, "h": 104.5, "l": 101.6, "c": 104.2},
        {"t": "10:02", "o": 104.2, "h": 107.0, "l": 104.0, "c": 106.8}, # Peak 107.0 (cHH)
        {"t": "10:03", "o": 106.5, "h": 106.6, "l": 104.2, "c": 104.4}, # Pullback to 104.2 (Confirmed Higher Low cHL = 104.2)
        {"t": "10:04", "o": 104.4, "h": 108.5, "l": 104.3, "c": 108.2}, # Uptrend peak (cHH = 108.5, last_cHL = 104.2)
        
        # The Structural Breakdown (Trend Flip)
        {"t": "10:05", "o": 108.0, "h": 108.2, "l": 105.0, "c": 105.2}, # Red candle
        {"t": "10:06", "o": 105.2, "h": 105.4, "l": 101.8, "c": 102.0}, # MASSIVE DUMP! Body closes (102.0) < last_cHL (104.2)!
        # At 10:06: TREND FLIPS TO DOWNTREND!
        # cHH (108.5) becomes Confirmed Lower High (cLH = 108.5)
        # Breakdown low (101.8) becomes Confirmed Lower Low (cLL = 101.8)
        # SELL FLIP ZONE IS FORMED: [Top: 108.5, Bottom: 101.8, 50% Mid: 105.15]

        # Retracement to 50% Mid of the Flip Zone
        {"t": "10:07", "o": 102.0, "h": 104.0, "l": 101.9, "c": 103.8},
        {"t": "10:08", "o": 103.8, "h": 105.3, "l": 103.6, "c": 105.0}, # Retests 105.15 -> LIMIT SELL FILL @ 105.15!
        
        # Continuation Downward
        {"t": "10:09", "o": 105.0, "h": 105.2, "l": 101.0, "c": 101.2},
        {"t": "10:10", "o": 101.2, "h": 101.4, "l": 99.0, "c": 99.2},  # Target Hit @ 99.81 (+0.8R)!
    ]

    sell_flip_zone = {
        "name": "SELL FLIP ZONE (Uptrend ➔ Downtrend)",
        "side": "SELL",
        "c_time": "10:06", "fill_time": "10:08", "exit_time": "10:10",
        "top": 108.5, "bottom": 101.8, "mid": 105.15,
        "tp": 99.81, "sl": 111.85,
        "outcome": "WIN (+0.8R)",
        "desc": "Created on 10:06 when body closed below last_cHL (104.2). Top = cLH (108.5), Bottom = cLL (101.8). 50% Mid @ 105.15 filled at 10:08."
    }

    # --- Scenario 2: DOWNTREND TO UPTREND FLIP ZONE ---
    candles_flip_buy = [
        # Downtrend establish
        {"t": "11:00", "o": 108.0, "h": 108.2, "l": 106.0, "c": 106.2},
        {"t": "11:01", "o": 106.2, "h": 106.4, "l": 103.5, "c": 103.8},
        {"t": "11:02", "o": 103.8, "h": 104.0, "l": 101.0, "c": 101.2}, # Low 101.0 (cLL)
        {"t": "11:03", "o": 101.5, "h": 103.8, "l": 101.4, "c": 103.6}, # Bounce to 103.8 (Confirmed Lower High cLH = 103.8)
        {"t": "11:04", "o": 103.6, "h": 103.7, "l": 99.5, "c": 99.8},   # Downtrend low (cLL = 99.5, last_cLH = 103.8)

        # The Structural Breakout (Trend Flip)
        {"t": "11:05", "o": 100.0, "h": 102.5, "l": 99.8, "c": 102.2},
        {"t": "11:06", "o": 102.2, "h": 106.2, "l": 102.0, "c": 106.0}, # MASSIVE RALLY! Body closes (106.0) > last_cLH (103.8)!
        # At 11:06: TREND FLIPS TO UPTREND!
        # cLL (99.5) becomes Confirmed Higher Low (cHL = 99.5)
        # Breakout high (106.2) becomes Confirmed Higher High (cHH = 106.2)
        # BUY FLIP ZONE IS FORMED: [Top: 106.2, Bottom: 99.5, 50% Mid: 102.85]

        # Retracement to 50% Mid of the Flip Zone
        {"t": "11:07", "o": 106.0, "h": 106.1, "l": 103.5, "c": 103.8},
        {"t": "11:08", "o": 103.8, "h": 103.9, "l": 102.6, "c": 102.8}, # Retests 102.85 -> LIMIT BUY FILL @ 102.85!

        # Continuation Upward
        {"t": "11:09", "o": 102.8, "h": 107.0, "l": 102.7, "c": 106.8},
        {"t": "11:10", "o": 106.8, "h": 109.0, "l": 106.5, "c": 108.8}, # Target Hit @ 108.21 (+0.8R)!
    ]

    buy_flip_zone = {
        "name": "BUY FLIP ZONE (Downtrend ➔ Uptrend)",
        "side": "BUY",
        "c_time": "11:06", "fill_time": "11:08", "exit_time": "11:10",
        "top": 106.2, "bottom": 99.5, "mid": 102.85,
        "tp": 108.21, "sl": 96.15,
        "outcome": "WIN (+0.8R)",
        "desc": "Created on 11:06 when body closed above last_cLH (103.8). Top = cHH (106.2), Bottom = cHL (99.5). 50% Mid @ 102.85 filled at 11:08."
    }

    return candles_flip_sell, sell_flip_zone, candles_flip_buy, buy_flip_zone


def main():
    c_sell, z_sell, c_buy, z_buy = build_flip_scenarios()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DTFX Trend Flip Zones Visualizer</title>
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
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 20px; line-height: 1.5; }}
  
  .header {{ border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 20px; }}
  .title {{ font-size: 22px; font-weight: 700; color: #fff; }}
  
  .tab-bar {{ display: flex; gap: 10px; margin-bottom: 20px; }}
  .tab-btn {{ background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-secondary); padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 700; transition: all 0.2s; }}
  .tab-btn:hover {{ color: #fff; background: #1C2B44; }}
  .tab-btn.active {{ background: #1C2B44; color: var(--accent-blue); border-color: var(--accent-blue); box-shadow: 0 0 10px rgba(56, 139, 253, 0.3); }}
  
  .chart-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 25px; }}
  .chart-title {{ font-size: 18px; font-weight: 700; margin-bottom: 10px; }}
  
  .flow-card {{ background: #0E131E; border: 1px solid #253147; border-radius: 8px; padding: 15px; margin-bottom: 15px; font-size: 13px; }}
  
  #chartDiv {{ width: 100%; height: 600px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">🔄 DTFX Trend Flip Zones: Reversal Zone Generation Engine</div>
  <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
    Demonstrating how a <b>Trend Flip</b> immediately forms a confirmed <b>Flip Zone</b> between the new extreme and the broken structural level.
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" id="tabSell" onclick="switchTab('sell')">1. Uptrend ➔ Downtrend Flip (SELL FLIP ZONE)</button>
  <button class="tab-btn" id="tabBuy" onclick="switchTab('buy')">2. Downtrend ➔ Uptrend Flip (BUY FLIP ZONE)</button>
</div>

<div class="flow-card" id="flowCard"></div>

<div class="chart-card">
  <div class="chart-title" id="chartHeaderTitle"></div>
  <div id="chartDiv"></div>
</div>

<script>
const sellCandles = {json.dumps(c_sell)};
const sellZone = {json.dumps(z_sell)};
const buyCandles = {json.dumps(c_buy)};
const buyZone = {json.dumps(z_buy)};

let currentTab = 'sell';

function switchTab(tab) {{
  currentTab = tab;
  document.getElementById("tabSell").className = `tab-btn ${{tab === 'sell' ? 'active' : ''}}`;
  document.getElementById("tabBuy").className = `tab-btn ${{tab === 'buy' ? 'active' : ''}}`;
  renderView();
}}

function renderView() {{
  if (currentTab === 'sell') {{
    document.getElementById("chartHeaderTitle").innerHTML = "🔴 Uptrend ➔ Downtrend Flip: SELL FLIP ZONE Creation";
    document.getElementById("flowCard").innerHTML = `
      <div style="font-weight: 700; color: #fff; margin-bottom: 6px;">Flip Zone Anatomy (Uptrend ➔ Downtrend):</div>
      <div style="color: var(--text-secondary); margin-bottom: 6px;">
        1. <b>Uptrend Base</b>: Peak @ 108.5 (cHH) and last confirmed higher low floor @ 104.2 (last_cHL).<br>
        2. <b>🚨 The Trend Flip Trigger (10:06)</b>: Candle body dumps and closes @ 102.0 (< 104.2). Uptrend breaks ➔ <b>Flips to Downtrend!</b><br>
        3. <b>⚡ SELL FLIP ZONE Created</b>:
           • <b>Zone Top</b> = Confirmed Lower High <code>cLH = 108.5</code> (the peak that initiated the dump).<br>
           • <b>Zone Bottom</b> = Confirmed Lower Low <code>cLL = 101.8</code> (the breakdown low).<br>
        4. <b>Trade Execution</b>: At 10:08, price pulls back to the 50% Midpoint (105.15) ➔ <b>Limit Sell Filled</b> ➔ Target Hit @ 99.81 (+0.8R)!
      </div>
    `;

    const times = sellCandles.map(c => c.t);
    const opens = sellCandles.map(c => c.o);
    const highs = sellCandles.map(c => c.h);
    const lows = sellCandles.map(c => c.l);
    const closes = sellCandles.map(c => c.c);

    const candleTrace = {{
      x: times, open: opens, high: highs, low: lows, close: closes,
      type: 'candlestick', name: 'Price',
      increasing: {{ line: {{ color: '#26A69A' }} }},
      decreasing: {{ line: {{ color: '#EF5350' }} }},
      showlegend: false
    }};

    const zoneTrace = {{
      x: [sellZone.c_time, sellZone.exit_time, sellZone.exit_time, sellZone.c_time, sellZone.c_time],
      y: [sellZone.top, sellZone.top, sellZone.bottom, sellZone.bottom, sellZone.top],
      fill: 'toself',
      fillcolor: 'rgba(218, 54, 51, 0.15)',
      line: {{ color: '#DA3633', width: 2 }},
      name: sellZone.name, mode: 'lines'
    }};

    const midTrace = {{
      x: [sellZone.c_time, sellZone.exit_time],
      y: [sellZone.mid, sellZone.mid],
      mode: 'lines',
      line: {{ color: '#388BFD', width: 2, dash: 'dot' }},
      name: `50% Mid Limit Sell (${{sellZone.mid}})`
    }};

    const lastChlLine = {{
      x: ['10:03', '10:06'],
      y: [104.2, 104.2],
      mode: 'lines',
      line: {{ color: '#F1E05A', width: 2, dash: 'dash' }},
      name: 'last_cHL Floor (104.2)'
    }};

    const annotations = [
      {{
        x: '10:03', y: 104.2,
        text: '📍 last_cHL Floor (104.2)', showarrow: true, arrowhead: 2,
        ax: 0, ay: 30, bgcolor: '#2A2612', bordercolor: '#F1E05A', font: {{ color: '#F1E05A', size: 10 }}
      }},
      {{
        x: '10:04', y: 108.5,
        text: '📍 Peak (108.5) ➔ Becomes cLH', showarrow: true, arrowhead: 2,
        ax: 0, ay: -30, bgcolor: '#2A1719', bordercolor: '#DA3633', font: {{ color: '#fff', size: 10 }}
      }},
      {{
        x: '10:06', y: 102.0,
        text: '🚨 FLIP TO DOWNTREND (Close < 104.2)<br>⚡ SELL FLIP ZONE FORMED [108.5 - 101.8]', showarrow: true, arrowhead: 2,
        ax: -40, ay: 40, bgcolor: '#2A1719', bordercolor: '#DA3633', font: {{ color: '#F87171', size: 10, weight: 'bold' }}
      }},
      {{
        x: '10:08', y: 105.15,
        text: '🎯 Limit Sell Fill @ 105.15', showarrow: true, arrowhead: 2,
        ax: 30, ay: -30, bgcolor: '#1C2B44', bordercolor: '#388BFD', font: {{ color: '#fff', size: 10 }}
      }},
      {{
        x: '10:10', y: 99.81,
        text: '🏁 TP HIT (+0.8R)', showarrow: true, arrowhead: 2,
        ax: 0, ay: 30, bgcolor: '#12281D', bordercolor: '#238636', font: {{ color: '#4EBE71', size: 10, weight: 'bold' }}
      }}
    ];

    const layout = {{
      margin: {{ l: 50, r: 50, t: 30, b: 40 }},
      paper_bgcolor: '#111622', plot_bgcolor: '#111622',
      xaxis: {{ type: 'category', gridcolor: '#20293A', color: '#7E8C9F' }},
      yaxis: {{ gridcolor: '#20293A', color: '#7E8C9F', autorange: true }},
      annotations: annotations, showlegend: true,
      legend: {{ orientation: 'h', x: 0, y: 1.12, font: {{ color: '#7E8C9F', size: 10 }} }}
    }};

    Plotly.newPlot('chartDiv', [candleTrace, lastChlLine, zoneTrace, midTrace], layout, {{ responsive: true }});

  }} else {{
    // BUY FLIP TAB
    document.getElementById("chartHeaderTitle").innerHTML = "🟢 Downtrend ➔ Uptrend Flip: BUY FLIP ZONE Creation";
    document.getElementById("flowCard").innerHTML = `
      <div style="font-weight: 700; color: #fff; margin-bottom: 6px;">Flip Zone Anatomy (Downtrend ➔ Uptrend):</div>
      <div style="color: var(--text-secondary); margin-bottom: 6px;">
        1. <b>Downtrend Base</b>: Valley @ 99.5 (cLL) and last confirmed lower high ceiling @ 103.8 (last_cLH).<br>
        2. <b>🚨 The Trend Flip Trigger (11:06)</b>: Candle body explodes and closes @ 106.0 (> 103.8). Downtrend breaks ➔ <b>Flips to Uptrend!</b><br>
        3. <b>⚡ BUY FLIP ZONE Created</b>:
           • <b>Zone Top</b> = Confirmed Higher High <code>cHH = 106.2</code> (the breakout high).<br>
           • <b>Zone Bottom</b> = Confirmed Higher Low <code>cHL = 99.5</code> (the valley that launched the rally).<br>
        4. <b>Trade Execution</b>: At 11:08, price pulls back to the 50% Midpoint (102.85) ➔ <b>Limit Buy Filled</b> ➔ Target Hit @ 108.21 (+0.8R)!
      </div>
    `;

    const times = buyCandles.map(c => c.t);
    const opens = buyCandles.map(c => c.o);
    const highs = buyCandles.map(c => c.h);
    const lows = buyCandles.map(c => c.l);
    const closes = buyCandles.map(c => c.c);

    const candleTrace = {{
      x: times, open: opens, high: highs, low: lows, close: closes,
      type: 'candlestick', name: 'Price',
      increasing: {{ line: {{ color: '#26A69A' }} }},
      decreasing: {{ line: {{ color: '#EF5350' }} }},
      showlegend: false
    }};

    const zoneTrace = {{
      x: [buyZone.c_time, buyZone.exit_time, buyZone.exit_time, buyZone.c_time, buyZone.c_time],
      y: [buyZone.top, buyZone.top, buyZone.bottom, buyZone.bottom, buyZone.top],
      fill: 'toself',
      fillcolor: 'rgba(56, 139, 253, 0.15)',
      line: {{ color: '#388BFD', width: 2 }},
      name: buyZone.name, mode: 'lines'
    }};

    const midTrace = {{
      x: [buyZone.c_time, buyZone.exit_time],
      y: [buyZone.mid, buyZone.mid],
      mode: 'lines',
      line: {{ color: '#388BFD', width: 2, dash: 'dot' }},
      name: `50% Mid Limit Buy (${{buyZone.mid}})`
    }};

    const lastClhLine = {{
      x: ['11:03', '11:06'],
      y: [103.8, 103.8],
      mode: 'lines',
      line: {{ color: '#F1E05A', width: 2, dash: 'dash' }},
      name: 'last_cLH Ceiling (103.8)'
    }};

    const annotations = [
      {{
        x: '11:03', y: 103.8,
        text: '📍 last_cLH Ceiling (103.8)', showarrow: true, arrowhead: 2,
        ax: 0, ay: -30, bgcolor: '#2A2612', bordercolor: '#F1E05A', font: {{ color: '#F1E05A', size: 10 }}
      }},
      {{
        x: '11:04', y: 99.5,
        text: '📍 Valley (99.5) ➔ Becomes cHL', showarrow: true, arrowhead: 2,
        ax: 0, ay: 30, bgcolor: '#1C2B44', bordercolor: '#388BFD', font: {{ color: '#fff', size: 10 }}
      }},
      {{
        x: '11:06', y: 106.0,
        text: '🚨 FLIP TO UPTREND (Close > 103.8)<br>⚡ BUY FLIP ZONE FORMED [106.2 - 99.5]', showarrow: true, arrowhead: 2,
        ax: -40, ay: -40, bgcolor: '#12281D', bordercolor: '#238636', font: {{ color: '#4EBE71', size: 10, weight: 'bold' }}
      }},
      {{
        x: '11:08', y: 102.85,
        text: '🎯 Limit Buy Fill @ 102.85', showarrow: true, arrowhead: 2,
        ax: -30, ay: 30, bgcolor: '#1C2B44', bordercolor: '#388BFD', font: {{ color: '#fff', size: 10 }}
      }},
      {{
        x: '11:10', y: 108.21,
        text: '🏁 TP HIT (+0.8R)', showarrow: true, arrowhead: 2,
        ax: 0, ay: -30, bgcolor: '#12281D', bordercolor: '#238636', font: {{ color: '#4EBE71', size: 10, weight: 'bold' }}
      }}
    ];

    const layout = {{
      margin: {{ l: 50, r: 50, t: 30, b: 40 }},
      paper_bgcolor: '#111622', plot_bgcolor: '#111622',
      xaxis: {{ type: 'category', gridcolor: '#20293A', color: '#7E8C9F' }},
      yaxis: {{ gridcolor: '#20293A', color: '#7E8C9F', autorange: true }},
      annotations: annotations, showlegend: true,
      legend: {{ orientation: 'h', x: 0, y: 1.12, font: {{ color: '#7E8C9F', size: 10 }} }}
    }};

    Plotly.newPlot('chartDiv', [candleTrace, lastClhLine, zoneTrace, midTrace], layout, {{ responsive: true }});
  }}
}}

renderView();
</script>
</body>
</html>
"""

    out_file = os.path.join(ARTIFACTS_DIR, "dtfx_trend_flip_zones_visualizer.html")
    with open(out_file, "w") as f:
        f.write(html_content)
    print(f"\nSaved Trend Flip Zones Visualizer to: {out_file}")


if __name__ == "__main__":
    main()
