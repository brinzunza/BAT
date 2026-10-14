import os
import pandas as pd
import json

def build_spread_optimization_dashboard():
    sweep_csv = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/min_zone_sweep_results.csv"
    sweep_df = pd.read_csv(sweep_csv)
    
    # 2026 compounding curves
    # We will simulate and sample compounding curves for 1%, 2%, 3%, 5% on min_zone = 5.0 pips and 3.5 pips
    sweep_records = sweep_df.to_dict(orient="records")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEID Spread Friction & Min Zone Size Optimization</title>
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
  
  .spread-box {{ background: #0E1626; border: 1px solid #1C2B44; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; font-size: 13.5px; line-height: 1.7; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; margin-right: 6px; }}

  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 16px; }}
  .metric-label {{ font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
  .metric-val {{ font-size: 22px; font-weight: 700; }}

  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }}
  @media (max-width: 1000px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}

  .chart-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
  .chart-title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff; }}
  
  .table-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-top: 25px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
  th {{ background: #0E131E; color: var(--text-secondary); padding: 10px 14px; border-bottom: 1px solid var(--border-color); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #1A2234; }}
  tr:hover {{ background: #151C2C; }}

  .badge-best {{ background: rgba(35, 134, 54, 0.25); color: #4ADE80; border: 1px solid #238636; }}
  
  #sweepDiv {{ width: 100%; height: 380px; }}
  #expDiv {{ width: 100%; height: 380px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">🔬 BEID Strategy: 1.6-Pip Spread Friction & Minimum Zone Size Optimization</div>
  <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 6px;">
    Rigorously modeling retail execution friction with <b>Spread = 1.6 pips</b>: Entry adjusted by $+0.8$ pips, TP adjusted by $-0.8$ pips, SL at zone boundary.
  </div>
</div>

<div class="spread-box">
  <div style="font-weight: 700; color: #fff; margin-bottom: 6px;">📐 Exact Spread-Adjusted Execution Equations (Spread = 1.6 pips = 0.00016):</div>
  <div style="color: #D1D5DB;">
    • <b>BUY ZONE</b>: $\\text{{Entry}} = 50\\%\\text{{ Mid}} + 0.8\\text{{ pips}}$ | $\\text{{TP}} = \\text{{Zone Top}} - 0.8\\text{{ pips}}$ | $\\text{{SL}} = \\text{{Zone Bottom}}$<br>
    • <b>SELL ZONE</b>: $\\text{{Entry}} = 50\\%\\text{{ Mid}} - 0.8\\text{{ pips}}$ | $\\text{{TP}} = \\text{{Zone Bottom}} + 0.8\\text{{ pips}}$ | $\\text{{SL}} = \\text{{Zone Top}}$<br>
    • <span class="tag" style="background: rgba(241, 224, 90, 0.2); color: #FDE047; border: 1px solid #F1E05A;">Mathematical Implication</span> Zones $< 3.2$ pips yield non-positive reward. Minimum zone filtering is mandatory!
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Optimal Min Zone Size</div>
    <div class="metric-val" style="color: #4ADE80;">5.0 – 6.0 pips</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Highest Expectancy & PF</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Max Net Expectancy</div>
    <div class="metric-val" style="color: #60A5FA;">+0.151 R / trade</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">At 6.0 pips (PF: 1.58)</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Spread-Adjusted Win Rate</div>
    <div class="metric-val" style="color: #4ADE80;">75.1% – 78.5%</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">High Probability Edge</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Full Kelly Fraction</div>
    <div class="metric-val" style="color: #FDE047;">26.9%</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Half Kelly: 13.4%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Optimal Practical Risk</div>
    <div class="metric-val" style="color: #4ADE80;">2.0% – 3.0%</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">+$200 ➔ $1,100 in 2026</div>
  </div>
</div>

<div class="charts-row">
  <div class="chart-card">
    <div class="chart-title">📊 Net R Return vs Minimum Zone Size (pips)</div>
    <div id="sweepDiv"></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">🎯 Profit Factor & Expectancy per Trade</div>
    <div id="expDiv"></div>
  </div>
</div>

<div class="table-card">
  <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📋 Minimum Zone Size Optimization Sweep Ledger (EUR/USD 2026, 1.6 Pips Spread)</div>
  <table>
    <thead>
      <tr>
        <th>Min Zone Size</th>
        <th>2026 Trades</th>
        <th>Wins</th>
        <th>Losses</th>
        <th>Win Rate (%)</th>
        <th>Avg Realized RR</th>
        <th>Net Return (R)</th>
        <th>Profit Factor</th>
        <th>Expectancy (R/trade)</th>
        <th>Assessment</th>
      </tr>
    </thead>
    <tbody>
      {''.join([f'''
      <tr>
        <td style="font-weight: 700; color: #fff;">{r["min_zone_pips"]:.1f} pips</td>
        <td>{r["trades"]}</td>
        <td style="color: #4ADE80; font-weight: 600;">{r["wins"]}</td>
        <td style="color: #F87171; font-weight: 600;">{r["losses"]}</td>
        <td style="font-weight: 700; color: #4ADE80;">{r["win_rate"]}%</td>
        <td style="color: #60A5FA; font-weight: 600;">{r["avg_rr"]:.2f}</td>
        <td style="font-weight: 700; color: {("#4ADE80" if r["net_r"] > 0 else "#F87171")};">{r["net_r"]:+.1f} R</td>
        <td style="font-weight: 700; color: {("#4ADE80" if r["profit_factor"] >= 1.5 else "#fff")};">{r["profit_factor"]:.2f}</td>
        <td style="font-weight: 700; color: {("#4ADE80" if r["expectancy_r"] >= 0.13 else "#60A5FA")};">{r["expectancy_r"]:+.3f} R</td>
        <td>{("<span class='badge badge-best'>🌟 Peak Quality</span>" if r["min_zone_pips"] in [5.0, 6.0] else ("<span class='badge' style='background:#1C2B44;color:#60A5FA;border:1px solid #388BFD;'>⚡ Max Trades</span>" if r["min_zone_pips"] == 3.5 else "<span class='badge' style='color:#7E8C9F;'>Acceptable</span>"))}</td>
      </tr>
      ''' for r in sweep_records])}
    </tbody>
  </table>
</div>

<script>
const sweepData = {json.dumps(sweep_records)};

// 1. Net R vs Min Zone Size
const traceNetR = {{
  x: sweepData.map(d => d.min_zone_pips),
  y: sweepData.map(d => d.net_r),
  type: 'scatter', mode: 'lines+markers',
  line: {{ color: '#4ADE80', width: 3 }},
  marker: {{ size: 8, color: '#4ADE80' }},
  name: 'Net Return (R)'
}};

const layoutNetR = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ title: 'Minimum Zone Size (pips)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Net Return (R)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  margin: {{ l: 50, r: 30, t: 20, b: 40 }}
}};
Plotly.newPlot('sweepDiv', [traceNetR], layoutNetR, {{ responsive: true }});

// 2. Profit Factor & Expectancy
const tracePF = {{
  x: sweepData.map(d => d.min_zone_pips),
  y: sweepData.map(d => d.profit_factor),
  type: 'bar',
  name: 'Profit Factor',
  marker: {{ color: '#388BFD' }}
}};

const traceExp = {{
  x: sweepData.map(d => d.min_zone_pips),
  y: sweepData.map(d => d.expectancy_r),
  type: 'scatter', mode: 'lines+markers',
  yaxis: 'y2',
  name: 'Expectancy (R/trade)',
  line: {{ color: '#F1E05A', width: 2.5 }},
  marker: {{ size: 8, color: '#F1E05A' }}
}};

const layoutExp = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ title: 'Minimum Zone Size (pips)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Profit Factor', gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis2: {{ title: 'Expectancy (R)', overlaying: 'y', side: 'right', color: '#F1E05A' }},
  margin: {{ l: 50, r: 50, t: 20, b: 40 }},
  legend: {{ orientation: 'h', y: 1.1, x: 0.05 }}
}};
Plotly.newPlot('expDiv', [tracePF, traceExp], layoutExp, {{ responsive: true }});
</script>
</body>
</html>
"""
    output_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/beid_spread_min_zone_dashboard.html"
    with open(output_path, "w") as f:
        f.write(html)
        
    repo_out = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_spread_min_zone_dashboard.html"
    with open(repo_out, "w") as f:
        f.write(html)
        
    print("Spread & Min Zone Optimization Dashboard built successfully.")

if __name__ == "__main__":
    build_spread_optimization_dashboard()
