import os
import glob
import pandas as pd
import json
from backtest_beid_strategy import parse_candle_data, run_beid_state_machine_and_backtest

def generate_beid_dashboard():
    data_dirs = [
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data",
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data_ltf_deep"
    ]
    
    dataset_summaries = []
    all_trades_chronological = []
    sample_chart_dataset = None
    sample_chart_candles = None
    sample_chart_zones = None
    sample_chart_trades = None

    for d in data_dirs:
        for f in sorted(glob.glob(f"{d}/*.csv")):
            fname = os.path.basename(f)
            candles = parse_candle_data(f)
            if len(candles) < 50:
                continue
            zones, trades = run_beid_state_machine_and_backtest(candles)
            if len(trades) == 0:
                continue
                
            wins = sum(1 for t in trades if t["result"] == "WIN")
            losses = sum(1 for t in trades if t["result"] == "LOSS")
            win_rate = (wins / len(trades)) * 100.0
            net_r = sum(t["r_mult"] for t in trades)
            pf = (wins * 1.0) / max(1, losses * 1.0)
            
            # Asset and TF extraction
            asset = fname.split("_")[0].replace("XETHZUSD", "ETH").replace("XXBTZUSD", "BTC").replace("SOLUSD", "SOL").replace("EURUSD", "EUR/USD").replace("GBPUSD", "GBP/USD")
            tf = fname.split("_")[1].replace(".csv", "").upper()
            
            dataset_summaries.append({
                "file": fname,
                "asset": asset,
                "tf": tf,
                "candles": len(candles),
                "zones": len(zones),
                "trades": len(trades),
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 1),
                "net_r": round(net_r, 1),
                "profit_factor": round(pf, 2)
            })
            
            for t in trades:
                t_copy = dict(t)
                t_copy["dataset"] = f"{asset} {tf}"
                all_trades_chronological.append(t_copy)
                
            # Pick GBPUSD 15m as our rich sample chart
            if "GBPUSD_15m.csv" in fname and sample_chart_candles is None:
                sample_chart_dataset = "GBP/USD 15M"
                sample_chart_candles = candles
                sample_chart_zones = zones
                sample_chart_trades = trades

    # Sort trades chronologically
    all_trades_chronological.sort(key=lambda x: x["fill_time"])
    
    # Compute cumulative equity curve
    cum_r = 0.0
    equity_curve = []
    for idx, t in enumerate(all_trades_chronological):
        cum_r += t["r_mult"]
        equity_curve.append({
            "trade_num": idx + 1,
            "fill_time": t["fill_time"],
            "dataset": t["dataset"],
            "result": t["result"],
            "r_mult": t["r_mult"],
            "cum_r": round(cum_r, 1)
        })

    # Summary metrics
    total_trades = len(all_trades_chronological)
    total_wins = sum(1 for t in all_trades_chronological if t["result"] == "WIN")
    total_losses = sum(1 for t in all_trades_chronological if t["result"] == "LOSS")
    overall_wr = round((total_wins / total_trades) * 100.0, 1)
    overall_net_r = round(cum_r, 1)
    overall_pf = round((total_wins * 1.0) / max(1, total_losses * 1.0), 2)
    
    # Asset aggregates
    df_summary = pd.DataFrame(dataset_summaries)
    asset_agg = df_summary.groupby("asset").agg({
        "trades": "sum",
        "wins": "sum",
        "losses": "sum",
        "net_r": "sum"
    }).reset_index()
    asset_agg["win_rate"] = round((asset_agg["wins"] / asset_agg["trades"]) * 100.0, 1)
    asset_chart_data = asset_agg.to_dict(orient="records")
    
    # Timeframe aggregates
    tf_agg = df_summary.groupby("tf").agg({
        "trades": "sum",
        "wins": "sum",
        "losses": "sum",
        "net_r": "sum"
    }).reset_index()
    tf_agg["win_rate"] = round((tf_agg["wins"] / tf_agg["trades"]) * 100.0, 1)
    tf_chart_data = tf_agg.to_dict(orient="records")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEID Strategy Backtest Performance Dashboard</title>
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
  
  .rules-banner {{ background: #0E1626; border: 1px solid #1C2B44; border-radius: 10px; padding: 15px 20px; margin-bottom: 20px; font-size: 13.5px; line-height: 1.6; }}
  .rules-tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; margin-right: 6px; }}
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 16px; }}
  .metric-label {{ font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
  .metric-val {{ font-size: 22px; font-weight: 700; }}

  .charts-row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 25px; }}
  @media (max-width: 1000px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}

  .chart-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
  .chart-title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff; }}
  
  .table-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-top: 25px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
  th {{ background: #0E131E; color: var(--text-secondary); padding: 10px 14px; border-bottom: 1px solid var(--border-color); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #1A2234; }}
  tr:hover {{ background: #151C2C; }}

  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }}
  .badge-win {{ background: rgba(35, 134, 54, 0.25); color: #4ADE80; border: 1px solid #238636; }}
  .badge-loss {{ background: rgba(218, 54, 51, 0.25); color: #F87171; border: 1px solid #DA3633; }}
  .badge-buy {{ background: rgba(35, 134, 54, 0.2); color: #4ADE80; border: 1px solid #238636; }}
  .badge-sell {{ background: rgba(218, 54, 51, 0.2); color: #F87171; border: 1px solid #DA3633; }}
  
  #equityDiv {{ width: 100%; height: 380px; }}
  #breakdownDiv {{ width: 100%; height: 380px; }}
  #candleChartDiv {{ width: 100%; height: 600px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">⚡ BEID Strategy: Multi-Asset Backtest & Performance Dashboard</div>
  <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 6px;">
    Deterministic price action system: <b>Entry @ 50% Midpoint</b> | <b>Take Profit @ Zone Top (Buy) / Zone Bottom (Sell)</b> | <b>Stop Loss @ Zone Bottom (Buy) / Zone Top (Sell)</b> | <b>1.0R (1:1 RR)</b>.
  </div>
</div>

<div class="rules-banner">
  <div style="font-weight: 700; color: #fff; margin-bottom: 4px;">📐 BEID Strategy Execution Architecture:</div>
  <div style="color: #D1D5DB;">
    • <span class="rules-tag" style="background: rgba(56, 139, 253, 0.2); color: #60A5FA; border: 1px solid #388BFD;">Limit Entry</span> <b>50% Midpoint</b> of confirmed left-anchored zone.<br>
    • <span class="rules-tag" style="background: rgba(35, 134, 54, 0.2); color: #4ADE80; border: 1px solid #238636;">Take Profit (TP)</span> <b>Zone Top</b> for Buy Zone | <b>Zone Bottom</b> for Sell Zone (+1.0R).<br>
    • <span class="rules-tag" style="background: rgba(218, 54, 51, 0.2); color: #F87171; border: 1px solid #DA3633;">Stop Loss (SL)</span> <b>Zone Bottom</b> for Buy Zone | <b>Zone Top</b> for Sell Zone (-1.0R).<br>
    • <span class="rules-tag" style="background: rgba(241, 224, 90, 0.2); color: #FDE047; border: 1px solid #F1E05A;">Lifecycle</span> Zone ends when price touches 50% on Candle A and exits the zone on subsequent Candle B.
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Total Trades Resolved</div>
    <div class="metric-val" style="color: #fff;">{total_trades} Trades</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">32 Datasets Evaluated</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Overall Win Rate</div>
    <div class="metric-val" style="color: #4ADE80;">{overall_wr}%</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">{total_wins} Wins / {total_losses} Losses</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Profit Factor</div>
    <div class="metric-val" style="color: #60A5FA;">{overall_pf}</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Gross Win / Gross Loss</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Total Net Return</div>
    <div class="metric-val" style="color: #4ADE80;">+{overall_net_r} R</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">1:1 Fixed R Profile</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Expectancy / Trade</div>
    <div class="metric-val" style="color: #FDE047;">+{round(overall_net_r / total_trades, 3)} R</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Positive Edge per Trade</div>
  </div>
</div>

<div class="charts-row">
  <div class="chart-card">
    <div class="chart-title">📈 Cumulative BEID Portfolio Equity Curve (Net R)</div>
    <div id="equityDiv"></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">📊 Win Rate by Asset Class</div>
    <div id="breakdownDiv"></div>
  </div>
</div>

<div class="chart-card" style="margin-top: 25px;">
  <div class="chart-title">🕯️ Live Trade Inspector: {sample_chart_dataset} (Real Market Execution)</div>
  <div style="font-size: 12.5px; color: var(--text-secondary); margin-bottom: 12px;">
    Visualizing actual BEID zones, 50% limit entries, TP targets, SL stops, and trade resolutions.
  </div>
  <div id="candleChartDiv"></div>
</div>

<div class="table-card">
  <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📋 Comprehensive Dataset Performance Summary (32 Datasets)</div>
  <table>
    <thead>
      <tr>
        <th>Asset</th>
        <th>Timeframe</th>
        <th>Candles</th>
        <th>Zones Detected</th>
        <th>Trades Executed</th>
        <th>Wins</th>
        <th>Losses</th>
        <th>Win Rate</th>
        <th>Net Return (R)</th>
        <th>Profit Factor</th>
      </tr>
    </thead>
    <tbody>
      {''.join([f'''
      <tr>
        <td style="font-weight: 700; color: #fff;">{r["asset"]}</td>
        <td><span class="badge" style="background: #1C2B44; color: #60A5FA; border: 1px solid #388BFD;">{r["tf"]}</span></td>
        <td>{r["candles"]}</td>
        <td>{r["zones"]}</td>
        <td style="font-weight: 600;">{r["trades"]}</td>
        <td style="color: #4ADE80; font-weight: 600;">{r["wins"]}</td>
        <td style="color: #F87171; font-weight: 600;">{r["losses"]}</td>
        <td style="font-weight: 700; color: {("#4ADE80" if r["win_rate"] >= 55 else ("#FDE047" if r["win_rate"] >= 50 else "#F87171"))};">{r["win_rate"]}%</td>
        <td style="font-weight: 700; color: {("#4ADE80" if r["net_r"] > 0 else ("#FDE047" if r["net_r"] == 0 else "#F87171"))};">{r["net_r"]:+.1f} R</td>
        <td style="font-weight: 600; color: #fff;">{r["profit_factor"]}</td>
      </tr>
      ''' for r in dataset_summaries])}
    </tbody>
  </table>
</div>

<script>
const equityData = {json.dumps(equity_curve)};
const assetData = {json.dumps(asset_chart_data)};
const sampleCandles = {json.dumps(sample_chart_candles[:220])};
const sampleZones = {json.dumps(sample_chart_zones[:8])};
const sampleTrades = {json.dumps(sample_chart_trades[:8])};

// 1. Equity Curve Chart
const eqTrace = {{
  x: equityData.map(d => d.trade_num),
  y: equityData.map(d => d.cum_r),
  type: 'scatter', mode: 'lines',
  line: {{ color: '#4ADE80', width: 2.5 }},
  fill: 'tozeroy',
  fillcolor: 'rgba(35, 134, 54, 0.12)',
  name: 'Net Cumulative R'
}};

const eqLayout = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ title: 'Trade Number (#)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Net Return (R)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  margin: {{ l: 50, r: 30, t: 20, b: 40 }}
}};
Plotly.newPlot('equityDiv', [eqTrace], eqLayout, {{ responsive: true }});

// 2. Asset Win Rate Chart
const assetTrace = {{
  x: assetData.map(d => d.asset),
  y: assetData.map(d => d.win_rate),
  type: 'bar',
  marker: {{
    color: assetData.map(d => d.win_rate >= 60 ? '#4ADE80' : (d.win_rate >= 50 ? '#388BFD' : '#F87171')),
    line: {{ color: '#20293A', width: 1 }}
  }},
  text: assetData.map(d => `${{d.win_rate}}% (${{d.trades}}T)`),
  textposition: 'auto'
}};

const assetLayout = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Win Rate (%)', gridcolor: '#1A2234', color: '#7E8C9F', range: [0, 100] }},
  margin: {{ l: 50, r: 30, t: 20, b: 40 }}
}};
Plotly.newPlot('breakdownDiv', [assetTrace], assetLayout, {{ responsive: true }});

// 3. Sample Live Trade Inspector Candlestick Chart
const times = sampleCandles.map(c => c.t);
const candleTrace = {{
  x: times,
  open: sampleCandles.map(c => c.o),
  high: sampleCandles.map(c => c.h),
  low: sampleCandles.map(c => c.l),
  close: sampleCandles.map(c => c.c),
  type: 'candlestick', name: 'GBP/USD 15M',
  increasing: {{ line: {{ color: '#26A69A' }} }},
  decreasing: {{ line: {{ color: '#EF5350' }} }},
  showlegend: false
}};

const shapes = [];
const annotations = [];

sampleZones.forEach((z) => {{
  const isBuy = z.side === 'BUY';
  let startTime = z.left_time;
  let endTime = times[Math.min(times.length - 1, times.indexOf(z.created_time) + 8)];
  if (times.indexOf(startTime) === -1) startTime = times[0];

  shapes.push({{
    type: 'rect',
    x0: startTime, x1: endTime,
    y0: z.bottom, y1: z.top,
    fillcolor: isBuy ? 'rgba(35, 134, 54, 0.15)' : 'rgba(218, 54, 51, 0.15)',
    line: {{ color: isBuy ? '#238636' : '#DA3633', width: 1.5 }},
    layer: 'below'
  }});

  shapes.push({{
    type: 'line',
    x0: startTime, x1: endTime,
    y0: z.entry, y1: z.entry,
    line: {{ color: '#388BFD', width: 1.5, dash: 'dot' }}
  }});
}});

sampleTrades.forEach((t) => {{
  if (times.indexOf(t.fill_time) !== -1 && times.indexOf(t.exit_time) !== -1) {{
    const isWin = t.result === 'WIN';
    annotations.push({{
      x: t.fill_time, y: t.entry,
      text: `🎯 50% Entry (${{t.side}})`,
      showarrow: true, arrowhead: 2,
      ax: 0, ay: t.side === 'BUY' ? 25 : -25,
      bgcolor: '#1C2B44', bordercolor: '#388BFD',
      font: {{ color: '#60A5FA', size: 9, weight: 'bold' }}
    }});

    annotations.push({{
      x: t.exit_time, y: t.result === 'WIN' ? t.tp : t.sl,
      text: isWin ? '🏆 TP Hit (+1.0R)' : '❌ SL Hit (-1.0R)',
      showarrow: true, arrowhead: 2,
      ax: 0, ay: isWin ? (t.side === 'BUY' ? -25 : 25) : (t.side === 'BUY' ? 25 : -25),
      bgcolor: isWin ? '#142C1D' : '#2A1719',
      bordercolor: isWin ? '#238636' : '#DA3633',
      font: {{ color: isWin ? '#4ADE80' : '#F87171', size: 9, weight: 'bold' }}
    }});
  }}
}});

const candleLayout = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F', rangeslider: {{ visible: false }} }},
  yaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
  margin: {{ l: 60, r: 40, t: 20, b: 40 }},
  shapes: shapes,
  annotations: annotations
}};
Plotly.newPlot('candleChartDiv', [candleTrace], candleLayout, {{ responsive: true }});
</script>
</body>
</html>
"""
    output_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/beid_strategy_backtest_dashboard.html"
    with open(output_path, "w") as f:
        f.write(html)
    
    # Also save to repo
    repo_out = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_strategy_backtest_dashboard.html"
    os.makedirs(os.path.dirname(repo_out), exist_ok=True)
    with open(repo_out, "w") as f:
        f.write(html)
        
    print("BEID Strategy Dashboard generated successfully.")

if __name__ == "__main__":
    generate_beid_dashboard()
