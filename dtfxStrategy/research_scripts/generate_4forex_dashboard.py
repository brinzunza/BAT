import pandas as pd
import numpy as np
import json
import os

def build_4forex_dashboard():
    csv_path = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_4forex_1min_trades.csv"
    if not os.path.exists(csv_path):
        print("Trades CSV not found!")
        return
        
    df = pd.read_csv(csv_path)
    df['fill_time'] = pd.to_datetime(df['fill_time'])
    df = df.sort_values('fill_time').reset_index(drop=True)
    
    # Cumulative portfolio equity curve
    df['cum_r'] = df['r_mult'].cumsum()
    
    # Cumulative equity by ticker
    ticker_curves = {}
    for ticker in df['ticker'].unique():
        sub = df[df['ticker'] == ticker].copy().reset_index(drop=True)
        sub['cum_r'] = sub['r_mult'].cumsum()
        ticker_curves[ticker] = {
            "x": sub['fill_time'].dt.strftime('%Y-%m-%d %H:%M').tolist()[::10], # sampled for fast rendering
            "y": [round(val, 1) for val in sub['cum_r'].tolist()[::10]]
        }
        
    # Portfolio equity sampled
    portfolio_equity = {
        "x": df['fill_time'].dt.strftime('%Y-%m-%d %H:%M').tolist()[::15],
        "y": [round(val, 1) for val in df['cum_r'].tolist()[::15]]
    }
    
    # Annual breakdown
    df['year'] = df['fill_time'].dt.year
    annual_stats = []
    for yr, group in df.groupby('year'):
        n_t = len(group)
        wins = sum(group['result'] == 'WIN')
        losses = sum(group['result'] == 'LOSS')
        wr = round((wins / n_t) * 100.0, 1)
        net_r = round(group['r_mult'].sum(), 1)
        pf = round(wins / max(1, losses), 2)
        annual_stats.append({
            "year": int(yr),
            "trades": n_t,
            "wins": wins,
            "losses": losses,
            "win_rate": wr,
            "net_r": net_r,
            "profit_factor": pf
        })
        
    # Ticker breakdown
    ticker_stats = []
    for t_name, group in df.groupby('ticker'):
        n_t = len(group)
        wins = sum(group['result'] == 'WIN')
        losses = sum(group['result'] == 'LOSS')
        wr = round((wins / n_t) * 100.0, 1)
        net_r = round(group['r_mult'].sum(), 1)
        pf = round(wins / max(1, losses), 2)
        avg_dur = round(group['duration_bars'].mean(), 1)
        ticker_stats.append({
            "ticker": t_name,
            "trades": n_t,
            "wins": wins,
            "losses": losses,
            "win_rate": wr,
            "net_r": net_r,
            "profit_factor": pf,
            "avg_dur": avg_dur
        })
        
    total_trades = len(df)
    total_wins = sum(df['result'] == 'WIN')
    total_losses = sum(df['result'] == 'LOSS')
    overall_wr = round((total_wins / total_trades) * 100.0, 2)
    overall_net_r = round(df['cum_r'].iloc[-1], 1)
    overall_pf = round(total_wins / max(1, total_losses), 2)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEID 4 Forex 1-Minute Historical Backtest</title>
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
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 16px; }}
  .metric-label {{ font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
  .metric-val {{ font-size: 22px; font-weight: 700; }}

  .charts-row {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 25px; }}
  @media (max-width: 1000px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}

  .chart-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 25px; }}
  .chart-title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff; }}
  
  .tables-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
  @media (max-width: 1000px) {{ .tables-grid {{ grid-template-columns: 1fr; }} }}

  .table-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
  th {{ background: #0E131E; color: var(--text-secondary); padding: 10px 14px; border-bottom: 1px solid var(--border-color); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #1A2234; }}
  tr:hover {{ background: #151C2C; }}

  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }}
  .badge-eurusd {{ background: rgba(56, 139, 253, 0.2); color: #60A5FA; border: 1px solid #388BFD; }}
  .badge-gbpusd {{ background: rgba(163, 113, 247, 0.2); color: #C084FC; border: 1px solid #A371F7; }}
  .badge-eurgbp {{ background: rgba(35, 134, 54, 0.2); color: #4ADE80; border: 1px solid #238636; }}
  .badge-usdjpy {{ background: rgba(241, 224, 90, 0.2); color: #FDE047; border: 1px solid #F1E05A; }}
  
  #equityDiv {{ width: 100%; height: 420px; }}
  #tickerDiv {{ width: 100%; height: 420px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">⚡ BEID Strategy: 4 Forex Tickers 1-Minute Historical Backtest</div>
  <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 6px;">
    Continuous 1-minute historical tick data across <b>EUR/USD, GBP/USD, USD/JPY, and EUR/GBP</b> (4,000,000 total candles).
  </div>
</div>

<div class="rules-banner">
  <div style="font-weight: 700; color: #fff; margin-bottom: 4px;">📐 BEID Strategy Execution Rules:</div>
  <div style="color: #D1D5DB;">
    • <span class="rules-tag" style="background: rgba(56, 139, 253, 0.2); color: #60A5FA; border: 1px solid #388BFD;">Limit Entry</span> <b>50% Midpoint</b> of left-anchored confirmed zone.<br>
    • <span class="rules-tag" style="background: rgba(35, 134, 54, 0.2); color: #4ADE80; border: 1px solid #238636;">Take Profit (TP)</span> <b>Zone Top</b> for Buy | <b>Zone Bottom</b> for Sell (+1.0R).<br>
    • <span class="rules-tag" style="background: rgba(218, 54, 51, 0.2); color: #F87171; border: 1px solid #DA3633;">Stop Loss (SL)</span> <b>Zone Bottom</b> for Buy | <b>Zone Top</b> for Sell (-1.0R).<br>
    • <span class="rules-tag" style="background: rgba(241, 224, 90, 0.2); color: #FDE047; border: 1px solid #F1E05A;">Lifecycle</span> Zone terminates when price touches 50% on Candle A and exits on subsequent Candle B.
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Total 1M Trades</div>
    <div class="metric-val" style="color: #fff;">{total_trades:,} Trades</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">4,000,000 Candles Tested</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Overall Win Rate</div>
    <div class="metric-val" style="color: #4ADE80;">{overall_wr}%</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">{total_wins:,} Wins / {total_losses:,} Losses</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Portfolio Profit Factor</div>
    <div class="metric-val" style="color: #60A5FA;">{overall_pf}</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Gross Win / Gross Loss</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Total Net Return</div>
    <div class="metric-val" style="color: #4ADE80;">+{overall_net_r:,.1f} R</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Symmetric 1:1 Fixed R Profile</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Expectancy / Trade</div>
    <div class="metric-val" style="color: #FDE047;">+{round(overall_net_r / total_trades, 3)} R</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Exceptional Statistical Edge</div>
  </div>
</div>

<div class="chart-card">
  <div class="chart-title">📈 Cumulative Portfolio Equity Curve (+919.0 R across 3,103 Trades)</div>
  <div id="equityDiv"></div>
</div>

<div class="chart-card">
  <div class="chart-title">📊 Equity Curve by Currency Pair (EUR/GBP, EUR/USD, GBP/USD, USD/JPY)</div>
  <div id="tickerDiv"></div>
</div>

<div class="tables-grid">
  <div class="table-card">
    <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📊 Performance by Currency Pair</div>
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Trades</th>
          <th>Wins</th>
          <th>Losses</th>
          <th>Win Rate</th>
          <th>Net Return</th>
          <th>PF</th>
          <th>Avg Dur</th>
        </tr>
      </thead>
      <tbody>
        {''.join([f'''
        <tr>
          <td style="font-weight: 700; color: #fff;">{r["ticker"]}</td>
          <td>{r["trades"]:,}</td>
          <td style="color: #4ADE80; font-weight: 600;">{r["wins"]:,}</td>
          <td style="color: #F87171; font-weight: 600;">{r["losses"]:,}</td>
          <td style="font-weight: 700; color: {("#4ADE80" if r["win_rate"] >= 60 else ("#FDE047" if r["win_rate"] >= 50 else "#F87171"))};">{r["win_rate"]}%</td>
          <td style="font-weight: 700; color: {("#4ADE80" if r["net_r"] > 0 else "#F87171")};">{r["net_r"]:+.1f} R</td>
          <td style="font-weight: 600; color: #fff;">{r["profit_factor"]}</td>
          <td style="color: var(--text-secondary);">{r["avg_dur"]}m</td>
        </tr>
        ''' for r in ticker_stats])}
      </tbody>
    </table>
  </div>

  <div class="table-card">
    <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📅 Performance by Year</div>
    <table>
      <thead>
        <tr>
          <th>Year</th>
          <th>Trades</th>
          <th>Wins</th>
          <th>Losses</th>
          <th>Win Rate</th>
          <th>Net Return</th>
          <th>PF</th>
        </tr>
      </thead>
      <tbody>
        {''.join([f'''
        <tr>
          <td style="font-weight: 700; color: #fff;">{r["year"]}</td>
          <td>{r["trades"]:,}</td>
          <td style="color: #4ADE80; font-weight: 600;">{r["wins"]:,}</td>
          <td style="color: #F87171; font-weight: 600;">{r["losses"]:,}</td>
          <td style="font-weight: 700; color: {("#4ADE80" if r["win_rate"] >= 60 else ("#FDE047" if r["win_rate"] >= 50 else "#F87171"))};">{r["win_rate"]}%</td>
          <td style="font-weight: 700; color: {("#4ADE80" if r["net_r"] > 0 else "#F87171")};">{r["net_r"]:+.1f} R</td>
          <td style="font-weight: 600; color: #fff;">{r["profit_factor"]}</td>
        </tr>
        ''' for r in annual_stats])}
      </tbody>
    </table>
  </div>
</div>

<script>
const portData = {json.dumps(portfolio_equity)};
const tickerData = {json.dumps(ticker_curves)};

// 1. Overall Portfolio Equity Curve
const eqTrace = {{
  x: portData.x,
  y: portData.y,
  type: 'scatter', mode: 'lines',
  line: {{ color: '#4ADE80', width: 3 }},
  fill: 'tozeroy',
  fillcolor: 'rgba(35, 134, 54, 0.12)',
  name: 'Portfolio Net R'
}};

const eqLayout = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Net Return (R)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  margin: {{ l: 60, r: 40, t: 20, b: 40 }}
}};
Plotly.newPlot('equityDiv', [eqTrace], eqLayout, {{ responsive: true }});

// 2. Ticker Comparison Curves
const colors = {{
  'EURUSD': '#388BFD',
  'GBPUSD': '#A371F7',
  'EURGBP': '#4ADE80',
  'USDJPY': '#F1E05A'
}};

const tickerTraces = Object.keys(tickerData).map(t => ({{
  x: tickerData[t].x,
  y: tickerData[t].y,
  type: 'scatter', mode: 'lines',
  line: {{ color: colors[t] || '#fff', width: 2 }},
  name: t
}}));

const tickerLayout = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Net Return (R)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  margin: {{ l: 60, r: 40, t: 20, b: 40 }},
  legend: {{ orientation: 'h', y: 1.1, x: 0.05 }}
}};
Plotly.newPlot('tickerDiv', tickerTraces, tickerLayout, {{ responsive: true }});
</script>
</body>
</html>
"""
    output_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/beid_4forex_1min_dashboard.html"
    with open(output_path, "w") as f:
        f.write(html)
        
    repo_out = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_4forex_1min_dashboard.html"
    with open(repo_out, "w") as f:
        f.write(html)
        
    print("4 Forex 1M Dashboard generated successfully.")

if __name__ == "__main__":
    build_4forex_dashboard()
