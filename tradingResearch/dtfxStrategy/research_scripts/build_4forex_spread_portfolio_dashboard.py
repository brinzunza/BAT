import pandas as pd
import numpy as np
import json
import os

def build_4forex_spread_portfolio_dashboard():
    csv_path = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_4forex_portfolio_spread_trades.csv"
    if not os.path.exists(csv_path):
        print("CSV not found!")
        return
        
    df = pd.read_csv(csv_path)
    df['fill_time_dt'] = pd.to_datetime(df['fill_time'])
    df = df.sort_values('fill_time_dt').reset_index(drop=True)
    df['cum_r'] = df['r_mult'].cumsum()
    
    # Portfolio equity sampled
    portfolio_equity = {
        "x": df['fill_time_dt'].dt.strftime('%Y-%m-%d %H:%M').tolist()[::15],
        "y": [round(val, 1) for val in df['cum_r'].tolist()[::15]]
    }
    
    # Ticker equity curves
    ticker_curves = {}
    ticker_stats = []
    for ticker, group in df.groupby('ticker'):
        sub = group.copy().reset_index(drop=True)
        sub['cum_r'] = sub['r_mult'].cumsum()
        ticker_curves[ticker] = {
            "x": sub['fill_time_dt'].dt.strftime('%Y-%m-%d %H:%M').tolist()[::10],
            "y": [round(val, 1) for val in sub['cum_r'].tolist()[::10]]
        }
        
        n_t = len(group)
        wins = sum(group['result'] == 'WIN')
        losses = sum(group['result'] == 'LOSS')
        wr = round((wins / n_t) * 100.0, 1)
        net_r = round(group['r_mult'].sum(), 1)
        gross_w = sum(group[group['result'] == 'WIN']['r_mult'])
        gross_l = abs(sum(group[group['result'] == 'LOSS']['r_mult']))
        pf = round(gross_w / max(0.001, gross_l), 2)
        avg_rr = round(group['raw_rr'].mean(), 2)
        avg_dur = round(group['duration_bars'].mean(), 1)
        
        ticker_stats.append({
            "ticker": ticker,
            "trades": n_t,
            "wins": wins,
            "losses": losses,
            "win_rate": wr,
            "avg_rr": avg_rr,
            "net_r": net_r,
            "pf": pf,
            "avg_dur": avg_dur
        })
        
    # Annual breakdown
    df['year'] = df['fill_time_dt'].dt.year
    annual_stats = []
    for yr, group in df.groupby('year'):
        n_t = len(group)
        wins = sum(group['result'] == 'WIN')
        losses = sum(group['result'] == 'LOSS')
        wr = round((wins / n_t) * 100.0, 1)
        net_r = round(group['r_mult'].sum(), 1)
        gross_w = sum(group[group['result'] == 'WIN']['r_mult'])
        gross_l = abs(sum(group[group['result'] == 'LOSS']['r_mult']))
        pf = round(gross_w / max(0.001, gross_l), 2)
        annual_stats.append({
            "year": int(yr),
            "trades": n_t,
            "wins": wins,
            "losses": losses,
            "win_rate": wr,
            "net_r": net_r,
            "pf": pf
        })

    total_trades = len(df)
    total_wins = sum(df['result'] == 'WIN')
    total_losses = sum(df['result'] == 'LOSS')
    overall_wr = round((total_wins / total_trades) * 100.0, 2)
    overall_net_r = round(df['cum_r'].iloc[-1], 1)
    gross_w = sum(df[df['result'] == 'WIN']['r_mult'])
    gross_l = abs(sum(df[df['result'] == 'LOSS']['r_mult']))
    overall_pf = round(gross_w / max(0.001, gross_l), 2)
    expectancy = round(overall_net_r / total_trades, 3)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEID 4-Forex Portfolio Backtest (1.6 Pips Spread)</title>
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
  
  .rules-banner {{ background: #0E1626; border: 1px solid #1C2B44; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; font-size: 13.5px; line-height: 1.7; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; margin-right: 6px; }}
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 16px; }}
  .metric-label {{ font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
  .metric-val {{ font-size: 22px; font-weight: 700; }}

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
  
  #equityDiv {{ width: 100%; height: 420px; }}
  #tickerDiv {{ width: 100%; height: 420px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">⚡ BEID 4-Forex Portfolio: Spread-Adjusted Backtest (1.6 Pips Spread)</div>
  <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 6px;">
    Portfolio backtest across <b>EUR/USD, EUR/GBP, GBP/USD, and USD/JPY</b> on 1-minute tick data with exact spread execution adjustments.
  </div>
</div>

<div class="rules-banner">
  <div style="font-weight: 700; color: #fff; margin-bottom: 6px;">📐 Spread-Adjusted Execution Rules (Spread = 1.6 pips = 0.00016):</div>
  <div style="color: #D1D5DB;">
    • <b>BUY ZONE</b>: Limit Entry @ <code>50% + 0.8 pips</code> | TP @ <code>Top - 0.8 pips</code> | SL @ <code>Bottom + 0.8 pips</code><br>
    • <b>SELL ZONE</b>: Limit Entry @ <code>50% - 0.8 pips</code> | TP @ <code>Bottom + 0.8 pips</code> | SL @ <code>Top - 0.8 pips</code><br>
    • <span class="tag" style="background: rgba(35, 134, 54, 0.2); color: #4ADE80; border: 1px solid #238636;">Risk Symmetry</span> Risk distance is exactly $0.5 \\times \\text{{Height}}$ | Reward distance is $0.5 \\times \\text{{Height}} - 1.6\\text{{ pips}}$.
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Total Portfolio Trades</div>
    <div class="metric-val" style="color: #fff;">{total_trades:,} Trades</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">4 Forex Tickers (1M Bars)</div>
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
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">After 1.6 pips spread on every trade</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Expectancy / Trade</div>
    <div class="metric-val" style="color: #FDE047;">+{expectancy:+.3f} R</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Solid Net Edge per Trade</div>
  </div>
</div>

<div class="chart-card">
  <div class="chart-title">📈 Combined Portfolio Equity Curve (+391.3 R across 2,439 Trades)</div>
  <div id="equityDiv"></div>
</div>

<div class="chart-card">
  <div class="chart-title">📊 Individual Equity Curves by Currency Pair (EUR/GBP, EUR/USD, GBP/USD, USD/JPY)</div>
  <div id="tickerDiv"></div>
</div>

<div class="tables-grid">
  <div class="table-card">
    <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📊 Performance by Currency Pair (Min Zone: 5.0 pips)</div>
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Trades</th>
          <th>Wins</th>
          <th>Losses</th>
          <th>Win Rate</th>
          <th>Avg RR</th>
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
          <td style="font-weight: 700; color: {("#4ADE80" if r["win_rate"] >= 65 else ("#FDE047" if r["win_rate"] >= 50 else "#F87171"))};">{r["win_rate"]}%</td>
          <td style="color: #60A5FA; font-weight: 600;">{r["avg_rr"]:.2f}</td>
          <td style="font-weight: 700; color: {("#4ADE80" if r["net_r"] > 0 else "#F87171")};">{r["net_r"]:+.1f} R</td>
          <td style="font-weight: 600; color: #fff;">{r["pf"]}</td>
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
          <td style="font-weight: 700; color: {("#4ADE80" if r["win_rate"] >= 65 else ("#FDE047" if r["win_rate"] >= 50 else "#F87171"))};">{r["win_rate"]}%</td>
          <td style="font-weight: 700; color: {("#4ADE80" if r["net_r"] > 0 else "#F87171")};">{r["net_r"]:+.1f} R</td>
          <td style="font-weight: 600; color: #fff;">{r["pf"]}</td>
        </tr>
        ''' for r in annual_stats])}
      </tbody>
    </table>
  </div>
</div>

<script>
const portData = {json.dumps(portfolio_equity)};
const tickerData = {json.dumps(ticker_curves)};

// 1. Portfolio Curve
const eqTrace = {{
  x: portData.x,
  y: portData.y,
  type: 'scatter', mode: 'lines',
  line: {{ color: '#4ADE80', width: 3 }},
  fill: 'tozeroy',
  fillcolor: 'rgba(35, 134, 54, 0.12)',
  name: 'Combined 4-Forex Portfolio'
}};

const eqLayout = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Net Return (R)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  margin: {{ l: 60, r: 40, t: 20, b: 40 }}
}};
Plotly.newPlot('equityDiv', [eqTrace], eqLayout, {{ responsive: true }});

// 2. Individual Curves
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
    out_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/beid_4forex_portfolio_spread_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
        
    repo_out = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_4forex_portfolio_spread_dashboard.html"
    with open(repo_out, "w") as f:
        f.write(html)
        
    print("4 Forex Portfolio Spread Dashboard generated successfully.")

if __name__ == "__main__":
    build_4forex_spread_portfolio_dashboard()
