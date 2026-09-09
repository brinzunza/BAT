import os
import pandas as pd
import numpy as np
import json
from backtest_beid_strategy import parse_candle_data, run_beid_state_machine_and_backtest

def run_eurusd_2026_simulation():
    csv_path = "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv"
    if not os.path.exists(csv_path):
        print("Data file not found!")
        return

    print("Loading EURUSD master 1-minute historical data...")
    df = pd.read_csv(csv_path)
    # Check date column
    date_col = 'date' if 'date' in df.columns else 'time'
    df['date_dt'] = pd.to_datetime(df[date_col])
    
    # Filter strictly for year 2026
    df_2026 = df[df['date_dt'].dt.year == 2026].copy().reset_index(drop=True)
    print(f"Found {len(df_2026):,} 1-minute candles in year 2026 for EUR/USD ({df_2026['date_dt'].min()} to {df_2026['date_dt'].max()}).")
    
    candles = []
    for idx, row in df_2026.iterrows():
        candles.append({
            "idx": idx,
            "t": str(row[date_col]),
            "o": float(row['open']),
            "h": float(row['high']),
            "l": float(row['low']),
            "c": float(row['close'])
        })
        
    zones, trades = run_beid_state_machine_and_backtest(candles)
    print(f"Generated {len(trades)} trades in 2026 on EUR/USD.")
    
    if len(trades) == 0:
        print("No trades found in 2026!")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['fill_time'] = pd.to_datetime(trades_df['fill_time'])
    trades_df = trades_df.sort_values('fill_time').reset_index(drop=True)
    
    # Let's simulate compounding growth of a $200 account across multiple risk sizing levels:
    # 1. Conservative (1% risk per trade)
    # 2. Moderate (2% risk per trade)
    # 3. Growth (3% risk per trade)
    # 4. Aggressive (5% risk per trade)
    # 5. Fixed Dollar (e.g. $10 risk per trade)
    
    risk_levels = [0.01, 0.02, 0.03, 0.05]
    sim_results = {}
    
    for r_pct in risk_levels:
        balance = 200.0
        peak = 200.0
        max_dd_pct = 0.0
        curve = [balance]
        dates = [trades_df['fill_time'].iloc[0].strftime('%Y-%m-%d %H:%M')]
        
        for idx, row in trades_df.iterrows():
            risk_amt = balance * r_pct
            pnl = risk_amt * row['r_mult']
            balance += pnl
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100.0
            if dd > max_dd_pct:
                max_dd_pct = dd
            curve.append(round(balance, 2))
            dates.append(row['exit_time'])
            
        sim_results[f"{int(r_pct*100)}% Risk"] = {
            "risk_pct": f"{int(r_pct*100)}%",
            "final_balance": round(balance, 2),
            "total_return_pct": round(((balance - 200.0) / 200.0) * 100.0, 1),
            "net_profit": round(balance - 200.0, 2),
            "max_dd_pct": round(max_dd_pct, 1),
            "curve": curve,
            "dates": dates
        }
        
    # Trade statistics
    wins = sum(trades_df['result'] == 'WIN')
    losses = sum(trades_df['result'] == 'LOSS')
    wr = round((wins / len(trades_df)) * 100.0, 1)
    net_r = round(trades_df['r_mult'].sum(), 1)
    pf = round(wins / max(1, losses), 2)
    
    # Monthly breakdown
    trades_df['month'] = trades_df['fill_time'].dt.strftime('%Y-%m')
    monthly_stats = []
    
    # Track 2% compound growth monthly
    b_track = 200.0
    for m, group in trades_df.groupby('month'):
        m_wins = sum(group['result'] == 'WIN')
        m_losses = sum(group['result'] == 'LOSS')
        m_wr = round((m_wins / len(group)) * 100.0, 1)
        m_r = round(group['r_mult'].sum(), 1)
        m_start_b = b_track
        for _, r in group.iterrows():
            b_track += b_track * 0.02 * r['r_mult']
        m_gain_pct = round(((b_track - m_start_b) / m_start_b) * 100.0, 1)
        
        monthly_stats.append({
            "month": m,
            "trades": len(group),
            "wins": m_wins,
            "losses": m_losses,
            "win_rate": m_wr,
            "net_r": m_r,
            "end_balance_2pct": round(b_track, 2),
            "monthly_gain_pct": m_gain_pct
        })
        
    print("\n" + "="*70)
    print(f"EUR/USD 2026 BEID STRATEGY PERFORMANCE ($200 Starting Account):")
    print(f"Total Trades: {len(trades_df)} | Win Rate: {wr}% ({wins}W / {losses}L)")
    print(f"Net R Return: {net_r:+.1f} R | Profit Factor: {pf}")
    print("-"*70)
    for k, v in sim_results.items():
        print(f"[{k}] Final: ${v['final_balance']:,.2f} | Return: +{v['total_return_pct']}% | Net Profit: +${v['net_profit']:,.2f} | Max DD: {v['max_dd_pct']}%")
    print("="*70)
    
    # Generate visual dashboard HTML
    build_2026_growth_dashboard(sim_results, monthly_stats, trades_df, wr, net_r, pf)

def build_2026_growth_dashboard(sim_results, monthly_stats, trades_df, wr, net_r, pf):
    dates_sample = sim_results["2% Risk"]["dates"][::5]
    curve_1pct = sim_results["1% Risk"]["curve"][::5]
    curve_2pct = sim_results["2% Risk"]["curve"][::5]
    curve_3pct = sim_results["3% Risk"]["curve"][::5]
    curve_5pct = sim_results["5% Risk"]["curve"][::5]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEID Strategy: $200 Account Growth (2026 EUR/USD 1M)</title>
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
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 15px; margin-bottom: 25px; }}
  .metric-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 16px; }}
  .metric-label {{ font-size: 11.5px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
  .metric-val {{ font-size: 24px; font-weight: 700; }}

  .chart-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 25px; }}
  .chart-title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff; }}
  
  .table-card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
  th {{ background: #0E131E; color: var(--text-secondary); padding: 10px 14px; border-bottom: 1px solid var(--border-color); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #1A2234; }}
  tr:hover {{ background: #151C2C; }}

  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }}
  
  #growthDiv {{ width: 100%; height: 480px; }}
</style>
</head>
<body>

<div class="header">
  <div class="title">💰 $200 Account Growth Simulation: 2026 EUR/USD BEID Strategy</div>
  <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 6px;">
    Compounding progression of a <b>$200 starting account</b> trading <b>EUR/USD 1-Minute BEID strategy</b> (Entry @ 50% Mid, TP @ Zone Top/Bottom, SL @ Zone Bottom/Top, 1:1 RR) throughout the entire year of 2026.
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">Starting Capital</div>
    <div class="metric-val" style="color: #60A5FA;">$200.00</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Initial Account Size</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">2026 Trades & Win Rate</div>
    <div class="metric-val" style="color: #fff;">{len(trades_df)} Trades ({wr}%)</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">{sum(trades_df['result'] == 'WIN')} Wins / {sum(trades_df['result'] == 'LOSS')} Losses</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Net R Generated</div>
    <div class="metric-val" style="color: #4ADE80;">+{net_r:,.1f} R</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">Profit Factor: {pf}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Final Equity (2% Risk)</div>
    <div class="metric-val" style="color: #4ADE80;">${sim_results['2% Risk']['final_balance']:,.2f}</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">+{sim_results['2% Risk']['total_return_pct']}% Return</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Final Equity (5% Risk)</div>
    <div class="metric-val" style="color: #FDE047;">${sim_results['5% Risk']['final_balance']:,.2f}</div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">+{sim_results['5% Risk']['total_return_pct']}% Return</div>
  </div>
</div>

<div class="chart-card">
  <div class="chart-title">📈 $200 Account Equity Compounding Curves in 2026 (1%, 2%, 3%, and 5% Risk per Trade)</div>
  <div id="growthDiv"></div>
</div>

<div class="table-card">
  <div style="font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff;">📅 Month-by-Month Compounding Ledger (2% Risk Baseline)</div>
  <table>
    <thead>
      <tr>
        <th>Month</th>
        <th>Trades</th>
        <th>Wins</th>
        <th>Losses</th>
        <th>Win Rate</th>
        <th>Net Return (R)</th>
        <th>Monthly Gain (%)</th>
        <th>Account Balance ($)</th>
      </tr>
    </thead>
    <tbody>
      {''.join([f'''
      <tr>
        <td style="font-weight: 700; color: #fff;">{m["month"]}</td>
        <td>{m["trades"]}</td>
        <td style="color: #4ADE80; font-weight: 600;">{m["wins"]}</td>
        <td style="color: #F87171; font-weight: 600;">{m["losses"]}</td>
        <td style="font-weight: 700; color: {("#4ADE80" if m["win_rate"] >= 60 else ("#FDE047" if m["win_rate"] >= 50 else "#F87171"))};">{m["win_rate"]}%</td>
        <td style="font-weight: 700; color: {("#4ADE80" if m["net_r"] > 0 else "#F87171")};">{m["net_r"]:+.1f} R</td>
        <td style="font-weight: 700; color: {("#4ADE80" if m["monthly_gain_pct"] > 0 else "#F87171")};">{m["monthly_gain_pct"]:+.1f}%</td>
        <td style="font-weight: 700; color: #60A5FA;">${m["end_balance_2pct"]:,.2f}</td>
      </tr>
      ''' for m in monthly_stats])}
    </tbody>
  </table>
</div>

<script>
const dates = {json.dumps(dates_sample)};
const curve1 = {json.dumps(curve_1pct)};
const curve2 = {json.dumps(curve_2pct)};
const curve3 = {json.dumps(curve_3pct)};
const curve5 = {json.dumps(curve_5pct)};

const t1 = {{
  x: dates, y: curve1,
  type: 'scatter', mode: 'lines',
  name: '1% Risk (Conservative)',
  line: {{ color: '#388BFD', width: 2 }}
}};

const t2 = {{
  x: dates, y: curve2,
  type: 'scatter', mode: 'lines',
  name: '2% Risk (Recommended)',
  line: {{ color: '#4ADE80', width: 3 }}
}};

const t3 = {{
  x: dates, y: curve3,
  type: 'scatter', mode: 'lines',
  name: '3% Risk (Growth)',
  line: {{ color: '#F1E05A', width: 2.5 }}
}};

const t5 = {{
  x: dates, y: curve5,
  type: 'scatter', mode: 'lines',
  name: '5% Risk (Aggressive)',
  line: {{ color: '#FF7B72', width: 2.5 }}
}};

const layout = {{
  paper_bgcolor: '#111622', plot_bgcolor: '#080B10',
  xaxis: {{ gridcolor: '#1A2234', color: '#7E8C9F' }},
  yaxis: {{ title: 'Account Balance ($)', gridcolor: '#1A2234', color: '#7E8C9F' }},
  margin: {{ l: 70, r: 40, t: 20, b: 40 }},
  legend: {{ orientation: 'h', y: 1.1, x: 0.05 }}
}};

Plotly.newPlot('growthDiv', [t1, t2, t3, t5], layout, {{ responsive: true }});
</script>
</body>
</html>
"""
    out_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/eurusd_2026_200_growth_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
        
    repo_out = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/eurusd_2026_200_growth_dashboard.html"
    with open(repo_out, "w") as f:
        f.write(html)
        
    print("EUR/USD 2026 Growth Dashboard generated successfully.")

if __name__ == "__main__":
    run_eurusd_2026_simulation()
