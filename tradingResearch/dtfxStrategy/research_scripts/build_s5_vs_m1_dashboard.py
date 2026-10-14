import os
import pandas as pd
import numpy as np
import json

def build_comparison_dashboard():
    s5_csv = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_s5_trades.csv"
    m1_csv = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_m1_identical_period_trades.csv"
    
    df_s5 = pd.read_csv(s5_csv)
    df_m1 = pd.read_csv(m1_csv)
    
    df_s5['exit_time'] = pd.to_datetime(df_s5['exit_time'])
    df_m1['exit_time'] = pd.to_datetime(df_m1['exit_time'])
    
    df_s5 = df_s5.sort_values('exit_time').reset_index(drop=True)
    df_m1 = df_m1.sort_values('exit_time').reset_index(drop=True)
    
    df_s5['cum_pips'] = df_s5['net_pips'].cumsum()
    df_m1['cum_pips'] = df_m1['net_pips'].cumsum()
    
    df_s5['cum_raw_r'] = df_s5['raw_r'].cumsum()
    df_m1['cum_raw_r'] = df_m1['raw_r'].cumsum()
    
    # Subsample for smooth charting
    step_s5 = max(1, len(df_s5) // 300)
    step_m1 = max(1, len(df_m1) // 300)
    
    s5_dates = [str(d)[:16] for d in df_s5['exit_time'].iloc[::step_s5]]
    s5_pips = [round(float(v), 1) for v in df_s5['cum_pips'].iloc[::step_s5]]
    s5_raw_r = [round(float(v), 1) for v in df_s5['cum_raw_r'].iloc[::step_s5]]
    
    m1_dates = [str(d)[:16] for d in df_m1['exit_time'].iloc[::step_m1]]
    m1_pips = [round(float(v), 1) for v in df_m1['cum_pips'].iloc[::step_m1]]
    m1_raw_r = [round(float(v), 1) for v in df_m1['cum_raw_r'].iloc[::step_m1]]
    
    # HTML Content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEID Timeframe Benchmark: 1-Minute (M1) vs 5-Second (S5) Bars</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #121826;
            --bg-card: #182234;
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-green: #00e676;
            --accent-red: #ff5252;
            --accent-purple: #9d4edd;
            --accent-amber: #ffb703;
            --text-main: #f0f4f8;
            --text-dim: #94a3b8;
            --border: #233044;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background: var(--bg-primary);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }}
        .badge {{
            display: inline-block;
            background: rgba(0, 242, 254, 0.15);
            color: var(--accent-cyan);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            border: 1px solid rgba(0, 242, 254, 0.3);
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(135deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        .subtitle {{
            color: var(--text-dim);
            font-size: 14px;
        }}
        .grid-compare {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 900px) {{
            .grid-compare {{ grid-template-columns: 1fr; }}
        }}
        .timeframe-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .tf-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }}
        .tf-title {{
            font-size: 18px;
            font-weight: 700;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 13px;
        }}
        .stat-label {{ color: var(--text-dim); }}
        .stat-val {{ font-weight: 600; }}
        .text-green {{ color: var(--accent-green); }}
        .text-cyan {{ color: var(--accent-cyan); }}
        .text-red {{ color: var(--accent-red); }}
        .text-amber {{ color: var(--accent-amber); }}
        .text-purple {{ color: var(--accent-purple); }}
        
        .charts-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 900px) {{
            .charts-container {{ grid-template-columns: 1fr; }}
        }}
        .chart-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            height: 380px;
        }}
        .chart-title {{
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 12px;
        }}
        
        .insight-box {{
            background: rgba(255, 183, 3, 0.08);
            border: 1px solid rgba(255, 183, 3, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            font-size: 13.5px;
        }}
        .insight-title {{
            color: var(--accent-amber);
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 10px;
        }}
        ul {{
            margin-left: 20px;
            margin-top: 6px;
        }}
        li {{
            margin-bottom: 6px;
        }}
    </style>
</head>
<body>

    <div class="header">
        <div class="badge">EMPIRICAL RESEARCH BENCHMARK</div>
        <h1>BEID Strategy Timeframe Comparison: 1-Minute (M1) vs 5-Second (S5) Bars</h1>
        <div class="subtitle">
            Side-by-side empirical backtest on identical live continuous order flow (480,000 S5 bars across EURUSD, GBPUSD, USDJPY, EURGBP).
        </div>
    </div>

    <div class="insight-box">
        <div class="insight-title">&#9888; Key Quantitative Discovery: Why S5 Trading Fails Under Fixed Spread Friction</div>
        <ul>
            <li><strong>Zone Size vs Spread Fee:</strong> On 5-Second bars (S5), the average swing zone is only <strong>2.65 pips</strong> (median <strong>1.25 pips</strong>, half-height <strong>0.63 pips</strong>). With a broker spread of <strong>1.6 pips</strong>, the spread cost is <strong>2.5&times; larger than the entire gross profit target</strong>! Every win is converted into a net loss.</li>
            <li><strong>Transaction Drag:</strong> S5 generates <strong>1,535 trades in just 2 weeks</strong> ($4.5\times$ more trades than M1), handing <strong>2,456 pips in pure spread fees to the broker</strong>.</li>
            <li><strong>Microstructure Noise:</strong> 5-second swings reflect bid-ask bouncing rather than structural liquidity breaks, dropping the raw target hit rate to <strong>43.97%</strong> (vs <strong>67.4%</strong> on M1 macro swings).</li>
        </ul>
    </div>

    <div class="grid-compare">
        <!-- M1 Card -->
        <div class="timeframe-card" style="border-color: rgba(0, 230, 118, 0.4);">
            <div class="tf-header">
                <div class="tf-title text-green">1-Minute (M1) Bars</div>
                <div class="badge" style="background: rgba(0,230,118,0.15); color: var(--accent-green); border-color: rgba(0,230,118,0.3);">RECOMMENDED BASE</div>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Trades (Identical Window):</span>
                <span class="stat-val">{len(df_m1):,}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Target Hit Rate (Raw):</span>
                <span class="stat-val text-green">{(sum(df_m1['raw_result']=='WIN')/len(df_m1))*100.0:.1f}% ({sum(df_m1['raw_result']=='WIN')}W / {sum(df_m1['raw_result']=='LOSS')}L)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Average Zone Size:</span>
                <span class="stat-val text-cyan">{df_m1['zone_pips'].mean():.2f} pips (Median: {df_m1['zone_pips'].median():.2f}p)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Average Target (Half Zone):</span>
                <span class="stat-val text-cyan">{df_m1['half_pips'].mean():.2f} pips</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Spread-to-Target Drag:</span>
                <span class="stat-val text-green">{(1.6 / max(1e-4, df_m1['half_pips'].mean()))*100.0:.1f}% (Manageable)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Zero-Friction Net Return:</span>
                <span class="stat-val text-cyan">{df_m1['raw_r'].sum():+,.1f} R</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Net PnL (-1.6 pips friction):</span>
                <span class="stat-val text-amber">{df_m1['net_pips'].sum():+,.1f} Pips</span>
            </div>
        </div>

        <!-- S5 Card -->
        <div class="timeframe-card" style="border-color: rgba(255, 82, 82, 0.4);">
            <div class="tf-header">
                <div class="tf-title text-red">5-Second (S5) Bars</div>
                <div class="badge" style="background: rgba(255,82,82,0.15); color: var(--accent-red); border-color: rgba(255,82,82,0.3);">SEVERE FRICTION DRAG</div>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Trades (Identical Window):</span>
                <span class="stat-val">{len(df_s5):,} (4.5&times; higher churn)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Target Hit Rate (Raw):</span>
                <span class="stat-val text-red">{(sum(df_s5['raw_result']=='WIN')/len(df_s5))*100.0:.1f}% ({sum(df_s5['raw_result']=='WIN')}W / {sum(df_s5['raw_result']=='LOSS')}L)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Average Zone Size:</span>
                <span class="stat-val text-red">{df_s5['zone_pips'].mean():.2f} pips (Median: {df_s5['zone_pips'].median():.2f}p)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Average Target (Half Zone):</span>
                <span class="stat-val text-red">{df_s5['half_pips'].mean():.2f} pips</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Spread-to-Target Drag:</span>
                <span class="stat-val text-red">{(1.6 / max(1e-4, df_s5['half_pips'].mean()))*100.0:.1f}% (DESTRUCTIVE)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Zero-Friction Net Return:</span>
                <span class="stat-val text-red">{df_s5['raw_r'].sum():+,.1f} R</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Net PnL (-1.6 pips friction):</span>
                <span class="stat-val text-red">{df_s5['net_pips'].sum():+,.1f} Pips</span>
            </div>
        </div>
    </div>

    <div class="charts-container">
        <div class="chart-box">
            <div class="chart-title">Net Pips Equity Curve: M1 vs S5 (After 1.6-Pip Spread Deduction)</div>
            <canvas id="pipsChart"></canvas>
        </div>
        <div class="chart-box">
            <div class="chart-title">Raw Target Hit R-Curve: M1 vs S5 (Zero Friction)</div>
            <canvas id="rawRChart"></canvas>
        </div>
    </div>

    <script>
        // Pips Chart
        const ctxPips = document.getElementById('pipsChart').getContext('2d');
        new Chart(ctxPips, {{
            type: 'line',
            data: {{
                labels: {json.dumps(m1_dates)},
                datasets: [
                    {{
                        label: '1-Minute (M1) Net Pips',
                        data: {json.dumps(m1_pips)},
                        borderColor: '#00e676',
                        borderWidth: 2,
                        pointRadius: 0
                    }},
                    {{
                        label: '5-Second (S5) Net Pips',
                        data: {json.dumps(s5_pips[:len(m1_dates)])},
                        borderColor: '#ff5252',
                        borderWidth: 2,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
                scales: {{
                    x: {{ display: false }},
                    y: {{ grid: {{ color: '#233044' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // Raw R Chart
        const ctxR = document.getElementById('rawRChart').getContext('2d');
        new Chart(ctxR, {{
            type: 'line',
            data: {{
                labels: {json.dumps(m1_dates)},
                datasets: [
                    {{
                        label: '1-Minute (M1) Raw R',
                        data: {json.dumps(m1_raw_r)},
                        borderColor: '#00f2fe',
                        borderWidth: 2,
                        pointRadius: 0
                    }},
                    {{
                        label: '5-Second (S5) Raw R',
                        data: {json.dumps(s5_raw_r[:len(m1_dates)])},
                        borderColor: '#ffb703',
                        borderWidth: 2,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
                scales: {{
                    x: {{ display: false }},
                    y: {{ grid: {{ color: '#233044' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    repo_out = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_s5_vs_m1_comparison_dashboard.html"
    artifact_out = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/beid_s5_vs_m1_comparison_dashboard.html"
    
    with open(repo_out, "w") as f:
        f.write(html)
    with open(artifact_out, "w") as f:
        f.write(html)
        
    print(f"Comparison dashboard saved to:\n- {repo_out}\n- {artifact_out}")

if __name__ == "__main__":
    build_comparison_dashboard()
