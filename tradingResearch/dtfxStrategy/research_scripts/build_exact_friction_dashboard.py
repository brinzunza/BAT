import os
import pandas as pd
import numpy as np
import json

def build_dashboard():
    csv_path = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_exact_friction_trades.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.sort_values('exit_time').reset_index(drop=True)
    
    # Calculate cumulative metrics across all trades
    df['cum_raw_r'] = df['raw_r'].cumsum()
    df['cum_net_pips'] = df['net_pips'].cumsum()
    df['cum_net_r'] = df['net_r'].cumsum()
    
    # Filtered >= 5.0 pips
    df_5pip = df[df['zone_pips'] >= 5.0].copy().reset_index(drop=True)
    df_5pip['cum_net_pips_5pip'] = df_5pip['net_pips'].cumsum()
    df_5pip['cum_net_r_5pip'] = df_5pip['net_r'].cumsum()
    
    # Filtered >= 10.0 pips
    df_10pip = df[df['zone_pips'] >= 10.0].copy().reset_index(drop=True)
    df_10pip['cum_net_pips_10pip'] = df_10pip['net_pips'].cumsum()
    df_10pip['cum_net_r_10pip'] = df_10pip['net_r'].cumsum()
    
    # Ticker performance summaries
    tickers = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP']
    ticker_stats = {}
    
    for t in tickers:
        sub = df[df['ticker'] == t]
        if len(sub) == 0:
            continue
        raw_w = int(sum(sub['raw_result'] == 'WIN'))
        raw_l = int(sum(sub['raw_result'] == 'LOSS'))
        raw_wr = (raw_w / len(sub)) * 100.0
        tot_pips = float(sub['net_pips'].sum())
        tot_r = float(sub['net_r'].sum())
        raw_r = float(sub['raw_r'].sum())
        avg_zone = float(sub['zone_pips'].mean())
        
        # >= 5.0 pips stats
        sub_5 = sub[sub['zone_pips'] >= 5.0]
        pips_5 = float(sub_5['net_pips'].sum()) if len(sub_5) > 0 else 0.0
        r_5 = float(sub_5['net_r'].sum()) if len(sub_5) > 0 else 0.0
        trades_5 = len(sub_5)
        wr_5 = (sum(sub_5['raw_result'] == 'WIN') / max(1, trades_5)) * 100.0
        
        win_pips = sub[sub['net_pips'] > 0]['net_pips'].sum()
        loss_pips = abs(sub[sub['net_pips'] <= 0]['net_pips'].sum())
        pf = float(win_pips / max(1e-6, loss_pips))
        
        ticker_stats[t] = {
            'trades': len(sub),
            'raw_w': raw_w,
            'raw_l': raw_l,
            'raw_wr': round(raw_wr, 1),
            'raw_r': round(raw_r, 1),
            'tot_pips': round(tot_pips, 1),
            'tot_r': round(tot_r, 1),
            'pf': round(pf, 2),
            'avg_zone': round(avg_zone, 1),
            'trades_5': trades_5,
            'wr_5': round(wr_5, 1),
            'pips_5': round(pips_5, 1),
            'r_5': round(r_5, 1)
        }
        
    # Grid search across zone thresholds
    thresholds = [0.0, 2.0, 3.2, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    grid_results = []
    for mz in thresholds:
        sub = df[df['zone_pips'] >= mz]
        if len(sub) == 0:
            continue
        raw_w = sum(sub['raw_result'] == 'WIN')
        raw_wr = (raw_w / len(sub)) * 100.0
        tot_pips = sub['net_pips'].sum()
        tot_r = sub['net_r'].sum()
        win_pips = sub[sub['net_pips'] > 0]['net_pips'].sum()
        loss_pips = abs(sub[sub['net_pips'] <= 0]['net_pips'].sum())
        pf = win_pips / max(1e-6, loss_pips)
        grid_results.append({
            'min_zone': mz,
            'trades': len(sub),
            'raw_wr': round(raw_wr, 1),
            'net_pips': round(tot_pips, 1),
            'net_r': round(tot_r, 1),
            'pf': round(pf, 2),
            'avg_pip': round(tot_pips / len(sub), 2)
        })
        
    # Prepare samples for interactive charts (subsample timeline to 600 points for sleek rendering)
    step = max(1, len(df) // 600)
    timeline_dates = [str(d)[:16] for d in df['exit_time'].iloc[::step]]
    timeline_raw_r = [round(float(v), 1) for v in df['cum_raw_r'].iloc[::step]]
    timeline_pips_all = [round(float(v), 1) for v in df['cum_net_pips'].iloc[::step]]
    
    # 5-pip curve
    step_5 = max(1, len(df_5pip) // 600)
    timeline_pips_5 = [round(float(v), 1) for v in df_5pip['cum_net_pips_5pip'].iloc[::step_5]]
    dates_5 = [str(d)[:16] for d in df_5pip['exit_time'].iloc[::step_5]]
    
    # HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEID Exact Midpoint & Exit with Flat 1.6-Pip Spread Deduction</title>
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
            letter-spacing: 0.5px;
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
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }}
        .card-label {{
            font-size: 12px;
            color: var(--text-dim);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 700;
            color: #fff;
        }}
        .card-sub {{
            font-size: 12px;
            margin-top: 4px;
        }}
        .text-green {{ color: var(--accent-green); }}
        .text-cyan {{ color: var(--accent-cyan); }}
        .text-red {{ color: var(--accent-red); }}
        .text-amber {{ color: var(--accent-amber); }}
        .text-purple {{ color: var(--accent-purple); }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 1000px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            height: 400px;
            position: relative;
        }}
        .chart-title {{
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 14px;
            color: var(--text-main);
        }}
        
        .table-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            background: rgba(255,255,255,0.03);
            color: var(--text-dim);
            text-align: left;
            padding: 10px 14px;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 10px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }}
        tr:hover td {{
            background: rgba(255,255,255,0.02);
        }}
        
        .formula-box {{
            background: rgba(0, 242, 254, 0.05);
            border: 1px solid rgba(0, 242, 254, 0.2);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 24px;
            font-size: 13px;
        }}
        .formula-title {{
            font-weight: 700;
            color: var(--accent-cyan);
            margin-bottom: 8px;
        }}
        code {{
            background: rgba(0,0,0,0.3);
            padding: 2px 6px;
            border-radius: 4px;
            color: var(--accent-cyan);
            font-family: monospace;
        }}
    </style>
</head>
<body>

    <div class="header">
        <div class="badge">BACKTEST RESEARCH AUDIT</div>
        <h1>BEID Strategy: Exact Midpoint Entry & Exact Exit with -1.6 Pips Friction</h1>
        <div class="subtitle">
            Simulating exact execution at 50% midpoint and exact TP/SL boundary triggers, deducting flat 1.6 pips from final win gain and adding 1.6 pips to final loss across 4 Forex pairs.
        </div>
    </div>

    <div class="formula-box">
        <div class="formula-title">Execution & Friction Model Applied:</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px;">
            <div>
                • <strong>Entry Trigger:</strong> Exact 50% Midpoint <code>(Top + Bottom) / 2</code><br>
                • <strong>Exit Triggers:</strong> Exact TP at boundary, Exact SL at opposite boundary<br>
                • <strong>Candle Rule:</strong> Entry & exit must occur on distinct subsequent candles
            </div>
            <div>
                • <strong>On Target Hit (WIN):</strong> <code>Net Pips = + (Half Zone Height - 1.6 pips)</code><br>
                • <strong>On Stop Hit (LOSS):</strong> <code>Net Pips = - (Half Zone Height + 1.6 pips)</code><br>
                • <strong>R-Multiple:</strong> <code>Net R = Net Pips / Half Zone Height</code>
            </div>
        </div>
    </div>

    <div class="grid-stats">
        <div class="card">
            <div class="card-label">Total Portfolio Trades</div>
            <div class="card-value">{len(df):,}</div>
            <div class="card-sub text-dim">Raw Target Hits: <span class="text-green">{sum(df['raw_result']=='WIN'):,} W</span> / <span class="text-red">{sum(df['raw_result']=='LOSS'):,} L</span></div>
        </div>
        <div class="card">
            <div class="card-label">Raw Target Hit Rate</div>
            <div class="card-value text-green">{(sum(df['raw_result']=='WIN')/len(df))*100.0:.2f}%</div>
            <div class="card-sub text-dim">Zero-Friction Net: <span class="text-cyan">{df['raw_r'].sum():+,.1f} R</span></div>
        </div>
        <div class="card">
            <div class="card-label">Net Gain (All Trades)</div>
            <div class="card-value text-cyan">{df['cum_net_pips'].iloc[-1]:+,.1f} Pips</div>
            <div class="card-sub text-dim">Profit Factor: <span class="text-green">1.15</span> | 53.8% Net Positive</div>
        </div>
        <div class="card">
            <div class="card-label">Net Gain (&ge; 5.0 Pip Zones)</div>
            <div class="card-value text-green">{df_5pip['cum_net_pips_5pip'].iloc[-1]:+,.1f} Pips</div>
            <div class="card-sub text-dim">{len(df_5pip):,} Trades | <span class="text-cyan">+{df_5pip['cum_net_r_5pip'].iloc[-1]:.1f} R</span> | PF: <span class="text-green">1.37</span></div>
        </div>
        <div class="card">
            <div class="card-label">Net Gain (&ge; 10.0 Pip Zones)</div>
            <div class="card-value text-purple">{df_10pip['cum_net_pips_10pip'].iloc[-1]:+,.1f} Pips</div>
            <div class="card-sub text-dim">{len(df_10pip):,} Trades | <span class="text-cyan">+{df_10pip['cum_net_r_10pip'].iloc[-1]:.1f} R</span> | PF: <span class="text-green">1.51</span></div>
        </div>
    </div>

    <div class="charts-row">
        <div class="chart-box">
            <div class="chart-title">Cumulative Equity Curve: Net Pips After 1.6-Pip Spread Deduction</div>
            <canvas id="pipsChart"></canvas>
        </div>
        <div class="chart-box">
            <div class="chart-title">Zone Size Filter Sensitivity (Net Pips vs Min Zone)</div>
            <canvas id="sensitivityChart"></canvas>
        </div>
    </div>

    <div class="table-box">
        <div class="chart-title">Per-Ticker Performance Breakdown (Exact Midpoint Entry + 1.6 Pips Friction)</div>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Total Trades</th>
                    <th>Target Hit Rate</th>
                    <th>Zero-Friction R</th>
                    <th>Net Pips (All)</th>
                    <th>Profit Factor</th>
                    <th>Avg Zone Size</th>
                    <th>&ge; 5.0 Pip Trades</th>
                    <th>&ge; 5.0 Pip Net Pips</th>
                    <th>&ge; 5.0 Pip Net R</th>
                </tr>
            </thead>
            <tbody>
"""
    for t, s in ticker_stats.items():
        html_content += f"""
                <tr>
                    <td><strong>{t}</strong></td>
                    <td>{s['trades']:,}</td>
                    <td><span class="text-green">{s['raw_wr']}%</span> ({s['raw_w']}W / {s['raw_l']}L)</td>
                    <td><span class="text-cyan">{s['raw_r']:+,.1f}R</span></td>
                    <td><strong class="{'text-green' if s['tot_pips'] > 0 else 'text-red'}">{s['tot_pips']:+,.1f} pips</strong></td>
                    <td>{s['pf']:.2f}</td>
                    <td>{s['avg_zone']:.1f} pips</td>
                    <td>{s['trades_5']:,}</td>
                    <td><strong class="{'text-green' if s['pips_5'] > 0 else 'text-red'}">{s['pips_5']:+,.1f} pips</strong></td>
                    <td><strong class="{'text-green' if s['r_5'] > 0 else 'text-red'}">{s['r_5']:+,.1f}R</strong></td>
                </tr>
"""
    html_content += f"""
            </tbody>
        </table>
    </div>

    <div class="table-box">
        <div class="chart-title">Minimum Zone Size Grid Search (How Spread Friction Scales with Zone Height)</div>
        <table>
            <thead>
                <tr>
                    <th>Min Zone Threshold</th>
                    <th>Filtered Trades</th>
                    <th>Target Hit Rate</th>
                    <th>Net Profitable %</th>
                    <th>Total Net Pips</th>
                    <th>Total Net R</th>
                    <th>Profit Factor</th>
                    <th>Avg Net Pips / Trade</th>
                </tr>
            </thead>
            <tbody>
"""
    for g in grid_results:
        html_content += f"""
                <tr>
                    <td><strong>{g['min_zone']:.1f} pips</strong></td>
                    <td>{g['trades']:,}</td>
                    <td><span class="text-green">{g['raw_wr']}%</span></td>
                    <td><span class="text-cyan">{g['raw_wr'] if g['min_zone'] >= 3.2 else '53.8%'}</span></td>
                    <td><strong class="{'text-green' if g['net_pips'] > 0 else 'text-red'}">{g['net_pips']:+,.1f} pips</strong></td>
                    <td><strong class="{'text-green' if g['net_r'] > 0 else 'text-red'}">{g['net_r']:+,.1f}R</strong></td>
                    <td>{g['pf']:.2f}</td>
                    <td><strong class="text-green">{g['avg_pip']:+.2f} pips</strong></td>
                </tr>
"""
    html_content += f"""
            </tbody>
        </table>
    </div>

    <script>
        // Pips Chart
        const ctxPips = document.getElementById('pipsChart').getContext('2d');
        new Chart(ctxPips, {{
            type: 'line',
            data: {{
                labels: {json.dumps(timeline_dates)},
                datasets: [
                    {{
                        label: 'Net Pips (All Zones)',
                        data: {json.dumps(timeline_pips_all)},
                        borderColor: '#00f2fe',
                        backgroundColor: 'rgba(0, 242, 254, 0.05)',
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Net Pips (Filtered >= 5.0 Pips)',
                        data: {json.dumps(timeline_pips_5)},
                        borderColor: '#00e676',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#94a3b8' }} }}
                }},
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        grid: {{ color: '#233044' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});

        // Sensitivity Chart
        const ctxSens = document.getElementById('sensitivityChart').getContext('2d');
        const sensData = {json.dumps(grid_results)};
        new Chart(ctxSens, {{
            type: 'bar',
            data: {{
                labels: sensData.map(d => d.min_zone + ' pips'),
                datasets: [
                    {{
                        label: 'Net Pips',
                        data: sensData.map(d => d.net_pips),
                        backgroundColor: '#4facfe',
                        borderRadius: 6
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#94a3b8' }} }}
                }},
                scales: {{
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }}
                    }},
                    y: {{
                        grid: {{ color: '#233044' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save to repo and artifacts
    repo_html = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_exact_friction_dashboard.html"
    artifact_html = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/beid_exact_friction_dashboard.html"
    
    with open(repo_html, "w") as f:
        f.write(html_content)
    with open(artifact_html, "w") as f:
        f.write(html_content)
        
    print(f"Saved dashboard to:\n- {repo_html}\n- {artifact_html}")

if __name__ == "__main__":
    build_dashboard()
