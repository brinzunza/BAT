import os
import pandas as pd
import numpy as np
import json

def build_dashboard():
    csv_path = "dtfxStrategy/research_results/optimal_beid_mc_grid_results.csv"
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return
        
    df = pd.read_csv(csv_path)
    
    # Valid filter: probability >= 70%, ruin <= 15%
    valid = df[(df['prob_reach_1m'] >= 70.0) & (df['ruin_rate'] <= 15.0)].copy()
    valid = valid.sort_values('med_days_to_1m').reset_index(drop=True)
    
    # Top 15 configurations
    top15 = valid.head(15).to_dict(orient='records')
    
    # Timeframe summary (best of each TF)
    tf_best = []
    for tf in sorted(df['timeframe_min'].unique()):
        sub_tf = valid[valid['timeframe_min'] == tf]
        if len(sub_tf) > 0:
            best_row = sub_tf.iloc[0].to_dict()
            tf_best.append(best_row)
            
    # Touches summary (best of each touch level on 3M)
    touches_3m = []
    for touch in sorted(df['max_touches'].unique()):
        sub_t = valid[(valid['timeframe_min'] == 3.0) & (valid['max_touches'] == touch)]
        if len(sub_t) > 0:
            touches_3m.append(sub_t.iloc[0].to_dict())
            
    # Min Pip summary (best of each min pip on 3M, 5 touches)
    min_pip_3m = []
    for mp in sorted(df['min_pip_range'].unique()):
        sub_p = valid[(valid['timeframe_min'] == 3.0) & (valid['max_touches'] == 5.0) & (valid['min_pip_range'] == mp)]
        if len(sub_p) > 0:
            min_pip_3m.append(sub_p.iloc[0].to_dict())
            
    # Risk % curve on #1 config (3M, 5 touches, 10 pips)
    risk_curve = []
    sub_risk = df[(df['timeframe_min'] == 3.0) & (df['max_touches'] == 5.0) & (df['min_pip_range'] == 10.0)].sort_values('risk_pct')
    for _, r in sub_risk.iterrows():
        risk_curve.append({
            'risk_pct': round(r['risk_pct'] * 100, 1),
            'med_months': round(r['med_months_to_1m'], 1) if not np.isnan(r['med_months_to_1m']) else None,
            'med_max_dd': round(r['med_max_dd'], 1),
            'prob_1m': round(r['prob_reach_1m'], 1),
            'ruin_rate': round(r['ruin_rate'], 1)
        })
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimal BEID Configuration: Quickest Time to $1M in Monte Carlo</title>
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
        .champion-box {{
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.1), rgba(0, 230, 118, 0.05));
            border: 1px solid rgba(0, 242, 254, 0.4);
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 24px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        }}
        .champ-title {{
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-cyan);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .champ-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }}
        .champ-item {{
            background: rgba(0,0,0,0.25);
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.06);
        }}
        .champ-item-label {{
            font-size: 11px;
            color: var(--text-dim);
            text-transform: uppercase;
            font-weight: 600;
        }}
        .champ-item-val {{
            font-size: 20px;
            font-weight: 700;
            color: #fff;
            margin-top: 4px;
        }}
        .text-green {{ color: var(--accent-green); }}
        .text-cyan {{ color: var(--accent-cyan); }}
        .text-red {{ color: var(--accent-red); }}
        .text-amber {{ color: var(--accent-amber); }}
        .text-purple {{ color: var(--accent-purple); }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 950px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
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
    </style>
</head>
<body>

    <div class="header">
        <div class="badge">GLOBAL MONTE CARLO OPTIMIZATION</div>
        <h1>BEID Strategy: Optimal Speed to Median $1,000,000</h1>
        <div class="subtitle">
            1,344 Parameter Configurations evaluated across 4 Forex pairs with flat 1.6-pip spread friction and $200 starting capital.
        </div>
    </div>

    <div class="champion-box">
        <div class="champ-title">
            <span>&#127942;</span> GLOBAL OPTIMAL CHAMPION CONFIGURATION (FASTEST TO $1M)
        </div>
        <div class="champ-grid">
            <div class="champ-item">
                <div class="champ-item-label">Optimal Timeframe</div>
                <div class="champ-item-val text-cyan">3-Minute (3M)</div>
            </div>
            <div class="champ-item">
                <div class="champ-item-label">Multi-Zone Touches</div>
                <div class="champ-item-val text-green">Up to 5 Touches (Re-arm)</div>
            </div>
            <div class="champ-item">
                <div class="champ-item-label">Min Zone Size</div>
                <div class="champ-item-val text-purple">&ge; 10.0 Pips</div>
            </div>
            <div class="champ-item">
                <div class="champ-item-label">Optimal Risk %</div>
                <div class="champ-item-val text-amber">6.0% / Trade (or 5.0%)</div>
            </div>
            <div class="champ-item">
                <div class="champ-item-label">Median Time to $1M</div>
                <div class="champ-item-val text-green">29.3 Months (2.4 yrs)</div>
            </div>
            <div class="champ-item">
                <div class="champ-item-label">Win Rate / PF</div>
                <div class="champ-item-val text-cyan">67.3% / 1.32 PF</div>
            </div>
            <div class="champ-item">
                <div class="champ-item-label">Probability of $1M</div>
                <div class="champ-item-val text-green">100.0% (0.0% Ruin)</div>
            </div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-box">
            <div class="chart-title">Timeframe Speed Comparison (Best Months to $1M by Bar Size)</div>
            <canvas id="tfChart"></canvas>
        </div>
        <div class="chart-box">
            <div class="chart-title">Risk % vs Speed to $1M & Drawdown (3M, 5 Touches, 10 Pips)</div>
            <canvas id="riskChart"></canvas>
        </div>
    </div>

    <div class="table-box">
        <div class="chart-title">Top 15 Fastest Parameter Configurations to Median $1,000,000 ($200 Start)</div>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Timeframe</th>
                    <th>Max Touches</th>
                    <th>Min Pip Filter</th>
                    <th>Risk %</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Trades / Year</th>
                    <th>Median Months to $1M</th>
                    <th>Median Max DD</th>
                    <th>Prob of $1M</th>
                </tr>
            </thead>
            <tbody>
"""
    for idx, c in enumerate(top15):
        html += f"""
                <tr>
                    <td><strong class="{'text-cyan' if idx==0 else 'text-main'}">#{idx+1}</strong></td>
                    <td><strong>{int(c['timeframe_min'])}M Bars</strong></td>
                    <td><span class="text-green">{int(c['max_touches'])} Touches</span></td>
                    <td>&ge; {c['min_pip_range']:.1f} pips</td>
                    <td><strong class="text-amber">{c['risk_pct']*100:.1f}%</strong></td>
                    <td><span class="text-green">{c['raw_win_rate']:.1f}%</span></td>
                    <td>{c['profit_factor']:.2f}</td>
                    <td>{c['trades_per_year']:,.0f}</td>
                    <td><strong class="text-green">{c['med_months_to_1m']:.1f} mo ({c['med_months_to_1m']/12:.2f} yr)</strong></td>
                    <td>{c['med_max_dd']:.1f}%</td>
                    <td><strong class="text-green">{c['prob_reach_1m']:.1f}%</strong></td>
                </tr>
"""
    html += f"""
            </tbody>
        </table>
    </div>

    <script>
        // TF Chart
        const ctxTF = document.getElementById('tfChart').getContext('2d');
        const tfData = {json.dumps(tf_best)};
        new Chart(ctxTF, {{
            type: 'bar',
            data: {{
                labels: tfData.map(d => d.timeframe_min + 'M Bars'),
                datasets: [
                    {{
                        label: 'Median Months to $1M (Lower is Faster)',
                        data: tfData.map(d => d.med_months_to_1m),
                        backgroundColor: tfData.map(d => d.timeframe_min === 3 ? '#00e676' : '#4facfe'),
                        borderRadius: 6
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
                scales: {{
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#233044' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        // Risk Chart
        const ctxRisk = document.getElementById('riskChart').getContext('2d');
        const riskData = {json.dumps(risk_curve)};
        new Chart(ctxRisk, {{
            type: 'line',
            data: {{
                labels: riskData.map(d => d.risk_pct + '%'),
                datasets: [
                    {{
                        label: 'Median Months to $1M',
                        data: riskData.map(d => d.med_months),
                        borderColor: '#00f2fe',
                        yAxisID: 'y',
                        borderWidth: 2,
                        pointRadius: 4
                    }},
                    {{
                        label: 'Median Max Drawdown %',
                        data: riskData.map(d => d.med_max_dd),
                        borderColor: '#ff5252',
                        yAxisID: 'y1',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
                scales: {{
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {{ color: '#233044' }},
                        ticks: {{ color: '#00f2fe' }},
                        title: {{ display: true, text: 'Months to $1M', color: '#00f2fe' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#ff5252' }},
                        title: {{ display: true, text: 'Max Drawdown %', color: '#ff5252' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    repo_html = "dtfxStrategy/research_results/beid_optimal_mc_speed_to_1m_dashboard.html"
    artifact_html = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/beid_optimal_mc_speed_to_1m_dashboard.html"
    
    with open(repo_html, "w") as f:
        f.write(html)
    with open(artifact_html, "w") as f:
        f.write(html)
        
    print(f"Saved dashboard to:\n- {repo_html}\n- {artifact_html}")

if __name__ == "__main__":
    build_dashboard()
