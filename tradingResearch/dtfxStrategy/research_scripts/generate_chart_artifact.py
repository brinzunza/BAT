import json
import os

with open('/tmp/usdjpy_chart_data.json') as f:
    data = json.load(f)

candles_json = json.dumps(data['candles'])
trades_json = json.dumps(data['trades'])

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DTFX Live Trading & Backtest Visualizer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #0f172a;
      color: #f8fafc;
    }}
    .trade-row:hover {{
      background-color: rgba(51, 65, 85, 0.5);
    }}
    .trade-row.active {{
      background-color: rgba(59, 130, 246, 0.2);
      border-left: 3px solid #3b82f6;
    }}
  </style>
</head>
<body class="min-h-screen p-6">
  <div class="max-w-7xl mx-auto space-y-6">
    <!-- Header -->
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-xl">
      <div>
        <div class="flex items-center gap-3">
          <h1 class="text-2xl font-bold text-white tracking-tight">USD/JPY DTFX Execution & Backtest Chart</h1>
          <span class="px-3 py-1 text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-full">1-Min M1 Live Feed</span>
          <span class="px-3 py-1 text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full">10-Pip Floor</span>
        </div>
        <p class="text-slate-400 text-sm mt-1">Interactive candlestick replay showing 50% midpoint limit entries, swing zones, stop loss, and take profit resolution.</p>
      </div>
      <div class="flex items-center gap-2">
        <button id="zoomInBtn" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-lg border border-slate-700 transition">Zoom In (+)</button>
        <button id="zoomOutBtn" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-lg border border-slate-700 transition">Zoom Out (-)</button>
        <button id="resetZoomBtn" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-lg border border-slate-700 transition">Reset View</button>
      </div>
    </div>

    <!-- KPI Metric Cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Analyzed Trades</div>
        <div id="kpiTotal" class="text-2xl font-bold text-white mt-1">--</div>
      </div>
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Win Rate</div>
        <div id="kpiWinRate" class="text-2xl font-bold text-emerald-400 mt-1">--</div>
      </div>
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Profit Factor</div>
        <div id="kpiPF" class="text-2xl font-bold text-blue-400 mt-1">--</div>
      </div>
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow">
        <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Return</div>
        <div id="kpiTotalR" class="text-2xl font-bold text-indigo-400 mt-1">--</div>
      </div>
    </div>

    <!-- Main Chart & Trade Panel Layout -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Interactive Chart View -->
      <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col">
        <div class="flex justify-between items-center mb-3">
          <div class="flex items-center gap-4 text-xs">
            <label class="flex items-center gap-1.5 cursor-pointer text-slate-300">
              <input type="checkbox" id="toggleTrades" checked class="rounded bg-slate-800 border-slate-700 text-blue-500 focus:ring-0">
              <span>Show Trade Markers</span>
            </label>
            <label class="flex items-center gap-1.5 cursor-pointer text-slate-300">
              <input type="checkbox" id="toggleBrackets" checked class="rounded bg-slate-800 border-slate-700 text-blue-500 focus:ring-0">
              <span>Show SL / TP Levels</span>
            </label>
          </div>
          <div id="hoverInfo" class="text-xs font-mono text-slate-400">Drag to Pan | Wheel to Zoom</div>
        </div>
        
        <div class="relative flex-1 w-full bg-slate-950 rounded-xl overflow-hidden border border-slate-800/80" style="min-height: 480px;">
          <canvas id="chartCanvas" class="w-full h-full cursor-crosshair"></canvas>
        </div>

        <div class="flex justify-between items-center text-xs text-slate-500 mt-3">
          <span>Click any trade in the inspector on the right to jump directly to it</span>
          <span id="rangeLabel">Loading...</span>
        </div>
      </div>

      <!-- Trade Inspector & Log Panel -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col" style="max-height: 590px;">
        <div class="flex justify-between items-center mb-3">
          <h3 class="font-bold text-white text-sm">Trade Inspector</h3>
          <span class="text-xs text-slate-400" id="tradeCountLabel">-- trades</span>
        </div>
        
        <div class="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar" id="tradeListContainer">
          <!-- Trade rows inserted dynamically -->
        </div>
      </div>
    </div>
  </div>

  <script>
    const candles = {candles_json};
    const trades = {trades_json};

    // Calculate metrics
    const wins = trades.filter(t => t.outcome === 'win').length;
    const losses = trades.filter(t => t.outcome === 'loss').length;
    const wr = trades.length ? ((wins / trades.length) * 100).toFixed(1) : '0.0';
    const totalR = trades.reduce((acc, t) => acc + (t.net_r || 0), 0).toFixed(2);
    const grossWin = trades.filter(t => t.net_r > 0).reduce((acc, t) => acc + t.net_r, 0);
    const grossLoss = Math.abs(trades.filter(t => t.net_r < 0).reduce((acc, t) => acc + t.net_r, 0));
    const pf = grossLoss > 0 ? (grossWin / grossLoss).toFixed(2) : 'Inf';

    document.getElementById('kpiTotal').textContent = trades.length + ' trades';
    document.getElementById('kpiWinRate').textContent = wr + '% (' + wins + 'W / ' + losses + 'L)';
    document.getElementById('kpiPF').textContent = pf;
    document.getElementById('kpiTotalR').textContent = (totalR > 0 ? '+' : '') + totalR + ' R';
    document.getElementById('tradeCountLabel').textContent = trades.length + ' setups';

    // Chart state
    const canvas = document.getElementById('chartCanvas');
    const ctx = canvas.getContext('2d');
    let viewStart = Math.max(0, candles.length - 120);
    let viewCount = 120;
    let isDragging = false;
    let dragStartX = 0;

    function resizeCanvas() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      drawChart();
    }}
    window.addEventListener('resize', resizeCanvas);

    function drawChart() {{
      const width = canvas.width / window.devicePixelRatio;
      const height = canvas.height / window.devicePixelRatio;
      ctx.clearRect(0, 0, width, height);

      if (!candles.length) return;

      const visible = candles.slice(viewStart, viewStart + viewCount);
      if (!visible.length) return;

      let minP = Math.min(...visible.map(c => c.low));
      let maxP = Math.max(...visible.map(c => c.high));
      const pad = (maxP - minP) * 0.1 || 0.1;
      minP -= pad;
      maxP += pad;

      const padTop = 20, padBottom = 30, padRight = 60, padLeft = 10;
      const plotW = width - padLeft - padRight;
      const plotH = height - padTop - padBottom;

      const pToY = p => padTop + plotH - ((p - minP) / (maxP - minP)) * plotH;
      const idxToX = i => padLeft + (i / (viewCount - 1)) * plotW;

      // Grid lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      for (let i = 0; i <= 5; i++) {{
        const y = padTop + (plotH / 5) * i;
        ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(width - padRight, y); ctx.stroke();
        const priceVal = maxP - (i / 5) * (maxP - minP);
        ctx.fillStyle = '#64748b'; ctx.font = '10px monospace'; ctx.fillText(priceVal.toFixed(3), width - padRight + 5, y + 3);
      }}
      ctx.setLineDash([]);

      const barW = Math.max(2, (plotW / viewCount) * 0.65);

      // Draw Candlesticks
      visible.forEach((c, idx) => {{
        const x = idxToX(idx);
        const yOpen = pToY(c.open);
        const yClose = pToY(c.close);
        const yHigh = pToY(c.high);
        const yLow = pToY(c.low);
        const isUp = c.close >= c.open;

        ctx.strokeStyle = isUp ? '#10b981' : '#ef4444';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, yHigh);
        ctx.lineTo(x, yLow);
        ctx.stroke();

        ctx.fillStyle = isUp ? '#10b981' : '#ef4444';
        const bodyTop = Math.min(yOpen, yClose);
        const bodyH = Math.max(1.5, Math.abs(yOpen - yClose));
        ctx.fillRect(x - barW / 2, bodyTop, barW, bodyH);
      }});

      // Draw Trades & Brackets
      const showTrades = document.getElementById('toggleTrades').checked;
      const showBrackets = document.getElementById('toggleBrackets').checked;

      if (showTrades) {{
        trades.forEach((t) => {{
          const cIdx = candles.findIndex(c => c.time === t.entry_time);
          if (cIdx >= viewStart && cIdx < viewStart + viewCount) {{
            const localIdx = cIdx - viewStart;
            const x = idxToX(localIdx);
            const yEntry = pToY(t.entry_price);
            const isBuy = t.side.toLowerCase() === 'buy';

            // Entry arrow marker
            ctx.fillStyle = isBuy ? '#38bdf8' : '#f43f5e';
            ctx.beginPath();
            if (isBuy) {{
              ctx.moveTo(x, yEntry + 14);
              ctx.lineTo(x - 6, yEntry + 24);
              ctx.lineTo(x + 6, yEntry + 24);
            }} else {{
              ctx.moveTo(x, yEntry - 14);
              ctx.lineTo(x - 6, yEntry - 24);
              ctx.lineTo(x + 6, yEntry - 24);
            }}
            ctx.fill();

            // Outcome badge
            ctx.fillStyle = t.outcome === 'win' ? '#10b981' : '#ef4444';
            ctx.font = 'bold 9px sans-serif';
            ctx.fillText(t.outcome.toUpperCase(), x - 10, isBuy ? yEntry + 34 : yEntry - 28);

            // Draw SL & TP lines
            if (showBrackets) {{
              const endX = Math.min(width - padRight, x + 80);
              // TP line (Green)
              ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
              ctx.lineWidth = 1.5;
              ctx.setLineDash([3, 2]);
              ctx.beginPath(); ctx.moveTo(x, pToY(t.target_price)); ctx.lineTo(endX, pToY(t.target_price)); ctx.stroke();
              // SL line (Red)
              ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
              ctx.beginPath(); ctx.moveTo(x, pToY(t.stop_price)); ctx.lineTo(endX, pToY(t.stop_price)); ctx.stroke();
              ctx.setLineDash([]);
            }}
          }}
        }});
      }}

      // Update Range Label
      document.getElementById('rangeLabel').textContent = visible[0].time + '  →  ' + visible[visible.length - 1].time;
    }}

    // Interaction controls
    document.getElementById('zoomInBtn').onclick = () => {{
      if (viewCount > 30) {{ viewCount = Math.floor(viewCount * 0.75); drawChart(); }}
    }};
    document.getElementById('zoomOutBtn').onclick = () => {{
      if (viewCount < candles.length) {{ viewCount = Math.min(candles.length, Math.floor(viewCount * 1.35)); drawChart(); }}
    }};
    document.getElementById('resetZoomBtn').onclick = () => {{
      viewCount = 120; viewStart = Math.max(0, candles.length - viewCount); drawChart();
    }};
    document.getElementById('toggleTrades').onchange = drawChart;
    document.getElementById('toggleBrackets').onchange = drawChart;

    // Pan with mouse drag
    canvas.addEventListener('mousedown', e => {{ isDragging = true; dragStartX = e.clientX; }});
    window.addEventListener('mouseup', () => {{ isDragging = false; }});
    canvas.addEventListener('mousemove', e => {{
      if (isDragging) {{
        const dx = e.clientX - dragStartX;
        const shift = Math.round((dx / canvas.width) * viewCount);
        if (shift !== 0) {{
          viewStart = Math.max(0, Math.min(candles.length - viewCount, viewStart - shift));
          dragStartX = e.clientX;
          drawChart();
        }}
      }}
    }});

    // Populate Trade List
    const listContainer = document.getElementById('tradeListContainer');
    trades.forEach((t) => {{
      const div = document.createElement('div');
      const isWin = t.outcome === 'win';
      div.className = 'trade-row p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs cursor-pointer transition';
      div.innerHTML = `
        <div class="flex justify-between items-center mb-1">
          <span class="font-bold ${{t.side === 'buy' ? 'text-blue-400' : 'text-rose-400'}}">${{t.side.toUpperCase()}} ${{t.symbol}}</span>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold ${{isWin ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'}}">${{t.outcome.toUpperCase()}} (${{t.net_r > 0 ? '+' : ''}}${{t.net_r}}R)</span>
        </div>
        <div class="flex justify-between text-slate-400 text-[11px]">
          <span>Entry: ${{t.entry_price.toFixed(3)}}</span>
          <span>SL: ${{t.stop_price.toFixed(3)}} | TP: ${{t.target_price.toFixed(3)}}</span>
        </div>
        <div class="text-[10px] text-slate-500 mt-1">${{t.entry_time}}</div>
      `;
      div.onclick = () => {{
        document.querySelectorAll('.trade-row').forEach(el => el.classList.remove('active'));
        div.classList.add('active');
        const cIdx = candles.findIndex(c => c.time === t.entry_time);
        if (cIdx !== -1) {{
          viewStart = Math.max(0, Math.min(candles.length - viewCount, cIdx - Math.floor(viewCount / 2)));
          drawChart();
        }}
      }};
      listContainer.appendChild(div);
    }});

    setTimeout(resizeCanvas, 50);
  </script>
</body>
</html>"""

target_path = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05/dtfx_trades_chart.html"
with open(target_path, "w") as f:
    f.write(html_template)
print("Successfully generated interactive visual chart artifact at:", target_path)
