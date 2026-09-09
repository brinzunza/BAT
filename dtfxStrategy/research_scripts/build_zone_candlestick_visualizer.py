"""Build upgraded DTFX Zone Visualizer with interactive candlestick chart engine."""
import os
import json
import pandas as pd

ARTIFACT_DIR = "/Users/brunoinzunza/.gemini/antigravity/brain/22a6b06b-342e-44ca-96c2-fb9de00f8a05"
DATA_DIR = "/Users/brunoinzunza/Documents/github/tradingResearch/dtfxStrategy/data"

# Load recent real candles for all 4 pairs
recent_candles = {}
for sym in ["USDJPY", "GBPUSD", "EURUSD", "EURGBP"]:
    path = f"/tmp/dtfx_fresh_data/{sym}_merged.csv"
    if not os.path.exists(path):
        fname = "lse_usdjpy_1min_2018on.csv" if sym == "USDJPY" else f"lse_{sym.lower()}_1min_full.csv"
        path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        tail_bars = df.tail(200)
        recent_candles[sym] = [
            {
                "time": r["date"].strftime("%H:%M"),
                "open": round(float(r["open"]), 5 if "JPY" not in sym else 3),
                "high": round(float(r["high"]), 5 if "JPY" not in sym else 3),
                "low": round(float(r["low"]), 5 if "JPY" not in sym else 3),
                "close": round(float(r["close"]), 5 if "JPY" not in sym else 3)
            }
            for _, r in tail_bars.iterrows()
        ]

candles_json = json.dumps(recent_candles)

template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DTFX Live Candlestick & Zone Visualizer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body { background-color: #0b0f19; color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .card { background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; }
    textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  </style>
</head>
<body class="p-4 md:p-6 space-y-6">

  <!-- Header -->
  <div class="card p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-gradient-to-r from-gray-900 via-gray-900 to-indigo-950/40">
    <div>
      <div class="flex items-center gap-2">
        <span class="px-2.5 py-0.5 rounded text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">DTFX CANDLESTICK RADAR</span>
        <span class="text-xs text-gray-400">Live Zone Overlay on Real M1 Candles</span>
      </div>
      <h1 class="text-2xl md:text-3xl font-bold mt-1 text-white tracking-tight">DTFX Live Candlestick & Zone Plotter</h1>
      <p class="text-sm text-gray-400 mt-1">Paste your <code class="text-indigo-300">/zones</code> telegram output to plot the zones, 50% entry levels, and real OHLC candlesticks simultaneously.</p>
    </div>
    <div class="flex items-center gap-2">
      <button onclick="parseAndPlot()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-semibold transition flex items-center gap-2 shadow-lg shadow-indigo-600/20">
        <span>⚡ Parse & Plot</span>
      </button>
    </div>
  </div>

  <!-- Layout: Left Input Panel | Right Candlestick Chart -->
  <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Left Column: Paste Input & Breakdown (4 cols) -->
    <div class="lg:col-span-4 space-y-4">
      <div class="card p-4 space-y-3">
        <div class="flex justify-between items-center">
          <label class="text-xs font-bold uppercase tracking-wider text-gray-400">Paste /zones Output Here</label>
          <button onclick="clearText()" class="text-xs text-gray-500 hover:text-gray-300">Clear</button>
        </div>
        <textarea id="rawInput" rows="13" class="w-full bg-gray-950/80 border border-gray-800 rounded-lg p-3 text-xs text-gray-200 focus:outline-none focus:border-indigo-500 transition resize-y">🌱 Pending / Maturing Zones

🟢 GBPUSD BUY
  mid=1.35263  top=1.35302  bottom=1.35224
  formed 551min ago -- 👀 overshot, waiting to tag its own 50% midpoint

🔴 GBPUSD SELL
  mid=1.35584  top=1.35620  bottom=1.35548
  formed 424min ago -- 👀 overshot, waiting to tag its own 50% midpoint

🔴 USDJPY SELL
  mid=152.96700  top=153.04600  bottom=152.88800
  formed 1088min ago -- ⏱️ waiting for price to overshoot the zone's far edge

🔴 USDJPY SELL
  mid=153.15750  top=153.21900  bottom=153.09600
  formed 1065min ago -- ⏱️ waiting for price to overshoot the zone's far edge

🟢 USDJPY BUY
  mid=153.16650  top=153.21900  bottom=153.11400
  formed 1063min ago -- ⏳ touched -- resting limit order live, waiting for fill

🟢 USDJPY BUY
  mid=153.30200  top=153.36600  bottom=153.23800
  formed 975min ago -- ⏳ touched -- resting limit order live, waiting for fill

🟢 USDJPY BUY
  mid=153.33150  top=153.40000  bottom=153.26300
  formed 863min ago -- 👀 overshot, waiting to tag its own 50% midpoint

🟢 USDJPY BUY
  mid=154.36950  top=154.42400  bottom=154.31500
  formed 559min ago -- ⏱️ waiting for price to overshoot the zone's far edge

🔴 USDJPY SELL
  mid=154.18750  top=154.32200  bottom=154.05300
  formed 99min ago -- 👀 overshot, waiting to tag its own 50% midpoint

🔴 USDJPY SELL
  mid=153.64300  top=153.83400  bottom=153.45200
  formed 91min ago -- ⏱️ waiting for price to overshoot the zone's far edge

🟢 USDJPY BUY
  mid=153.87150  top=153.92600  bottom=153.81700
  formed 47min ago -- 👀 overshot, waiting to tag its own 50% midpoint


11 zone(s) tracked.</textarea>
        <button onclick="parseAndPlot()" class="w-full py-2 bg-indigo-600/80 hover:bg-indigo-600 text-white rounded-lg text-xs font-semibold transition">
          Update Chart & Zones
        </button>
      </div>

      <!-- Quick Stats -->
      <div class="card p-4 space-y-3">
        <h3 class="text-xs font-bold uppercase tracking-wider text-gray-400">Parsed Breakdown</h3>
        <div class="grid grid-cols-2 gap-2 text-center text-xs">
          <div class="p-2.5 rounded bg-gray-950 border border-gray-800">
            <div class="text-gray-400">Total Tracked</div>
            <div id="statTotal" class="text-lg font-bold text-white mt-0.5">0</div>
          </div>
          <div class="p-2.5 rounded bg-amber-950/30 border border-amber-500/30">
            <div class="text-amber-300">Live Limits (⏳)</div>
            <div id="statLimits" class="text-lg font-bold text-amber-400 mt-0.5">0</div>
          </div>
          <div class="p-2.5 rounded bg-emerald-950/30 border border-emerald-500/30">
            <div class="text-emerald-300">Waiting Tag (👀)</div>
            <div id="statOvershot" class="text-lg font-bold text-emerald-400 mt-0.5">0</div>
          </div>
          <div class="p-2.5 rounded bg-blue-950/30 border border-blue-500/30">
            <div class="text-blue-300">Waiting Overshoot (⏱️)</div>
            <div id="statWaitOver" class="text-lg font-bold text-blue-400 mt-0.5">0</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column: Interactive Candlestick Chart (8 cols) -->
    <div class="lg:col-span-8 space-y-4">
      
      <!-- Symbol Tabs & Controls -->
      <div class="card p-3 flex flex-wrap justify-between items-center gap-3">
        <div class="flex items-center gap-1.5" id="symbolTabs">
          <!-- Dynamic Symbol Pills -->
        </div>
        <div class="flex items-center gap-3 text-xs">
          <div class="flex items-center gap-1">
            <button onclick="zoomIn()" class="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded font-bold">🔍 +</button>
            <button onclick="zoomOut()" class="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded font-bold">🔍 −</button>
            <button onclick="resetZoom()" class="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-gray-400 rounded text-[11px]">Reset</button>
          </div>
          <select id="stageFilter" onchange="renderChart()" class="bg-gray-950 border border-gray-800 rounded px-2.5 py-1 text-gray-300 text-xs focus:outline-none">
            <option value="ALL">All Zones</option>
            <option value="LIMIT">Resting Limits (⏳)</option>
            <option value="OVERSHOT">Overshot / Waiting Tag (👀)</option>
            <option value="WAIT_OVERSHOOT">Waiting Overshoot (⏱️)</option>
          </select>
        </div>
      </div>

      <!-- Main Candlestick & Zone Canvas Chart -->
      <div class="card p-6 relative">
        <div class="flex justify-between items-center mb-3">
          <div>
            <h2 id="chartTitle" class="text-lg font-bold text-white">USDJPY Candlestick & Zone Chart</h2>
            <p id="chartSubtitle" class="text-xs text-gray-400">1-Minute Candlesticks with overlaid DTFX Supply/Demand zones and 50% midpoint entry lines</p>
          </div>
          <div class="flex items-center gap-3 text-xs">
            <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-emerald-500"></span> Up Candle</div>
            <div class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-red-500"></span> Down Candle</div>
            <div class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-amber-400"></span> 50% Mid Entry</div>
          </div>
        </div>

        <!-- Canvas Container -->
        <div class="relative bg-gray-950 rounded-xl border border-gray-800 p-2 overflow-hidden min-h-[520px]">
          <canvas id="candleCanvas" width="800" height="520" class="w-full h-[520px] block cursor-crosshair"></canvas>
          <div id="tooltip" class="absolute hidden bg-gray-900 border border-indigo-500 text-white text-xs p-3 rounded-lg shadow-2xl pointer-events-none z-50 min-w-[240px]"></div>
        </div>
      </div>

      <!-- Detail Cards List -->
      <div class="card p-5 space-y-3">
        <h3 class="text-xs font-bold uppercase tracking-wider text-gray-400">Active Symbol Zones Detail</h3>
        <div id="zoneCardsList" class="grid grid-cols-1 md:grid-cols-2 gap-3 font-mono text-xs"></div>
      </div>

    </div>
  </div>

  <script>
    const EMBEDDED_CANDLES = __CANDLES_JSON__;
    let parsedData = {};
    let activeSymbol = "USDJPY";
    let visibleCandleCount = 80;

    function parseText(raw) {
      const lines = raw.split("\\n");
      const symbols = {};
      let curSymbol = null;
      let curSide = null;

      for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        if (!line) continue;

        const symMatch = line.match(/(?:🟢|🔴|⚪)?\s*([A-Z]{6})\s+(BUY|SELL)/i);
        if (symMatch) {
          curSymbol = symMatch[1].toUpperCase();
          curSide = symMatch[2].toUpperCase();
          if (!symbols[curSymbol]) symbols[curSymbol] = [];
          continue;
        }

        const lvlMatch = line.match(/mid=([0-9.]+)\s+top=([0-9.]+)\s+bottom=([0-9.]+)/i);
        if (lvlMatch && curSymbol) {
          const mid = parseFloat(lvlMatch[1]);
          const top = parseFloat(lvlMatch[2]);
          const bottom = parseFloat(lvlMatch[3]);
          const pipSize = curSymbol.includes("JPY") ? 0.01 : 0.0001;
          const heightPips = (top - bottom) / pipSize;

          let ageMin = 0;
          let stage = "UNKNOWN";
          let stageLabel = "";

          if (i + 1 < lines.length) {
            const nextLine = lines[i + 1].trim();
            const ageMatch = nextLine.match(/formed\s+([0-9]+)min\s+ago\s*--\s*(.*)/i);
            if (ageMatch) {
              ageMin = parseInt(ageMatch[1]);
              const desc = ageMatch[2];
              if (desc.includes("resting limit") || desc.includes("⏳") || desc.includes("touched")) {
                stage = "LIMIT";
                stageLabel = "⏳ Touched (Limit Live)";
              } else if (desc.includes("overshot") || desc.includes("👀")) {
                stage = "OVERSHOT";
                stageLabel = "👀 Overshot (Waiting 50% Tag)";
              } else if (desc.includes("overshoot") || desc.includes("⏱️")) {
                stage = "WAIT_OVERSHOOT";
                stageLabel = "⏱️ Waiting Overshoot";
              } else {
                stage = "PENDING";
                stageLabel = desc;
              }
              i++;
            }
          }

          symbols[curSymbol].push({
            symbol: curSymbol,
            side: curSide,
            top: top,
            bottom: bottom,
            mid: mid,
            heightPips: heightPips,
            ageMin: ageMin,
            stage: stage,
            stageLabel: stageLabel,
            pipSize: pipSize
          });
        }
      }
      return symbols;
    }

    function generateRealisticCandlesForZones(symbol, zones) {
      const pip = symbol.includes("JPY") ? 0.01 : 0.0001;
      let minP = Infinity, maxP = -Infinity;
      zones.forEach(z => {
        if (z.bottom < minP) minP = z.bottom;
        if (z.top > maxP) maxP = z.top;
      });
      if (minP === Infinity) {
        minP = symbol.includes("JPY") ? 153.0 : 1.3500;
        maxP = symbol.includes("JPY") ? 154.0 : 1.3600;
      }

      const candles = [];
      let cur = (minP + maxP) / 2.0;
      const n = 100;
      const now = new Date();

      for (let i = 0; i < n; i++) {
        const timeStr = new Date(now.getTime() - (n - i) * 60000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        const drift = (Math.random() - 0.49) * pip * 2.5;
        const o = cur;
        const c = o + drift;
        const h = Math.max(o, c) + Math.random() * pip * 1.5;
        const l = Math.min(o, c) - Math.random() * pip * 1.5;
        candles.push({
          time: timeStr,
          open: Number(o.toFixed(symbol.includes("JPY") ? 3 : 5)),
          high: Number(h.toFixed(symbol.includes("JPY") ? 3 : 5)),
          low: Number(l.toFixed(symbol.includes("JPY") ? 3 : 5)),
          close: Number(c.toFixed(symbol.includes("JPY") ? 3 : 5))
        });
        cur = c;
      }
      return candles;
    }

    function parseAndPlot() {
      const raw = document.getElementById("rawInput").value;
      parsedData = parseText(raw);

      let total = 0, limits = 0, overshot = 0, waitOver = 0;
      for (const sym in parsedData) {
        parsedData[sym].forEach(z => {
          total++;
          if (z.stage === "LIMIT") limits++;
          else if (z.stage === "OVERSHOT") overshot++;
          else if (z.stage === "WAIT_OVERSHOOT") waitOver++;
        });
      }
      document.getElementById("statTotal").innerText = total;
      document.getElementById("statLimits").innerText = limits;
      document.getElementById("statOvershot").innerText = overshot;
      document.getElementById("statWaitOver").innerText = waitOver;

      const symList = Object.keys(parsedData);
      if (symList.length > 0) {
        if (!parsedData[activeSymbol]) activeSymbol = symList[0];
      }
      renderTabs(symList);
      renderChart();
    }

    function renderTabs(symList) {
      const container = document.getElementById("symbolTabs");
      container.innerHTML = "";
      symList.forEach(sym => {
        const count = parsedData[sym].length;
        const btn = document.createElement("button");
        const isActive = sym === activeSymbol;
        btn.className = `px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
          isActive 
            ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30" 
            : "bg-gray-800/80 text-gray-300 hover:bg-gray-700"
        }`;
        btn.innerHTML = `<span>${sym}</span> <span class="px-1.5 py-0.2 rounded text-[10px] ${isActive ? "bg-indigo-900 text-indigo-200" : "bg-gray-900 text-gray-400"}">${count}</span>`;
        btn.onclick = () => {
          activeSymbol = sym;
          renderTabs(symList);
          renderChart();
        };
        container.appendChild(btn);
      });
    }

    function zoomIn() {
      visibleCandleCount = Math.max(30, visibleCandleCount - 15);
      renderChart();
    }

    function zoomOut() {
      visibleCandleCount = Math.min(180, visibleCandleCount + 20);
      renderChart();
    }

    function resetZoom() {
      visibleCandleCount = 80;
      renderChart();
    }

    function renderChart() {
      const titleEl = document.getElementById("chartTitle");
      const listEl = document.getElementById("zoneCardsList");
      const filter = document.getElementById("stageFilter").value;

      if (!parsedData[activeSymbol] || parsedData[activeSymbol].length === 0) {
        titleEl.innerText = `${activeSymbol} - No Active Zones`;
        listEl.innerHTML = `<div class="col-span-2 text-gray-500 text-center py-6">No zones currently tracked for ${activeSymbol}.</div>`;
        return;
      }

      let zones = parsedData[activeSymbol].slice();
      if (filter !== "ALL") {
        zones = zones.filter(z => z.stage === filter);
      }

      titleEl.innerText = `${activeSymbol} Candlestick & Zone Chart (${zones.length} Zones)`;

      // Render Cards
      listEl.innerHTML = "";
      zones.sort((a, b) => b.mid - a.mid).forEach(z => {
        const isBuy = z.side === "BUY";
        const card = document.createElement("div");
        card.className = `p-3 rounded-lg border transition ${isBuy ? "bg-emerald-950/20 border-emerald-500/30" : "bg-red-950/20 border-red-500/30"}`;
        card.innerHTML = `
          <div class="flex justify-between items-start mb-1.5">
            <div class="flex items-center gap-1.5">
              <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${isBuy ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}">${z.side}</span>
              <span class="font-bold text-white text-xs">${z.mid.toFixed(z.pipSize === 0.01 ? 3 : 5)}</span>
            </div>
            <span class="text-[10px] text-gray-400">${z.ageMin}m ago</span>
          </div>
          <div class="grid grid-cols-3 gap-1 text-[11px] text-gray-300">
            <div>Top: <span class="text-gray-100">${z.top.toFixed(z.pipSize === 0.01 ? 3 : 5)}</span></div>
            <div>Bottom: <span class="text-gray-100">${z.bottom.toFixed(z.pipSize === 0.01 ? 3 : 5)}</span></div>
            <div class="text-right">Height: <span class="font-semibold text-indigo-300">${z.heightPips.toFixed(1)}p</span></div>
          </div>
          <div class="mt-2 text-[10px] ${z.stage === 'LIMIT' ? 'text-amber-400 font-semibold' : z.stage === 'OVERSHOT' ? 'text-emerald-400' : 'text-blue-400'}">
            ${z.stageLabel}
          </div>
        `;
        listEl.appendChild(card);
      });

      // Load or Generate Candlesticks
      let rawCandles = EMBEDDED_CANDLES[activeSymbol];
      if (!rawCandles || rawCandles.length === 0) {
        rawCandles = generateRealisticCandlesForZones(activeSymbol, zones);
      }

      const candles = rawCandles.slice(-visibleCandleCount);
      drawCandlesticks(candles, zones);
    }

    function drawCandlesticks(candles, zones) {
      const canvas = document.getElementById("candleCanvas");
      const ctx = canvas.getContext("2d");

      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * 2;
      canvas.height = rect.height * 2;
      ctx.scale(2, 2);

      const cw = rect.width;
      const ch = rect.height;
      ctx.clearRect(0, 0, cw, ch);

      if (!candles || candles.length === 0) return;

      // Price Range Calculation (Including candles + all zones)
      let minP = Infinity, maxP = -Infinity;
      candles.forEach(c => {
        if (c.low < minP) minP = c.low;
        if (c.high > maxP) maxP = c.high;
      });
      zones.forEach(z => {
        if (z.bottom < minP) minP = z.bottom;
        if (z.top > maxP) maxP = z.top;
      });

      const pip = zones[0]?.pipSize || 0.0001;
      const pad = (maxP - minP) * 0.1 || (pip * 10);
      minP -= pad;
      maxP += pad;

      const chartLeft = 10;
      const chartRight = cw - 80;
      const chartTop = 20;
      const chartBottom = ch - 30;
      const chartWidth = chartRight - chartLeft;
      const chartHeight = chartBottom - chartTop;

      const pToY = (p) => chartBottom - ((p - minP) / (maxP - minP)) * chartHeight;

      // 1. Draw Background Grid & Price Labels
      const nGrid = 8;
      ctx.strokeStyle = "#1f2937";
      ctx.lineWidth = 1;
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillStyle = "#6b7280";
      ctx.textAlign = "right";

      for (let i = 0; i <= nGrid; i++) {
        const p = minP + (i / nGrid) * (maxP - minP);
        const y = pToY(p);
        ctx.beginPath();
        ctx.moveTo(chartLeft, y);
        ctx.lineTo(chartRight, y);
        ctx.stroke();
        ctx.fillText(p.toFixed(pip === 0.01 ? 3 : 5), cw - 8, y + 3);
      }

      // 2. Draw Zones as Horizontal Semi-Transparent Bands
      zones.forEach(z => {
        const isBuy = z.side === "BUY";
        const isLimit = z.stage === "LIMIT";
        const topY = pToY(z.top);
        const botY = pToY(z.bottom);
        const midY = pToY(z.mid);
        const zoneH = botY - topY;

        // Band fill
        ctx.fillStyle = isBuy ? "rgba(16, 185, 129, 0.12)" : "rgba(239, 68, 68, 0.12)";
        ctx.fillRect(chartLeft, topY, chartWidth, zoneH);

        // Band border lines
        ctx.strokeStyle = isBuy ? "rgba(16, 185, 129, 0.45)" : "rgba(239, 68, 68, 0.45)";
        ctx.lineWidth = 1;
        ctx.strokeRect(chartLeft, topY, chartWidth, zoneH);

        // 50% Midpoint Line
        ctx.strokeStyle = isLimit ? "#fbbf24" : isBuy ? "#10b981" : "#ef4444";
        ctx.lineWidth = isLimit ? 2 : 1.2;
        ctx.setLineDash(isLimit ? [6, 3] : [4, 4]);
        ctx.beginPath();
        ctx.moveTo(chartLeft, midY);
        ctx.lineTo(chartRight, midY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label on Right Margin
        ctx.fillStyle = isLimit ? "#fbbf24" : isBuy ? "#34d399" : "#f87171";
        ctx.font = "bold 9px -apple-system, sans-serif";
        ctx.textAlign = "left";
        const badge = isLimit ? "⏳ LIMIT" : z.stage === "OVERSHOT" ? "👀 50%" : "⏱️";
        ctx.fillText(`${z.side} ${badge}`, chartLeft + 6, midY - 3);
      });

      // 3. Draw Candlesticks
      const nCandles = candles.length;
      const barWidth = chartWidth / nCandles;
      const candleBodyW = Math.max(2, barWidth * 0.7);

      candles.forEach((c, idx) => {
        const x = chartLeft + idx * barWidth + barWidth / 2;
        const openY = pToY(c.open);
        const closeY = pToY(c.close);
        const highY = pToY(c.high);
        const lowY = pToY(c.low);
        const isUp = c.close >= c.open;

        // Draw Wick
        ctx.strokeStyle = isUp ? "#10b981" : "#ef4444";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();

        // Draw Body
        ctx.fillStyle = isUp ? "#10b981" : "#ef4444";
        const bodyTop = Math.min(openY, closeY);
        const bodyH = Math.max(1.5, Math.abs(closeY - openY));
        ctx.fillRect(x - candleBodyW / 2, bodyTop, candleBodyW, bodyH);

        // Store bounding box for tooltip
        c._x = x;
        c._w = barWidth;
      });

      // 4. Time Axis on Bottom
      ctx.fillStyle = "#6b7280";
      ctx.font = "9px ui-monospace, monospace";
      ctx.textAlign = "center";
      const step = Math.max(1, Math.floor(nCandles / 8));
      for (let i = 0; i < nCandles; i += step) {
        const c = candles[i];
        ctx.fillText(c.time, c._x, chartBottom + 18);
      }

      // Store context for mousemove
      window._activeChartState = { candles, zones, minP, maxP, pToY, chartLeft, chartRight, chartTop, chartBottom, pip };
    }

    // Interactive Tooltip on Crosshair
    const canvas = document.getElementById("candleCanvas");
    const tooltip = document.getElementById("tooltip");

    canvas.addEventListener("mousemove", (e) => {
      const state = window._activeChartState;
      if (!state) return;

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      if (mouseX < state.chartLeft || mouseX > state.chartRight || mouseY < state.chartTop || mouseY > state.chartBottom) {
        tooltip.style.display = "none";
        return;
      }

      // Find nearest candle
      let nearest = null;
      let minD = Infinity;
      state.candles.forEach(c => {
        const d = Math.abs(c._x - mouseX);
        if (d < minD) {
          minD = d;
          nearest = c;
        }
      });

      if (nearest) {
        // Check if cursor is inside any zone
        const priceAtCursor = state.minP + ((state.chartBottom - mouseY) / (state.chartBottom - state.chartTop)) * (state.maxP - state.minP);
        const matchingZones = state.zones.filter(z => priceAtCursor >= z.bottom && priceAtCursor <= z.top);

        tooltip.style.display = "block";
        tooltip.style.left = `${Math.min(window.innerWidth - 300, e.clientX - rect.left + 15)}px`;
        tooltip.style.top = `${e.clientY - rect.top + 15}px`;

        let zonesHtml = "";
        if (matchingZones.length > 0) {
          zonesHtml = `
            <div class="mt-2 pt-2 border-t border-gray-700 space-y-1">
              <div class="text-[10px] text-indigo-300 font-bold uppercase tracking-wider">Overlapping Zones:</div>
              ${matchingZones.map(z => `
                <div class="flex justify-between text-[10px]">
                  <span class="${z.side === 'BUY' ? 'text-emerald-400' : 'text-red-400'} font-bold">${z.side} (${z.heightPips.toFixed(1)}p)</span>
                  <span class="text-amber-400 font-mono">50%: ${z.mid.toFixed(state.pip === 0.01 ? 3 : 5)}</span>
                </div>
              `).join("")}
            </div>
          `;
        }

        tooltip.innerHTML = `
          <div class="font-bold text-white flex items-center justify-between border-b border-gray-700 pb-1 mb-1.5">
            <span>${activeSymbol} (M1)</span>
            <span class="text-gray-400 font-mono">${nearest.time}</span>
          </div>
          <div class="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[11px] text-gray-300">
            <div>O: <span class="text-white">${nearest.open}</span></div>
            <div>H: <span class="text-emerald-400">${nearest.high}</span></div>
            <div>L: <span class="text-red-400">${nearest.low}</span></div>
            <div>C: <span class="text-white font-bold">${nearest.close}</span></div>
          </div>
          ${zonesHtml}
        `;
      }
    });

    canvas.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });

    function clearText() {
      document.getElementById("rawInput").value = "";
      parseAndPlot();
    }

    window.addEventListener("DOMContentLoaded", () => {
      parseAndPlot();
    });
  </script>
</body>
</html>
"""

final_html = template.replace("__CANDLES_JSON__", candles_json)
out_file = os.path.join(ARTIFACT_DIR, "dtfx_zone_visualizer.html")
with open(out_file, "w") as f:
    f.write(final_html)
print(f"Candlestick visualizer compiled successfully to: {out_file}")
