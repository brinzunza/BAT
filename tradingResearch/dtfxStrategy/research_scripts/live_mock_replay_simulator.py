"""Interactive Live Mock Replay Backtesting Engine with Candlestick Charting.
Aesthetic inspired by minimal modern portfolio (JetBrains Mono, Inter, obsidian cards, crisp micro-borders).
"""
import os
import sys
import time
import random
import threading
import argparse
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "tradingResearch", "dtfxStrategy", "data")

sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "tradingResearch", "dtfxStrategy", "research_scripts"))
from dtfx_eurusd_backtest_strict import find_pivots_strict as find_pivots, run_structure_fsm as run_fsm

# Instruments
INSTRUMENT_FILES = {
    "EUR_USD": os.path.join(DATA_DIR, "lse_eurusd_1min_full.csv"),
    "GBP_USD": os.path.join(DATA_DIR, "lse_gbpusd_1min_full.csv"),
    "EUR_GBP": os.path.join(DATA_DIR, "lse_eurgbp_1min_full.csv"),
    "USD_JPY": os.path.join(DATA_DIR, "lse_usdjpy_1min_2018on.csv"),
}

PIP_SIZES = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "EUR_GBP": 0.0001,
    "USD_JPY": 0.01,
}

MIN_RETRACEMENT = 2
N_MULTIPLIER = 3.343496
MIN_FILL_AGE_MINUTES = 202  # 3.366 hours
MAX_TOUCHES = 4
START_BALANCE = 10000.00
RISK_PCT = 0.05

class MockReplayEngine:
    def __init__(self, symbol="EUR_USD", window_bars=3000, speed=30.0):
        self.symbol = symbol
        self.pip_size = PIP_SIZES.get(symbol, 0.0001)
        self.window_bars = window_bars
        self.speed = speed
        self.delay = 1.0 / max(speed, 1.0)
        self.running = False
        self.lock = threading.Lock()

        self.df_full = None
        self.times = None
        self.opens = None
        self.highs = None
        self.lows = None
        self.closes = None
        self.total_bars = 0
        
        self.start_bar_idx = 0
        self.current_bar_idx = 0
        self.warmup_bars = 500

        self.active_zones = []
        self.open_positions = []
        self.closed_trades = []
        self.balance_long = START_BALANCE / 2.0
        self.balance_short = START_BALANCE / 2.0
        self.initial_balance = START_BALANCE
        self.recent_bars = []

        self.load_data()
        self.pick_random_window()

    def load_data(self):
        filepath = INSTRUMENT_FILES.get(self.symbol)
        if not filepath or not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return
        df = pd.read_csv(filepath)
        for col in df.columns:
            if col.lower() in ("time", "date", "datetime", "timestamp"):
                df["date"] = pd.to_datetime(df[col])
                break
        df = df.set_index("date").sort_index()
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        self.df_full = df[["open", "high", "low", "close"]].dropna()
        self.times = self.df_full.index.astype(str).values
        self.opens = self.df_full["open"].values
        self.highs = self.df_full["high"].values
        self.lows = self.df_full["low"].values
        self.closes = self.df_full["close"].values
        self.total_bars = len(self.df_full)

    def pick_random_window(self, seed=None):
        with self.lock:
            if seed is not None:
                random.seed(seed)
            max_start = self.total_bars - self.window_bars - 100
            if max_start <= 1000:
                self.start_bar_idx = 1000
            else:
                self.start_bar_idx = random.randint(1000, max_start)
            
            self.current_bar_idx = self.start_bar_idx + self.warmup_bars
            self.active_zones = []
            self.open_positions = []
            self.closed_trades = []
            self.balance_long = self.initial_balance / 2.0
            self.balance_short = self.initial_balance / 2.0
            self.recent_bars = []

            self._warmup_state()

    def _warmup_state(self):
        warmup_df = self.df_full.iloc[self.start_bar_idx : self.current_bar_idx]
        pivots = find_pivots(warmup_df, min_retracement=MIN_RETRACEMENT)
        _, zones = run_fsm(pivots)

        current_time = pd.Timestamp(self.times[self.current_bar_idx])
        valid_zones = []
        for z in zones:
            z_top = z["top"]
            z_bot = z["bottom"]
            z_mid = z["mid"]
            z_side = z["side"]
            z_created = pd.Timestamp(z["created_at"])
            
            c_idx = np.searchsorted(self.df_full.index, z_created, side="right")
            sub_lows = self.lows[c_idx : self.current_bar_idx]
            sub_highs = self.highs[c_idx : self.current_bar_idx]
            
            invalid = False
            if len(sub_lows) > 0:
                if z_side == "buy" and np.min(sub_lows) <= z_bot:
                    invalid = True
                elif z_side == "sell" and np.max(sub_highs) >= z_top:
                    invalid = True
            
            if not invalid:
                age_mins = (current_time - z_created).total_seconds() / 60.0
                valid_zones.append({
                    "id": z.get("id", len(valid_zones) + 1),
                    "side": z_side,
                    "top": round(z_top, 5),
                    "bottom": round(z_bot, 5),
                    "mid": round(z_mid, 5),
                    "created_at": str(z_created),
                    "age_minutes": round(age_mins, 1),
                    "touches": 0,
                    "matured": age_mins >= MIN_FILL_AGE_MINUTES,
                    "status": "ARMED" if age_mins >= MIN_FILL_AGE_MINUTES else "MATURING"
                })

        self.active_zones = valid_zones[-12:]
        for idx in range(max(self.start_bar_idx, self.current_bar_idx - 120), self.current_bar_idx + 1):
            self.recent_bars.append({
                "time": self.times[idx],
                "open": float(self.opens[idx]),
                "high": float(self.highs[idx]),
                "low": float(self.lows[idx]),
                "close": float(self.closes[idx])
            })

    def step(self):
        with self.lock:
            if self.current_bar_idx >= self.start_bar_idx + self.window_bars or self.current_bar_idx >= self.total_bars - 1:
                self.running = False
                return False

            self.current_bar_idx += 1
            idx = self.current_bar_idx
            bar_time = self.times[idx]
            dt_time = pd.Timestamp(bar_time)
            b_open = float(self.opens[idx])
            b_high = float(self.highs[idx])
            b_low = float(self.lows[idx])
            b_close = float(self.closes[idx])

            self.recent_bars.append({
                "time": bar_time,
                "open": b_open,
                "high": b_high,
                "low": b_low,
                "close": b_close
            })
            if len(self.recent_bars) > 160:
                self.recent_bars.pop(0)

            # 1. Update Open Positions & Check TP / SL
            closed_this_bar = []
            for pos in self.open_positions:
                side = pos["side"]
                sl = pos["stop_loss"]
                tp = pos["take_profit"]
                entry = pos["entry_price"]
                risk_usd = pos["risk_usd"]

                if side == "buy":
                    pips = (b_close - entry) / self.pip_size
                    unrealized_r = (b_close - entry) / (entry - sl) if entry != sl else 0
                    hit_sl = b_low <= sl
                    hit_tp = b_high >= tp
                else:
                    pips = (entry - b_close) / self.pip_size
                    unrealized_r = (entry - b_close) / (sl - entry) if entry != sl else 0
                    hit_sl = b_high >= sl
                    hit_tp = b_low <= tp

                pos["current_price"] = b_close
                pos["unrealized_pips"] = round(pips, 1)
                pos["unrealized_r"] = round(unrealized_r, 2)
                pos["unrealized_usd"] = round(unrealized_r * risk_usd, 2)

                if hit_sl or hit_tp:
                    is_win = hit_tp and not (hit_sl and b_open <= sl)
                    realized_r = 1.0 if is_win else -1.0
                    realized_usd = risk_usd if is_win else -risk_usd

                    if side == "buy":
                        self.balance_long += realized_usd
                    else:
                        self.balance_short += realized_usd

                    self.closed_trades.append({
                        "id": pos["id"],
                        "side": side,
                        "entry_time": pos["entry_time"],
                        "exit_time": bar_time,
                        "entry_price": entry,
                        "exit_price": tp if is_win else sl,
                        "pnl_r": realized_r,
                        "pnl_usd": round(realized_usd, 2),
                        "outcome": "WIN" if is_win else "LOSS",
                        "touch": pos["touch"]
                    })
                    closed_this_bar.append(pos)

            for pos in closed_this_bar:
                self.open_positions.remove(pos)

            # 2. Update Active Zones Age & Status
            retained_zones = []
            for z in self.active_zones:
                created = pd.Timestamp(z["created_at"])
                age_mins = (dt_time - created).total_seconds() / 60.0
                z["age_minutes"] = round(age_mins, 1)
                z["matured"] = age_mins >= MIN_FILL_AGE_MINUTES

                invalidated = False
                if z["side"] == "buy" and b_low <= z["bottom"]:
                    invalidated = True
                elif z["side"] == "sell" and b_high >= z["top"]:
                    invalidated = True

                if invalidated:
                    continue

                if z["matured"] and z["touches"] < MAX_TOUCHES:
                    z["status"] = "ARMED"
                    mid_hit = (b_low <= z["mid"]) if z["side"] == "buy" else (b_high >= z["mid"])
                    
                    has_open_for_zone = any(p["zone_id"] == z["id"] for p in self.open_positions)
                    if mid_hit and not has_open_for_zone:
                        z["touches"] += 1
                        sizing_base = 2.0 * min(self.balance_long, self.balance_short)
                        risk_usd = sizing_base * RISK_PCT
                        
                        zone_width = z["top"] - z["bottom"]
                        risk_pips = (N_MULTIPLIER * zone_width / 2.0)
                        
                        entry_p = z["mid"]
                        sl_p = entry_p - risk_pips if z["side"] == "buy" else entry_p + risk_pips
                        tp_p = entry_p + risk_pips if z["side"] == "buy" else entry_p - risk_pips

                        self.open_positions.append({
                            "id": f"T-{len(self.closed_trades) + len(self.open_positions) + 1}",
                            "zone_id": z["id"],
                            "side": z["side"],
                            "touch": z["touches"],
                            "entry_time": bar_time,
                            "entry_price": round(entry_p, 5),
                            "stop_loss": round(sl_p, 5),
                            "take_profit": round(tp_p, 5),
                            "current_price": b_close,
                            "risk_usd": round(risk_usd, 2),
                            "unrealized_pips": 0.0,
                            "unrealized_r": 0.0,
                            "unrealized_usd": 0.0
                        })
                else:
                    z["status"] = "MATURING" if not z["matured"] else "TOUCH_CAP"

                retained_zones.append(z)

            # 3. Detect new structure zones
            if idx % 5 == 0:
                lookback_df = self.df_full.iloc[max(0, idx - 1000) : idx + 1]
                pivots = find_pivots(lookback_df, min_retracement=MIN_RETRACEMENT)
                _, new_zones = run_fsm(pivots)
                if new_zones:
                    latest = new_zones[-1]
                    exists = any(abs(z["top"] - latest["top"]) < 1e-5 and abs(z["bottom"] - latest["bottom"]) < 1e-5 for z in retained_zones)
                    if not exists:
                        created = pd.Timestamp(latest["created_at"])
                        age_m = (dt_time - created).total_seconds() / 60.0
                        retained_zones.append({
                            "id": len(self.closed_trades) + len(retained_zones) + 1,
                            "side": latest["side"],
                            "top": round(latest["top"], 5),
                            "bottom": round(latest["bottom"], 5),
                            "mid": round(latest["mid"], 5),
                            "created_at": str(created),
                            "age_minutes": round(age_m, 1),
                            "touches": 0,
                            "matured": age_m >= MIN_FILL_AGE_MINUTES,
                            "status": "ARMED" if age_m >= MIN_FILL_AGE_MINUTES else "MATURING"
                        })

            self.active_zones = retained_zones[-8:]
            return True

    def get_state(self):
        with self.lock:
            sizing_base = 2.0 * min(self.balance_long, self.balance_short)
            combined_bal = self.balance_long + self.balance_short
            floating_pnl_usd = sum(p.get("unrealized_usd", 0.0) for p in self.open_positions)
            total_equity = combined_bal + floating_pnl_usd

            wins = sum(1 for t in self.closed_trades if t["outcome"] == "WIN")
            total_c = len(self.closed_trades)
            win_rate = (wins / total_c * 100.0) if total_c > 0 else 0.0
            tot_r = sum(t["pnl_r"] for t in self.closed_trades)

            curr_time = self.times[self.current_bar_idx] if self.current_bar_idx < self.total_bars else "N/A"
            curr_price = float(self.closes[self.current_bar_idx]) if self.current_bar_idx < self.total_bars else 0.0

            return {
                "symbol": self.symbol,
                "current_time": curr_time,
                "current_price": round(curr_price, 5),
                "running": self.running,
                "speed": self.speed,
                "progress_bar": f"{self.current_bar_idx - self.start_bar_idx} / {self.window_bars}",
                "account": {
                    "balance_long": round(self.balance_long, 2),
                    "balance_short": round(self.balance_short, 2),
                    "sizing_base": round(sizing_base, 2),
                    "combined_balance": round(combined_bal, 2),
                    "floating_pnl_usd": round(floating_pnl_usd, 2),
                    "total_equity": round(total_equity, 2),
                    "realized_pnl_r": round(tot_r, 1),
                    "total_trades": total_c,
                    "win_rate": round(win_rate, 1)
                },
                "active_zones": self.active_zones,
                "open_positions": self.open_positions,
                "closed_trades": self.closed_trades[-8:],
                "recent_bars": self.recent_bars[-75:]
            }

app = Flask(__name__)
engine = None

# Minimalist Black & White UI with colored Candlesticks & Dollar Values
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>bruninvestor // live market replay</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #000000;
            --surface: #0a0a0a;
            --surface-elevated: #141414;
            --border: #262626;
            --border-hover: #404040;
            --text: #ffffff;
            --text-secondary: #a3a3a3;
            --text-tertiary: #525252;
            
            /* ONLY Candlesticks and Dollar Values are colored */
            --candle-green: #10b981;
            --candle-red: #ef4444;
            --dollar-green: #10b981;
            --dollar-red: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, sans-serif;
            padding: 24px 32px;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }

        .font-mono { font-family: 'JetBrains Mono', monospace; }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }

        .brand {
            display: flex;
            align-items: baseline;
            gap: 12px;
        }

        .brand-title {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #ffffff;
        }

        .brand-badge {
            font-size: 11px;
            color: #ffffff;
            background: #171717;
            border: 1px solid #333333;
            padding: 2px 8px;
            border-radius: 9999px;
            letter-spacing: 0.5px;
            font-weight: 500;
        }

        .clock-ticker {
            font-size: 12px;
            color: var(--text-tertiary);
            letter-spacing: 0.5px;
        }

        /* Controls */
        .controls-bar {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        select, button {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 7px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.15s ease;
            outline: none;
        }

        select:hover, button:hover {
            border-color: var(--border-hover);
            background: var(--surface-elevated);
        }

        button.primary {
            background: #ffffff;
            color: #000000;
            border-color: #ffffff;
            font-weight: 600;
        }
        button.primary:hover {
            background: #e5e5e5;
        }

        button.danger {
            background: #171717;
            color: #ffffff;
            border-color: #525252;
        }

        /* KPI Cards Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        .kpi-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 20px;
            transition: border-color 0.2s;
        }
        .kpi-card:hover { border-color: var(--border-hover); }

        .kpi-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 6px;
            font-weight: 500;
        }

        .kpi-value {
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        /* Main 2-Column Split */
        .main-layout {
            display: grid;
            grid-template-columns: 1fr 360px;
            gap: 24px;
        }

        /* Candlestick Canvas Container */
        .chart-container {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 24px;
            position: relative;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .chart-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .price-live {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #ffffff;
        }

        #candlestickCanvas {
            width: 100%;
            height: 420px;
            display: block;
            border-radius: 6px;
            background: #000000;
        }

        /* Panels & Tables */
        .card-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 24px;
        }

        .panel-heading {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        th {
            text-align: left;
            padding: 8px 12px;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-tertiary);
            border-bottom: 1px solid var(--border);
            font-weight: 600;
        }

        td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-family: 'JetBrains Mono', monospace;
            color: #ffffff;
        }

        /* Pure Monochrome Pills */
        .pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            border: 1px solid var(--border);
            background: #171717;
            color: #ffffff;
        }
        .pill-win { border-color: #525252; color: #ffffff; }
        .pill-loss { border-color: #333333; color: #a3a3a3; }
        .pill-armed { background: #ffffff; color: #000000; font-weight: 700; border: none; }
        .pill-maturing { background: #1a1a1a; color: #737373; }

        /* ONLY Dollar Values are Colored */
        .dollar-pos { color: var(--dollar-green); font-weight: 700; }
        .dollar-neg { color: var(--dollar-red); font-weight: 700; }
        .dollar-neutral { color: #ffffff; font-weight: 700; }

        /* Zone Card */
        .zone-card {
            background: #0d0d0d;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 8px;
            transition: all 0.15s;
        }
        .zone-card:hover { border-color: var(--border-hover); background: #141414; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span class="brand-title font-mono">bruninvestor</span>
            <span class="brand-badge font-mono">PROD v1.0 • REPLAY</span>
            <span class="clock-ticker font-mono" id="currentTimeLabel">--</span>
        </div>
        <div class="controls-bar">
            <select id="symbolSelect" onchange="changeSymbol()">
                <option value="EUR_USD">EUR / USD</option>
                <option value="GBP_USD">GBP / USD</option>
                <option value="EUR_GBP">EUR / GBP</option>
                <option value="USD_JPY">USD / JPY</option>
            </select>
            <select id="speedSelect" onchange="changeSpeed()">
                <option value="5">5x speed</option>
                <option value="15">15x speed</option>
                <option value="30" selected>30x speed</option>
                <option value="60">60x speed</option>
                <option value="120">120x speed</option>
            </select>
            <button class="primary" id="btnPlay" onclick="togglePlay()">play ▶</button>
            <button onclick="stepBar()">step ⏭</button>
            <button onclick="resetRandom()">🎲 random date</button>
        </div>
    </header>

    <!-- Top KPI Grid -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label font-mono">Combined Balance</div>
            <div class="kpi-value font-mono dollar-neutral" id="kpiBalance">$10,000.00</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label font-mono">Live Floating P&L</div>
            <div class="kpi-value font-mono" id="kpiFloating">$0.00</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label font-mono">Sizing Base (2×min)</div>
            <div class="kpi-value font-mono dollar-neutral" id="kpiSizingBase">$10,000.00</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label font-mono">Cumulative Edge</div>
            <div class="kpi-value font-mono" style="color: #ffffff;" id="kpiRealizedR">+0.0 R</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label font-mono">Closed Win Rate</div>
            <div class="kpi-value font-mono" style="color: #ffffff;" id="kpiWinRate">0.0% (0)</div>
        </div>
    </div>

    <!-- Main 2-Column Layout -->
    <div class="main-layout">
        <!-- Left: Candlestick Chart & Open Positions -->
        <div>
            <div class="chart-container">
                <div class="chart-header">
                    <span class="chart-title font-mono" id="chartSymbol">1-MIN CANDLESTICK STREAM</span>
                    <span class="price-live font-mono" id="priceLive">1.00000</span>
                </div>
                <canvas id="candlestickCanvas"></canvas>
            </div>

            <!-- Active Open Positions -->
            <div class="card-panel">
                <div class="panel-heading font-mono">
                    <span>Live Active Positions (<span id="openCount">0</span>)</span>
                    <span style="color: #a3a3a3;">1:1 BRACKET EXECUTION</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Trade ID</th>
                            <th>Side</th>
                            <th>Touch</th>
                            <th>Entry Price</th>
                            <th>Stop Loss</th>
                            <th>Take Profit</th>
                            <th>Current</th>
                            <th>Floating (Pips)</th>
                            <th>Floating (R)</th>
                            <th>Floating ($)</th>
                        </tr>
                    </thead>
                    <tbody id="openPositionsTbody">
                        <tr><td colspan="10" style="text-align: center; color: var(--text-tertiary); padding: 24px;">No open positions currently active.</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Closed Executions Log -->
            <div class="card-panel">
                <div class="panel-heading font-mono">
                    <span>Recent Closed Trades</span>
                    <span style="color: var(--text-tertiary);">EXECUTIONS</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Side</th>
                            <th>Entry Time</th>
                            <th>Exit Time</th>
                            <th>Outcome</th>
                            <th>P&L (R)</th>
                            <th>P&L ($)</th>
                        </tr>
                    </thead>
                    <tbody id="closedTradesTbody">
                        <tr><td colspan="7" style="text-align: center; color: var(--text-tertiary); padding: 24px;">No closed trades yet in this window.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Right: Active Valid Zones & Sub-Accounts -->
        <div>
            <div class="card-panel">
                <div class="panel-heading font-mono">
                    <span>Active Zones (<span id="zoneCount">0</span>)</span>
                    <span style="color: #a3a3a3;">STRUCTURE FSM</span>
                </div>
                <div id="zonesList">
                    <div style="color: var(--text-tertiary); font-size: 12px; text-align: center; padding: 24px;">Scanning structure...</div>
                </div>
            </div>

            <div class="card-panel">
                <div class="panel-heading font-mono"><span>Account Balancing</span></div>
                <div style="font-size: 12px; line-height: 2;" class="font-mono">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-secondary);">Long Sub-Account:</span>
                        <span id="longBal" class="dollar-neutral">$5,000.00</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-secondary);">Short Sub-Account:</span>
                        <span id="shortBal" class="dollar-neutral">$5,000.00</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-top: 1px solid var(--border); padding-top: 8px; margin-top: 8px;">
                        <span style="color: #ffffff;">Risk / Trade (5%):</span>
                        <span id="riskPerTrade" class="dollar-neutral" style="font-weight: 700;">$500.00</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('candlestickCanvas');
        const ctx = canvas.getContext('2d');
        let isRunning = false;
        let lastState = null;

        function resizeCanvas() {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            if (lastState) renderCandlesticks(lastState);
        }
        window.addEventListener('resize', resizeCanvas);

        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                lastState = data;
                updateDashboard(data);
                renderCandlesticks(data);
            } catch (e) {
                console.error(e);
            }
        }

        function renderCandlesticks(data) {
            const rect = canvas.getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;

            ctx.clearRect(0, 0, width, height);

            const bars = data.recent_bars;
            if (!bars || bars.length === 0) return;

            let minP = Infinity;
            let maxP = -Infinity;
            for (let b of bars) {
                if (b.low < minP) minP = b.low;
                if (b.high > maxP) maxP = b.high;
            }

            for (let z of (data.active_zones || [])) {
                if (z.bottom < minP) minP = z.bottom;
                if (z.top > maxP) maxP = z.top;
            }
            for (let p of (data.open_positions || [])) {
                if (p.stop_loss < minP) minP = p.stop_loss;
                if (p.take_profit > maxP) maxP = p.take_profit;
            }

            const pRange = maxP - minP || 0.001;
            const padTop = 30;
            const padBottom = 30;
            const chartH = height - padTop - padBottom;

            function getY(price) {
                return padTop + chartH * (1 - (price - minP) / pRange);
            }

            const nBars = bars.length;
            const candleW = Math.max(2, (width - 80) / nBars);
            const bodyW = Math.max(1.5, candleW * 0.7);

            // 1. Grid Lines (Subtle Monochrome)
            ctx.strokeStyle = '#171717';
            ctx.fillStyle = '#525252';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';

            const steps = 5;
            for (let i = 0; i <= steps; i++) {
                const p = minP + (pRange * i) / steps;
                const y = getY(p);
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width - 60, y);
                ctx.stroke();
                ctx.fillText(p.toFixed(5), width - 8, y + 3);
            }

            // 2. Active Zones (Subtle Monochromatic Gray Bands)
            for (let z of (data.active_zones || [])) {
                const yTop = getY(z.top);
                const yBot = getY(z.bottom);
                const yMid = getY(z.mid);

                ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
                ctx.fillRect(0, Math.min(yTop, yBot), width - 60, Math.abs(yBot - yTop));

                // Midpoint line
                ctx.strokeStyle = '#404040';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(0, yMid);
                ctx.lineTo(width - 60, yMid);
                ctx.stroke();
                ctx.setLineDash([]);

                // Midpoint Tag
                ctx.fillStyle = '#737373';
                ctx.fillText(`${z.side.toUpperCase()} MID: ${z.mid.toFixed(5)}`, width - 70, yMid - 4);
            }

            // 3. Active Position Brackets (Monochrome with dashed indicators)
            for (let p of (data.open_positions || [])) {
                const yEntry = getY(p.entry_price);
                const ySL = getY(p.stop_loss);
                const yTP = getY(p.take_profit);

                // SL Line
                ctx.strokeStyle = '#525252';
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(0, ySL);
                ctx.lineTo(width - 60, ySL);
                ctx.stroke();
                ctx.fillStyle = '#a3a3a3';
                ctx.fillText(`SL: ${p.stop_loss.toFixed(5)}`, width - 70, ySL - 3);

                // TP Line
                ctx.strokeStyle = '#525252';
                ctx.beginPath();
                ctx.moveTo(0, yTP);
                ctx.lineTo(width - 60, yTP);
                ctx.stroke();
                ctx.fillText(`TP: ${p.take_profit.toFixed(5)}`, width - 70, yTP - 3);

                // Entry Line
                ctx.strokeStyle = '#ffffff';
                ctx.beginPath();
                ctx.moveTo(0, yEntry);
                ctx.lineTo(width - 60, yEntry);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // 4. Candlesticks (COLOR RETAINED)
            for (let i = 0; i < nBars; i++) {
                const b = bars[i];
                const x = 20 + i * candleW;
                const isBull = b.close >= b.open;
                const color = isBull ? '#10b981' : '#ef4444';

                const yHigh = getY(b.high);
                const yLow = getY(b.low);
                const yOpen = getY(b.open);
                const yClose = getY(b.close);

                // Draw Wick
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(x, yHigh);
                ctx.lineTo(x, yLow);
                ctx.stroke();

                // Draw Body
                ctx.fillStyle = color;
                const bodyY = Math.min(yOpen, yClose);
                const bodyH = Math.max(1.5, Math.abs(yClose - yOpen));
                ctx.fillRect(x - bodyW / 2, bodyY, bodyW, bodyH);
            }

            // 5. Current Price Line Pulse
            const lastBar = bars[bars.length - 1];
            const currY = getY(lastBar.close);

            ctx.strokeStyle = '#404040';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(0, currY);
            ctx.lineTo(width - 60, currY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Price badge on right axis
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(width - 62, currY - 9, 60, 18);
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 10px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText(lastBar.close.toFixed(5), width - 32, currY + 4);
        }

        function updateDashboard(data) {
            document.getElementById('currentTimeLabel').innerText = `${data.symbol.replace('_', '/')} • ${data.current_time} (${data.progress_bar})`;
            document.getElementById('priceLive').innerText = data.current_price.toFixed(5);
            document.getElementById('chartSymbol').innerText = `${data.symbol.replace('_', '/')} LIVE CANDLESTICKS (1-MIN)`;

            const acc = data.account;
            document.getElementById('kpiBalance').innerText = '$' + acc.combined_balance.toLocaleString(undefined, {minimumFractionDigits:2});
            
            const flEl = document.getElementById('kpiFloating');
            flEl.innerText = (acc.floating_pnl_usd >= 0 ? '+$' : '-$') + Math.abs(acc.floating_pnl_usd).toFixed(2);
            flEl.className = 'kpi-value font-mono ' + (acc.floating_pnl_usd >= 0 ? 'dollar-pos' : 'dollar-neg');

            document.getElementById('kpiSizingBase').innerText = '$' + acc.sizing_base.toLocaleString(undefined, {minimumFractionDigits:2});
            document.getElementById('kpiRealizedR').innerText = (acc.realized_pnl_r >= 0 ? '+' : '') + acc.realized_pnl_r + ' R';
            document.getElementById('kpiWinRate').innerText = `${acc.win_rate}% (${acc.total_trades})`;

            document.getElementById('longBal').innerText = '$' + acc.balance_long.toLocaleString(undefined, {minimumFractionDigits:2});
            document.getElementById('shortBal').innerText = '$' + acc.balance_short.toLocaleString(undefined, {minimumFractionDigits:2});
            document.getElementById('riskPerTrade').innerText = '$' + (acc.sizing_base * 0.05).toLocaleString(undefined, {minimumFractionDigits:2});

            // Open Positions Table
            const opBody = document.getElementById('openPositionsTbody');
            document.getElementById('openCount').innerText = data.open_positions.length;
            if (data.open_positions.length === 0) {
                opBody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-tertiary); padding: 24px;">No open positions currently active.</td></tr>';
            } else {
                opBody.innerHTML = data.open_positions.map(p => `
                    <tr>
                        <td style="font-weight:700;">${p.id}</td>
                        <td><span class="pill">${p.side.toUpperCase()}</span></td>
                        <td>T#${p.touch}</td>
                        <td>${p.entry_price.toFixed(5)}</td>
                        <td style="color:var(--text-secondary);">${p.stop_loss.toFixed(5)}</td>
                        <td style="color:var(--text-secondary);">${p.take_profit.toFixed(5)}</td>
                        <td style="font-weight:700;">${p.current_price.toFixed(5)}</td>
                        <td style="color:#ffffff;">${p.unrealized_pips >= 0 ? '+' : ''}${p.unrealized_pips}</td>
                        <td style="color:#ffffff;">${p.unrealized_r >= 0 ? '+' : ''}${p.unrealized_r} R</td>
                        <td class="${p.unrealized_usd >= 0 ? 'dollar-pos' : 'dollar-neg'}">${p.unrealized_usd >= 0 ? '+$' : '-$'}${Math.abs(p.unrealized_usd).toFixed(2)}</td>
                    </tr>
                `).join('');
            }

            // Closed Trades Table
            const clBody = document.getElementById('closedTradesTbody');
            if (data.closed_trades.length === 0) {
                clBody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-tertiary); padding: 24px;">No closed trades yet in this window.</td></tr>';
            } else {
                clBody.innerHTML = data.closed_trades.slice().reverse().map(t => `
                    <tr>
                        <td>${t.id}</td>
                        <td><span class="pill">${t.side.toUpperCase()}</span></td>
                        <td style="color:var(--text-tertiary);">${t.entry_time.split(' ')[1] || t.entry_time}</td>
                        <td style="color:var(--text-tertiary);">${t.exit_time.split(' ')[1] || t.exit_time}</td>
                        <td><span class="pill ${t.outcome === 'WIN' ? 'pill-win' : 'pill-loss'}">${t.outcome}</span></td>
                        <td style="color:#ffffff;">${t.pnl_r > 0 ? '+' : ''}${t.pnl_r} R</td>
                        <td class="${t.pnl_usd > 0 ? 'dollar-pos' : 'dollar-neg'}">${t.pnl_usd > 0 ? '+$' : '-$'}${Math.abs(t.pnl_usd).toFixed(2)}</td>
                    </tr>
                `).join('');
            }

            // Active Zones
            const zCont = document.getElementById('zonesList');
            document.getElementById('zoneCount').innerText = data.active_zones.length;
            if (data.active_zones.length === 0) {
                zCont.innerHTML = '<div style="color: var(--text-tertiary); font-size: 12px; text-align: center; padding: 24px;">No uninvalidated zones found.</div>';
            } else {
                zCont.innerHTML = data.active_zones.slice().reverse().map(z => `
                    <div class="zone-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span class="font-mono" style="font-weight: 700; font-size: 12px; color: #ffffff;">${z.side.toUpperCase()} ZONE #${z.id}</span>
                            <span class="pill ${z.status === 'ARMED' ? 'pill-armed' : 'pill-maturing'}">${z.status}</span>
                        </div>
                        <div class="font-mono" style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary);">
                            <span>Mid: <b style="color: #ffffff;">${z.mid.toFixed(5)}</b></span>
                            <span>Age: ${(z.age_minutes/60).toFixed(1)}h</span>
                            <span>Touches: ${z.touches}/4</span>
                        </div>
                        <div class="font-mono" style="font-size: 10px; color: var(--text-tertiary); margin-top: 4px;">
                            [${z.bottom.toFixed(5)} - ${z.top.toFixed(5)}]
                        </div>
                    </div>
                `).join('');
            }

            isRunning = data.running;
            const btn = document.getElementById('btnPlay');
            btn.innerText = isRunning ? 'pause ⏸' : 'play ▶';
            btn.className = isRunning ? 'danger' : 'primary';
        }

        async function togglePlay() {
            await fetch('/api/' + (isRunning ? 'pause' : 'play'), { method: 'POST' });
            fetchState();
        }

        async function stepBar() {
            await fetch('/api/step', { method: 'POST' });
            fetchState();
        }

        async function changeSpeed() {
            const spd = document.getElementById('speedSelect').value;
            await fetch('/api/speed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speed: parseFloat(spd) })
            });
        }

        async function changeSymbol() {
            const sym = document.getElementById('symbolSelect').value;
            await fetch('/api/symbol', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: sym })
            });
            fetchState();
        }

        async function resetRandom() {
            await fetch('/api/reset', { method: 'POST' });
            fetchState();
        }

        window.onload = () => {
            resizeCanvas();
            fetchState();
            setInterval(fetchState, 100);
        };
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_DASHBOARD)

@app.route("/api/state")
def api_state():
    return jsonify(engine.get_state())

@app.route("/api/play", methods=["POST"])
def api_play():
    engine.running = True
    return jsonify({"status": "running"})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    engine.running = False
    return jsonify({"status": "paused"})

@app.route("/api/step", methods=["POST"])
def api_step():
    res = engine.step()
    return jsonify({"stepped": res})

@app.route("/api/speed", methods=["POST"])
def api_speed():
    data = request.json or {}
    engine.speed = float(data.get("speed", 30.0))
    engine.delay = 1.0 / max(engine.speed, 1.0)
    return jsonify({"speed": engine.speed})

@app.route("/api/symbol", methods=["POST"])
def api_symbol():
    data = request.json or {}
    sym = data.get("symbol", "EUR_USD")
    engine.symbol = sym
    engine.load_data()
    engine.pick_random_window()
    return jsonify({"symbol": engine.symbol})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    engine.pick_random_window()
    return jsonify({"status": "reset"})

def background_playback_loop():
    while True:
        if engine and engine.running:
            more = engine.step()
            if not more:
                engine.running = False
        time.sleep(engine.delay if engine else 0.05)

def main():
    global engine
    parser = argparse.ArgumentParser(description="Live Mock Replay Simulator")
    parser.add_argument("--symbol", default="EUR_USD", help="Symbol: EUR_USD, GBP_USD, EUR_GBP, USD_JPY")
    parser.add_argument("--speed", type=float, default=30.0, help="Replay speed multiplier")
    parser.add_argument("--port", type=int, default=8050, help="Web dashboard port")
    args = parser.parse_args()

    engine = MockReplayEngine(symbol=args.symbol, speed=args.speed)

    t = threading.Thread(target=background_playback_loop, daemon=True)
    t.start()
    print(f"\n" + "=" * 90)
    print(f"  🚀 BRUNINVESTOR LIVE CANDLESTICK REPLAY STARTED")
    print(f"  🌐 Open in your browser: http://localhost:{args.port}")
    print(f"  ⚡ Running at {args.speed}x Speed on {args.symbol}")
    print("=" * 90 + "\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
