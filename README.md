<div style="margin-bottom: 20px;">
  <img src="bdavid.png" alt="BAT Trading System" width="100%" style="display: block;"/>
</div>

# BAT [Bruninvestor Algorithmic Trading]

BAT is a comprehensive algorithmic trading system that supports both **stocks and cryptocurrencies** with multiple trading strategies and execution modes.

## Disclaimer
This project is for educational purposes only. It is not financial advice. Always conduct thorough research or consult with a professional financial advisor before making financial or investment decisions.

## Repository layout

```
configs/        data.yaml  model.yaml  training.yaml
data/           raw/  processed/  external/              (contents git-ignored)
src/
  data/         download, preprocess, features (indicators + build_features), dataset, dataloader, providers/ (Polygon/Alpaca/OANDA/synth)
  models/       mlp  cnn  lstm  transformer  factory  regime_switching/ (Bayesian HMM)
  training/     trainer  losses  optimizers  schedulers  callbacks
  evaluation/   metrics  walk_forward  backtest (BacktestEngine)
  inference/    predict  export  api (FastAPI)
  execution/    live_engine  brokers  ib_broker           (live/paper trading)
  strategies/   rule-based strategies (MA, RSI, MACD, Bollinger, SMC, ...)
  core/         hexagonal domain/ports/adapters/application (see docs/architecture)
  analytics/    prodlive/  queries over the PRODLIVETRADING bot database
  ui/           classic prompt-based CLI + matplotlib live chart (python3 main.py --classic)
  tui/          full-screen terminal UI (Textual): charts, tables, backtest/live/research/ML/analytics pages
  utils/        device  checkpoint  logging  seed
experiments/    notebooks, scripts/, research/ (parameter optimisation, zone analysis)
tests/  checkpoints/  runs/  docs/
tradingResearch/  strategy research workspace (kept in place; ~23 GB of data, hard-coded paths)
train.py  evaluate.py  predict.py  main.py (terminal UI; --classic for the old CLI)
```

```bash
python3 train.py --name run1 --override model.name=transformer      # trains, writes checkpoints/run1.pt and runs/run1/metrics.jsonl
python3 evaluate.py --ckpt checkpoints/run1.pt                      # held-out test split
python3 evaluate.py --walk-forward --cost 0.0001                    # expanding-window walk-forward
python3 predict.py --ckpt checkpoints/run1.pt --bars recent.csv
python3 main.py                                                     # terminal UI (backtest / live / research / ML / analytics)
python3 main.py --classic                                           # the original prompt-based CLI
```

## Trading Modes

### 1. Backtesting Mode
- Test strategies on historical data
- Custom time periods
- Risk-free strategy validation
- Interactive performance charts
- Detailed trade analysis

### 2. Live Trading Mode
- Real-time trading execution
- Live candlestick charts
- Automated signal processing
- Real-time P&L tracking

### 3. Research and Optimization
- Data collection for different tickers
- Parameter tuning and testing
- Efficient optimization for paramters

## Features

- **Unified Trading Platform** - Trade both stocks and crypto from one interface
- **Multiple Strategies** - Bollinger Bands, RSI, MACD, Moving Averages
- **Dual Trading Modes** - Backtesting and Live Trading
- **Real-time Charts** - Interactive candlestick charts with indicators
- **Multiple Brokers** - Alpaca Paper Trading and Simulated Broker
- **Comprehensive Analysis** - Detailed performance metrics and visualizations

## Quick Start

### Main Application 
```bash
pip install -r requirements.txt
python3 main.py            # full-screen terminal UI
python3 main.py --classic  # original prompt-based CLI
```
The terminal UI covers the whole project from one window: backtesting with interactive charts and risk metrics,
live trading (offline demo market, simulated broker, Alpaca / OANDA / IB), dataset management and parameter
optimisation, model training / evaluation, and production analytics. See [docs/tui.md](docs/tui.md) for the page
guide, keyboard shortcuts and how to extend it. It falls back to the classic CLI automatically when there is no
interactive terminal or `textual` is missing.

## Monorepo layout & live-bot analytics

`tradingResearch/` (strategy research, `quant_engine`, BEID/DTFX studies; merged here with its full git history) lives alongside the BAT engine.
The old path `~/Documents/GitHub/tradingResearch` is a symlink to `BAT/tradingResearch` so hard-coded absolute paths keep working.

`src/analytics/prodlive/` queries the live bot (**PRODLIVETRADING**, separate repo/VM) through a read-only snapshot of its SQLite store:
```bash
export PRODLIVE_SSH="user@vm-host:/home/user/PRODLIVETRADING/logs/practice/trades.db"   # or PRODLIVE_DB=/local/path/trades.db
python3 -m src.analytics.prodlive sync                      # consistent sqlite-backup copy of the bot DB -> src/analytics/prodlive/data/latest.db
python3 -m src.analytics.prodlive summary [--symbol EUR_USD --since 2026-09-01 --json]
python3 -m src.analytics.prodlive by symbol|side|entry_hour_utc|outcome|weekday|month
python3 -m src.analytics.prodlive weekly | equity | trades | events | open | plot
python3 -m src.analytics.prodlive expect --win-rate 0.45 --avg-r 0.1     # live vs backtest expectation (z / t)
python3 -m src.analytics.prodlive sql "select symbol,count(*) from trades group by 1"
```
In Python: `from src.analytics.prodlive import queries as Q; Q.summary(); Q.breakdown("symbol")`.
