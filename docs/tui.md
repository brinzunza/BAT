# BAT terminal UI

`python3 main.py` opens a full-screen terminal application built with [Textual](https://textual.textualize.io).
`python3 main.py --classic` starts the original prompt-based CLI; the launcher also falls back to it when there is no
interactive terminal or `textual` is not installed. Run the UI module directly with `python3 -m src.tui`.

Everything is drawn with text: braille dots (2×4 sub-cell resolution) for lines, areas and bars, half-block glyphs for
candlesticks, background colours for heat maps. No images, no plotting library, any truecolor terminal works.

## Pages

| Key | Page | What it does |
|---|---|---|
| `1` | Home | Dataset overview with trend sparklines, recent backtests, connection / ML / production status, activity feed. |
| `2` | Backtest | Form-driven runs on local CSVs or a connected provider (**Run ▶** or `ctrl+r`). A **run log** reports every step — data checks, indicator problems, signal counts, warnings — and the simulation is **replayed bar by bar** (KPI cards, equity, drawdown, price chart and a trade tape update live; pick the speed or press Skip). Tabs: **Overview** (16 KPI cards, equity vs buy-and-hold, drawdown), **Price & signals** (pannable candlestick chart with indicators and trade markers), **Trades** (sortable round-trip table, Enter jumps the chart to the trade), **Analysis** (P&L distribution, by hour / weekday, monthly heat map), **History** (compare runs, export CSV/JSON). |
| `3` | Live Trading | Offline demo market or Alpaca / OANDA / Synth data; simulated broker or Alpaca / IB. Live candles, indicator pane, equity, position / account panels, trade table, event log. Runs in the background while you use other pages. |
| `4` | Research | **Datasets** (inspect quality, preview, resample & save), **Optimizer** (grid search with train/test split, robustness flag, heat map, "apply to backtest"), **Download** (provider → `data/raw`). |
| `5` | ML Lab | Train (`train.py` as a cancellable job with live loss chart and command preview), compare runs, evaluate (test split / walk-forward), predict, inspect checkpoints. |
| `6` | Analytics | PRODLIVETRADING snapshot: KPIs, equity, breakdowns, periods, trades, open state, vs-backtest z-scores, read-only SQL console. |
| `7` | Connections | Test and connect providers / brokers from `.env` or typed credentials (memory only). |
| `8` | Logs | Every event from every page, filterable and saveable. |

Global keys: `ctrl+p` command palette · `ctrl+r` run the active page · `ctrl+s` start/stop live session · `ctrl+b`
sidebar · `?` help · `esc` leave a text field · `ctrl+q` quit.
On the price chart: `←/→` pan, `+`/`−` or mouse wheel zoom, `Home`/`End`/`0` jump, hover for OHLC.
The layout adapts to the terminal width (sidebar hides and cards stack below ~125 columns).

## When something looks wrong

The Backtest run log (`src/tui/diagnostics.py`) explains the common silent failures in plain language: missing columns,
too few bars for the strategy, NaN/duplicate/out-of-order data, indicators that hold text instead of numbers, zero
signals, signals that never became trades, positions left open, and short-sample statistics. Errors also raise a
notification and land in the Logs page. The replay is a playback of an already-computed result, so the final numbers
are identical with or without it.

## Safety

* Real-money brokers (Alpaca non-paper, IB live port) require typing `TRADE LIVE` before a session starts; closing a
  position on a real broker asks again.
* Stopping a session offers "close position & stop" or "stop, leave open"; quitting while trading asks first.
* Credentials are never written to disk. `.env` values are read at start-up; typed values live in memory.
* Everything works offline: the demo market + simulated broker never leave the machine.

## Architecture

```
src/tui/
  app.py            shell: navigation, status bar, command palette, themes, breakpoints
  bat.tcss          layout & theme (colours come from the active Textual theme)
  state.py          shared state: connections, dataset cache, backtest history, live session, event log
  charts.py         pure functions data -> rich Text (line/area, candles, volume, bars, histogram, heat map, sparklines)
  report.py         backtest analytics: mark-to-market equity, drawdown, round trips, Sharpe/Sortino/Calmar, …
  strategies.py     strategy registry: ParamSpec / StrategySpec drive forms, validation and the optimizer
  datasets.py       dataset discovery, loading, resampling, quality stats
  services/         capture (thread-aware stdout), connections, demo feed, live session,
                    optimizer, mlops (runs, checkpoints, subprocess jobs)
  widgets/          Plot, PriceChart (pan/zoom/hover), cards, parameter form, event log, modals
  pages/            one module per page
```

Design rules that keep it maintainable:

* **Engines are untouched.** The UI calls `BacktestService`, `LiveTradingEngine`, the providers/brokers, `train.py` and
  `evaluate.py` exactly as the classic CLI does. `LiveSession` drives the engine from a worker thread and publishes
  snapshots; the engine's `print` output is routed per-thread into the event log (`services/capture.py`).
* **Pure logic first.** Charts, reports, the registry, the optimizer and the services have no Textual dependency and are
  unit-tested (`tests/test_tui_logic.py`); pages are covered by pilot tests (`tests/test_tui_app.py`).
* **Never block the UI thread.** Disk scans, backtests, sweeps, downloads and predictions run in workers; training and
  evaluation run as subprocesses so they can be cancelled.

## Extending

* **New strategy:** implement the usual interface, then add one `StrategySpec` to `STRATEGIES` in `src/tui/strategies.py`
  (parameters, bounds, validation). It appears in Backtest, Live and the Optimizer, and indicator columns are
  auto-placed on the chart (price-scale overlays vs separate pane).
* **New provider / broker:** add a `ConnSpec` and a `_connect_<key>` method in `services/connections.py`; for live
  data add a `BarSource` in `services/live.py`.
* **New page:** create a `Widget` in `pages/`, add a `PageDef` in `app.py:page_defs()`. Optional hooks: `on_show()`,
  `action_run()` (ctrl+r), `action_primary()` (ctrl+s), `push(entry)` for log entries.

## Known limitations

* Annualised statistics (CAGR, Sharpe, Sortino, Calmar) are computed from the sample's own period count; on windows
  shorter than ~60 days they are flagged as unreliable on the Overview tab.
* The optimizer evaluates the test segment cold (no indicator warm-up from the train segment).
* The live loop is tick-based (default 60 s for real providers); there is no websocket streaming.
