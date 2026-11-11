# Backtest engine API

One entry point, one result object, no side effects.

```python
from src.core.api import run_backtest, BacktestConfig

outcome = run_backtest(bars, strategy, BacktestConfig(
    initial_cash=10_000, trading_mode="long_short", position_fraction=0.5, spread_pips=1.0, symbol="C:EURUSD"))

outcome.summary        # dict of headline metrics (return, cagr, sharpe, sortino, max_drawdown, win_rate, profit_factor, ...)
outcome.trades         # DataFrame, one row per execution (BUY / CLOSE / SELL_SHORT / CLOSE_SHORT)
outcome.round_trips    # DataFrame, one row per completed trade
outcome.equity         # Series, mark-to-market equity per bar
outcome.signal_data    # the strategy's output frame (bars + indicators + signal columns)
outcome.to_dict()      # JSON-safe digest (NaN -> null, timestamps -> ISO, down-sampled equity, capped trades)
```

## Inputs

| Input | Contract |
|---|---|
| `bars` | DataFrame with at least `Close` and `timestamp`. Built-in strategies also need `Open High Low Volume`. Never mutated. |
| `strategy` | Anything satisfying `src.core.ports.trading.Strategy`: `name`, `generate_signals(df) -> df`, `get_signal_names() -> {"buy": col, "sell": col}`, `get_required_lookback() -> int`. Optional (used by the UI): `get_indicators()`, `validate_data(df)`. |
| `config` | `BacktestConfig`, validated on construction (`initial_cash > 0`, `trading_mode in {long_only, long_short}`, `0 < position_fraction <= 1`, `spread_pips >= 0`). `symbol` matters: symbols starting with `C:` get the forex spread model. |

## What the engine guarantees (and checks)

- **Deterministic and pure**: no network, disk, env vars, printing, or global state. Same inputs, same outputs.
- **Strategies are not trusted.** The returned frame must be a DataFrame, have one row per input bar, keep `Close` and
  `timestamp`, and contain the declared buy/sell columns, otherwise `ValueError`. Missing values in a signal column
  (NaN/None) mean "no signal"; signal columns are coerced to plain booleans.
- **Fill model**: a signal on bar *i* executes at bar *i*'s close; bar 0 is never traded; bars with `Close <= 0` are skipped.
  No commissions or slippage; only the forex spread. See `src/core/application/backtest.py`.
- **Accounting** is done by `Portfolio` (`src/core/domain/portfolio.py`), which raises on illegal state changes.

## Layers

```
src/core/api.py                 facade: run_backtest -> BacktestOutcome
src/core/application/backtest.py  BacktestService (the simulation loop)
src/core/analytics/report.py    metrics, equity curve, round trips (pure pandas/numpy)
src/core/domain/                value objects + Portfolio
src/core/ports/trading.py       Strategy / MarketDataProvider / ExecutionGateway protocols
src/core/adapters/legacy.py     adapters for the older providers/brokers
```

`src.core` imports nothing from `src.tui`, providers or brokers.
The legacy `src.evaluation.backtest.BacktestEngine` is a compatibility shim over this API plus matplotlib/plotly plotting.

## Adding a strategy

Hand-written: a class in `src/strategies/` implementing the protocol above, plus one `StrategySpec` in
`src/tui/strategies.py`.
