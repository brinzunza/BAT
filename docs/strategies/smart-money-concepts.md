# Smart Money Concepts strategy

This is BAT's causal, backtestable implementation of the supplied SMC
specification. Give it one-minute OHLCV data; it resamples that data to 15
minutes for higher-timeframe structure by default.

## Implemented rules

* Swing highs/lows use the requested `n` candles on each side and become known
  only `n` candles later, preventing look-ahead bias.
* BOS/CHOCH uses candle closes rather than wick crossings.
* Three-candle FVGs produce higher-timeframe points of interest.
* The strategy arms only after an opposing liquidity sweep overlaps that POI,
  then waits for a lower-timeframe close through a confirmed swing.
* Bullish entries emit `Buy Signal`; bearish entries emit `Sell Signal`.

## Backtest parameters

`structure_timeframe` (default `15min`), `swing_window` (default `2`),
`liquidity_lookback` (default `60` bars), and `poi_expiry_bars` (default `60`).
Use BAT **long/short** mode to test both directions.

## Deliberate boundary

BAT's current backtester accepts market-style buy/sell signals only. It cannot
yet model the specification's limit entry at a micro order block, hard stop,
two partial take-profits, or cancellation of a pending order. Those behaviours
remain a future execution-adapter/risk-management enhancement; this strategy
does not pretend to simulate them. Kafka, TimescaleDB, and live session storage
are likewise deployment concerns and are not required to run a deterministic
historical backtest.
