# DTFX Strategy — Extended Research Checkpoint

This checkpoint captures a long research thread that goes well beyond `dtfx_strategy.ipynb`
(the original committed notebook). It covers: a stricter swing-validation rule, full-history
backtests via the London Strategic Edge data API, a battery of diagnostics, an indicator-based
filter that was validated across 7 instruments, and a live-tracking check on recent performance.

Scripts live in `research_scripts/`. Summary tables in `research_results/`. Key charts in
`research_results/figures/`. Note: the large raw minute-bar datasets (multi-GB, pulled from
Polygon.io and London Strategic Edge) are NOT included here — only the code needed to
re-fetch and re-derive them.

## 1. The strict pivot rule

Replaced the original centered-rolling-window swing detection with a stricter rule:
- A swing candidate only extends when a candle's **close** (never a wick) beats the current
  candidate's own high/low.
- A candidate is only validated as an actual swing once **3 consecutive opposite-color
  candles** follow it without being extended.

This is a drop-in replacement for `find_pivots` + `build_zigzag`, feeding directly into the
unchanged `run_structure_fsm`. See `research_scripts/dtfx_eurusd_backtest_strict.py` and the
synthetic demos `strict_pivots_demo.py` / `strict_multiswing_demo.py`.

Applying this to EUR/USD 2024-2026 substantially improved results over the loose rule
(e.g. original variant: net return flipped from -30% to +27% after simulated costs).

## 2. Full-history testing via London Strategic Edge

Got access to a data API (`lse_live_...` key) with much deeper history than Polygon's ~2-year
minute-data window — EUR/USD and several other instruments back to 2003-2009, BTC/USD back to
2017. This unlocked the single most important finding of the whole project:

**The edge is a post-2021 regime phenomenon, not a timeless property of the strategy.**

- EUR/USD 2009-2026 (17 years, strict rule): win rate sat at ~28% (statistical breakeven,
  given ~2.5R avg win) for **every year from 2009 to 2020**. Starting in 2021 it jumped to
  34-39% and stayed there through 2026.
- XAU/USD 2006-2026 (20 years): **no edge at any point**, including the post-2021 window
  where the FX pairs worked. Confirms this isn't universal across asset classes.
- A pre-2020 sweep across 7 more instruments (SPX500, ES.F futures, AAPL, USD/JPY, AUD/USD)
  found the same ~28% breakeven win rate everywhere, no exceptions. See
  `dtfx_sweep_pre2020.py` and `dtfx_xauusd_lse_full_history.py`.

Plausible mechanism: the 2022+ Fed/ECB/BOE rate-divergence supercycle produced unusually
strong, sustained USD-pair trends that a retracement-continuation strategy is built to
exploit. 2009-2020 (low-rate, range-bound) offered no such trends.

## 3. Diagnostics (`dtfx_diagnostics*.py`, `dtfx_crosspair_diagnostic.py`)

- **Buy vs sell**: sells outperform buys 2-3x in total R, consistent with the USD-strength
  backdrop driving the post-2021 edge.
- **Hour of day / month / day of week**: real patterns exist but several turned out to be
  regime artifacts (e.g. hours 1 and 7 UTC looked bad pooled across all years, but were
  strongly *positive* within the 2021-2026 regime specifically — pooling across regimes
  smuggled in a false signal). Lesson: never filter on a "worst N buckets by rank" basis —
  only exclude buckets that are actually net negative, and check regime-stability before
  trusting any bucket-level pattern.
- **Position-in-trend** (1st, 2nd, ... swing since last flip): looked like a clean, robust,
  cross-regime pattern on EUR/USD alone. Tested on 7 more instruments
  (`dtfx_postrend_sweep.py`) — the shape was different every time (sometimes early swings
  best, sometimes middle, sometimes late-swings worst). Did not generalize. Explicit
  negative result, kept here so it isn't rediscovered and re-trusted later.
- **4H trend alignment**: counterintuitively, trading *against* the 4H trend slightly
  outperformed trading with it in aggregate, though the effect is asymmetric (aligned sells
  are the best bucket, aligned buys are the worst).
- **News proximity** (NFP/FOMC/CPI): not a loss zone — trades near major releases performed
  *better* than average, not worse.

## 4. The validated filter

`dtfx_indicator_diagnostics.py` tested moving-average trend filters, RSI mean-reversion,
ATR volatility regime, and cross-pair momentum confirmation. Two held up:

- **Exclude trades where RSI(14) at entry is in the 45-55 "neutral" zone.**
- **Exclude trades where ATR(14) / its own 2000-bar rolling average < 0.7 ("compressed
  volatility").**

Combined, tested with the exact same unchanged rule on 7 instruments, longest history
available for each (`dtfx_filter_test_generic.py`, `dtfx_usdjpy_filter_test.py`):

| Instrument | Span | Baseline sum R | Filtered sum R | Change |
|---|---|---|---|---|
| EUR/USD | 17.0y | 3,253 | 4,002 | +23.0% |
| USD/JPY | 16.9y | 1,754 | 2,752 | +56.9% |
| GBP/USD | 23.6y | 2,546 | 3,274 | +28.6% |
| EUR/GBP | 23.6y | 8,491 | 5,966 | **-29.7%** |
| SPX500 | 23.4y | -94 | +340 | flips loser to winner |
| AAPL | 22.9y | 98 | 147 | +49.7% |
| BTC/USD | 9.0y | 314 | 1,913 | **+508.7%** |

6 of 7 improve substantially. EUR/GBP is a genuine, understood exception: unlike every USD
pair, its baseline already has strong, persistent edge across the *entire* 23-year history
(not just post-2021) — since it isn't exposed to the USD-divergence trend, the buckets this
filter excludes aren't dead weight there, they're carrying real profit.

Full stats (win/loss counts, Sharpe, max drawdown, 1%-risk compounded equity) for all 7 are
in `research_results/dtfx_final_stats_table.csv`, generated by `dtfx_full_report.py`. Note:
the compounded dollar totals in that file are a mathematical artifact of trade count at 1%
compounding (some reach into the quadrillions) — not literal achievable outcomes. Sharpe and
max drawdown are the numbers that actually describe risk-adjusted quality; every instrument
has a max drawdown above 55%, so this is a high-variance strategy regardless of edge.

## 5. Live-tracking check (2026-08-17/18 and trailing 4 weeks)

Two bad days in a row (~-17R and -18R total across all 7 instruments) initially looked like
a possible drawdown starting. Zooming out to the trailing 28 days showed it wasn't — total
was +62R over 4 weeks, and the two bad days were mostly the tail end of trends already
visible in each instrument's own trajectory (USD/JPY and SPX500 grinding down for weeks;
BTC/USD and GBP/USD barely dipped).

That said, USD/JPY and SPX500 showed **genuine, monotonic degradation** across rolling
windows (full-history → 1yr → 6mo → 3mo → 28d each worse than the last) — a real signal,
not noise. Root causes differ:
- **USD/JPY**: 3 months of pure chop (net move ~0%, but 15.7% cumulative absolute daily
  movement) inverted the ATR signal — high volatility now means whipsaw, not trend, so the
  historically-strongest "expanding ATR" bucket became the worst bucket.
- **SPX500**: still trending steadily up, but the moderate-volatility "just barely passes
  filter" bucket collapsed while the high-conviction ATR>1.3 bucket held up exactly at its
  historical average. Untested hypothesis: raising the ATR floor from 0.7 to 1.3 specifically
  for SPX500 might recover most of the recent damage. Not yet backtested.

## Open items / next steps

- SPX500 ATR-floor-1.3 hypothesis above — not yet tested.
- No true forward/paper-trading validation yet — everything so far is historical.
- Broker research done for EUR/USD execution: OANDA (clean API, free demo account, ~1 pip
  spread, US-accessible) vs Interactive Brokers (tighter ~0.1-0.2 pip verified spread, no
  forex market-data fee, but heavier account-opening process and some reported API
  reliability issues). No live/paper deployment started yet.
- Raw minute-bar data (Polygon + LSE) used throughout is cached locally in the session
  scratchpad only, not committed here — re-run the fetch scripts to regenerate.
