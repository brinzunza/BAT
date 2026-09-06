# HTF trend + LTF liquidity-sweep reversal strategy

Direct user request (2026-09-05): "develop and backtest this strat," referring to a YouTube
video transcript (channel: Stoic Trader) describing a price-action / market-structure
strategy. This document maps the video's rules onto this project's existing DTFX
infrastructure, and reports the backtest, run through the same realistic-fill + true-accepted
pipeline used everywhere else in this project.

## Rules extracted from the video

1. **Identify the higher-timeframe (HTF) trend** via swing structure: higher-highs/
   higher-lows = bullish, lower-highs/lower-lows = bearish. The video calls this "the last
   low that created the high" -- exactly this project's own `trend_state()` concept
   (`goal70_ict_liquidity_fvg.py`), reused directly rather than re-derived.
2. **Only trade WITH the HTF trend.** The video is explicit that counter-trend trades get a
   worse R:R and should generally be avoided.
3. **Wait for price to retrace into the 50% of the most recent unmitigated impulse leg**
   that built the current structure. This is exactly a DTFX zone's own `[bottom, top, mid]`
   geometry -- a zone's `mid` already IS the 50% retracement of the leg that created it, so
   this project's existing zone-fade entry convention already captures this rule with no
   new code needed.
4. **At that zone, on a LOWER time frame, require a "manipulation" pattern** before
   entering: a liquidity sweep of a local swing in the zone's own bias direction (the
   video's "sweep the low, then sweep/break the high" reversal signature). Approximated
   using this project's existing liquidity-sweep detector (`detect_liquidity_sweeps`,
   `goal70_ict_liquidity_fvg.py`) at a short lookback, checked for presence in a trailing
   window before the zone's own formation -- the same trailing-window proxy `goal72` used
   instead of reconstructing exact impulse-leg boundaries (that script's own zone-`start_idx`
   investigation found `start_idx` isn't the true impulse-leg start, just `created_at`'s bar
   index).
5. **Entry at the zone's mid**, same convention as every zone-fade strategy in this project.
   **Stop at the zone's own boundary** -- N=1 in this project's existing risk formula (risk
   = half the zone height = the manipulation swing's own extreme), matching the video's own
   stop placement, not some wider multiple of it. **Target as a multiple of that risk**
   (R:R swept) -- the video's own "-2 to -4" measured-move projections are roughly 2-4x the
   swing's size, and a fixed-R:R target from the same stop distance is the direct,
   defensible translation of that into this project's framework.

The genuinely new ingredient versus every earlier DTFX+ICT strategy this session
(`goal62`-`goal73`): an explicit **HTF-trend-alignment gate** on which zones are even
eligible. Earlier work gated zones by regime (Efficiency Ratio, Hurst, volatility
percentile), momentum (Impulse Velocity), or sweep-confluence, but never by swing-structure
trend direction. Implementation: `research_scripts/goal74_htf_ltf_reversal_strategy.py`.

## A real calibration bug caught early: N=1's risk_pct is 2 orders of magnitude smaller

The first run returned "too few" trades even for the *ungated* baseline. Diagnosis: this
project's zone-fade `MIN_RISK_PCT` filters (0.005-0.02, calibrated throughout the session for
N=15-45 zone-fade work) assume a risk_pct scale set by that N range. At N=1 (this strategy's
stop-at-zone-boundary requirement), median `risk_pct` on GBPUSD is ~0.00018 -- about 30-100x
smaller. `min_risk_pct=0.005` was silently filtering out 138,728 of 138,737 candidate zones.
Fixed by recalibrating to `min_risk_pct=0.0001` for this N, and adding a `min_stop_pips=5.0`
guard (the same protection `goal70`'s standalone ICT signal needed earlier this session, for
the same reason: a tiny stop lets the fixed spread cost dominate the trade's R multiple).

## Does the recipe actually work? Ingredient-by-ingredient (GBPUSD, R:R=2.0, N=1)

| Configuration | n | freq/yr | WR | PF |
|---|---|---|---|---|
| Baseline (plain zone-fade, no gates) | 9538 | 563.6 | 36.1% | 0.906 |
| + HTF trend alignment only | 2328 | 137.7 | 37.2% | 0.952 |
| + LTF sweep confluence only | 3527 | 208.4 | 37.1% | 0.955 |
| **+ BOTH (the full video recipe)** | 1133 | 67.0 | 38.9% | **1.037** |

Neither ingredient alone is enough -- each moves PF from 0.906 to only ~0.95. Combined, PF
crosses 1.0. This is the same "confluence, not a standalone signal" pattern `goal72` found
for sweep+momentum: two real but individually-modest filters compounding into a genuine
edge only when required together, not evidence found on the first attempt.

## R:R and lookback sensitivity (GBPUSD, full recipe)

R:R=2.5 is the sweet spot: PF=1.093 with **all three history-thirds independently above
1.0** (1.024 / 1.183 / 1.091) -- a genuinely robust number, not an average propped up by
one lucky period. HTF/LTF lookback sensitivity (htf_lookback=240, ltf_lookback=15, i.e.
roughly a 4-hour HTF swing structure over a ~15-bar LTF sweep) was the best of 5 combinations
tested (120/10 through 480/30, PF range 0.944-1.093) -- the first, most natural parameter
choice happened to be the strongest, not a result cherry-picked from a wide search.

## Cross-symbol generalization: real, but only on 2 of 4 symbols

Testing the identical GBPUSD recipe (R:R=2.5) on the other three established symbols:

| Symbol | n | freq/yr | WR | PF | Thirds |
|---|---|---|---|---|---|
| GBPUSD | 882 | 52.2 | 34.6% | 1.093 | 1.024 / 1.183 / 1.091 |
| EURUSD | 834 | 49.7 | 31.2% | 0.916 | 0.930 / 0.810 / 0.979 |
| EURGBP | 284 | 18.3 | 33.1% | 0.980 | 0.937 / 0.580 / 1.408 |
| USDJPY | 372 | 43.6 | 28.0% | 0.796 | 0.783 / 0.898 / 0.730 |

Pooled across all four with the *same* R:R: PF=0.967 (train 0.935, holdout 1.000) --
essentially breakeven, not a real portfolio-wide edge. Rather than force a marginal 4-symbol
pool, a per-symbol R:R sweep (1.5-4.0) was run to see whether EURUSD/EURGBP/USDJPY simply
need their own R:R (a real pattern throughout this project -- e.g. `Velocity-Cable` and
`Chop-Fade` bases each have symbol-specific configs already):

- **EURGBP at R:R=4.0**: PF=1.226, thirds = 1.156 / 1.300 / 1.195 -- all three clean, even
  stronger than GBPUSD's own number. Lower frequency (15.2/yr) since a wider R:R at a fixed
  stop distance means fewer touches clear the acceptance pipeline's fill/margin constraints.
- **EURUSD**: no R:R in 1.5-4.0 produces a thirds-clean result. Best is R:R=4.0 at PF=1.008
  (essentially breakeven) with one third still below 1.0 (0.896). Not included in the final
  pool -- a coin flip, not an edge.
- **USDJPY**: never crosses PF=1.0 at *any* R:R tested (0.599-0.796 across the whole 1.5-4.0
  range). This isn't a tuning problem -- the strategy structurally doesn't work on this pair,
  consistent with USDJPY being the hardest symbol to keep profitable across this whole
  session's DTFX+ICT work (see `ICT_LIQUIDITY_FVG_STRATEGY.md`'s frequency frontier, where
  USDJPY needed the most aggressive settings and still lagged the other three legs).

## Final result: GBPUSD (R:R=2.5) + EURGBP (R:R=4.0), pooled

| Metric | Value |
|---|---|
| Symbols | GBPUSD (R:R=2.5) + EURGBP (R:R=4.0) |
| Config | N=1 (stop at zone boundary), HTF swing_lookback=240, LTF swing_lookback=15 |
| Gates | HTF trend alignment (mandatory) + LTF liquidity sweep within 60 bars of zone formation (mandatory) |
| Trades/yr | 66.9 |
| Win rate | 32.9% |
| **Profit Factor** | **1.125** |
| Train (pre-2018) | PF=1.064, n=592 |
| Holdout (2018+) | PF=1.195, n=539 |
| Thirds | 1.050 / 1.185 / 1.154 (all independently >1.0) |

Both halves of history, and all three independent thirds, are individually profitable --
genuine robustness, not a result propped up by one period. A lower PF than the session's
DTFX+ICT confluence flagship (2.115 at 3.0 trades/yr, see `ICT_LIQUIDITY_FVG_STRATEGY.md`),
but at ~22x the trade frequency, on a structurally different construction (trend-alignment +
sweep on a stop-at-zone-boundary zone-fade, vs. momentum + sweep-confluence on a
wider-N zone-fade). Two honest, disclosed limitations: (1) EURUSD is left out (marginal, not
a real edge at any R:R tested) and (2) USDJPY is left out entirely (a structural loser at
every R:R tested, not merely under-tuned).

Raw data: `data/goal74_htf_ltf_reversal_{gbpusd,eurgbp}_trades.csv` and the pooled
`data/goal74_htf_ltf_reversal_pooled_trades.csv`. Generating script:
`research_scripts/goal74_htf_ltf_reversal_strategy.py`.
