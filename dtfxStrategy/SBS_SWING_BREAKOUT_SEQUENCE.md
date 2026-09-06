# SBS: swing breakout sequence

Direct user request (2026-09-05): "develop and backtest" a third YouTube video's strategy
(Stoic Trader, "SBS: swing breakout sequence"). Structurally different from the two prior
videos (`STOIC_HTF_LTF_REVERSAL_STRATEGY.md`) -- this one is explicitly NOT trend-dependent,
a standalone, claimed-fractal 5-step liquidity-trap sequence.

## The pattern, as described

1. **Breakout**: price breaks a swing high/low (a wick counts, no candle close required).
2. **First pullback**: price retraces back into the broken swing's own range -- "the first
   stop," where early breakout traders get squeezed first.
3. **New high** (or equal highs): price pushes to a genuine new extreme beyond the
   breakout's own high/low, trapping continuation/breakout buyers.
4. **Liquidation of #2**: price reverses hard and sweeps back through the #2 pullback
   level, stopping out both the step-3 breakout-buyer traps and anyone who bought the #2
   dip in isolation.
5. **Reversal (the entry)**: after the sweep at #4, a double-bottom/double-top forms --
   entry via a stop order at the level that reversal breaks, not on the raw sweep itself.
6. **Target**: back toward, but explicitly not through, the #3 high (the video: "I don't
   even set it towards the high... sometimes we have a big pullback right before the high
   gets taken").

Mapped onto this project's DTFX zone framework: a zone's own `[bottom, top, mid]` supplies
steps 1-2 (the breakout leg and its mid as "the first pullback" level); steps 3-5 require a
genuinely new, sequential state machine tracking what happens to price AFTER a zone is
touched -- something no earlier strategy this session needed, since every prior DTFX
strategy treated a zone's touch as the terminal event, not the midpoint of a longer
sequence. Implemented as a numba-JIT'd per-zone scan (`_scan_sbs_sequences` in
`research_scripts/goal76_sbs_sequence.py`) since a pure-Python loop over ~140K zones x up
to a 1440-bar window each would be too slow. A stop-order entry is mechanically different
from every zone-fade strategy elsewhere in this project (a resting limit order): a stop
order always crosses the spread once triggered, so cost is charged via entry-side
repricing unconditionally, never the fill-depth-dependent "genuinely free" rule used for
limit-order zone-fades.

## Two real bugs caught before trusting the result

**Bug 1 -- degenerate near-zero targets.** The first working version showed a spectacular
raw signal on GBPUSD (n=32,106, WR=71.0%, PF=1.735). Diagnosing before trusting it: 12.8%
of trades resolved within 1 minute of entry, with a 99.98% win rate among that subset --
median `target_r=0.11` there (the target was only 11% of the stop distance, essentially
already reached the moment the stop order triggered). Fixed with a `min_target_r=0.5`
floor. Removing this subset barely moved the headline PF (1.735 -> 1.652 on GBPUSD raw) --
the core signal wasn't propped up by it -- but it's a real degenerate-target class worth
excluding on principle.

**Bug 2 -- non-causal reversal-point selection (a genuine lookahead bug).** After fixing
bug 1, results looked even stronger and more believable: WR 54-62% at a proper ~1.8 average
R:R, PF 1.376-1.944 across all four symbols, **every one of 12 history-thirds profitable**,
train and holdout both strong (1.783 / 1.627 pooled). This is when the result should have
been trusted -- except a closer read of the implementation found the bug: step 5's reversal
point was chosen as the GLOBAL MINIMUM low across the *entire* forward-looking
`reversal_lookback` window, computed by first scanning the whole window, then placing the
entry trigger relative to that point. Identifying "the lowest low in the next 30 bars"
requires knowing those 30 bars in advance -- something no real-time trader placing a stop
order bar-by-bar could ever know. It's a subtler form of lookahead than the more familiar
"literally reads a future price" kind, but it's still lookahead: the algorithm was
implicitly allowed to pick the *best possible* stop-placement point in hindsight before
deciding whether the setup worked.

**Fixed with a genuinely causal, streaming rewrite**: at each new bar, first check whether
it triggers entry using ONLY the reference level established from strictly earlier bars,
and only afterward update the running low/reference for future iterations -- a bar's own
extremes are never used to trigger its own entry decision. This is the single most
important fix of this whole exercise, and it **completely reversed the result**:

| | Before fix (lookahead) | After fix (causal) |
|---|---|---|
| GBPUSD raw WR / PF | 56.1% / 1.808 | 30.0% / 0.699 |
| EURUSD PF | 1.661 | 0.709 |
| GBPUSD PF | 1.944 | 0.692 |
| EURGBP PF | 1.376 | 0.668 |
| USDJPY PF | 1.759 | 0.540 |
| Pooled PF | 1.696 | 0.666 |
| Train / holdout | 1.783 / 1.627 | 0.678 / 0.640 |

Every symbol flips from a strong winner to a clear loser. A 12-point parameter sweep on
GBPUSD (`reversal_lookback` in {15, 30, 60, 120} x `target_buffer_pips` in {1, 3, 5}) after
the fix stayed in a tight, consistently losing band -- PF 0.677-0.727, WR 33-36% -- across
every combination. This is not a tuning problem; it's a robust non-edge.

## Conclusion

The SBS sequence, implemented as literally and faithfully as the video describes and
verified genuinely causal (no forward-looking information anywhere in the entry, stop, or
target logic), **does not have positive expectancy on this data** -- PF~0.67-0.73,
consistent across all four established symbols and a 12-point parameter sweep. This is a
real, negative, honestly-reported result: the strategy's apparent strength in the buggy
intermediate version was entirely a lookahead artifact, not a genuine pattern the market
respects. Given the video's own framing -- explicit skepticism of "cherry-picked hindsight
screenshots" from other traders, while showing only its own curated chart examples -- this
finding is a fitting, if ironic, empirical check: a pattern that looks compelling in
retrospective chart annotations can still fail a forward-only, bar-by-bar backtest.

No trades CSV persisted for this result (a confirmed non-edge isn't a candidate for
deployment). Generating script: `research_scripts/goal76_sbs_sequence.py`.
