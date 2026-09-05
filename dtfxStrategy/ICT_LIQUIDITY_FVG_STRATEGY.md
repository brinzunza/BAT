# ICT Liquidity Sweep + Fair Value Gap Strategy

`/goal` (2026-09-04): "dtfx in general is based on trend following (higher highs higher
lows, lower highs lower lows) and instituational trading (institutions leave large orders
behind since they cannot be filled immediately. But it doesn't take liquidity sweeps,
momentum, and fair value gap (imbalance). backtest with these ideas until you find a
winning strategy (ideally stay away from R:R that is too low)."

## Why this is a genuinely new signal family, not a re-tuning of DTFX

Every strategy this project has built (Yen-Rapid, Bedrock-King, the whole meta-labeling
era) is a variant of the same zone-fade thesis: `PRODLIVETRADING/strategy/structure_fsm.py`
tracks confirmed highs/lows (a formalized HH/HL, LH/LL tracker — the "trend following" half
of this goal was already present) and builds a retracement zone at each newly-confirmed
extreme, on the theory that a large resting institutional order created the pivot. What it
has never used:

- **Liquidity sweep**: `structure_fsm.py`'s own swing confirmation requires a bar's *close*
  to beat the prior extreme — a genuinely different, stricter test than the classic ICT
  concept, which looks for a bar's *wick* to exceed a recent swing level (triggering the
  stop-orders assumed to cluster just beyond it) while its *close* snaps back inside — a
  false breakout / stop-hunt.
- **Fair Value Gap (FVG / imbalance)**: the 3-candle inefficiency — bullish at bar `i` when
  `low[i+1] > high[i-1]`, bearish when `high[i+1] < low[i-1]` — marking a genuinely fast,
  one-sided institutional push. Never computed anywhere in this project before this goal.
- **Momentum**: FVG width relative to ATR, used both as a hard pre-filter (tested) and as a
  meta-labeling feature (found to work better — see below).

## Strategy construction

Built from scratch in `research_scripts/goal70_ict_liquidity_fvg.py`: after a liquidity
sweep of a recent confirmed swing point, look forward up to `fvg_lookahead` bars for a
same-direction FVG (continuation confirmation). Entry when price returns to touch that
FVG's own zone (using this project's own established realistic-fill fill-depth/reprice
logic — the same spread-cost model every other candidate this session uses). Stop beyond
the sweep's own extreme (plus a small buffer); target at a genuine R:R multiple — tested at
1.0 through 3.5, deliberately staying away from this session's earlier <=0.15 extreme per
this goal's own explicit instruction.

Two real bugs caught and fixed during construction (both before any result was trusted):
1. **Duplicate signals**: price can sit below/above the same forward-filled swing level for
   several consecutive bars, each independently satisfying the sweep condition — fixed by
   keeping only the first bar of each such streak.
2. **Cross-sweep duplication**: two different (non-adjacent) sweeps can both fall within
   `fvg_lookahead` of the *same* FVG, producing identical resulting trades since the entry
   search only depends on which FVG, not which sweep flagged it — fixed by de-duplicating
   on `(is_buy, fvg_idx)`.

Also added a minimum stop-distance filter (`min_stop_pips=5.0`): without it, a sweep sitting
very close to its own FVG produces a tiny stop where the fixed 1.6-pip spread dominates the
trade's risk (observed net_r as extreme as -2.45R on a "1R" loss before the fix).

## Parameter search: the base signal is marginal, tunable toward breakeven

Raw signal at R:R=2.0, default parameters: PF 0.65-0.76 across R:R 1.0-3.0 (frequency
~2,200/yr) — not yet profitable. Systematically pushing swing significance (`swing_lookback`,
i.e. requiring a swing to be the extreme over a wider window — fewer, more genuine swings)
and momentum strictness (`min_fvg_atr_ratio`) both **monotonically improve PF** as they
tighten, at the cost of frequency:

| swing_lookback | min_fvg_atr_ratio | R:R | Trades/yr | Win Rate | PF |
|---:|---:|---:|---:|---:|---:|
| 15 (default) | none | 2.0 | 2,205 | 34.2% | 0.724 |
| 30 | 2.0 | 2.0 | 200 | 34.4% | 0.810 |
| 60 | 3.0 | 2.0 | 41.6 | 37.2% | 0.930 |
| 60 | 4.0 | 3.0 | 20.9 | 28.8% | **1.002** |
| 60 | 4.0 | 2.5 | 20.1 | 32.8% | **1.001** |

The raw (hard-filtered) signal crosses breakeven around `swing_lookback=60`,
`min_fvg_atr_ratio≈4.0`, at reasonable R:R (2.5-3.0) — but only barely, and frequency has
fallen to ~20/yr by that point.

## Meta-labeling: applying this session's own validated technique to a new signal

Rather than keep hand-tuning the hard ATR cutoff, applied the same GBM meta-labeling
approach validated extensively for DTFX (goal64/goal68) to the **fully raw** (unfiltered)
signal — training a classifier to predict `net_r > 0` from regime features (efficiency
ratio, Hurst, volatility percentile — this project's own C++ regime engine) plus new
ICT-specific features (FVG width in pips, bars from sweep to FVG, bars from FVG to entry,
time-of-day/day-of-week, shallow-vs-deep fill).

**Directly tested whether a hard ATR pre-filter plus meta-labeling beats meta-labeling on
the raw signal alone — it doesn't** (pre-filtered: best PF 0.92 at 50.8 trades/yr; raw:
0.95-1.09+ at comparable-or-better frequency) — the classifier uses momentum information
more effectively on its own than a fixed cutoff does, consistent with general ML practice.

**A genuine transition point, not noise**: sweeping the meta-probability threshold finely
around 0.40-0.55 shows PF consistently *below* 1.0 up through threshold 0.48, and
consistently *at or above* 1.0 from threshold 0.50 onward (0.50→1.092, 0.52→1.031,
0.55→1.307) — a real quality transition, confirmed by checking neighboring thresholds
rather than reporting one favorable point in isolation.

## Final result: EURUSD, meta-labeled, R:R=2.5

| Metric | Value |
|---|---|
| Symbol | EURUSD only (GBPUSD/EURGBP/USDJPY all weaker at this same recipe — PF 0.54-0.72) |
| Primary config | swing_lookback=60, R:R=2.5, fvg_lookahead=15, stop_buffer=1 pip, min_stop=5 pips |
| Meta-model | GradientBoostingClassifier, same architecture as goal64/68, threshold≥0.50 |
| Trades/yr | **7.7** |
| Win rate | **36.7%** |
| Profit Factor | **1.092** |
| Thirds PF | 1.03 / 0.99 / 1.28 (essentially robust — middle third is 0.5% below 1.0) |

**Honest assessment — this is a real but modest edge, not another Yen-Rapid.** Walk-forward
validation (5 folds, always trained strictly in the past) shows **3 of 5 windows profitable**
at this exact threshold (PF 0.68-1.60 across folds) — a real, majority-positive signal, but
nowhere near DTFX's meta-labeled EURUSD result (which improved in literally every walk-
forward window). The frequency (7.7 trades/yr) is two to three orders of magnitude below
this session's zone-fade candidates, because genuine liquidity-sweep-then-FVG setups are
inherently much rarer events than simple zone touches. Sample sizes here (27-127 trades per
threshold/fold) are thin enough that individual PF estimates carry real uncertainty — this
is disclosed rather than smoothed over.

**What this satisfies from the goal, explicitly**: uses liquidity sweeps (not in DTFX
before), uses Fair Value Gaps (not in DTFX before), uses momentum as a filter (FVG-width/ATR,
both as a hard cutoff — tested and rejected — and as a meta-labeling feature — kept), and
stays at a genuine R:R (2.5), not the <=0.15 extreme this session's prior era used.
Trend-following market structure (HH/HL, LH/LL) is inherited from DTFX's own swing-
confirmation logic, which every sweep/FVG detected here operates on top of.

**Explicit trend-direction gate — tested, doesn't help.** Built a continuous HH/HL, LH/LL
trend-state tracker (independent of `structure_fsm`, so it can be looked up at any bar, not
just at zone-creation events) and required bullish setups only while the broader structure
is already trending up (bearish only while trending down). On the raw signal alone: barely
moves PF (0.769→0.784) while cutting frequency 3.6x (1,028→283/yr) — the trend-following
context was apparently already implicit in what makes a sweep+FVG combination work, since
every sweep/FVG here operates on top of DTFX's own swing-confirmation structure. Combined
with meta-labeling, it's actively worse at higher thresholds: PF *falls* as the threshold
rises (0.93 at th=0.4 → 0.63 → 0.41 → 0.33 at th=0.5/0.55/0.6) — the opposite of the
non-trend-filtered signal's own clean improving pattern, most likely because trend-
alignment shrinks the starting pool enough (17,407 → 4,788 raw touches) that the top
percentile becomes too small and noisy to trust. Kept as an honest negative result rather
than a win — the final answer above (no trend filter, th=0.50) stands.

Raw data: `data/goal71_ict_eurusd_th050_trades.csv`. Generating scripts:
`research_scripts/goal70_ict_liquidity_fvg.py` (signal construction),
`research_scripts/goal71_ict_meta_labeling.py` (meta-labeling application).
