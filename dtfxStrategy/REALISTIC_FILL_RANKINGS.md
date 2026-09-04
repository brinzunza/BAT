# Realistic-Fill Rankings — 2026-09-04

Supersedes every prior DTFX zone-fade ranking in this project (including
`trading-data/STRATEGY_RANKINGS.md` and this repo's own `goal47`/`goal51`/`goal52` numbers)
for one reason: all of them assumed a resting limit order fills for free, exactly at the
zone's own mid, the instant price reaches it. That assumption broke live — a real EUR/GBP
order sat resting well past when the old backtest would have called it filled, because a
BUY limit only actually fills once the ASK reaches it, a SELL limit once the BID does, not
the MID this project's 1-minute OHLC data is built from.

## What changed

Two corrections, applied together, both at the user's explicit direction:

1. **Fill depth requirement.** A touch only counts as a genuine, free limit fill if the
   entry bar's low (buy) / high (sell) reaches a full 1.6 pips (`SPREAD_PIPS_TYPICAL`)
   beyond the zone's mid — not just touching mid, which the old model treated as
   guaranteed. A touch that reaches mid but falls short of that depth is kept, not
   discarded (discarding it would just as unrealistically undercount frequency as
   assuming a free fill overcounts quality) — priced instead as a market-order fill 1.6
   pips worse, in the adverse direction. Stop/target stay anchored to the zone's own mid;
   a worse entry then costs on both sides by construction (less room to target, more
   distance to stop). Win/loss itself is not re-simulated from tick data this project
   doesn't have — only the entry price, and the R-multiple that follows from it.
2. **Universal spread cost.** Every trade now pays the 1.6-pip spread, not just
   stop-outs. The old "passive-fill target" exemption was always paired with "and entry
   is free too" — once entry isn't free, a win's own exit doesn't get a free pass either.

Full derivation and code: `research_scripts/goal51_single_symbol_closest_price_backtest.py`
(the fill model itself) and `research_scripts/goal53_realistic_fill_leaderboard_and_search.py`
/ `goal54_realistic_fill_search.py` (this ranking's own generation).

Scope: **EUR/GBP only**, matching PRODLIVETRADING's current single-symbol deployment
(2026-09-03, retired the 4-symbol portfolio the same day the concurrency policy changed —
see `strategy/engine.py`'s own APEX-ULTRA docstring entry in that repo).

## Every existing strategy fails

Reproduced from `trading-data/strategy_rankings.csv`'s own published (N, R:R) per
strategy (that repo has no scripts of its own — data/results only). Excluded:
**Causal-Confluence** (its name implies a causal-discovery signal beyond pure zone
geometry; no source exists in this repo to reproduce it honestly, and an ML-discovered
filter not built causally is exactly the lookahead risk this exclusion is meant to catch)
and **goal70_final_winner** (an internal placeholder name in the leaderboard's own CSV,
not a described strategy). age_h/touches aren't recorded in that CSV for most rows —
standardized to Apex-Ultra's own (age_h=0.08h, touches=20) except where independently
known (flagged `*` below); this is a best-effort reproduction of a leaderboard whose own
generating code isn't available to audit, not a byte-for-byte replay of it.

| Strategy | N | R:R | PF (corrected) | WR | Shallow-touch % |
|---|---:|---:|---:|---:|---:|
| SOMEGA (live) | 3.34 | 1.00 | 0.378 | 47.9% | 93.5% |
| SOMEGA Robust | 3.34 | 1.00 | 0.378 | 47.9% | 93.5% |
| CANA+ | 3.00 | 1.00 | 0.341 | 45.6% | 93.5% |
| CANA (faithful) | 3.00 | 1.00 | 0.341 | 45.6% | 93.5% |
| DTFX-Ultra / SOMEGA-Ultra / Hybrid-Ultra | 6.00 | 0.40 | 0.276 | 52.9% | 93.8% |
| Quality-King | 7.00 | 0.35 | 0.290 | 54.3% | 93.8% |
| SOMEGA-NarrowTarget | 3.34 | 0.70 | 0.257 | 45.4% | 93.7% |
| CANA-NarrowTarget | 3.00 | 0.80 | 0.264 | 44.6% | 93.6% |
| DTFX-NarrowTarget | 2.00 | 0.70 | 0.106 | 25.6% | 93.6% |
| DTFX Robust | 2.00 | 1.00 | 0.209 | 36.3% | 93.3% |
| Titan-76 | 2.00 | 0.76 | 0.126 | 27.2% | 93.5% |
| Speed-King | 7.00 | 0.20 | 0.095 | 30.6% | 94.1% |
| Velocity-King * (deployed until 2026-09-03) | 5.00 | 0.20 | 0.047 | 18.4% | 93.3% |
| Apex-King * (deployed before that) | 4.50 | 0.18 | 0.028 | 13.9% | 92.7% |
| Vortex-King / Hyper-King | 4.00 | 0.15 | 0.009 | 7.1% | 93.1% |
| **Apex-Ultra * (deployed 2026-09-03, this table's motivation)** | 4.00 | 0.15 | **0.009** | 7.1% | 93.1% |
| Steadfast-King * | 5.00 | 0.15 | 0.017 | 9.9% | 93.8% |

`*` = independently known age_h/touches, not standardized. Full per-strategy output:
`data/goal53_leaderboard_realistic_fill.csv`.

**Zero of the 17 reproducible strategies clear PF = 1.0.** The best (SOMEGA/CANA family,
R:R≈1.0) only gets to 0.34–0.38 — losing roughly 65 cents for every dollar risked, not
close to breakeven. This isn't sensitive to which specific strategy: 92–94% of every
strategy's own "touches" never reach the full spread depth this model requires, meaning
the old backtests were crediting a free, mid-priced fill to trades that would realistically
have needed a market order (or wouldn't have filled at all). Mechanically, the shared root
cause is `MIN_RISK_PCT` (0.000170) letting zones through with stops barely wider than a
single 1.6-pip spread — charging even one spread against a stop that thin consumes most or
all of a trade's entire R before any edge gets a chance to show, and these strategies'
tiny targets (R:R 0.15–0.4) are frequently *smaller* than the spread itself.

**Corroborating real evidence, not just backtest math**: PRODLIVETRADING's two $100
sub-accounts, which have run Apex-King → Velocity-King → Apex-Ultra in sequence, currently
sit around $91/$99 — a real decline, not the growth the old free-fill backtests predicted.
Direction matches this corrected model, not the one that was being trusted.

## The fix: raise the minimum-risk floor, not just N or R:R

A blind grid search over N and R:R alone (holding `MIN_RISK_PCT` fixed) never clears
PF=1.0. Sweeping `MIN_RISK_PCT` jointly is what actually matters — it excludes zones too
thin to survive a real spread, which N and R:R alone can't reliably do since a "wider
stop" from a bigger N can still be paired with a naturally tiny zone. Initial results
pointed toward extreme parameters (N=50, in-sample PF up to 1.67) — rejected after a
2018 train/holdout check: those collapsed out-of-sample (holdout PF 1.06–1.14 against
in-sample 1.58–1.67), the textbook signature of overfitting a big search grid against a
shrinking sample (as few as 130-170 trades at the most extreme points). Requiring BOTH
halves to hold up, with an adequate sample in each, is what the winner below survived.

## New leader: Bedrock-King

**N=22.0, R:R=0.75, age=0.08h, touches=20, MIN_RISK_PCT=0.020** (vs. Apex-Ultra's
0.00017 — over 100x wider minimum stop). Named for what actually makes it survive: a
stop floor wide enough that a fixed 1.6-pip spread cost can't erode it away, unlike
every strategy above.

- **Train/holdout (2018-01-01 split): PF 1.242 (n=290, pre-2018) vs. PF 1.264 (n=205,
  2018+)** — consistent, not reversed; this is what separates it from the rejected N=50
  region.
- **Full history**: 495 trades / 16.0 years — **31 trades/yr, ~1 every 12 days** — a
  completely different cadence from Apex-Ultra's 8,658/yr. WR 63.23%, PF 1.251,
  mean net_r +0.093.
- **Sharpe 1.91**, excess kurtosis +5.06 (fat right tail — a handful of large wins carry
  real weight), Calmar 0.28, max drawdown (1%-risk reference curve) −10.0%.
- **Positive-period rate** (period-reset, $100/$100 each period): Day 57.5% (n=369),
  Week 53.2% (n=252), **Month 50.0% (n=124)**, Year 58.8% (n=17).
- MC race-to-$1M did not converge within a bounded 20,000-trade simulation at this
  frequency — this is not a fast-growth strategy the way the old (illusory) high-frequency
  ones claimed to be; "years to $1M" isn't the right lens for it.

**Honest trade-offs, not just the win**: a 50% monthly positive rate means real, extended
flat-to-losing stretches are normal at this cadence, not a red flag on their own. Excess
kurtosis of +5 means the P&L is genuinely lumpy — a small number of trades carry a
disproportionate share of the return, which cuts both ways in a live account. Only 17
independent years of data back the Year-level consistency figure. And this is still a
single-symbol (EUR/GBP), single-search-pass result — the neighborhood around N=22–25,
R:R=0.5–1.0, MIN_RISK_PCT=0.013–0.023 all showed similarly consistent train/holdout
behavior (see `data/goal54b_extended_search.csv`), suggesting a genuine region rather
than a fragile point, but a finer sweep and the other 3 symbols haven't been checked yet.

## Composite score

The 3-/4-/5-factor formulas `trading-data/STRATEGY_RANKINGS.md` used (Speed to $1M,
normalized WR/Frequency, MC drawdown, audited Consistency) don't transfer cleanly here:
Speed-to-$1M doesn't converge for Bedrock-King, and is meaningless for every strategy
above it (PF<1 means never reaching $1M, not "slowly"). Rather than force those numbers,
this ranking uses PF itself as the primary gate — everything below 1.0 is **not viable**,
full stop, regardless of its old composite score — with Bedrock-King the only entry that
clears it:

| Rank | Strategy | PF | Viable? |
|---|---|---:|---|
| 1 | **Bedrock-King** | **1.251** | **Yes** |
| 2 | SOMEGA (live) / SOMEGA Robust | 0.378 | No |
| 3 | CANA+ / CANA (faithful) | 0.341 | No |
| ... | (all others) | <0.30 | No |

## Next steps

1. Halt Apex-Ultra's live entries (has no real edge under this model) — separate,
   more urgent action from whether Bedrock-King gets deployed.
2. Finer sweep around Bedrock-King's neighborhood (N=20–28, R:R=0.5–1.0,
   MIN_RISK_PCT=0.013–0.025) to check for a materially better nearby point.
3. Check the other 3 symbols (EURUSD, GBPUSD, USDJPY) under this same corrected model
   and search methodology — EUR/GBP won every PRE-correction comparison run this
   session; that conclusion needs revisiting from scratch now that the fill model itself
   has changed.
4. Full margin/MC validation (goal52-style) on Bedrock-King before any deployment
   decision, plus a real practice-account smoke test given how different its trading
   cadence is from anything previously run live.

Raw data: `data/goal53_leaderboard_realistic_fill.csv` (leaderboard reproduction),
`data/goal54_realistic_fill_search.csv` + `data/goal54b_extended_search.csv` (search
sweeps), `data/goal55_extreme_n_sweep.csv` (the rejected N-up-to-50 region) +
`data/goal55_train_holdout_validation.csv` (the 2018 split that separated Bedrock-King
from it). Generating scripts: `research_scripts/goal5{1,3,4,5}_*.py`.
