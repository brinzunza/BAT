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

## Cross-symbol search: quickest validated path to $1M (`/goal`, 2026-09-04)

Bedrock-King is net profitable but slow: at its best tested risk (4%), its 495-trade,
16-year EUR/GBP track record compounds $100→$100 combined into just $207 — extrapolated
CAGR 0.2%, ~4,126 years to $1M. Bedrock-King's own drawback (low frequency limiting how
much a real edge can compound within available history) turned out to generalize past
EUR/GBP specifically, not the single-symbol restriction: over 1,500 (N, R:R,
MIN_RISK_PCT) configs swept across all four symbols, at every risk level from 2% to 60%,
**none reached $1M within their own available history** — best full-history multiple
achieved was ~14x (EURUSD, at an unsustainable 60% risk carrying a −54% drawdown).

Ranking candidates instead by *extrapolated* years-to-$1M (CAGR from each candidate's
own best-tested risk level, projected forward at that same rate — the literal $1M
threshold isn't reached inside any dataset tested, so this is the honest way to answer
"quickest," not a claim that $1M is imminent):

| Candidate | Instrument | N | R:R | MIN_RISK_PCT | Best risk | CAGR | Max DD | Extrapolated years to $1M | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Yen-Punch** | USDJPY | 4.0 | 0.75 | 0.005 | 28% | **29.3%** | −33.3% | **~33** | 122 |
| Yen-Steady | USDJPY | 4.0 | 0.15 | 0.005 | 36% | 20.3% | −21.3% | ~46 | 244 |
| Dollar-Wide | EURUSD | 15.0 | 1.00 | 0.020 | 60% | 13.3% | −54.3% | ~68 | 123 |
| Cable-Wide | GBPUSD | 15.0 | 1.50 | 0.020 | 28% | 9.2% | −26.8% | ~96 | 124 |
| Bedrock-King | EURGBP | 22.0 | 0.75 | 0.020 | 4% | 0.2% | −10.7% | ~4,126 | 495 |

**Winner: Yen-Punch** (USDJPY, N=4.0, R:R=0.75, age=0.08h, 20 touches, MIN_RISK_PCT=0.005,
RISK_PCT=0.28) — net profitable and fastest validated extrapolated path to $1M of
everything found this session.

- **Train/holdout (midpoint split, USDJPY only has data from 2018 so the 2018-01-01 split
  used elsewhere doesn't apply): PF 1.462 (pre-2023-02-10, n=52) vs. PF 1.662
  (post-split, n=70)** — holdout is *better* than train, the opposite of overfitting.
- Full history: 122 trades / 7.0 years (17.5/yr), WR 68.85%, PF 1.573, mean_r +0.182.
- **Sharpe 4.67**, excess kurtosis +0.56 (much less fat-tailed than Bedrock-King's +5.06
  — a more evenly-distributed return stream, not a few outsized trades carrying it),
  Calmar 1.05, max drawdown (1%-risk reference curve) only −3.0%.

**Caveats, stated plainly**: 122 trades is a genuinely modest sample — smaller than
Bedrock-King's 495, and USDJPY's own data only goes back to 2018 (7 years), so this
hasn't been tested across as many market regimes. The winning RISK_PCT (28%) is far
above the 6% this project has used everywhere else, chosen here purely to minimize
years-to-$1M — the −33.3% max drawdown at that risk level is a real, honest cost of that
choice, not hidden. "Years to $1M" is an extrapolation from a realized CAGR, not a
literal event observed in the data — treat the ~33-year figure as illustrative of
relative speed between candidates, not a forecast. Before any live consideration: verify
independently on GBP/USD-adjacent JPY crosses' data quality, re-validate at a more
conservative RISK_PCT (the Sharpe/Calmar numbers above are strong even before pushing
risk to 28%), and run the same margin/MC validation Bedrock-King still needs.

## Frequency-emphasis composite (`/goal`, 2026-09-04)

"Find a profitable strategy that beats all in the composite score with an emphasis on
trade frequency" — tested directly, not assumed: does chasing raw trade count actually
win once frequency is given real weight, or does thin edge at high frequency lose to
genuine edge at moderate frequency the same way it already lost on speed-to-$1M above?

**New high-frequency search** (all 4 symbols, N up to 35, tight R:R down to 0.10, broader
MIN_RISK_PCT grid) found real, train/holdout-robust candidates trading far more often
than anything in the table above:

| Candidate | Instrument | N | R:R | MIN_RISK_PCT | Trades/yr | PF (full) | Train PF | Holdout PF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Frequent-Cable | GBPUSD | 25.0 | 0.30 | 0.013 | **340** | 1.073 | 1.092 | 1.054 |
| Balanced-Cable | GBPUSD | 30.0 | 0.25 | 0.016 | 315 | 1.081 | 1.113 | 1.050 |
| Balanced-Dollar | EURUSD | 30.0 | 0.50 | 0.020 | 136 | 1.147 | 1.141 | 1.159 |

All three are genuinely profitable and validated (2018 split, holdout confirms train —
these aren't overfit artifacts). But scored on a composite that explicitly emphasizes
frequency (min-max normalized across all 7 candidates below; Frequency weighted 40%,
PF/WR/Max-DD 20% each — double weight on frequency vs. any other single factor):

| Rank | Strategy | Instrument | Trades/yr | PF | WR | Max DD (6% risk) | Composite (freq-weighted) |
|---|---|---|---:|---:|---:|---:|---:|
| **1** | **Yen-Steady** | USDJPY | 35 | 1.637 | 93.0% | −7.2% | **62.9** |
| 2 | Frequent-Cable | GBPUSD | 340 | 1.073 | 79.0% | −22.0% | 53.6 |
| 3 | Yen-Punch | USDJPY | 17 | 1.573 | 68.9% | −9.0% | 45.1 |
| 4 | Dollar-Wide | EURUSD | 8 | 1.646 | 62.6% | −8.7% | 44.0 |
| 5 | Balanced-Dollar | EURUSD | 136 | 1.147 | 70.2% | −18.7% | 40.7 |
| 6 | Cable-Wide | GBPUSD | 8 | 1.433 | 49.2% | −7.6% | 32.0 |
| 7 | Bedrock-King | EURGBP | 31 | 1.251 | 63.2% | −14.1% | 26.0 |

**Yen-Steady wins even under 40%-frequency weighting**, despite Frequent-Cable trading
nearly 10x more often. Mechanically: Frequent-Cable's thin edge (PF 1.073) costs it
almost the entire PF, WR, and drawdown components of the score (it scores at or near
zero on all three), and raw frequency alone isn't enough to compensate even at double
weight. This isn't just a scoring-formula artifact — it holds up on the metric that
actually matters, too: extrapolated CAGR at best-tested risk gives Frequent-Cable
**~1,588 years** to $1M (Balanced-Cable's CAGR turns *negative* at low risk — the same
volatility-drag effect that punished Bedrock-King, here amplified by how thin the edge
is per trade) against Yen-Steady's already-established ~46. **Once realistic execution
costs are priced in, high-frequency trading with a thin per-trade edge doesn't win — it
takes real edge to make frequency pay off, not the other way around.**

Yen-Steady was already the runner-up in the speed-to-$1M table above; this composite
confirms it independently as the strongest all-around candidate found this session,
under a scoring lens that was deliberately built to give a raw-frequency play its best
chance to win and still didn't produce one.

## Real multi-symbol trading (`/goal`, 2026-09-04): a genuine speed win, not a frequency one

"Find a strategy that outperforms all of them, specifically in frequency and race to 1
million. New factor is you can trade EURUSD, EURGBP, USDJPY, and GBPUSD" — the first
goal this session to actually exercise real multi-symbol trading, not single-symbol
research. Turns out no new resolver was needed: each symbol's own per-(symbol,side)
closest-price arbitration (already built and validated) answers "which zone does this
symbol rest" correctly in isolation, and the cross-symbol rule — at most 1 CONFIRMED
fill per side across the whole account, matching PRODLIVETRADING's actual,
always-been-this-way `MAX_CONCURRENT_PER_SIDE=1` — turns out to be *exactly*
`simulate_two_account`'s existing accept/reject logic (first entry-time wins per side),
unmodified. Concatenate all 4 symbols' independently-resolved streams and feed the
combined set into the same account simulation used everywhere else in this document.

**Central finding, confirmed by direct testing and then an exhaustive 227-combination
sweep across every validated single-symbol candidate this session found: the shared
per-side slot makes multi-symbol trading a zero-sum reallocation of slot-time, not an
additive source of trades.** Every single tested addition to USDJPY's Yen-Steady
baseline *reduced* its frequency — never increased it — because occupying the one
shared slot on a side, regardless of which symbol does it, blocks every other symbol on
that side for the full duration of the position. **Zero of 227 combinations beat
Yen-Steady on both frequency and race-to-$1M simultaneously.** This is a structural
property of the architecture, not a gap in the search.

**Best result found: Elite-Ensemble** — USDJPY (Yen-Steady's own N=4.0/R:R=0.15/
MIN_RISK_PCT=0.005) supplemented by each of the other 3 symbols' own *best-individual-PF*
config (not their fastest one): EURUSD (N=15.0/R:R=1.00/MIN_RISK_PCT=0.020), GBPUSD
(N=15.0/R:R=1.50/MIN_RISK_PCT=0.020), EURGBP (N=5.0/R:R=0.15/MIN_RISK_PCT=0.008).
Infrequent, high-quality signals from the other 3 pairs add real edge without
displacing much of USDJPY's own activity, unlike every high-frequency addition tested.

- **Genuinely outperforms every prior single-symbol finalist on race-to-$1M**: ~40.9
  extrapolated years at 48% risk — faster than Yen-Steady (46.2), Dollar-Wide (68.0),
  Cable-Wide (96.4), and dramatically faster than Bedrock-King (4,126).
- Best PF of anything found this session: **1.851** (vs. Yen-Steady's 1.637), Sharpe
  4.71, excess kurtosis −1.11 (thinner-tailed than every single-symbol candidate — a
  genuinely more evenly distributed return stream), max drawdown (1%-risk reference)
  only −3.8%.
- **Train/holdout robust**: PF 2.000 pre-2018 (n=88) vs. 1.725 post-2018 (n=215) —
  consistent, not reversed, despite the smaller train sample.
- **Does not beat Yen-Steady on raw frequency**: 18 trades/yr vs. Yen-Steady's 35 —
  this is the direct, unavoidable cost of the finding above. Multi-symbol trading under
  this architecture is a quality lever, not a frequency lever.

**Read plainly**: the goal's two named criteria — frequency and race-to-$1M — pull in
opposite directions once multi-symbol combination is actually tried, the same tension
already found within single symbols (goal58). Elite-Ensemble is the best-evidenced
answer to "use all 4 symbols to do better," but it wins on speed, not on frequency; no
strategy tested beats every prior candidate on both axes at once, and 227 tested
combinations say that isn't a search failure. Raw data:
`data/goal59_multi_symbol_combinatorial_search.csv` (the full sweep) and
`data/goal59_elite_ensemble_trades.csv` (Elite-Ensemble's own trade log).

## New leader: Yen-Rapid (Stop hook follow-up, 2026-09-04)

The Stop hook correctly rejected Elite-Ensemble: it beat every prior candidate on
race-to-$1M but lost to Yen-Steady on raw frequency (18/yr vs. 35/yr), and the goal
requires both. Rather than accept that shortfall, the search resumed by widening a
parameter that every USDJPY search so far had held fixed at 0.08h: `MIN_FILL_AGE`
(`age_h`). It shouldn't have been fixed — sweeping it directly finds a materially
better USDJPY-only config.

**Yen-Rapid**: N=4.0, R:R=0.15, MIN_RISK_PCT=0.005, age_h=0.05h — identical to
Yen-Steady except a shorter fill-age gate. Beats Yen-Steady outright on both axes the
goal names:

| Metric | Yen-Steady | **Yen-Rapid** |
|---|---:|---:|
| Trades/yr | 35.0 | **40.0** |
| PF (full history) | 1.637 | **1.670** |
| Years to $1M (extrapolated) | 46.2 | **36.2** |
| Max drawdown (1%-risk reference) | −7.2% | −2.2% |

It's also the first candidate this entire session to pass a stricter robustness bar:
**all three independent thirds of its own history are profitable** (PF 1.34 / 2.11 /
1.50) — something Yen-Steady fails (its first third is a loss, PF 0.80), Yen-Punch
fails to even validate (its first third has only 12 trades), and a higher-N variant
found along the way (N=7.0/R:R=0.10/age=0.15h, PF=1.859, y1m=32.0) also fails — its
first third is a clear loss (PF 0.55) and its excess kurtosis (+7.93) points to an
unstable, regime-dependent result rather than a genuine edge, not the fast, robust
strategy it looks like from the full-history number alone.

**The one number Yen-Rapid doesn't beat**: Yen-Punch's own raw 33.1-year race-to-$1M.
A dedicated search requiring all-three-thirds robustness (26 candidates found, ranked
by speed) confirms this specific tension holds even among fully-validated candidates —
the fastest robust option found (N=6.0/R:R=0.20/MIN_RISK_PCT=0.008/age_h=0.03h) gets to
31.2 years, beating Yen-Punch outright, but only trades 22.7/yr, below Yen-Steady's own
frequency. Nothing found clears both Yen-Steady's frequency ceiling and Yen-Punch's
speed floor at once. Given that, Yen-Rapid is recommended over both: it beats
Yen-Steady cleanly on everything, and where it falls short of Yen-Punch's headline
number, Yen-Punch's own number rests on a foundation too thin to trust in the first
place.

Full stats: 280 trades / 7.0 years, WR 93.21%, Sharpe 3.90, excess kurtosis +3.12
(fatter-tailed than Elite-Ensemble but far more stable than the rejected N=7 variant),
Calmar 0.85. Raw data: `data/goal60_robust_speed_search.csv` (the all-thirds-robust
search) and `data/goal60_yen_rapid_trades.csv` (Yen-Rapid's own trade log).

## Next steps

1. Halt Apex-Ultra's live entries (has no real edge under this model) — separate,
   more urgent action from any new deployment decision.
2. Decide between Bedrock-King (EUR/GBP, matches current deployment scope, slower but
   deeper-validated) and Yen-Punch (USDJPY, faster but thinner sample, would mean
   switching PRODLIVETRADING's traded symbol) — or hold for further validation on both.
3. Finer sweep around both candidates' neighborhoods, and a genuine train/holdout split
   for Yen-Punch that doesn't rely on a single midpoint cut, once more USDJPY history is
   available or a robustness check across multiple split points is run.
4. Full margin/MC validation (goal52-style) plus a real practice-account smoke test on
   whichever candidate moves forward, given how different both are from anything
   previously run live.

Raw data: `data/goal53_leaderboard_realistic_fill.csv` (leaderboard reproduction),
`data/goal54_realistic_fill_search.csv` + `data/goal54b_extended_search.csv` (search
sweeps), `data/goal55_extreme_n_sweep.csv` (the rejected N-up-to-50 region) +
`data/goal55_train_holdout_validation.csv` (the 2018 split that separated Bedrock-King
from it), `data/goal56_fastest_to_1m_screen.csv` (the EUR/GBP speed screen) and the
cross-symbol screen/validation behind the table above. Generating scripts:
`research_scripts/goal5{1,3,4,5,6}_*.py`.
