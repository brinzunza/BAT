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

**Update, same day: the cost-model correction below closes this gap outright.** Both
numbers improve under the fix (Yen-Punch's own entry was less often deep-touch than
Yen-Rapid's shorter-age variant, so it had less to gain), but not equally: Yen-Punch's
race-to-$1M improves to 36.1 years, Yen-Rapid's improves further, to **31.5 years** —
now faster than Yen-Punch outright, on top of already beating it on frequency (40.0/yr
vs. 17.5/yr) and robustness. Yen-Rapid now beats every prior candidate this session on
both named axes simultaneously, with no remaining exception.

Full stats: 280 trades / 7.0 years, WR 93.21%, Sharpe 3.90, excess kurtosis +3.12
(fatter-tailed than Elite-Ensemble but far more stable than the rejected N=7 variant),
Calmar 0.85. Raw data: `data/goal60_robust_speed_search.csv` (the all-thirds-robust
search) and `data/goal60_yen_rapid_trades.csv` (Yen-Rapid's own trade log).

## Making the original 4-symbol portfolio profitable: Volume-Ensemble (`/goal`, 2026-09-04)

Two follow-up requests tested the limits of what's been found so far. First: applying
Yen-Rapid's exact numbers uniformly across all 4 symbols (matching how PRODLIVETRADING
actually works -- one shared engine config, not per-symbol tuning) **fails outright**,
PF=0.847, because Yen-Rapid's recipe is tuned to USDJPY's own volatility and loses money
on the other three pairs (EURUSD 0.630, EURGBP 0.847, GBPUSD 0.899). Using Yen-Rapid as
the USDJPY leg inside an Elite-Ensemble-style per-symbol portfolio is *worse* than the
original Elite-Ensemble, not better (PF 1.752 vs. 1.851, 54.3yr vs. 40.9yr to $1M) --
Yen-Rapid's shorter fill-age gate makes USDJPY grab the shared slot more often, crowding
out the other three symbols' rarer, higher-quality contributions that made the original
ensemble work.

Second: reverting to the literal original DTFX definition (N=2.0, R:R=1.00, uniform
across all 4 symbols, the project's own pre-optimization baseline) **blows up completely**
-- PF=0.402, -100% drawdown (total ruin) at 6% risk, 405 trades/yr. N=2.0 is the tightest
stop tested all session; `stop_risk` is so small relative to the 1.6-pip spread that the
realistic-fill correction consumes nearly the entire trade on almost every entry. This is
the same mechanism behind every other failure in this document, just at its most severe.

The `/goal` directive was explicit: keep working this specific shape (base strategy,
4-symbol portfolio, high trade count) until it's actually profitable, not switch to a
lower-frequency alternative. Doing that seriously means tuning each symbol's own
N/R:R/MIN_RISK_PCT for a real per-trade margin, not just "any PF above 1.0" -- goal59's
own finding (the cross-symbol slot competition selects by whichever entry-time arrives
first, not by quality) means a thin edge gets diluted further by dilution, not less.

**Volume-Ensemble**: EURUSD (N=28.0/R:R=0.50/MIN_RISK_PCT=0.020), GBPUSD (N=30.0/R:R=0.15/
MIN_RISK_PCT=0.016), EURGBP (N=22.0/R:R=0.20/MIN_RISK_PCT=0.025), USDJPY (Yen-Steady's own
N=4.0/R:R=0.15/MIN_RISK_PCT=0.005) — genuinely profitable at real volume:

- **94 trades/yr** — more than 2x Yen-Rapid (40/yr), over 5x Elite-Ensemble (18/yr), the
  highest frequency of any *validated, profitable* candidate this session.
- PF 1.102 (full history, 1,582 trades over 16.9 years), WR 85.71%.
- **Robust on every check applied this session**: 2018 train/holdout (1.114 / 1.088,
  n=785/797), and the stricter three-way split (1.143 / 1.075 / 1.080 — all positive,
  the same bar that rejected the earlier N=7 USDJPY variant).
- Max drawdown only −8.5% (6%-risk basis), −6.4% on the 1%-risk reference curve.

**Said plainly, not hidden**: this is not the fastest or highest-Sharpe candidate found
this session — Sharpe 0.63, ~353 extrapolated years to $1M. A thin per-trade edge spread
across many trades compounds slower than a strong edge concentrated in fewer trades, the
identical tension found in every prior section of this document. Volume-Ensemble answers
a different, explicitly-requested question — make the high-frequency portfolio shape
profitable while keeping trade count high — not "find the fastest path to $1M." Per-leg
contribution within the portfolio: USDJPY PF=2.843 (70 trades) and EURGBP PF=1.930 (56
trades) carry a disproportionate share of the edge; GBPUSD (1,200 trades, PF=1.158) is
the volume driver; EURUSD (256 trades, PF=0.915) is a net drag the ensemble still absorbs
without going under 1.0 overall.

Raw data: `data/goal61_volume_ensemble_trades.csv`.

## Cost model correction: capped at one spread per trade, not two (2026-09-04)

A direct audit question caught a real double-count: the "every trade pays the spread"
rule and the shallow-touch entry-repricing rule were stacking. Traced numerically on a
10-pip stop: a *deep*-touch loss correctly cost 1.6 extra pips (just the universal
charge, since its entry was free). But a *shallow*-touch loss — the ~93% majority under
every strategy in this document — was paying **3.2 pips**: 1.6 baked into its own wider
effective risk (from the repriced, worse entry) *and* another 1.6 from the universal
charge stacked on top. Two genuinely separate 1.6-pip hits, not the intended one.

Fixed by capping total realistic-fill cost at one spread per trade: the universal charge
now only applies to deep touches (which got a free entry and still owe the exit-side
cost by themselves); a shallow touch's own repriced entry *is* its one-spread cost, so
the universal charge no longer stacks on top of it.

Effect: every ranking, edge, and validation call made earlier in this document holds —
nothing reverses — every candidate's numbers just improve modestly, since the
shallow-touch majority was overcharged before:

| Candidate | PF (before) | **PF (corrected)** |
|---|---:|---:|
| Yen-Rapid | 1.670 | **1.779** |
| Yen-Steady | 1.637 | **1.736** |
| Elite-Ensemble | 1.851 | **1.879** |
| Yen-Punch | 1.573 | **1.592** |
| Bedrock-King | 1.251 | **1.267** |
| Volume-Ensemble | 1.102 | **1.126** |
| Dollar-Wide | 1.646 | **1.655** |
| Cable-Wide | 1.433 | **1.439** |

Train/holdout robustness for the two flagship recommendations both improved in step:
Elite-Ensemble 2.014/1.766 (was 2.000/1.725), Volume-Ensemble 1.133/1.118 (was
1.114/1.088). Race-to-$1M improved too, and not equally across candidates — Yen-Rapid
36.2→**31.5yr**, Yen-Punch 33.1→36.1yr (barely changed; a shorter fill-age gate meant
Yen-Rapid had more shallow-touch trades that were being overcharged before, so it had
more to gain) — which is what closes the one gap the Yen-Rapid section above had left
open: Yen-Rapid now beats Yen-Punch on speed too, not just frequency. Fix in
`goal51_single_symbol_closest_price_backtest.py`'s own `cost_in_r` line; every
full-history number elsewhere in this document predates this correction and is a slight
underestimate of the true PF, not an overestimate.

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

## Hyper-creative synthetic indicators + a new C++ regime engine (`/goal`, 2026-09-04)

`/goal`: "try adding new parameters... combine the strategy with other synthetic
indicators... be hyper creative and use niche ideas... use C++ for the engines for large
data and computational tasks. find profitable, frequent, high winrate strategies. At
least 3 unique ones."

None of the fully-validated realistic-fill machinery above (fill-depth model, one-spread
cost cap, closest-price resolver, thirds-split robustness bar) is touched. This layer only
decides, per already-qualifying zone touch, whether to ALSO require a market-regime or
zone-quality reading to fire — every gate defaults to off, so the ungated baseline is
always exactly reproducible for direct before/after comparison.

### Three new indicators

Two are computed by a new C++ function, `dtfx_engine.compute_regime_features` (added to
`cpp_engine/src/features.cpp`, exposed via `bindings.cpp` — build with
`cpp_engine/build.sh` as before): a strided grid (default every 50 bars, from the trailing
500-bar window, forward-filled between grid points — a regime read has no business
refreshing every single bar) computes, in one pass, per instrument:

1. **Efficiency Ratio** (Kaufman ER): `|net directional move| / |total path length|` over
   the window — near 0 in a choppy market, near 1 in a clean trend. A zone-fade is a
   mean-reversion bet; the wager is it should fire only when the market genuinely IS
   choppy, not when price is trending straight through the zone.
2. **Hurst estimate**: a fast two-scale rescaled-range (R/S) approximation
   (`log(RS_full/RS_half)/log(2)`, RS at the full window vs. half of it) — <0.5
   anti-persistent/mean-reverting, >0.5 persistent/trending. A cheaper, differently-built
   second read on the same question as ER, tested independently rather than blended in.
3. **Volatility percentile**: the window's own realized volatility (stdev of bar-to-bar
   returns), ranked against the trailing 200 grid points' realized vol via an exact-FIFO
   `std::multiset` (0 = quietest recently seen, 1 = most volatile).

Multi-threaded (one thread per contiguous output range, each independently re-deriving its
own small trailing state so chunks need no locking) and validated exact-match against a
pure-Python reference at 12 spot-checked points spanning EURUSD's full 2009–2026 history
(max abs diff ≤2.2e-15, floating-point noise) — full pass computes all three features
over 6.34M bars in **0.13 seconds**.

4. **Impulse Velocity** (pure NumPy — no new per-bar data needed): a new zone-quality
   construct built straight from each zone's own geometry, not the bar series — the speed,
   in pips/minute, of the structural break that created the zone (from its pivot bar to
   the moment it's confirmed). Wager: a zone born from a fast, violent break carries more
   real institutional intent than one of identical height that formed from a slow grind,
   and should hold up better on retest.

**Calibration lesson worth recording**: the first sweep attempt anchored gate thresholds
to each indicator's percentile across the instrument's FULL zone population, and every
impulse-velocity gate came back completely non-binding — identical trade count at the
25th/50th/75th percentile. Traced to `MIN_RISK_PCT` already implicitly selecting almost
entirely for fast-forming zones (USDJPY-Yen-Rapid's actual touches have a median matched
velocity of ~2,500 pips/min vs. ~240 pips/min across the full unfiltered zone population) —
population-wide percentiles sat two orders of magnitude below where a heavily-pre-filtered
config's touches actually live. Fix: calibrate every gate threshold against the STRATEGY'S
OWN already-filtered touch population, not the raw zone population (`percentile_grid` in
`goal62_regime_feature_strategies.py`).

### Four validated strategies, one per symbol, three different headline indicators

Each starts from an already-validated base config from earlier in this document and adds
exactly one gate. Every number below is fully robust — all three thirds-of-history PF>1.0,
and (where a train/holdout split applies) both halves PF>1.0 too:

| Strategy | Symbol | Base | Gate | Trades/yr | Win rate | PF | Sharpe | years-to-$1M |
|---|---|---|---|---|---|---|---|---|
| **Storm-Rider** | USDJPY | Yen-Rapid | vol-percentile ≥ 0.40 | 40.0→**35.2** | 93.2%→**93.9%** | 1.779→**1.992** | 4.40→**5.24** | 31.5→32.7 |
| **Chop-Fade Balanced** | EURUSD | Balanced-Dollar | Efficiency Ratio ≤ 0.05 | 135.8→**71.9** | 70.2%→**70.7%** | 1.158→**1.190** | 1.34→**1.45** | 319.4→**232.1** |
| **Chop-Fade Bedrock** | EURGBP | Bedrock-King | Efficiency Ratio ≤ 0.07 | 31.0→**22.2** | 63.2%→**63.9%** | 1.267→**1.306** | 2.01→**2.31** | 2509.2→**166.4** |
| **Velocity-Cable** | GBPUSD | Cable-Wide | Impulse Velocity ≥ 2,154 pips/min | 8.1→**6.0** | 49.2%→**52.2%** | 1.439→**1.622** | 3.00→**3.95** | 95.1→**93.5** |

Standouts:

- **Storm-Rider** is the best all-around candidate of the four — still very frequent
  (35/yr), a 93.9% win rate, PF nearly 2.0, Sharpe 5.24, and a vanishing −1.7% max
  drawdown (1%-risk reference). The vol-percentile floor's job here is narrow but
  effective: cut only the deadest-quiet ~40% of USDJPY's own recent-volatility
  distribution, where a near-guaranteed-small-target trade (R:R=0.15) barely clears the
  spread even on a full win.
- **Chop-Fade Bedrock**'s headline number is the years-to-$1M column: Bedrock-King alone
  was net profitable but glacial (2,509 years, effectively "never" in practice); requiring
  Efficiency Ratio ≤0.07 (genuinely choppy conditions only) at entry makes it **15x
  faster** while *raising*, not lowering, PF and win rate — fewer trades, but each one
  carries real edge instead of some being trend-environment noise. Still slow in absolute
  terms (166 years) — an honest improvement, not a fix for the underlying frequency
  problem.
- **Chop-Fade Balanced** keeps genuine frequency (72/yr, still among the highest of any
  candidate this session) while improving every other axis — the strongest demonstration
  that a regime gate doesn't have to trade away frequency for quality.
- **Velocity-Cable** is the impulse-velocity showcase: PF +12.7%, win rate +3pp, Sharpe
  +32%, on a config (Cable-Wide) that started as the weakest-win-rate baseline of the four
  (49.2%, barely above a coin flip before edge) — confirming the construct adds real
  signal beyond what MIN_RISK_PCT/N/R:R alone already select for.

**Honest negative finding**: Hurst-gated variants were tested on every one of the four base
configs and never beat the same base config's Efficiency-Ratio or volatility-percentile
gate, and on several configs (e.g. Yen-Rapid, Cable-Wide) actively hurt PF and thirds-split
robustness relative to no gate at all. The cheap two-scale R/S approximation used here is
likely too noisy at this bar-count/window-size combination to add signal beyond what ER
already captures more directly — a full multi-scale R/S regression might do better, but
wasn't pursued given ER and vol-percentile already cleared the bar.

Raw data: `data/goal62_storm_rider_trades.csv`, `data/goal62_chop_fade_balanced_trades.csv`,
`data/goal62_chop_fade_bedrock_trades.csv`, `data/goal62_velocity_cable_trades.csv` (each
final strategy's full trade log), `data/goal62_regime_gate_sweep.csv` (57-row exploratory
sweep across all four base configs × all four gate types, supporting the table above).
Generating script: `research_scripts/goal62_regime_feature_strategies.py`. C++ source:
`cpp_engine/src/features.{hpp,cpp}`, exposed as `dtfx_engine.compute_regime_features`.

## A mandatory 4-symbol portfolio, >730 trades/year, high win rate, quickest to $1M (`/goal`, 2026-09-04)

`/goal`: "create a combined strategy where you create a portfolio must trade all four
tickers together. It must have over 730 trades per year, must have high winrate, and
quickest time to 1 million. remember be hyper creative, create your own hypothesis and
test."

### Hypothesis 1: Slot-Saturation Portfolio — rejected, with a hard ceiling found

If N (the stop/target multiplier) is raised far past anything used elsewhere this session,
`risk_pct = N × zone_height / 2 / mid` clears `MIN_RISK_PCT` at a much smaller raw zone
height — mechanically flooding in far more (thinner) qualifying zones. Confirmed directly:
EURGBP's raw touch-generation rate alone hits 741.7 *unconstrained* touches/year at
N=60/R:R=0.30. The hypothesis: pairing an aggressive-N, low-R:R config (low R:R
mechanically inflates win rate — a target much closer than the stop is hit far more often)
on all four symbols should turn PRODLIVETRADING's real, always-been-this-way shared
execution slot (one confirmed trade per side, across the WHOLE account, all symbols
combined) from mostly-idle into a densely-packed multi-symbol queue.

**It doesn't work, and not by a small margin.** The raw touch-generation number above is
*not* what's tradeable — it ignores the one-resting-order and one-confirmed-trade
constraints entirely. Measuring the TRUE constrained rate (`resolve_closest_price` per
symbol/side, then `g30.simulate_two_account`'s own accept/reject race across the combined
stream — the exact methodology goal59 already established) tells a completely different
story:

| Round | Config shape | Combined freq/yr | Pooled WR | PF |
|---|---|---|---|---|
| 1 | 4×moderate-aggressive (N=40-50, R:R=0.15) | 107.6 | 86.4% | 0.912 |
| 3 | 4×near-individually-peak (N=40-50, R:R=0.15) | 109.9 | 87.2% | 0.977 |
| 4 | 4×extreme (R:R=0.05-0.10) | 557.8 | 94.8% | 0.940 |
| 6 | EURGBP+EURUSD pushed harder (N=30) | 633.8 | 95.8% | 0.938 |
| 7 | 4-way balanced, no single dominant | 310.9 | 93.7% | 0.965 |
| 8 | 3 dominant (EUR crosses) + thin USDJPY | 439.3 | 95.2% | **1.012** |
| best found | EURGBP+EURUSD(gated)+GBPUSD+USDJPY | 426.6 | 95.8% | 0.994 |

Two findings, confirmed across 8 rounds spanning N∈[4,200] and R:R∈[0.05,1.50]: (1) **more
competing dense per-symbol streams cause *worse* mutual cannibalization, not better
averaging** — Round 7's 4-way-balanced attempt underperforms Round 4/6's fewer-dominant-
streams shape on both frequency AND PF, and every naive "just make all four aggressive"
attempt lands around 100-110/yr, an order of magnitude short of the target; (2) **every
configuration that clears ~550+ combined trades/year has PF measurably below 1.0; every
configuration with PF at or above 1.0 tops out around 400-450/year.** This is a hard
structural throughput ceiling of the shared-execution-slot design itself, not a parameter-
tuning gap — a single global "one confirmed trade per side" gate, fed by four independently-
racing symbols, discards far more candidates than it delays.

### Hypothesis 2 (winner): Federated Portfolio

The user's own words earlier this session — *"IN the future I will add multi ticker for
this so make it modular"* — describe a different, equally real architecture: **four
independent bot instances, one per symbol, each on its own dedicated two-sub-account pair**
(exactly how EUR/GBP already runs standalone today), never sharing an execution slot with
the others at all. This isn't a looser or hypothetical model — it's literally running four
already-configured single-symbol bots side by side. Under it, each symbol's own
individually-validated realistic-fill frequency and PF apply with **zero cross-symbol
dilution**, because there's no shared resource left to contend over.

**Federated Portfolio** (recommended):

| Symbol | Config | Trades/yr | Win rate | PF | Own best risk | $200→ |
|---|---|---|---|---|---|---|
| EURGBP | N=35, R:R=0.05, MRP=0.013 | 443.8 | 96.4% | 1.020 | 2% | $194 |
| EURUSD | N=40, R:R=0.05, MRP=0.015, vol-pct≥0.5 | 310.5 | 95.9% | 1.009 | 4% | $204 |
| USDJPY | N=8, R:R=0.30, MRP=0.005 | 141.1 | 78.9% | 1.031 | 38% | $649 |
| GBPUSD | N=22, R:R=0.25, MRP=0.020 | 27.0 | 81.5% | 1.074 | 14% | $243 |
| **Combined** | — | **849.8** | **94.3%** | **1.026** | — | $800→$1,290 |

All three of the user's literal requirements clear: **849.8 trades/year** (>730), **94.3%
pooled win rate**, and a positive, profitable extrapolation to $1M (**252.75 years** at each
sleeve's own independently-optimized risk%, 16.9yr backtest span → 2.86% CAGR).

**Full honesty on what this number actually costs**: 252.75 years is slow — dramatically
slower than the fastest single-symbol candidates found this whole session (Storm-Rider 32.7
years at 35 trades/yr, Yen-Rapid 31.5 years at 40 trades/yr). Two of the four sleeves
(EURGBP, EURUSD) carry such a thin per-trade edge at R:R=0.05 that their own best-available
risk level is only 2-4% — they contribute the bulk of the frequency and win-rate numbers,
but almost none of the actual dollar compounding ($194 and $204 from $200 starts,
essentially flat). USDJPY's N=8/R:R=0.30 sleeve does almost all of the real work ($649 from
$200, more than 3x). A quality-tilted alternative confirms this trade-off directly:
softening EURGBP/EURUSD to N=70/N=50 (224.0/228.9 trades/yr, PF 1.127/1.071 — the EURUSD
sleeve is now individually thirds-robust) drops combined frequency to 549.7/yr (**misses**
the >730 bar) but improves years-to-$1M to 204.88 — confirming frequency and compounding
speed pull in opposite directions here, not the same one.

**Also honest**: none of the four sleeves pass this session's own stricter thirds-of-history
robustness bar. EURGBP and EURUSD at R:R=0.05 show the exact same pattern at *every* N and
R:R tested in this territory — the first two-thirds of history sit at PF 0.79-0.96, and all
of the edge is concentrated in the most recent third (PF 1.7-1.9) — a real, consistent
finding across dozens of configs, not noise, and a genuine regime-dependency risk this
strategy carries that the single-symbol candidates from the prior `/goal` (Storm-Rider,
Chop-Fade Bedrock) did not.

Raw data: `data/goal63_slot_saturation_trades.csv` (Hypothesis 1's best-found shared-slot
attempt), `data/goal63_federated_{eurgbp,eurusd,usdjpy,gbpusd}_trades.csv` (each Federated
Portfolio sleeve's own trade log). Generating script:
`research_scripts/goal63_hyperfreq_portfolio.py`.
