# Causal DTFX discovery — registered before candidate results

Quote-activity experiment: acquire all timestamp-selected preceding hours for
hourly zone-failure signals in2013–early2018. Require >=95% complete15-minute
contexts in each nonempty instrument/period before fitting. Quote features at
1/5/15 minutes include available-size imbalance, directional repricing, microprice,
observed spread, quote rate and mid variation, plus activity/imbalance changes.
Execution still uses fixed1.6pip spread, raw M1 SL-first exits and300-observation
timeout. Test1R/2R exits. Train-only feature-quantile singles and pairs of four
leaders; risk .005/.01/.02/.04/.06/.08/.12. Require full-training n>=200 and
positive growth/PF>1 in both chronological halves with n>=80 each before selecting
highest training growth. Growth is only a screening proxy for speed, not a claimed
time-to$1M. Validate unchanged rules with n>=200, PF>1 and ending equity>200.
Validation is reused; eligible candidates still require NAV and later frozen
comparison. No quote experiment results have informed this declaration.

Round 10: false-break recapture. After confirmed DTFX zone availability, wait
up to three calendar days for the first far-boundary breach. Require a directional
completed close back inside the zone within three timeframe intervals. Enter
next open; stop beyond the observed extreme, with minimum distance three spreads.
One chance per zone. Test 1/5/15/60-minute source candles, 240 raw-observation
holding limit and RR .5/1/2. Features: penetration/ATR, close recovery fraction,
recapture latency, risk/ATR and width/ATR. Train-only quantile singles and pairs
of four leading features; risks .005/.01/.02/.04. Same pre-2018 selection gates.
This changes the entry sequence and structural stop; it is not merely a filter
on the previous midpoint entry or an inversion of its trade outcomes.

Round 9: spatial field of live confirmed DTFX zones. Remove zones after a known
far-boundary wick or seven calendar days; activate newly confirmed zones only
after their confirmation candle closes. Use ATR-scaled exponential densities,
nearest-zone distances above/below, directional attraction, spatial dispersion,
live count and recent birth count. Ten features oriented to trade direction.
Apply to six structural mechanisms at 5/15/60 minutes. Raw one-minute SL/TP
execution with horizon (floor(240/TF)+1)*TF observations, explicitly different
from calendar time across missing bars. Fit single feature quantiles and pairs
of four train-leading features, risks .005/.01/.02/.04, on 2013–2015 only.
Validate one train-selected policy per family on 2016–early2018, requiring
PF>1 and at least 200 trades in each. No later results inform fitting; any
later-period evaluation remains project-contaminated and must be labeled.

Round 8 addition: resolved-shadow regime evidence. For each instrument and side,
track exponentially weighted net R and dispersion with memory 16/64/256 resolved
opportunities. Only exits strictly before entry may update evidence, independent
of actual account acceptance. Require 32 prior resolved observations. Select
dispersion penalty 0/.5/1, mean edge 0/.05/.1/.2 and risk .005/.01/.02/.04 on
2013–2015 only, then validate 2016–early2018. Dispersion penalty is a heuristic,
not a confidence interval: overlapping shadow outcomes are dependent. Evaluate
six existing structural mechanisms at each available timeframe; retain all failed
tests. No post-2018 evidence or outcomes are used.

Objective: discover a DTFX-based strategy exceeding the current four-factor leaderboard
under EURUSD/GBPUSD/EURGBP/USDJPY, separate $100 buy/sell accounts, risk proportional to
the smaller balance, 50x leverage, 1.6-pip spread and one filled position per side.
No live trading changes are part of this research.

Research families: range-weighted directional pressure (a price-activity proxy, NOT
volume), rejection absorption, directional efficiency, downside/upside semivariance,
volatility compression/shocks, multi-horizon displacement and cross-currency agreement.
Test individual mechanisms, then train-selected combinations. DTFX geometry remains
the base. Baselines include Velocity (5/.2/.2h/4) and Hyper (4/.15/.1h/16).

Primary ranking objective is the published 30% speed / 30% win rate / 20% frequency /
20% drawdown safety composite. Report each component and Pareto comparisons separately;
a composite victory is not dominance on every metric. First-stage selection uses
chronological log-growth, win rate, frequency and drawdown rather than expensive MC.

Data partitions: pre-2016 discovery; 2016 through 2018-03-08 validation; 2018-03-09
onward locked evaluation, opened only after selecting a candidate and its risk.
Training trades must close before a boundary; 7-day entry embargo at boundaries.
USDJPY has no discovery-period data and is an instrument transfer test, not a fitted
instrument. All feature thresholds are fixed a priori or estimated on discovery only.
The later period has been repeatedly used by older research; it is a held-back period
for THIS run, not a pristine project-wide holdout. Do not claim otherwise.

Causality: zone eligibility starts on the bar AFTER pivot confirmation. Preserve the
extreme timestamp only as the structural-age reference. All feature inputs end on the
previous completed candle. Cross-pair inputs must be as-of joins, never nearest/future.
Maintain a shadow DTFX opportunity process independent of account acceptance: a skipped
signal does not change the shadow process, and future outcomes are used only as they
resolve in chronological account simulation. Explicitly document this strategy rule.

First-stage trade outcomes use the existing shared simulator for comparability (entry
bar exits omitted, stop wins ambiguous later bars, losing trades charged 1.6 pips).
This is only exploratory evidence. A candidate cannot be declared a verified winner
until same-bar execution, unresolved trades, bid/ask execution, confirmation causality,
and the benchmark comparison are checked. Closed-equity drawdown is labeled as such.
Pending-order capacity is an additional live-parity validation, not silently confused
with the requested one-OPEN-position-per-side constraint.

All long loops (zone detection, features, trade walks, account sweeps, Monte Carlo) run
in C++. Python orchestrates I/O and small tables. Await command completion; no busy
polling. Preserve existing research and rankings until a victory is verified.

## Round 2 — registered after execution audit, before structural results

The original 534 bid/ask indicator combinations failed the discovery/validation gates.
Change mechanisms rather than tune that losing family: completed-candle midpoint reclaim,
confirmed zone-failure reversal, and favorable-edge continuation breakout. Each enters
at the NEXT open on the appropriate ask/bid. Use structural/ATR/spread-scaled stops
(max of half zone width, 2 ATR60, 3 spreads), exits at 0.5/1R for reclaim, 1/2R for
reversal and breakout, with a 240-observation time exit. One shadow opportunity per
zone, three-day observation search. Apply the same pre-2018 indicator-family discovery
and validation gates. No post-2018 structural results are used for selection.

## Round 3 — synthetic expected-return learning

All six structural families failed the initial gates. Fit a 64-stump gradient-boosted
expected-R score using only closed outcomes before 2013. Select a score threshold and
risk from out-of-time 2013–2015 forecasts. Refit on closed outcomes before 2016 and
validate unchanged thresholds/risk during 2016–early 2018. Minimum leaf size 256 or 1%
of training rows, 16 training-only quantile bins, learning rate 0.1, bounded regression
targets [-3,3] (account P&L remains unbounded). One model per structural family.
Post-2018 data excluded from fitting and selection. Failure is recorded, not promoted.

## Round 4 — completed-bar scale transfer

Test the same six structural mechanisms on 5-, 15-, and 60-minute OHLC aggregations
computed in C++. Candles remain labeled by OPEN time, with entry after signal-candle
close. Cross-currency inputs continue to use raw one-minute bars as-of entry. Keep the
four-hour exit horizon (240/timeframe observations) and one shadow opportunity per zone.
No indicators use an unfinished aggregated candle. Selection remains pre-2018.
Discovery scoring now distinguishes losing candidates (negative log growth penalized)
instead of tying every unprofitable rule at -100; final profitability gates are unchanged.

## Round 5 — shared multi-scale failure regime

Learned scores on 5/15/60-minute families did not pass. Pool 2R zone-failure reversal
signals across those scales. Transform feature distributions to per-scale percentiles
using only closed pre-2013 observations; discover shared rules on 2013–2015 and validate
2016–early 2018. Create a causal cross-scale state machine: agreement and opposition
are sums of exp(-age/1h) for the last signal on each scale, truncated after one hour.
All simultaneous completed signals are observable as a group. Resolve competing signals
using known consensus times volatility percentile, never subsequent return. Test fixed
percentile levels, then pairs of distinct train-selected families, with 0.5–10% risk.
Require at least 200 accepted trades and PF>1 in both discovery and validation. This is
a distinct pooled strategy with its own declared gate; it does not promote the sparse
single-scale candidate or retroactively change its failed fold gates. Use the new NAV
replay before opening any later data. Later data are physically excluded from this run.

## Round 6 — currency graph dislocation

The pooled consensus strategy did not pass. Test price-level information absent from
the earlier momentum indicators: log(EURUSD / (GBPUSD * EURGBP)), its causal EMA120
innovation and normalized surprise, and each instrument's implied fair-price distance
in pips. Use completed, contemporaneous candles only (foreign staleness <=60 seconds;
feature age <=120 seconds). Test both reversion and continuation directions because a
vendor dislocation need not resolve through the traded leg. USDJPY receives no fabricated
triangular feature and does not trade these rules; the allowed universe is unchanged.
Apply to DTFX reclaim, failure, and expansion on 1/5/15-minute bars. Fixed economic
thresholds in pip units, plus stop-cost quality gates; fit risk and rule on 2013–2015,
validate 2016–early 2018, requiring PF>1 and >=200 accepted trades in each. No later
outcomes are loaded into this experiment. Quotes from the three other legs are signals,
not additional positions or pooled collateral.

## Round 7 — DTFX response forecasts and bidirectional policy

At the birth of each confirmed 5/15-minute DTFX zone, forecast the normalized gross
mid-price move over the next 60 or 240 observed one-minute bars. Features include
causal pressure/volatility/geometry, zone displacement and confirmation age, cyclical
UTC hour/day and instrument identity. A depth-three histogram boosted model (80
trees, 16 bins, learning rate .05, L2=20, leaf size >=max(100,n/200)) learns interactions
and predicts magnitude and direction rather than an already-fixed strategy's result.
Regress targets clipped to [-5,5] for robustness; execution P&L is not clipped.

Train before 2013, forecast 2013–2015. Enter in or against the DTFX direction only
when absolute predicted gross R covers spread/R plus a fixed extra edge from
{0,.05,.1,.2,.4}. Next-open bid/ask execution, 1R native stop and TP from {.25,.5,1}R,
with a horizon exit. Fit risk from {.005,.01,.02,.04,.06,.08}. Refit before 2016 and
validate only the single selected policy per timeframe/horizon on 2016–early 2018.
All sample generation and fitting physically excludes post-2018 bars. Require >1 PF
and >=200 accepted trades in discovery and validation, then NAV verification before
any later evaluation. No performance assertion until these gates pass.
