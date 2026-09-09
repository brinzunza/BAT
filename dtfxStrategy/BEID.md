# BEID — refined strategy and empirical validation

Updated 2026-09-09. **Best candidate identified; reliable profitability remains unproven. Live deployment is not justified yet.**

The strongest training-selected refinement is **40-pip flip zones, aligned with the latest completed hourly trend, one touch per zone, one position across the four-pair portfolio**. It improves substantially on the baseline and has positive historical net R in each tested period at 1.6-pip costs. However, its 2016–2020 result turns negative at 2.4 pips and all later-sample confidence intervals include negative expectancy. Do not confuse a profitable historical sample with an established trading edge.

## Revised rules

1. **Signal:** use completed 3-minute candles and the versioned BEID state machine in `beid_conservative_audit.py`. Preserve the running-extreme/pullback confirmation logic and elapsed 48-hour pullback timeout. Keep only `BUY_FLIP` / `SELL_FLIP` zones, formed when a close breaks the active structural floor/ceiling.
2. **Size filter:** require total zone width >=40 pips. A pip is 0.01 for USDJPY and 0.0001 for the other three pairs.
3. **Hourly alignment:** calculate a 20-observation EMA from completed hourly closes, with at least 20 observations. A buy requires latest hourly close > EMA and EMA > previous EMA. A sell requires both inequalities reversed. Apply this condition at the first entry touch using only information available then; cancel the setup if it fails. This is an EMA trend proxy, not a higher-timeframe BEID structural signal.
4. **Entry and exits:** midpoint limit, stop at the opposite zone boundary, target at the near profit boundary. Gross target distance equals initial stop distance. No re-entry/rearming; no optimized stop or target multiplier.
5. **Timing:** activate only after the zone-confirmation candle closes. Expire unfilled orders after 48 hours. Research replay preserves the original coarse expiry eligibility; exact minute/quote expiry remains an execution validation requirement. Stops remain active after entry, including its fill candle.
6. **Portfolio:** EURUSD, GBPUSD, USDJPY and EURGBP; all hours; at most one open position in total. Ties use a fixed symbol order and known entry/stop/target/type values. Do not reuse capacity within the same execution bar. Selection does not use future exit outcomes.
7. **Risk:** no live allocation until the validation gates below pass. Increasing risk cannot fix an uncertain or negative edge. Final sizing must include costs, gaps, account-currency conversion and the user's drawdown tolerance.

The inherited signal implementation uses the last opposite-colour candle's open, not necessarily the immediately preceding candle. Its timeout re-anchors the extreme; it does not fully implement the earlier prose's “structural origin” fallback. These are explicit implementation limitations, not a claim of an authoritative institutional methodology.

## Evidence for the refinement

All instruments were restarted at 2003-01-01. The year 2003 is warmup; scored training is 2004–2015. Signals use 3-minute bars, execution is replayed on the underlying one-minute bars. Within-minute ambiguity remains adverse: stop first; a fill-minute target is credited only when the close proves a post-entry crossing. Stops that gap use the worse of the stop and open. Unresolved positions are marked at the final observed close.

The fixed grid was {20,30,40}-pip zones × {all,continuation,flip} × {all hours,07–17 source time} × {any hourly direction,aligned}: **36 configurations**. Selection used only pre-2016 mean R at 2.4-pip costs, requiring 500+ trades and positive mean R. The earlier 18-configuration experiment is also part of the research history; these trials must not be ignored when assessing selection bias. No candidate was chosen on later performance.

R is net price P&L divided by initial entry-to-stop distance. R profit factor is positive R divided by absolute negative R; it is not a dollar return. Fixed 1.6/2.4/3.2-pip deductions are scenarios, not verified executable broker costs. Costs and sizing explain why aggregate pips can disagree with aggregate R.

| Configuration | Period | Trades | PF at 1.6p | Net R at 1.6p | PF at 2.4p | PF at 3.2p |
|---|---|---:|---:|---:|---:|---:|
| 20p baseline | 2004–2015 | 13,794 | 0.962 | -273.91 | 0.871 | 0.788 |
| 20p baseline | 2016–2020 | 3,558 | 0.854 | -279.48 | 0.764 | 0.683 |
| 20p baseline | 2021–March 2026 | 4,831 | 0.941 | -146.97 | 0.844 | 0.755 |
| 20p baseline | April 2026 onward | 182 | 0.576 | -50.14 | 0.517 | 0.463 |
| 40p flips + hourly alignment | 2004–2015 | 921 | 1.383 | +147.64 | 1.309 | 1.239 |
| 40p flips + hourly alignment | 2016–2020 | 138 | 1.017 | +1.16 | 0.954 | 0.894 |
| 40p flips + hourly alignment | 2021–March 2026 | 243 | 1.225 | +24.36 | 1.155 | 1.089 |
| 40p flips + hourly alignment | April 2026 onward | 6 | 1.828 | +1.71 | 1.745 | 1.665 |

These are reused historical research partitions, not pristine holdouts. Trades crossing a partition end are excluded from its score while still occupying capacity in the continuous portfolio. The latest six-trade sample is too small for a reliable conclusion. Different instruments have different last observation dates, listed in the input manifest.

### Uncertainty and remaining failure

For the selected candidate's **387 post-2015 trades**, a 2,000-replicate, seven-calendar-day circular block bootstrap gives a 95% interval for mean net R of approximately **[−0.034, +0.168] at 1.6 pips** and **[−0.064, +0.138] at 2.4 pips**. Twenty-eight-day blocks also cross zero. These are conditional exploratory intervals, not adjusted for all model selection. The 2016–2020 stress result is negative. Both facts prevent a claim of a dependable edge.

The available five-second bid/ask caches cover roughly late August to September 9, 2026. A cold-start replay with 24-hour warmup produced **zero eligible candidate trades**. It provides no profitability evidence. The quote replay uses ask for buy entry, bid for sell entry, and the opposite quote for exit; spread is embedded and is not deducted again.

## Critical audit corrections

- The old backtest discarded some orders when the fill candle also reached the stop, ignored fill-bar exits, and chose ambiguous wins from closing direction. It also checked expiry after entry. Those conventions can misstate execution.
- The old 25-zone cap combined with post-hoc filtering can suppress eligible trades. The revised signal study applies zone admission before simulation and removes that cap; the portfolio is separately constrained.
- The saved legacy >=20-pip ledger contains **64,798 trades**, not the old document's **45,475**. The former's pip PF is 1.293 but R PF only 1.120. The old headline table is not reproducible from that ledger.
- Raw EURUSD bars contain **1965.0001 at 2001-06-07 23:32** and **−0.0001 at 2001-09-11 20:12**. Flat candles can be internally consistent yet wrong. These corrupt observations invalidate full-history performance estimates, including the first conservative audit's negative estimates.
- The new raw-input diagnostic flags those two rows and no comparable rows in the other three instruments. It checks finite/positive prices, OHLC consistency, and broad deviations from a prior median. It does not establish complete vendor or timezone provenance. New full-history runs fail closed on flags.

The 2003 restart avoids propagating corrupted 2001 signal state. No source data was changed and no isolated losing trades were deleted. Full-history outputs and earlier documents remain archived as invalidated diagnostics. The revised results change several conventions, so they do not isolate the causal effect of any one fix.

## Profitability hurdle and next validation

For midpoint entries with stop distance d=zone width/2, symmetric gross barriers and fixed round-trip cost c:

`E[R] = 2p − 1 − c/d`, so `p_break_even = (1 + c/d)/2`.

At a 40-pip zone, the break-even win rate is 54% with 1.6-pip costs and 56% with 2.4-pip costs. Use trade-specific cost/risk ratios when zone widths vary; gaps and financing worsen the hurdle. A wider zone reduces cost drag but does not prove prediction skill.

Keep this candidate frozen. The next useful evidence is executable historical bid/ask replay with complete quote coverage, commission, slippage, financing and actual portfolio occupancy—not another sweep against the reused late data. A hash-ordered public quote probe retrieved 21 of 32 attempted entry hours (22 trade-entry minutes); an immediate retry did not improve coverage. The median entry-minute spread was 0.8 pips, the 90th percentile 1.57 pips, and the maximum 5.2 pips. Missing downloads returned HTTP 503 or timeouts. These partial, cross-source cost observations do not include commission/slippage and cannot establish BEID profitability or justify deleting high-cost trades.

Before deployment, require positive stressed validation expectancy, a positive lower confidence bound that accounts for clustered trades and selection, sufficient independent observations, and acceptable mark-to-market chronological drawdown under margin constraints. Then freeze code and collect genuinely new forward observations. A proposed small risk cap such as 0.25% is only a research setting, not a proven safe recommendation. No compounding-to-$1M claim is supported.

## Reproduce

From the `tradingResearch` repository:

```sh
python3 dtfxStrategy/research_scripts/beid_data_quality.py
python3 dtfxStrategy/research_scripts/beid_clean_htf_study.py
python3 dtfxStrategy/research_scripts/beid_validation_uncertainty.py
python3 dtfxStrategy/research_scripts/beid_bidask_check.py
python3 dtfxStrategy/research_scripts/beid_quote_cost_probe.py --limit 32
python3 -m unittest discover -s dtfxStrategy/research_scripts -p 'test_beid_*.py' -v
```

The clean study requires pandas, NumPy and Numba. Outputs are under `dtfxStrategy/research_results/beid_conservative_audit/clean_2003_htf/`; `evaluation.json` and `validation_uncertainty.json` preserve results even though the repository ignores CSV files. `NEXT_EXPERIMENT.md` records the protocol. Source coverage and the raw-data diagnostic are in the parent directory. Never use the invalidated full-history caches to support deployment.

**Twelve regression tests pass**, covering fill-bar losses, stop/target ordering, gaps, expiry, final marking, malformed prices, completed-hour causality, and bid/ask-side execution. Those tests verify implementation cases, not a profitable edge. Quote-level coverage, portfolio mark-to-market drawdown, financing, full provenance and untouched forward evidence remain incomplete.

Primary references: [NFA on hypothetical-performance limitations](https://www.nfa.futures.org/rulebooksql/rules.aspx?RuleID=9025&Section=9) and [OANDA on spread-plus-commission costs](https://help.oanda.com/us/en/faqs/spreads-commission.htm). Neither validates BEID or the assumed flat costs.
