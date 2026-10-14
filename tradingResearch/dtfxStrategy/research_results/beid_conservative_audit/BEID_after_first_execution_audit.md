# BEID: execution audit and revised research strategy

Updated 2026-09-09. **Status: profitability not established; research only.**

The previous document overstated the evidence for live profitability. A conservative re-test of the four available FX histories produces losses for the 20-pip baseline. A small filter search finds a stronger training configuration, but it fails subsequent validation. Neither configuration qualifies for deployment. Increasing risk cannot repair negative expectancy.

## Raw-data correction discovered during finer replay

The raw EURUSD history contains `1965.0001` on 2001-06-07 23:32 and `−0.0001` on 2001-09-11 20:12. These flat, internally consistent candles passed the original OHLC checks. The full-history results below are now **invalidated as estimates of real performance**, including negative results: corrupted prices affect signal state and execution. Preserve them as diagnostic evidence only.

A raw-input audit flags those two rows and no rows in the other three instruments under finite/positive/OHLC-consistency and broad prior-median checks. Passing these checks is not proof of complete data quality. New full-history runs now fail closed on suspect inputs. The next study restarts all instruments in 2003, uses that year as warmup, and scores from 2004 with one-minute execution and a preregistered completed-hour trend filter. No source data was overwritten and no isolated losing trades were dropped.

## What changed in the evidence

The prior “fixed” engine repaired a frozen pullback state but retained important execution problems. See `dtfxStrategy/research_scripts/verify_beid_full_history_fixed.py`, function `backtest_beid_fixed`:

- An unfilled buy is discarded when the bar low reaches its stop, before checking whether the same bar filled its midpoint entry. Sells have the symmetric problem. A resting order can fill and lose on that bar; deleting it excludes losses.
- Stops and targets are ignored on the fill bar. A subsequent-candle constraint is not a valid model of a protective stop that is already active.
- When a later bar touches both stop and target, closing direction chooses the winner. OHLC does not establish that price path.
- Expiry is checked after the entry condition, allowing an expired order to fill. Bar-count expiry also spans weekends differently from elapsed-time expiry.
- The 25-zone cap and post-hoc zone filtering mean rejected small zones can suppress eligible large zones. Unresolved trades disappear at the dataset end.

The new audit retains the inherited swing generator but changes all these execution conventions, uses a single touch, and applies a minimum 20-pip admission rule before simulating orders. Therefore its difference from the old table is **not an isolated estimate of fill-bar bias**. Among 36,106 resulting signal trades, 1,849 stop out on their fill bar. Those events must not be silently removed.

### The old table is not reproducible from the saved ledger

The file named `beid_fixed_pre2021_trades.csv` contains later trades too. Its >=20-pip subset has **64,798 trades**, versus **45,475** in the previous document. Its pip profit factor is **1.293**, but its risk-unit profit factor is only **1.120**. Mean return falls from **+0.05589 R** at 1.6-pip costs to **+0.00530 R** at 2.4 pips and **−0.04529 R** at 3.2 pips. These are diagnostics of an execution-biased ledger, not investable returns.

The previous summary mixed win rates from different filters, called pooled positive results “consistent across all 26 years,” and described 1–2% risk as sustainable without sufficient drawdown evidence. Those claims are withdrawn. The four symbols also have different history coverage; consult the data manifest rather than assuming 26 years for each.

## Re-tested results

R denotes net price P&L divided by the original entry-to-stop distance. Profit factor below is **sum of positive R / absolute sum of negative R**, appropriate to comparing equal initial stop-risk trades. It is not a dollar portfolio return. Costs are a fixed round-trip pip deduction; 2.4 and 3.2 pips are stress scenarios, not measured broker quotes.

All portfolio rows allow only one open trade across all four symbols. Ties use known entry/stop/target/type information and a fixed symbol ordering. A slot is reused only on a later bar. Parameter selection uses pre-2016 trades with exits before the boundary: zone floors {20,30,40}, zone types {all,continuation,flip}, and hours {all,07:00–17:00 source time}, giving **18 candidates**. Selection maximizes training mean R at 2.4-pip costs, requiring at least 500 trades and positive mean R. Risk and target distances were not optimized.

The selected candidate is **40-pip minimum, flip zones only, all hours, one touch**. It is rejected after validation.

| Configuration | Period | Trades | PF at 1.6p | Mean R at 1.6p | PF at 2.4p |
|---|---|---:|---:|---:|---:|
| 20p baseline | Before 2016 | 16,567 | 0.916 | -0.0455 | 0.828 |
| 20p baseline | 2016–2020 | 3,512 | 0.851 | -0.0804 | 0.761 |
| 20p baseline | 2021–March 2026 | 4,739 | 0.910 | -0.0468 | 0.816 |
| 20p baseline | April 2026 onward | 176 | 0.583 | -0.2700 | 0.523 |
| 40p flips | Before 2016 | 2,443 | 1.273 | +0.1199 | 1.202 |
| 40p flips | 2016–2020 | 355 | 0.961 | -0.0196 | 0.904 |
| 40p flips | 2021–March 2026 | 555 | 1.018 | +0.0088 | 0.960 |
| 40p flips | April 2026 onward | 17 | 0.794 | -0.1151 | 0.750 |

These are historical research partitions, **not pristine out-of-sample evidence**: this repository has already researched later periods extensively. Trades crossing a partition's end are excluded from that partition's statistics; the continuous portfolio still accounts for their occupied slot. The last partition is particularly small for the selected candidate. No annual return, Sharpe ratio, probability of ruin, or live-profit claim follows from this table.

## Refined BEID specification

### 1. Preserve a causal structure signal

Use completed 3-minute candles. Track running extrema; a pullback confirms when a later close breaks its frozen extreme. Continuation zones span the confirmed extreme and the pullback extreme. Flip zones span the broken structural floor/ceiling and opposite running extreme. Entry remains the midpoint; initial stop and target remain the opposite boundaries.

The inherited detector uses the last opposite-colour candle's open, which need not be the immediately preceding candle. Its 48-hour timeout re-anchors the running extreme. It does not fully implement the prior prose's “structural origin” fallback. Treat the code as an explicitly versioned research definition; resolve these signal-definition differences before implementing a production engine. Do not describe it as an authoritative institutional standard.

### 2. Execute orders causally

- Create an order only after the confirmation candle has closed; activate it on the next bar. Test completed-bar time alignment explicitly when adding higher timeframes.
- Admit zones >=20 pips for the baseline. Keep the 40-pip flip configuration as a rejected research comparison, not a recommended live setting.
- Use one touch per zone. Expire an unfilled order after 48 elapsed hours, checked before any fill. Expiry does not cancel protection on an open position.
- A touched entry can stop out on the same bar. If both barriers are possible, assign the stop first until finer quotes establish ordering.
- On a fill bar, credit a target only when its close proves price crossed the target after entry. Otherwise defer an ambiguous target. This is a deliberately adverse OHLC convention, not a reconstruction of the true path.
- Gap stops execute at the worse of the stop and bar open. No favorable gap improvement is credited on limits. Mark unresolved open positions at the final close and report them separately from normal exits.
- Before live use, replace price-touch fills and flat friction with executable bid/ask quotes, spread, commission, slippage, financing, size constraints, and account-currency conversion. Do not subtract spread twice when using actual bid/ask execution.

### 3. Use a cost hurdle, not just a pip floor

For midpoint entries let zone width be Z, stop distance d=Z/2, and round-trip cost c. With exact symmetric barriers and no gaps:

`E[R] = 2p − 1 − c/d`, and `p_break_even = (1 + c/d)/2`.

At Z=20 pips and c=1.6, break-even is 58%; at c=2.4 it is 62%. At Z=40 and c=1.6 it is 54%. With variable stop widths, use the average trade-specific cost/risk ratio rather than dividing cost by the average zone. Gaps and financing can make the hurdle worse.

Reject a proposed setup unless a conservative, separately validated estimate of its conditional expectancy exceeds all-in costs with a margin. A larger zone lowers the cost hurdle but does not establish a predictive edge. Small zones are not universally “guaranteed losses”; if d<=c even a full target cannot produce a positive net result under this simplified model.

### 4. Separate edge validation from risk sizing

Current live allocation to the tested configurations: **zero**. This is a rejection decision, not a profitable strategy.

After a candidate passes independent validation, simulate chronological equity with bid/ask mark-to-market, aggregate currency exposure, margin and financing. Size from total expected stop loss including costs and gap allowance. A small initial research limit such as 0.25% per position is only a proposed cap, not an empirically established safe level. Set the actual cap from the user's acceptable drawdown and correlated block-bootstrap stress results. Never increase risk simply because pooled PF exceeds 1.5.

### 5. Further research and acceptance requirements

The most useful next experiment is finer execution data: retain the same signal and compare one-minute or quote-level fill ordering against this adverse 3-minute model. Obtain provenance for timestamps, bid/ask convention, vendor overlaps, duplicate resolution, and coverage before trusting session filters. The current inputs pass duplicate-timestamp and OHLC consistency checks, which do not establish those facts.

For subsequent signal experiments, preregister a small set of hypotheses, such as alignment with fully completed higher-timeframe structure or a causal rejection-confirmed entry. Log every attempted variant. A changed entry needs fresh target/stop and cost accounting; it cannot inherit the old midpoint statistics. Do not mine the reused late partition and then relabel it a holdout.

A candidate may advance to paper trading only after positive net expectancy in later validation, positive performance under adverse costs, adequate independent trade clusters, and acceptable chronological drawdown. Require a positive lower confidence bound on mean net R using calendar-block resampling that preserves simultaneous currency exposure; preregister block sizes and report sensitivity. Then freeze the implementation and gather genuinely new forward observations with executable quotes. Failing any gate means reject or continue research, not deploy.

## Reproduction and limitations

Run from the `tradingResearch` repository:

```sh
python3 dtfxStrategy/research_scripts/beid_conservative_audit.py
python3 -m unittest discover -s dtfxStrategy/research_scripts -p test_beid_conservative_audit.py -v
```

Outputs are in `dtfxStrategy/research_results/beid_conservative_audit/`: source coverage manifest, per-symbol trades, training grid, selected portfolio ledgers, and evaluation. `--evaluate-cache` re-evaluates saved ledgers without regenerating signals; use only caches from the same execution version. CSV files are ignored by the existing repository settings; the Markdown table and JSON summary preserve the headline results. The original document is archived alongside them.

Six execution regression tests cover fill-bar losses, ambiguous barriers, proven fill-bar targets, adverse stop gaps, elapsed-time expiry, and end-of-data marking. Tests passing does not validate predictive profitability. The audit does not yet establish quote-level execution, source timezone provenance, mark-to-market portfolio drawdown, live feasibility, or untouched forward performance. It also changes multiple legacy conventions at once; isolate each change before attributing causal performance differences.

Primary references: [NFA on limitations of hypothetical performance](https://www.nfa.futures.org/rulebooksql/rules.aspx?RuleID=9025&Section=9) explains hindsight and execution limitations. [OANDA's spread-plus-commission description](https://help.oanda.com/us/en/faqs/spreads-commission.htm) confirms that applicable spread and commission both enter trading costs. Neither source validates BEID or establishes the flat cost assumptions used here.
