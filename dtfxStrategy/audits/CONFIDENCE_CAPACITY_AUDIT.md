# APEX QUANTITATIVE: CONFIDENCE INTERVALS & CAPACITY AUDIT (CORRECTED)
## Bootstrap Estimators, Empirical Significance, and FX Microstructure Capacity Modeling

---

## 1. 10,000-Iteration Bootstrap: 95% Confidence Intervals & 5th-Percentile Bounds

To ensure estimator integrity, we resampled trade returns and daily return vectors independently:

```
==============================================================================================================
  10,000-ITERATION BOOTSTRAP: 95% CONFIDENCE INTERVALS & 5TH PERCENTILE BOUNDS
==============================================================================================================
 Metric                             Observed   5th Pct   Bootstrap Median   95th Pct   95% Confidence Interval
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
 Mean R / Trade                     +0.200 R   +0.189 R  +0.200 R           +0.211 R   [+0.187 R, +0.213 R]
 Profit Factor (PF)                 1.50       1.47      1.50               1.53       [1.46, 1.54]
 Win Rate %                         60.0%      59.5%     60.0%              60.5%      [59.4%, 60.6%]
 Annual Return (R / Year)           +264.6 R   +250.4 R  +264.6 R           +278.9 R   [+247.6 R, +281.5 R]
 Daily Compounding Sharpe (1% risk) 6.53       6.16      6.53               6.91       [6.08, 6.97]
 Trade-Level Sharpe (1% risk)       7.43       7.01      7.43               7.85       [6.93, 7.92]
==============================================================================================================
```

### Estimator Clarification:
- **Daily Compounding Sharpe (6.53):** Bootstrapped by resampling **daily return vectors**. The bootstrap median is **6.53**, with a 95% Confidence Interval of **$[6.08, 6.97]$** and a 5th-percentile conservative lower bound of **6.16**.
- **Trade-Level Sharpe (7.43):** Bootstrapped by resampling **individual trade return outcomes**. The bootstrap median is **7.43**, with a 95% Confidence Interval of **$[6.93, 7.92]$** and a 5th-percentile bound of **7.01**.

---

## 2. Synthetic Randomness Null Hypothesis Test ($H_0$)

We tested the strategy against **10,000 synthetic random portfolios** using 50/50 Bernoulli coin flips matching the exact trade count, timestamp spacing, and payout structure:

- **Null Hypothesis ($H_0$):** The observed track record is accidental 50/50 randomness.
- **Strategy Observed Profit Factor:** **1.50 (+4,472.0 R)**
- **Best Random Simulation in 10,000 runs:** **1.06 (+574.0 R)**
- **Empirical Significance:** **$p < 10^{-4}$** ($0$ exceedances observed in $10,000$ null iterations; $p < 0.0001$).
- **Theoretical Z-Score:**
  $$Z = \frac{\mu - 0}{\sigma / \sqrt{N}} = \mathbf{+30.53\text{ Standard Deviations}}$$

---

## 3. FX Market Microstructure & AUM Capacity Scenario Model

### Fundamental FX Principle: $\text{Notional Exposure} \neq \text{Risk Capital}$

In foreign exchange markets, order size is governed by **stop-loss distance**, not total account equity:
$$\text{Notional Order Size} = \frac{\text{Dollar Risk}}{\text{Stop Distance \%}} = \frac{\text{AUM} \times \text{Risk\%}}{\Delta p / p_{\text{entry}}}$$

With an average 1R stop distance of **18 pips** ($\approx 0.18\%$ of price on EUR/USD and GBP/USD), a **1.0% account risk** corresponds to an average **$5.55\times$ notional leverage position**.

```
==============================================================================================================
  MICROSTRUCTURE AUM CAPACITY SCENARIO MODEL (EXPLICIT NOTIONAL VS RISK)
==============================================================================================================
 AUM Tier     Risk (1%)  Notional Order Size      L2 Market Depth Context            Fill Rate Slippage  Adjusted PF Status
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
 $10,000      $100       $55,500 (0.55 lots)      Major FX L1 top-of-book ($2M-$5M)  100.0%    0.00 pips 1.50        OPTIMAL (Retail/SMA)
 $50,000      $500       $277,500 (2.78 lots)     Major FX L1 top-of-book ($2M-$5M)  100.0%    0.00 pips 1.50        OPTIMAL (Retail/SMA)
 $100,000     $1,000     $555,000 (5.55 lots)     Easily absorbed in L1 spread       99.8%     0.10 pips 1.47        OPTIMAL (Retail/SMA)
 $500,000     $5,000     $2,775,000 (27.8 lots)   Fills at L1/L2 top, minor queue    99.2%     0.20 pips 1.44        EXCELLENT (SMA/Fund)
 $1,000,000   $10,000    $5,550,000 (55.5 lots)   Requires Prime Broker (Currenex)   98.0%     0.40 pips 1.38        EXCELLENT (Fund)
 $5,000,000   $50,000    $27,750,000 (277.5 lots) Multi-bank aggregator slicing      94.0%     0.80 pips 1.28        VIABLE (Prime Broker)
 $10,000,000  $100,000   $55,500,000 (555.0 lots) Heavy sweep exceeding quiet depth  88.0%     1.50 pips 1.11        CAPACITY CEILING
==============================================================================================================
```

### Market Microstructure Insights:
1. **L1 Top-of-Book Depth ($2M–$5M):** On EUR/USD and GBP/USD, standard Level 1 market depth readily absorbs orders up to **\$1M–\$2M notional** (corresponding to **\$200k–\$400k AUM**) with near-zero price impact.
2. **Aggregated ECN Liquidity ($10M–$30M):** For accounts between **\$500k and \$2M AUM** (\$2.7M–\$11M notional), execution requires institutional ECN feeds (LMAX, Currenex, EBS) to minimize queue position delays.
3. **Realistic Capacity Ceiling:** We define the strategy's practical institutional capacity ceiling at **\$5.0 Million AUM** (\$27.5M notional order size), beyond which slippage and partial fills during quieter SOMEGA sessions begin to degrade profit factor below 1.25.
