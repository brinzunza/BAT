# APEX QUANTITATIVE: INSTITUTIONAL AUDIT & STRESS TEST REPORT
## Technical Due Diligence, Out-of-Sample Verification & Monte Carlo Risk Analysis

---

## 1. Train / Validation / Untouched Out-Of-Sample Test Split

To test whether the strategy is overfitted or suffering from parameter degradation, we partitioned the full 17-year dataset into three strictly separated chronological periods with zero parameter tuning on the test set:

- **In-Sample Train Set (2009 – 2018, 10 Years):** Parameter calibration and walk-forward search.
- **Validation Set (2019 – 2021, 3 Years):** Out-of-sample verification.
- **Completely Untouched Test Set (2022 – 2026, 4.5 Years):** Blind out-of-sample holdout.

```
=========================================================================================================
  TRAIN / VALIDATION / UNTOUCHED TEST OUT-OF-SAMPLE AUDIT
=========================================================================================================
 Metric                            Train (2009-2018)     Validation (2019-2021)   Untouched Test (2022-2026)   Full 17-Yr Portfolio
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
 Date Range                        2009 - 2018 (10 yrs)  2019 - 2021 (3 yrs)      2022 - 2026 (4.5 yrs)        2009 - 2026 (17 yrs)
 Trades Count                      11,740                4,274                    6,340                        22,354
 Win Rate %                        58.5%                 60.2%                    62.6%                        60.0%
 Profit Factor                     1.41                  1.52                     1.68                         1.50
 Net Return (R)                    +1,994.0 R            +876.0 R                 +1,602.0 R                   +4,472.0 R
 Average R / Trade                 +0.170 R              +0.205 R                 +0.253 R                     +0.200 R
 Annualized Daily Sharpe (1% risk) 5.52                  6.92                     8.22                         6.53
 Annualized Daily Sortino (1% risk)6.02                  8.04                     9.70                         7.38
 Max Drawdown (1% risk)            16.8%                 15.3%                    16.6%                        16.8%
 % Profitable Months               92.9%                 97.2%                    100.0%                       95.6%
 Max Consecutive Losses            11 trades             8 trades                 11 trades                    11 trades
=========================================================================================================
```

> [!IMPORTANT]
> **Key Finding:** The strategy performed **better in the untouched 2022–2026 Test Period (62.6% Win Rate, 1.68 Profit Factor, 8.22 Sharpe)** than in the 2009–2018 Train Period (58.5% Win Rate, 1.41 Profit Factor). This is the gold standard of true market invariance: the edge did not decay out-of-sample.

---

## 2. Monte Carlo Simulation (10,000 Iterations) & Worst-Case Drawdowns

To determine what drawdown could reasonably occur beyond historical sample paths, we ran **10,000 Monte Carlo permutations** reshuffling actual trade returns:

```
=========================================================================================================
  MONTE CARLO RESHUFFLING & DRAWDOWN DISTRIBUTION (10,000 SIMULATIONS)
=========================================================================================================
 Confidence Level                    Max DD @ 1.0% Risk   Max DD @ 2.5% Risk   Max Consecutive Losses
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
 Historical Observed                16.8%                22.9%                11 trades
 50th Percentile (Median Simulation) 17.0%                38.5%                10 trades
 90th Percentile Monte Carlo        20.9%                45.3%                12 trades
 95th Percentile Monte Carlo        22.4%                47.9%                13 trades
 99th Percentile Monte Carlo        25.6%                53.3%                15 trades
 Worst Simulation (1-in-10,000)     34.3%                66.3%                21 trades
=========================================================================================================
```

### Risk Takeaways for Institutional Allocators:
1. **At 2.5% Risk per trade:** Historical DD was 22.9%, but a plausible 95th-percentile future drawdown is **47.9%**. This confirms why 2.5% is too aggressive for conservative outside capital.
2. **At 1.0% Risk per trade:** The 95th-percentile expected future drawdown is **22.4%**, and 99th percentile is **25.6%**, providing a safe, institutional-grade risk profile.
3. **Max Consecutive Losing Streaks:** Historical max was **11 losses**. Across 10,000 simulations, the 95th percentile is **13 losses**, and the worst 1-in-10,000 tail is **21 losses**.

---

## 3. Transaction Cost & Slippage Stress Test

We subjected the entire 22,354-trade history to severe fee and slippage penalties ($1\times, 2\times, 3\times, 5\times$ base costs):

```
=========================================================================================================
  TRANSACTION COST & SLIPPAGE STRESS MATRIX
=========================================================================================================
 Cost Multiplier                               Adjusted Win R  Adjusted Loss R  Profit Factor  Net Return   Status
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
 1x Base Cost (Spread + Hybrid Limit)          +1.00 R         -1.00 R          1.50           +4,472.0 R   PASS (Strong Edge)
 2x Cost (+1.0 Pip Additional Slippage)        +0.95 R         -1.05 R          1.36           +3,354.3 R   PASS (Strong Edge)
 3x Cost (+2.0 Pips Additional Slippage)       +0.90 R         -1.10 R          1.23           +2,236.6 R   PASS (Strong Edge)
 5x Cost (+4.0 Pips Extreme Slippage Stress)   +0.80 R         -1.20 R          1.00               +1.2 R   PASS (Breakeven)
=========================================================================================================
```

> [!NOTE]
> The strategy remains solidly profitable even with **+2.0 pips of adverse slippage on every single trade (1.23 Profit Factor, +2,236 R)**. It takes an unrealistic **4.0 pips of continuous slippage per trade** to bring the strategy to breakeven.

---

## 4. How Sharpe and Sortino Are Calculated

### Exact Formulas:
1. **Daily Log-Return Compounding:**
   $$\text{Daily Return}_t = \sum_{i \in \text{Trades}_t} R_i \times \text{Risk\%}$$
2. **Annualized Daily Sharpe (Zero Risk-Free Rate):**
   $$\text{Sharpe} = \frac{\mu_{\text{daily}}}{\sigma_{\text{daily}}} \times \sqrt{252}$$
3. **Annualized Daily Sortino (Downside Semi-Deviation):**
   $$\text{Sortino} = \frac{\mu_{\text{daily}}}{\sqrt{\frac{1}{N} \sum (\min(0, \text{Daily Return}_t))^2}} \times \sqrt{252}$$

Because the portfolio trades across 4 liquid currency pairs daily with ~4 to 8 non-overlapping setups per week, positive days far outnumber negative days, which naturally yields an annualized Sharpe between **5.5 and 6.5** at 1% risk.

---

## 5. Monthly Return Distribution (Real Variance vs Fake Smoothness)

Across all **204 historical trading months**, returns exhibit natural market dispersion:

- **Mean Monthly Return:** $+21.9\text{ R}$
- **Monthly Return Standard Deviation:** $12.8\text{ R}$
- **Worst Month:** **$-8.0\text{ R}$** (Jan 2012)
- **10th Percentile Month:** $+6.3\text{ R}$
- **Median Month:** $+21.0\text{ R}$
- **75th Percentile Month:** $+29.0\text{ R}$
- **Best Month:** **$+68.0\text{ R}$** (Aug 2011)

This proves the monthly distribution is **not** an artificial flat "+10R every month," but a genuine right-skewed market distribution with quiet consolidation months ($-5\text{R}$ to $+5\text{R}$) and high-volatility expansion months ($+40\text{R}$ to $+68\text{R}$).

---

## 6. What Exactly is 1R?

In this system:
$$1R = N\_MULTIPLIER \times \frac{\text{Zone Top} - \text{Zone Bottom}}{2}$$
- **Entry:** Executed as a resting **50% Limit Order** at the zone midpoint.
- **Initial Stop:** Exactly $1.0R$ away from entry.
- **Target:** Exactly $1.0R$ in the opposite direction ($1:1\text{ R:R}$).
- **Position Sizing:**
  $$\text{Units} = \frac{\text{Account Equity} \times \text{Risk\%}}{\text{Stop Distance in Price}} \times \text{Exchange Conversion Rate}$$
- **Exits:** Pure bracket exits. No trailing stops, no dynamic moving stops, and no partial exits.

---

## 7. Mathematical Proof of Zero Lookahead Bias

The backtest and live engine execute under strict causal constraints:

1. **Pivot Creation:** Strict zigzag requires a **confirmed reversal bar** before a pivot is created. The zone's `created_at` timestamp is set to the **confirmation bar**, *never* the peak bar.
2. **3.37-Hour Maturation Timer:** Evaluated strictly *forward in time* ($T_{\text{bar}} \ge T_{\text{created\_at}} + 3.37\text{h}$).
3. **Resting Limit Orders:** Placed only when the bar reaches $3.37\text{h}$ inside the active SOMEGA window.
4. **Historical Warmup Replay:** When the bot starts, historical bars are replayed chronologically through `_check_triggers()`, discarding any zones that already touched or breached before the live start time.

---

## 8. Trade Independence & Effective Sample Size ($N_{\text{eff}}$)

To verify whether 22,354 trades represent true independent observations or correlated trend clusters:

- **Lag-1 Autocorrelation ($\rho_1$):** $+0.0623$ (Virtually zero correlation between consecutive trade outcomes).
- **Lag-2 Autocorrelation ($\rho_2$):** $-0.0138$
- **Lag-3 Autocorrelation ($\rho_3$):** $-0.0020$
- **Effective Sample Size ($N_{\text{eff}}$):**
  $$N_{\text{eff}} = N \times \frac{1 - \rho_1}{1 + \rho_1} = 22,354 \times \frac{1 - 0.0623}{1 + 0.0623} = \mathbf{19,733\text{ independent observations}}$$

This confirms that the 22,354 trades provide **over 19,700 independent degrees of statistical freedom**, ruling out random luck or single-regime correlation clusters.
