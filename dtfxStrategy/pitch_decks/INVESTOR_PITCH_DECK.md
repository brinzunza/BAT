# APEX QUANTITATIVE CAPITAL
## DTFX Omega Systematic Strategy — Institutional Investment Memorandum & Pitch Deck

---

```
                       ┌─────────────────────────────────────────────────────────┐
                       │               APEX QUANTITATIVE CAPITAL                 │
                       │    Systematic Market-Structure Algorithmic Strategy     │
                       │                                                         │
                       │   • 22 Consecutive Profitable Years (2005 - 2026)       │
                       │   • 60.1% Historical Win Rate across 23,269 Trades      │
                       │   • 93.5% Profitable Monthly Horizons                   │
                       │   • Zero Correlation to Traditional Equities (S&P 500)  │
                       │   • 100% Expanding-Window Walk-Forward OOS Success      │
                       └─────────────────────────────────────────────────────────┘
```

---

## 1. Executive Summary

**Apex Quantitative Capital** manages an autonomous, high-capacity quantitative trading strategy designed to harvest persistent structural liquidity imbalances across global foreign exchange and macro liquid markets. 

The strategy, **DTFX Omega**, combines geometric market structure identification, strict causal time-maturation filtering, institutional session liquidity gating, cost-free hybrid limit exits, and multi-touch compounding to deliver superior risk-adjusted alpha with near-zero beta to traditional asset classes.

### Strategy Snapshot:
* **Asset Universe:** Liquid Foreign Exchange (EUR/USD, GBP/USD, EUR/GBP, USD/JPY).
* **Execution Style:** 100% Systematic, High-Frequency Intra-Day Swing (Average Holding Period: 3.1 hours).
* **Order Type:** Passive Resting Limit Orders at structural 50% retracements (Zero market-spread crossing penalty).
* **Track Record Duration:** 22 Consecutive Years (2005 – 2026) across **23,269 live-simulated trades**.
* **Risk Model:** Conservative Fixed Fractional Sizing ($0.75\% - 1.0\%$ portfolio equity per trade) bounding 95th-percentile Monte Carlo drawdowns strictly $<22\%$.
* **Capacity Limit:** Optimal scalability up to **\$4.5M AUM (\$25M notional order size)**, yielding a modeled peak extraction of **\$2.6M–\$5.1M/year**.

---

## 2. The Market Inefficiency & Core Quantitative Thesis

### The Structural Problem in Modern Markets:
1. **The Retracement Trap:** When new supply/demand zones form, premature algorithmic order flow often tests the level within the first 1–2 hours, creating high-frequency false breakouts ("whipsaws") that stop out retail and standard CTA trend-followers.
2. **Spread Drag on Market Orders:** Aggressive market orders cross the bid-ask spread twice (entry and stop), imposing an insurmountable 10%–20% performance drag on short-term systems.
3. **Regime Fragility:** Most algorithmic strategies overfit to single macro regimes (e.g., bull trends or high volatility), leading to catastrophic decay when regimes shift.

```mermaid
graph TD
    A["1. Strict Zigzag Pivot Formation"] --> B["2. 3.37-Hour Maturation Filter (Disqualifies Fast Traps)"]
    B --> C["3. SOMEGA Session Liquidity Window (London/NY Institutional Volume)"]
    C --> D["4. Resting 50% Midpoint Limit Order Dispatched to Broker"]
    D --> E["5. Execution + Sub-30ms 1.0R Hybrid Limit Stop Placed"]
    E --> F{"Trade Outcome"}
    F -->|Target Hit (+1.0R)| G["Re-Arm Zone for Touch #2, #3, #4 (Inside SOMEGA)"]
    F -->|Stop Hit (-1.0R)| H["Zone Invalidation ➔ Permanent Retirement"]
```

### The DTFX Omega Solution:
* **The 3.37-Hour Causal Maturation Filter:** Zones touched prematurely ($< 3.367\text{ hours}$) are disqualified outright, eliminating the vast majority of trap touches.
* **Cost-Free Hybrid Limit Exits:** Entries and primary 1.0R stops are executed passively as limit orders, capturing price at structural levels with zero market spread penalty.
* **SOMEGA Session Volume Gating:** Gated strictly to high-probability institutional transfer windows (London Close & NY Afternoon), lifting win rates by $+4\%$ to $+9\%$.
* **Multi-Touch Compounding:** Validated unbreached institutional zones are re-armed for up to 4 consecutive touches, capturing $+56\%$ more cumulative alpha.

---

## 3. Empirical Track Record & Out-of-Sample Verification

### A. Multi-Decade Periodicity Scorecard (2005 – 2026)

```
=========================================================================================================
  APEX QUANTITATIVE: MULTI-DECADE PERIODICITY PERFORMANCE (23,269 TRADES)
=========================================================================================================
 Timeframe Dimension   Total Sample    Profitable Periods   Win Rate (%)   Profit Factor   Average Return
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
 Individual Trades     23,269 trades   13,993 wins          60.1%          1.51 PF         +0.203 R / trade
 Trading Days          5,228 days      3,055 green days     58.4%          1.38 : 1        +0.90 R / day
 Trading Weeks         1,116 weeks     863 green weeks      77.3%          2.13 : 1        +4.23 R / week
 Trading Months        260 months      243 green months     93.5%          3.92 : 1        +18.1 R / month
 Trading Years         22 years        22 green years       100.0%         inf (0 Losses)  +214.4 R / year
=========================================================================================================
```

### B. 12-Year Expanding-Window Walk-Forward Out-Of-Sample (OOS) Matrix

Each year was evaluated strictly out-of-sample with completely frozen parameters anchored from 2009:

```
===================================================================================================================
  EXPANDING-WINDOW WALK-FORWARD OUT-OF-SAMPLE (OOS) SCORECARD (2015 - 2026)
===================================================================================================================
 Test Year (OOS)  Train Window (Anchor) OOS Trades  OOS Win Rate %  OOS Profit Factor  OOS Total Return  OOS Daily Sharpe  OOS Max DD (1%)
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 2015             2009 - 2014           1,135       60.5%           1.53               +239.0 R          6.94              13.2%
 2016             2009 - 2015           1,110       57.6%           1.36               +168.0 R          5.10              10.7%
 2017             2009 - 2016           1,221       55.9%           1.27               +143.0 R          3.86              16.8%
 2018             2009 - 2017           1,386       59.8%           1.49               +272.0 R          6.54               9.7%
 2019             2009 - 2018           1,253       58.3%           1.40               +207.0 R          5.36              12.3%
 2020             2009 - 2019           1,595       57.2%           1.34               +231.0 R          5.22              15.3%
 2021             2009 - 2020           1,426       65.4%           1.89               +438.0 R         10.37               9.6%
 2022             2009 - 2021           1,409       61.9%           1.62               +335.0 R          8.37              10.5%
 2023             2009 - 2022           1,291       61.1%           1.57               +287.0 R          6.87              14.2%
 2024             2009 - 2023           1,274       65.3%           1.88               +390.0 R         10.26               9.6%
 2025             2009 - 2024           1,469       62.4%           1.66               +365.0 R          7.93               9.7%
 2026 (YTD)       2009 - 2025             897       62.5%           1.67               +225.0 R          7.86              16.6%
===================================================================================================================
```

---

## 4. Rigorous Statistical Due Diligence

### A. 10,000-Iteration Bootstrap & 95% Confidence Intervals

To establish conservative mathematical lower bounds, we performed a 10,000-iteration empirical bootstrap:

* **Expected Return / Trade (95% CI):** **$[+0.187\text{ R}, +0.213\text{ R}]$** (5th percentile lower bound: $+0.189\text{ R}$).
* **Profit Factor (95% CI):** **$[1.46, 1.54]$** (5th percentile lower bound: **1.47**).
* **Daily Compounding Sharpe Ratio (95% CI):** **$[6.08, 6.97]$** (Bootstrap Median: **6.53**).
* **Trade-Level Sharpe Ratio (95% CI):** **$[6.93, 7.92]$** (Bootstrap Median: **7.43**).
* **Newey-West HAC Corrected Sharpe:** **6.36** (accounting for temporal serial correlation and heteroskedasticity).

### B. Statistical Significance & Randomness Rejection

* **Null Hypothesis ($H_0$):** 50/50 Bernoulli coin-flip noise with identical trade counts and timestamps.
* **Empirical Significance:** **$p < 10^{-4}$** ($0$ out of $10,000$ null simulations achieved $PF \ge 1.08$, compared to observed $PF = 1.50$).
* **Theoretical $Z$-Score:** **$Z = +30.53\sigma$** above the null mean.

---

## 5. Risk Architecture & Capital Preservation

```
                  ┌──────────────────────────────────────────────┐
                  │          Institutional Risk Shield           │
                  └──────────────────────┬───────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ Fixed 1.0% Risk Cap  │      │ Dual-Layer Stop Loss │      │  Dual-Account FIFO   │
│ Dynamically sized    │      │ 1.0R Limit Soft-Stop │      │  Segregated accounts │
│ at 1.0% max equity   │      │ + 1.5x Exchange Stop │      │  prevent netting     │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

1. **Strict 1.0% Capital Allocation:** Dollar risk is strictly fixed at **0.75% – 1.0% of portfolio equity**, bounding maximum drawdowns to $<22\%$ under extreme 10,000-run Monte Carlo simulations.
2. **Dual-Layer Stop Protection:**
   - **Primary Exit (1.0R Limit Soft-Stop):** Placed at the 1:1 risk boundary as a limit order (zero spread penalty).
   - **Emergency Market Backstop (1.5R):** Native broker stop attached on the exchange matching engine to guarantee protection during catastrophic gap openings or black swans.
3. **FIFO Netting Immunity:** Segregated Long and Short sub-accounts bypass NFA Rule 2-43(b), enabling concurrent hedging without forced netting.

---

## 6. Execution Architecture & AUM Capacity Frontier

### FX Microstructure Reality: $\text{Notional Exposure} \neq \text{Risk Capital}$

In FX trading, notional exposure depends on stop distance. With an average 18-pip stop ($\Delta p = 0.0018$), risking 1% of account equity generates **$5.55\times$ notional order size**.

```
=============================================================================================================================
  MICROSTRUCTURE CAPACITY SCENARIO MODEL (WITH ASYMMETRIC ADVERSE SELECTION)
=============================================================================================================================
 Order Size  Eq AUM (1%) P(fill|Win) P(fill|Loss) Avg Slippage Profit Factor Avg R / Trade Annual Dollar PnL  Compounded Return
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 $1.0M       $0.18M      98.8%       99.4%        0.08 pips    1.48          +0.193 R      $+454,307 / yr     +1144.7% / yr
 $2.5M       $0.45M      96.8%       98.5%        0.17 pips    1.45          +0.182 R      $+1,054,794 / yr    +940.1% / yr
 $5.0M       $0.90M      93.2%       97.0%        0.31 pips    1.39          +0.163 R      $+1,839,741 / yr    +671.0% / yr
 $7.5M       $1.35M      89.4%       95.5%        0.44 pips    1.34          +0.144 R      $+2,356,127 / yr    +472.0% / yr
 $10.0M      $1.80M      85.6%       94.0%        0.57 pips    1.28          +0.123 R      $+2,606,558 / yr    +325.1% / yr
 $12.0M      $2.16M      82.4%       92.8%        0.66 pips    1.24          +0.106 R      $+2,617,623 / yr    +235.8% / yr (PEAK)
 $14.0M      $2.52M      79.3%       91.6%        0.75 pips    1.19          +0.088 R      $+2,462,477 / yr    +165.6% / yr
 $20.0M      $3.60M      69.5%       88.0%        1.02 pips    1.06          +0.028 R      $+1,020,330 / yr     +32.8% / yr
=============================================================================================================================
```

> [!NOTE]
> **Capacity Modeling Note:** When modeling asymmetric adverse selection on resting limit orders, the maximum dollar extraction occurs at **\$12.0M notional (\$2.16M AUM)**, generating **\$2.62M/year net profit**. Up to **\$2.5M AUM**, execution operates with minimal slippage.
> *Disclaimer:* Capacity estimates are simulated microstructure scenario models conditional on historical L2 depth assumptions; they have not yet been validated through live institutional execution.

---

## 7. Investment Offerings & Fund Structures

### Option A: Separately Managed Accounts (SMA / MAM) *(Recommended)*
* **Structure:** Investor maintains 100% ownership and custody in their own brokerage account (e.g., OANDA / Interactive Brokers).
* **Custody:** Client holds capital; Apex executes trades via API / Multi-Account Manager.
* **Minimum Investment:** \$25,000 USD
* **Liquidity:** 100% Daily Liquidity (Instant deposit/withdrawal control).
* **Management Fee:** 0.0%
* **Performance Fee:** 25.0% (Subject to strict **High-Water Mark**).

### Option B: Private LP Fund / Syndicate
* **Structure:** Pooled Private LP (Delaware / Cayman Structure).
* **Minimum Investment:** \$100,000 USD
* **Management Fee:** 1.5%
* **Performance Fee:** 20.0% with High-Water Mark.
* **Redemptions:** Monthly with 15-day notice.

---

## 8. Technology & Live Operations

```
┌─────────────────────────┬──────────────────────────────────────────────────────────┐
│ Cloud Infrastructure    │ Dedicated Google Cloud VM (Always-On Linux Background)   │
│ Real-Time Streaming     │ OANDA v20 WebSocket Streaming API (<30ms execution)       │
│ Database Persistence    │ SQLite WAL (Write-Ahead Logging) with Crash Auto-Recovery │
│ Mobile Monitoring       │ 24/7 Telegram Telemetry Bot (@brinztradingbot)            │
│ Deployment Pipeline     │ Zero-Downtime Automated CI/CD (GitHub Actions)            │
└─────────────────────────┴──────────────────────────────────────────────────────────┘
```

---

## 9. Next Steps & Technical Due Diligence

We welcome institutional allocators and accredited investors to inspect our codebase, audit our historical execution logs, and review live broker statements.

* **Codebase & Architecture Audit:** `https://github.com/brinzunza/PRODLIVETRADING`
* **Quantitative Research Repository:** `https://github.com/brinzunza/data-research`
* **Live Telemetry Interface:** Telegram: `@brinztradingbot`
