# APEX QUANTITATIVE: WALK-FORWARD OOS & ADVERSE SELECTION AUDIT
## Expanding-Window Out-of-Sample Verification, Asymmetric Fill Modeling & Strategy Freeze

---

## 1. Expanding-Window Walk-Forward Out-of-Sample Scorecard (2015 to 2026)

To eliminate data-snooping and hindsight bias, we conducted an expanding-window walk-forward test anchored at 2009. Each test year was evaluated strictly out-of-sample with completely frozen parameters:

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

> [!IMPORTANT]
> **Key Finding:** Across **12 consecutive out-of-sample years**, **100% of walk-forward test periods were profitable** (zero unprofitable OOS years), with Profit Factors ranging from **1.27 to 1.89**.

---

## 2. Dense Capacity Sweep with Asymmetric Adverse Selection ($1M to $50M)

### The Adverse Selection Reality of Passive Limit Orders:
1. **Winning Trades ($P(\text{fill}\mid\text{Win})$):** Price grazes the 50% midpoint and rapidly reverses toward target. Large resting orders suffer queue exhaustion and partial fills.
2. **Losing Trades ($P(\text{fill}\mid\text{Loss})$):** High-volume institutional momentum slices violently through the level to stop-loss, guaranteeing near-100% fill rates on losing trades.

```
=============================================================================================================================
  DENSE ORDER-SIZE CAPACITY SWEEP WITH ASYMMETRIC ADVERSE SELECTION ($1M TO $50M)
=============================================================================================================================
 Order Size  Eq AUM (1%) P(fill|Win) P(fill|Loss) Avg Slippage Profit Factor Avg R / Trade Annual Dollar PnL (Linear) Compounded Return
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 $1.0M       $0.18M      98.8%       99.4%        0.08 pips    1.48          +0.193 R      $+454,307 / yr             +1144.7% / yr
 $2.5M       $0.45M      96.8%       98.5%        0.17 pips    1.45          +0.182 R      $+1,054,794 / yr            +940.1% / yr
 $5.0M       $0.90M      93.2%       97.0%        0.31 pips    1.39          +0.163 R      $+1,839,741 / yr            +671.0% / yr
 $7.5M       $1.35M      89.4%       95.5%        0.44 pips    1.34          +0.144 R      $+2,356,127 / yr            +472.0% / yr
 $10.0M      $1.80M      85.6%       94.0%        0.57 pips    1.28          +0.123 R      $+2,606,558 / yr            +325.1% / yr
 $12.0M      $2.16M      82.4%       92.8%        0.66 pips    1.24          +0.106 R      $+2,617,623 / yr            +235.8% / yr (PEAK)
 $14.0M      $2.52M      79.3%       91.6%        0.75 pips    1.19          +0.088 R      $+2,462,477 / yr            +165.6% / yr
 $16.0M      $2.88M      76.0%       90.4%        0.84 pips    1.15          +0.069 R      $+2,143,134 / yr            +110.4% / yr
 $18.0M      $3.24M      72.8%       89.2%        0.93 pips    1.10          +0.049 R      $+1,661,694 / yr             +67.0% / yr
 $20.0M      $3.60M      69.5%       88.0%        1.02 pips    1.06          +0.028 R      $+1,020,330 / yr             +32.8% / yr
 $22.0M      $3.96M      66.2%       86.8%        1.11 pips    1.01          +0.006 R        $+221,266 / yr              +5.7% / yr
 $24.0M      $4.32M      62.9%       85.6%        1.19 pips    0.96          -0.018 R        $-733,225 / yr             -15.6% / yr
 $26.0M      $4.68M      59.5%       84.4%        1.28 pips    0.92          -0.043 R      $-1,840,839 / yr             -32.5% / yr
 $30.0M      $5.40M      52.7%       82.0%        1.44 pips    0.82          -0.098 R      $-4,506,056 / yr             -56.6% / yr
 $50.0M      $9.00M      40.0%       70.0%        2.22 pips    0.67          -0.200 R     $-12,405,596 / yr             -74.8% / yr
=============================================================================================================================
```

> [!CAUTION]
> **Institutional Capacity Limit:**
> - When factoring in **Asymmetric Adverse Selection**, the true mathematical peak of the dollar extraction curve occurs at **\$12.0 Million notional order size (\$2.16 Million AUM at 1% risk)**, extracting **\$2.62 Million USD net profit per year**.
> - Beyond **\$14 Million notional (\$2.5M AUM)**, adverse fill selection rapidly destroys the statistical edge, crossing breakeven at \$22M notional.
> - *Disclaimer:* Capacity estimates are simulated microstructure scenario models conditional on historical L2 depth assumptions; they have not yet been validated through live institutional execution.

---

## 3. Formal Production Strategy Freeze: `APEX v1.0 (PROD)`

To permanently resolve data-snooping and over-optimization concerns, the production strategy is formally frozen:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   APEX v1.0 PRODUCTION STRATEGY MANIFEST                 │
├──────────────────────────────────────────────────────────────────────────┤
│ Version:              APEX v1.0 (PROD)                                   │
│ Release Date:         August 27, 2026                                    │
│ Codebase Status:      FROZEN (No further parameter modifications)        │
│ Repository Commit:    PRODLIVETRADING @ main                             │
├──────────────────────────────────────────────────────────────────────────┤
│ CORE FROZEN PARAMETERS:                                                  │
│ • MIN_RETRACEMENT:    2 (Strict Zigzag Confirmation)                     │
│ • N_MULTIPLIER:       3.343496 (1:1 Risk-to-Reward Boundary)              │
│ • MIN_FILL_AGE:       3.366682 hours (202 minutes Maturation Trap Filter)│
│ • MIN_RISK_PCT:       0.000170 (Minimum Volatility Floor)                │
│ • MAX_ZONE_TOUCHES:   4 (Multi-Touch Re-Arming Cap)                      │
│ • HYBRID_BACKSTOP:    1.5x (Native Exchange Emergency Market Backstop)   │
│ • SOMEGA SESSIONS:    EURGBP (18-23 UTC), GBPUSD (20-22), USDJPY (19-22) │
└──────────────────────────────────────────────────────────────────────────┘
```
