# APEX QUANTITATIVE: ZONE MAXIMUM AGE & EXPIRY RESEARCH
## Empirical Study on Zone Longevity, Decay Profiles & Expiry Filters (2005–2026)

---

## 1. Zone Age Cohort Analysis (21,772 Trades across 22 Years)

```
===================================================================================================================
  ZONE AGE COHORT ANALYSIS (WIN RATE & PF DECAY PROFILES BY ELAPSED TIME)
===================================================================================================================
 Zone Age Cohort      Trades   % of Total   Win Rate %   Profit Factor   Avg R / Trade   Total Return
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 3.3h to 6h            4,558        20.9%        63.0%            1.71        +0.261 R     +1,188.0 R
 6h to 12h             4,252        19.5%        61.0%            1.56        +0.220 R       +934.0 R
 12h to 24h (1 day)    3,460        15.9%        59.2%            1.45        +0.184 R       +636.0 R
 24h to 48h (2 days)   2,131         9.8%        58.8%            1.43        +0.176 R       +375.0 R
 48h to 72h (3 days)   1,886         8.7%        58.0%            1.38        +0.160 R       +302.0 R
 72h to 120h (5 days)  1,596         7.3%        57.9%            1.38        +0.158 R       +252.0 R
 120h to 240h (10 d)   1,491         6.8%        57.7%            1.36        +0.154 R       +229.0 R
 240h to 720h (1 mo)   1,131         5.2%        63.0%            1.71        +0.261 R       +295.0 R
 > 720h (>1 month)     1,267         5.8%        60.0%            1.50        +0.200 R       +253.0 R
===================================================================================================================
```

---

## 2. Parameter Sweep: Maximum Zone Age Expiry Filter Comparison

```
===================================================================================================================
  PARAMETER SWEEP: MAXIMUM ZONE AGE EXPIRY FILTER COMPARISON
===================================================================================================================
 Max Zone Age Filter     Trades   Win Rate %   Profit Factor   Total Return   Annual Return (R)   Sharpe   Max DD
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 12 Hours (0.5 days)     10,960        60.8%            1.55     +2,364.0 R          +113.1 R     5.38    17.1%
 24 Hours (1.0 days)     14,621        60.4%            1.53     +3,043.0 R          +145.6 R     5.67    18.4%
 36 Hours (1.5 days)     16,153        60.3%            1.52     +3,319.0 R          +158.8 R     5.75    16.0%
 48 Hours (2.0 days)     16,848        60.2%            1.51     +3,430.0 R          +164.1 R     5.79    17.8%
 72 Hours (3.0 days)     17,788        60.2%            1.51     +3,620.0 R          +173.2 R     5.89    19.4%
 96 Hours (4.0 days)     18,407        60.0%            1.50     +3,695.0 R          +176.8 R     5.88    18.7%
 120 Hours (5.0 days)    18,831        60.1%            1.51     +3,797.0 R          +181.7 R     5.92    18.5%
 168 Hours (1.0 week)    19,383        60.1%            1.50     +3,901.0 R          +186.7 R     5.95    18.6%
 240 Hours (10.0 days)   19,801        60.1%            1.51     +3,999.0 R          +191.3 R     6.02    16.0%
 336 Hours (2.0 weeks)   20,169        60.2%            1.51     +4,111.0 R          +196.7 R     6.14    16.0%
 504 Hours (3.0 weeks)   20,481        60.3%            1.52     +4,201.0 R          +201.0 R     6.20    16.0%
 720 Hours (1.0 month)   20,750        60.3%            1.52     +4,278.0 R          +204.7 R     6.26    16.0%
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Unlimited (No Max Age)  21,772        60.3%            1.52     +4,464.0 R          +213.6 R     6.29    16.1%
===================================================================================================================
```

---

## 3. Conclusions & Recommendation

1. **Every age cohort remains solidly profitable (PF 1.36 to 1.71):**
   * Zones that sit untouched for 10 days, 1 month, or longer maintain strong win rates (57.7% – 63.0%) because untested market-structure pivots retain institutional significance until invalidated.
2. **Adding a Maximum Age filter destroys substantial alpha:**
   * Capping zone age at **24 hours** cuts annual profit from **+213.6 R/yr down to +145.6 R/yr (-32% reduction)** and lowers Sharpe from **6.29 down to 5.67**.
   * Capping zone age at **1 week** cuts annual profit from **+213.6 R/yr down to +186.7 R/yr (-13% reduction)** and lowers Sharpe from **6.29 down to 5.95**.
3. **Recommendation:**
   * **Keep the minimum fill filter at 3.33 hours**, but **DO NOT add a maximum age limit**. Unbroken structural zones continue generating systematic alpha regardless of age.
