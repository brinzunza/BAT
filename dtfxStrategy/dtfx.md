# DTFX Strategy Specification

## 1. Executive Overview & Core Philosophy

The **DTFX Strategy** is a mathematically strict, deterministic price action and structural state machine. It eliminates subjective chart interpretations by enforcing exact candlestick body-close rules for:
1. **Trend Identification & Continuity** (Tracking Swings, Higher Highs, Higher Lows, Lower Lows, and Lower Highs).
2. **Pullback Initiation & Pivot Lock-In** (Candle body closing strictly beyond the preceding counter-colored candle's body).
3. **Zone Formation & Left-Anchoring** (Anchoring Buy, Sell, and Reversal Flip Zones from their structural origin).
4. **Strict Mitigation & Lifecycle Termination** (Zones remain active until price touches the $50\%$ Midpoint on one candle and leaves the zone on a **subsequent, different candle**).

---

## 2. Structural State Machine Definitions

At any given candle index $t$, the market exists in one of two discrete states:
$$\text{State}_t \in \{\text{UPTREND}, \text{DOWNTREND}\}$$

### State Variables:
* `pHH`: Potential Higher High (highest high reached during an impulse upward).
* `cHH`: Confirmed Higher High (locked in once a valid pullback starts).
* `pHL`: Potential Higher Low (lowest low reached during a pullback downward).
* `cHL`: Confirmed Higher Low (locked in once a breakout closes above `cHH`).
* `last_cHL`: The most recent confirmed Higher Low floor (crucial structural invalidation floor).
* `pLL`: Potential Lower Low (lowest low reached during an impulse downward).
* `cLL`: Confirmed Lower Low (locked in once a valid pullback starts).
* `pLH`: Potential Lower High (highest high reached during a pullback upward).
* `cLH`: Confirmed Lower High (locked in once a breakdown closes below `cLL`).
* `last_cLH`: The most recent confirmed Lower High ceiling (crucial structural invalidation ceiling).

---

## 3. Uptrend Mechanics & Buy Zone Creation

### 3.1. Impulse & Potential Higher High (`pHH`)
* While in an **UPTREND**, as price moves upward with green candles ($\text{Close}_t \ge \text{Open}_t$):
$$\text{pHH} = \max(\text{pHH}, \text{High}_t)$$

### 3.2. Strict Pullback Initiation (`cHH` Lock-In)
* A pullback is **strictly triggered** when a red candle appears ($\text{Close}_t < \text{Open}_t$) whose **body closes strictly below the body of the most recent green candle**:
$$\text{Close}_{\text{red}} < \text{Open}_{\text{prev green}}$$
* When this condition is met:
  1. $\text{pHH}$ is officially locked as **`cHH` (Confirmed Higher High)**.
  2. The pullback begins tracking **`pHL` (Potential Higher Low)**:
$$\text{pHL} = \text{Low}_t$$

### 3.3. Breakout Continuation & BUY ZONE Generation
* Price can continue pulling down during an uptrend as long as its candle body does **not** close below `last_cHL`.
* Once a green candle body **closes strictly above `cHH`**:
$$\text{Close}_t > \text{cHH}$$
* **Immediate Actions**:
  1. `pHL` becomes confirmed as **`cHL` (Confirmed Higher Low)**.
  2. `last_cHL` is updated: $\text{last\_cHL} = \text{cHL}$.
  3. A **BUY ZONE** is confirmed and created:
$$\text{Zone Top} = \text{cHH}$$
$$\text{Zone Bottom} = \text{cHL}$$
$$\text{50\% Midpoint Entry} = \frac{\text{cHH} + \text{cHL}}{2}$$
  4. Current price begins tracking the next `pHH`.

---

## 4. Downtrend Mechanics & Sell Zone Creation

### 4.1. Impulse & Potential Lower Low (`pLL`)
* While in a **DOWNTREND**, as price moves downward with red candles ($\text{Close}_t < \text{Open}_t$):
$$\text{pLL} = \min(\text{pLL}, \text{Low}_t)$$

### 4.2. Strict Pullback Initiation (`cLL` Lock-In)
* A pullback is **strictly triggered** when a green candle appears ($\text{Close}_t \ge \text{Open}_t$) whose **body closes strictly above the body of the most recent red candle**:
$$\text{Close}_{\text{green}} > \text{Open}_{\text{prev red}}$$
* When this condition is met:
  1. $\text{pLL}$ is officially locked as **`cLL` (Confirmed Lower Low)**.
  2. The pullback begins tracking **`pLH` (Potential Lower High)**:
$$\text{pLH} = \text{High}_t$$

### 4.3. Breakdown Continuation & SELL ZONE Generation
* Price can continue pulling up during a downtrend as long as its candle body does **not** close above `last_cLH`.
* Once a red candle body **closes strictly below `cLL`**:
$$\text{Close}_t < \text{cLL}$$
* **Immediate Actions**:
  1. `pLH` becomes confirmed as **`cLH` (Confirmed Lower High)**.
  2. `last_cLH` is updated: $\text{last\_cLH} = \text{cLH}$.
  3. A **SELL ZONE** is confirmed and created:
$$\text{Zone Top} = \text{cLH}$$
$$\text{Zone Bottom} = \text{cLL}$$
$$\text{50\% Midpoint Entry} = \frac{\text{cLH} + \text{cLL}}{2}$$
  4. Current price begins tracking the next `pLL`.

---

## 5. Trend Flips & Reversal Flip Zones

A **Trend Flip** occurs when price violates the established structural floor/ceiling of the prevailing trend. When a flip occurs, an instant **Flip Zone** is created between the broken structural level and the reversal extreme.

### 5.1. Uptrend $\to$ Downtrend Flip (`SELL FLIP ZONE`)
* **Trigger Condition**: In an Uptrend, a red candle body closes strictly below the last confirmed higher low floor:
$$\text{Close}_t < \text{last\_cHL}$$
* **State Transition**:
  * Trend flips immediately from **UPTREND $\to$ DOWNTREND**.
  * The highest peak that launched the reversal dump becomes the first confirmed lower high: $\text{cLH} = \text{Peak}$.
  * The low of the breakdown becomes the initial $\text{pLL}$.
* **⚡ SELL FLIP ZONE Boundaries**:
$$\text{Zone Top} = \text{cLH (Top Red Peak)}$$
$$\text{Zone Bottom} = \text{last\_cHL (Yellow Broken Floor)}$$
$$\text{50\% Midpoint Limit Sell} = \frac{\text{cLH (Top Red)} + \text{last\_cHL (Yellow Floor)}}{2}$$

### 5.2. Downtrend $\to$ Uptrend Flip (`BUY FLIP ZONE`)
* **Trigger Condition**: In a Downtrend, a green candle body closes strictly above the last confirmed lower high ceiling:
$$\text{Close}_t > \text{last\_cLH}$$
* **State Transition**:
  * Trend flips immediately from **DOWNTREND $\to$ UPTREND**.
  * The lowest trough that launched the reversal surge becomes the first confirmed higher low: $\text{cHL} = \text{Trough}$.
  * The high of the breakout becomes the initial $\text{pHH}$.
* **⚡ BUY FLIP ZONE Boundaries**:
$$\text{Zone Top} = \text{last\_cLH (Yellow Broken Ceiling)}$$
$$\text{Zone Bottom} = \text{cHL (Bottom Green Trough)}$$
$$\text{50\% Midpoint Limit Buy} = \frac{\text{last\_cLH (Yellow Ceiling)} + \text{cHL (Bottom Green)}}{2}$$

---

## 6. Zone Anchoring & Exact Lifecycle Rules

```
    [Origin Candle: Left-Most Part]
    ├───┬───────────────────────────────────┬───────────────┐
    │   │                                   │ 50% Midpoint  │
    │   │  ZONE BOX (Starts from left)      ├ - - - - - - - ┤ 🎯 Candle A (Touches 50%)
    │   │                                   │               │
    └───┴───────────────────────────────────┴───────▲───────┴───────────────▲
                                                    │                       │
                                           Validation Breakout     🏆 Candle B (Leaves Zone)
                                           (Leaves zone up)        Zone terminates at Candle B!
```

### 6.1. Left-Most Origin Anchoring
* Every zone box starts horizontally from the **left-most candle timestamp** of the structural swing that originated it:
  * **Buy Zone**: Starts at the earlier of $\text{cHH}$ timestamp or $\text{cHL}$ timestamp.
  * **Sell Zone**: Starts at the earlier of $\text{cLH}$ timestamp or $\text{cLL}$ timestamp.
  * **Flip Zone**: Starts at the earlier of the reversal peak/trough or the broken $\text{last\_cHL} / \text{last\_cLH}$ timestamp.

### 6.2. Strict $50\%$ Touch & Exit Rule (Different Candles)
A zone is **VALID and ACTIVE** until price touches the $50\%$ Midpoint and subsequently leaves the zone:
1. **Shallow Retracements Keep Zone Active**:
   * If price enters the zone but does **not** reach the $50\%$ Midpoint, the zone is **not** consumed and remains $100\%$ active.
2. **Candle A ($50\%$ Touch)**:
   * Price reaches the $50\%$ line ($\text{Low} \le \text{50\% Mid}$ for Buy; $\text{High} \ge \text{50\% Mid}$ for Sell) at candle index $t_{\text{touch}}$.
   * The zone remains active while price is testing liquidity inside the zone.
3. **Candle B (Subsequent Exit & Zone Termination)**:
   * On a **subsequent, different candle** ($t_{\text{exit}} > t_{\text{touch}}$), price leaves the zone in the trade direction:
     * **Buy Zone**: $\text{Close} > \text{Zone Top}$ or $\text{High} > \text{Zone Top}$.
     * **Sell Zone**: $\text{Close} < \text{Zone Bottom}$ or $\text{Low} < \text{Zone Bottom}$.
   * **Result**: The trade is completed (Win/Resolved) and the **zone is immediately invalidated/mitigated, terminating right at Candle B ($t_{\text{exit}}$)**.
4. **Structural Invalidation**:
   * If a candle body closes beyond the invalidation boundary ($\text{Close} < \text{Zone Bottom}$ for Buy; $\text{Close} > \text{Zone Top}$ for Sell), the zone is immediately invalidated and ends at that candle.

---

## 7. Order Execution & Risk Management Protocol

| Parameter | Buy Zone Specification | Sell Zone Specification |
| :--- | :--- | :--- |
| **Order Type** | Limit Buy Order | Limit Sell Order |
| **Entry Price** | $\text{Zone 50\% Midpoint} = \frac{\text{Top} + \text{Bottom}}{2}$ | $\text{Zone 50\% Midpoint} = \frac{\text{Top} + \text{Bottom}}{2}$ |
| **Stop Loss (SL)** | $\text{Zone Bottom} - 0.4 \times (\text{Top} - \text{Bottom})$ | $\text{Zone Top} + 0.4 \times (\text{Top} - \text{Bottom})$ |
| **Take Profit (TP)** | $\text{Entry} + 0.8 \times (\text{Entry} - \text{SL})$ ($+0.8\text{R}$) | $\text{Entry} - 0.8 \times (\text{SL} - \text{Entry})$ ($+0.8\text{R}$) |
| **Zone Extension** | Left Origin $\to$ Candle B (Exit Candle) | Left Origin $\to$ Candle B (Exit Candle) |
| **Stale Zone Purge** | Automatically terminated once Candle B exits | Automatically terminated once Candle B exits |
