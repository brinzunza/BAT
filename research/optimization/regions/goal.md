- Low volatility ranging market →
  Narrow bands (low std_multiplier)
  work best
  - High volatility trending market →
  Wide bands (high std_multiplier)
  avoid whipsaws
  - Choppy sideways market → Shorter
  SMA periods catch quick reversions
  - Strong trend → Mean reversion
  might fail entirely

  The insight: The "best" parameters
  change based on market conditions.

  How Clustering Solves This

  Step 1: Feature Extraction

  For each time window (e.g., last 100
   bars), calculate:
  Window 1: volatility=0.02,
  trend_strength=0.1, volume_ratio=1.2
  Window 2: volatility=0.15,
  trend_strength=0.8, volume_ratio=2.5
  Window 3: volatility=0.03,
  trend_strength=0.2, volume_ratio=0.9

  Step 2: Clustering

  K-Means groups similar market
  conditions together:
  Cluster 0 (Low Vol Ranging):
  Windows 1, 3, 7, 12...
  Cluster 1 (High Vol Trending):
  Windows 2, 5, 9...
  Cluster 2 (Choppy):
  Windows 4, 6, 10...

  Step 3: Regime-Specific Optimization

  Instead of finding ONE parameter
  set, you find THREE:
  Regime 0 (Low Vol):   SMA=10,
  std_mult=1.5  →  P&L = $5,000
  Regime 1 (High Vol):  SMA=50, 
  std_mult=3.0  →  P&L = $8,000
  Regime 2 (Choppy):    SMA=5,
  std_mult=1.0  →  P&L = $2,000

  Step 4: Adaptive Trading

  During live trading:
  1. Calculate current market features
  2. Determine which cluster/regime
  we're in
  3. Use the parameter set optimized
  for THAT regime
  4. Switch parameters when regime
  changes

  Concrete Example with Your BTC Data

  Let's say you analyze
  /datasets/X_BTCUSD_minute_2025-01-01
  _to_2025-09-01.csv:

  Without Clustering (current 
  approach):
  - Test all parameters on entire
  dataset
  - Best result: SMA=20, std_mult=2.0
  → P&L = $10,000
  - But this AVERAGES performance
  across all conditions

  With Clustering:
  Jan-Feb (Low vol, ranging): Use
  SMA=8, std_mult=1.2  → P&L = $6,000
  Mar-Apr (High vol spike):   Use 
  SMA=40, std_mult=3.5 → P&L = $12,000
  May-Jun (Trending):         Don't
  trade mean reversion → P&L = $0
  (avoid losses)
  Jul-Aug (Moderate vol):     Use
  SMA=15, std_mult=2.0 → P&L = $5,000
  Total: $23,000 vs $10,000 (2.3x
  improvement)

  What You Actually Achieve

  1. Better performance: Each regime
  gets parameters tailored to its
  characteristics
  2. Risk management: Identify when
  your strategy doesn't work (trending
   regime) and avoid trading
  3. Robustness: Less overfitting
  because you're not forcing one
  solution across all markets
  4. Market understanding: Clustering
  reveals WHAT market conditions exist
   in your data
  5. Adaptive system: Automatically
  adjusts to changing markets

  The Machine Learning Enhancement

  Once you have regimes identified,
  Random Forest learns:
  - "When volatility > 3% and trend <
  0.2 → use Regime 0 parameters"
  - "When volume spikes > 2x and price
   momentum strong → use Regime 1
  parameters"

  This is faster than re-running
  clustering every time and provides
  smooth predictions for new market
  states.

  Practical Benefit for Your Research

  Instead of asking: "What's the best 
  parameter for BTC?"

  You ask: "What are the distinct 
  market conditions in BTC, and what's
   best for each?"

  This is especially powerful if you
  want to:
  - Trade multiple pairs (BTC, ETH,
  SOL) - they might share regime
  characteristics
  - Adapt to 2025 crypto markets that
  behave differently than 2024
  - Build a production system that
  doesn't need constant
  re-optimization