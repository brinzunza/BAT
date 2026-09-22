#pragma once
// Synthetic regime-feature engine (2026-09-04, /goal "hyper creative... niche ideas... use
// C++ for the engines for large data and computational tasks"). Computes, on a strided grid
// over the full multi-year 1-minute bar history (recomputing every `stride` bars from the
// trailing `window` bars, then held constant/forward-filled between grid points -- a regime
// signal has no business updating every single bar anyway), three niche technical constructs
// used as ZONE-TOUCH GATES in goal62+ (not zone-detection inputs -- the DTFX zone geometry
// itself is untouched; these only decide whether a touch that already qualifies is ALSO
// taken):
//
//   - efficiency_ratio: Kaufman's Efficiency Ratio -- |net directional move| / |total path
//     length| over the window. Near 0 in a choppy/mean-reverting market, near 1 in a clean
//     trend. A zone-fade (retracement) entry is fundamentally a mean-reversion bet, so the
//     creative wager here is that it should do BETTER when ER is LOW, not high.
//   - hurst_est: a fast two-scale rescaled-range (R/S) Hurst estimate (RS at `window` vs
//     `window`/2, Hurst = log(RS_full/RS_half)/log(2)) -- a coarser, much cheaper cousin of
//     full multi-scale R/S regression, but the same idea: <0.5 anti-persistent/mean-reverting,
//     >0.5 persistent/trending. A second, differently-constructed read on the same "is this
//     regime fadeable" question as efficiency_ratio, kept separate (not blended) so the two
//     can be compared and combined independently downstream.
//   - vol_percentile: where the window's own realized volatility (stdev of bar-to-bar
//     returns) ranks against the trailing `vol_lookback_points` grid points' realized vol --
//     0=quietest recently seen, 1=most volatile. The wager: dead-quiet regimes barely clear
//     the spread even on a "win", and violent regimes blow through stops and reprice fills
//     badly -- so a strategy might only want the middle band.
//
// Parallelized across n_threads (default: hardware_concurrency), each thread owning a
// contiguous slice of the output grid and independently re-deriving the small amount of
// trailing state (the vol_lookback window) it needs to seed at its own start boundary --
// every grid point only ever reads backward through the read-only input arrays, so chunks
// are fully independent and need no locking.
#include <cstdint>
#include <vector>

struct RegimeFeatures {
    std::vector<double> efficiency_ratio;  // length n_bars, NaN until enough history
    std::vector<double> hurst_est;
    std::vector<double> vol_percentile;
    std::vector<double> realized_vol;
};

RegimeFeatures compute_regime_features(const double* high, const double* low, const double* close,
                                        int64_t n_bars, int64_t window, int64_t stride,
                                        int64_t vol_lookback_points, int n_threads);
