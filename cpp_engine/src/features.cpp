#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <set>
#include <thread>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Rescaled-range statistic over ret[start..start+len) (inclusive start, len returns):
// mean-adjust, cumulative-sum, R = max-min of the cumulative series (including the implicit
// leading 0), S = sample stdev of the returns. Returns R/S, or 0 if S==0 or len<2.
double rescaled_range(const double* ret, int64_t start, int64_t len) {
    if (len < 2) return 0.0;
    double mean = 0.0;
    for (int64_t t = 0; t < len; ++t) mean += ret[start + t];
    mean /= static_cast<double>(len);

    double cum = 0.0, cmin = 0.0, cmax = 0.0, var = 0.0;
    for (int64_t t = 0; t < len; ++t) {
        double d = ret[start + t] - mean;
        cum += d;
        cmin = std::min(cmin, cum);
        cmax = std::max(cmax, cum);
        var += d * d;
    }
    double s = std::sqrt(var / static_cast<double>(len - 1));
    double r = cmax - cmin;
    return s > 0.0 ? r / s : 0.0;
}

// One thread's independent pass over grid indices [g_lo, g_hi) into bar index
// i = g * stride + (window - 1), writing into the shared (disjoint-range) output arrays.
// Seeds its own trailing vol_lookback multiset by walking backward from g_lo alone -- no
// dependency on any other thread's progress.
void worker(const double* close, int64_t n_bars, int64_t window, int64_t stride,
            int64_t vol_lookback_points, const std::vector<double>& ret, int64_t g_lo,
            int64_t g_hi, RegimeFeatures* out) {
    std::multiset<double> vol_window;  // trailing realized_vol values, size <= vol_lookback_points
    std::deque<double> vol_order;      // same values, oldest-first -- gives multiset exact FIFO eviction

    // Seed: walk backward from g_lo collecting up to vol_lookback_points prior grid points'
    // realized_vol (skipping any that don't have a full `window` of history yet).
    {
        std::vector<double> seed;
        seed.reserve(vol_lookback_points);
        for (int64_t g = g_lo - 1; g >= 0 && static_cast<int64_t>(seed.size()) < vol_lookback_points; --g) {
            int64_t i = g * stride + (window - 1);
            if (i < window) break;  // no more history-qualifying grid points behind us
            int64_t ws = i - window + 1;  // first index into `ret` for this window (ret[k] = close[k]-close[k-1])
            if (ws < 1) break;
            double mean = 0.0;
            for (int64_t t = 0; t < window; ++t) mean += ret[ws + t];
            mean /= static_cast<double>(window);
            double var = 0.0;
            for (int64_t t = 0; t < window; ++t) {
                double d = ret[ws + t] - mean;
                var += d * d;
            }
            seed.push_back(std::sqrt(var / static_cast<double>(window - 1)));
        }
        // seed collected newest-first; insert oldest-first so vol_order is a true FIFO queue
        for (auto it = seed.rbegin(); it != seed.rend(); ++it) {
            vol_window.insert(*it);
            vol_order.push_back(*it);
        }
    }

    for (int64_t g = g_lo; g < g_hi; ++g) {
        int64_t i = g * stride + (window - 1);
        if (i >= n_bars) break;
        int64_t ws = i - window + 1;
        if (ws < 1) continue;  // not enough history for a full window of returns yet -> NaN stays

        double net = std::fabs(close[i] - close[ws - 1]);
        double path = 0.0;
        for (int64_t t = 0; t < window; ++t) path += std::fabs(ret[ws + t]);
        out->efficiency_ratio[i] = path > 0.0 ? net / path : 0.0;

        double rs_full = rescaled_range(ret.data(), ws, window);
        int64_t half = window / 2;
        double rs_half = half >= 2 ? rescaled_range(ret.data(), i - half + 1, half) : 0.0;
        out->hurst_est[i] = (rs_full > 0.0 && rs_half > 0.0)
                                 ? std::log(rs_full / rs_half) / std::log(2.0)
                                 : kNaN;

        double mean = 0.0;
        for (int64_t t = 0; t < window; ++t) mean += ret[ws + t];
        mean /= static_cast<double>(window);
        double var = 0.0;
        for (int64_t t = 0; t < window; ++t) {
            double d = ret[ws + t] - mean;
            var += d * d;
        }
        double vol = std::sqrt(var / static_cast<double>(window - 1));
        out->realized_vol[i] = vol;

        vol_window.insert(vol);
        vol_order.push_back(vol);
        if (static_cast<int64_t>(vol_window.size()) > vol_lookback_points) {
            double oldest = vol_order.front();
            vol_order.pop_front();
            vol_window.erase(vol_window.find(oldest));  // erase exactly one instance
        }

        auto lower = vol_window.lower_bound(vol);
        int64_t rank = std::distance(vol_window.begin(), lower);
        out->vol_percentile[i] = vol_window.size() > 1
                                      ? static_cast<double>(rank) / static_cast<double>(vol_window.size() - 1)
                                      : 0.5;
    }
}

}  // namespace

RegimeFeatures compute_regime_features(const double* /*high*/, const double* /*low*/,
                                        const double* close, int64_t n_bars, int64_t window,
                                        int64_t stride, int64_t vol_lookback_points,
                                        int n_threads) {
    RegimeFeatures out;
    out.efficiency_ratio.assign(n_bars, kNaN);
    out.hurst_est.assign(n_bars, kNaN);
    out.vol_percentile.assign(n_bars, kNaN);
    out.realized_vol.assign(n_bars, kNaN);
    if (n_bars < window + 1 || window < 4 || stride < 1) return out;

    std::vector<double> ret(n_bars, 0.0);
    for (int64_t k = 1; k < n_bars; ++k) ret[k] = close[k] - close[k - 1];

    int64_t n_grid = (n_bars - (window - 1) + stride - 1) / stride;
    if (n_grid <= 0) return out;

    int hw = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 1;
    hw = static_cast<int>(std::min<int64_t>(hw, std::max<int64_t>(1, n_grid)));

    std::vector<std::thread> pool;
    int64_t chunk = (n_grid + hw - 1) / hw;
    for (int t = 0; t < hw; ++t) {
        int64_t g_lo = t * chunk, g_hi = std::min(n_grid, g_lo + chunk);
        if (g_lo >= g_hi) continue;
        pool.emplace_back(worker, close, n_bars, window, stride, vol_lookback_points,
                           std::cref(ret), g_lo, g_hi, &out);
    }
    for (auto& th : pool) th.join();

    // Forward-fill each feature from its own grid points across the intervening bars (a
    // regime read is only ever refreshed every `stride` bars; the bars in between hold the
    // most recent value, not NaN or a naive per-bar recompute).
    double last_er = kNaN, last_h = kNaN, last_vp = kNaN, last_rv = kNaN;
    for (int64_t i = 0; i < n_bars; ++i) {
        if (!std::isnan(out.efficiency_ratio[i])) last_er = out.efficiency_ratio[i];
        else out.efficiency_ratio[i] = last_er;
        if (!std::isnan(out.hurst_est[i])) last_h = out.hurst_est[i];
        else out.hurst_est[i] = last_h;
        if (!std::isnan(out.vol_percentile[i])) last_vp = out.vol_percentile[i];
        else out.vol_percentile[i] = last_vp;
        if (!std::isnan(out.realized_vol[i])) last_rv = out.realized_vol[i];
        else out.realized_vol[i] = last_rv;
    }
    return out;
}
