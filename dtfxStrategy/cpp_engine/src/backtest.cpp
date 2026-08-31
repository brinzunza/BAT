#include "backtest.hpp"
#include <algorithm>
#include <thread>

namespace {

// Simulates zones [i_begin, i_end) into a private SimResult. Each zone is fully independent
// (no shared state), so splitting the zone range across threads is safe and preserves the
// exact per-zone trade order -- callers concatenate chunk results in ascending chunk order,
// which reproduces the identical single-threaded/numba output order exactly (validated in
// cpp_engine/validate.py).
void simulate_worker(const double* zone_top, const double* zone_bottom, const double* zone_mid,
                      const uint8_t* zone_is_buy, const int64_t* zone_start_idx,
                      const int64_t* zone_created_ns, int64_t i_begin, int64_t i_end,
                      const int64_t* times_ns, const double* lows, const double* highs, int64_t n_bars,
                      int64_t min_fill_age_ns, double stop_mult, double target_mult,
                      bool require_bounce, double probe_depth, double exc_mult,
                      int64_t max_touches, int64_t max_window, SimResult* local_out) {
    const int64_t chunk_zones = i_end - i_begin;
    const int64_t max_out = chunk_zones * (max_touches > 0 ? max_touches : 1);
    std::vector<int64_t> touch_num(max_out);
    std::vector<int64_t> entry_idx(max_out);
    std::vector<int64_t> exit_idx(max_out);
    std::vector<uint8_t> is_win(max_out);
    std::vector<double> mid_out(max_out);
    std::vector<double> stop_risk_out(max_out);
    int64_t out_count = 0;

    for (int64_t i = i_begin; i < i_end; ++i) {
        const double top = zone_top[i], bottom = zone_bottom[i], mid = zone_mid[i];
        const bool is_buy = zone_is_buy[i] != 0;
        const double zone_width = top - bottom;
        const double half_width = zone_width / 2.0;
        const double stop_risk = stop_mult * half_width;
        const double target_risk = target_mult * half_width;
        const double invalidation = is_buy ? bottom : top;
        const double probe_lvl = is_buy ? (top - probe_depth * zone_width) : (bottom + probe_depth * zone_width);
        const double exc_target = is_buy ? (top + exc_mult * zone_width) : (bottom - exc_mult * zone_width);
        const double stop_level = is_buy ? (mid - stop_risk) : (mid + stop_risk);
        const double target_level = is_buy ? (mid + target_risk) : (mid - target_risk);
        const int64_t created_ns = zone_created_ns[i];

        int64_t j = zone_start_idx[i];
        if (j >= n_bars) continue;
        const int64_t window_end = std::min(j + max_window, n_bars);

        bool zone_dead = false;
        bool in_trade = false;
        int64_t touch_count = 0;
        bool in_probe = false;
        bool had_bounce = false;
        int64_t cur_entry_j = -1;

        while (j < window_end && touch_count < max_touches) {
            const double lo = lows[j];
            const double hi = highs[j];

            if (!zone_dead) {
                const bool breached = is_buy ? (lo <= invalidation) : (hi >= invalidation);
                if (breached) {
                    zone_dead = true;
                } else if (require_bounce) {
                    const bool probed = is_buy ? (lo <= probe_lvl) : (hi >= probe_lvl);
                    if (probed) in_probe = true;
                    if (in_probe) {
                        const bool bounced = is_buy ? (hi >= exc_target) : (lo <= exc_target);
                        if (bounced) had_bounce = true;
                    }
                }
            }

            if (in_trade) {
                const bool tp_hit = is_buy ? (hi >= target_level) : (lo <= target_level);
                const bool sl_hit = is_buy ? (lo <= stop_level) : (hi >= stop_level);
                if (tp_hit || sl_hit) {
                    const bool win = tp_hit && !sl_hit;
                    touch_num[out_count] = touch_count;
                    entry_idx[out_count] = cur_entry_j;
                    exit_idx[out_count] = j;
                    is_win[out_count] = win ? 1 : 0;
                    mid_out[out_count] = mid;
                    stop_risk_out[out_count] = stop_risk;
                    ++out_count;
                    in_trade = false;
                    if (win && touch_count < max_touches && !zone_dead) {
                        in_probe = false;
                        had_bounce = false;
                    } else {
                        break;
                    }
                }
            } else {
                if (zone_dead) break;
                const bool touched_mid = is_buy ? (lo <= mid) : (hi >= mid);
                if (touch_count == 0) {
                    if (touched_mid) {
                        const int64_t age_ns = times_ns[j] - created_ns;
                        if (age_ns < min_fill_age_ns) break;
                        in_trade = true;
                        touch_count = 1;
                        cur_entry_j = j;
                    }
                } else {
                    if ((!require_bounce) || had_bounce) {
                        if (touched_mid) {
                            in_trade = true;
                            ++touch_count;
                            cur_entry_j = j;
                        }
                    }
                }
            }

            ++j;
        }
    }

    touch_num.resize(out_count);
    entry_idx.resize(out_count);
    exit_idx.resize(out_count);
    is_win.resize(out_count);
    mid_out.resize(out_count);
    stop_risk_out.resize(out_count);
    local_out->touch_num = std::move(touch_num);
    local_out->entry_idx = std::move(entry_idx);
    local_out->exit_idx = std::move(exit_idx);
    local_out->is_win = std::move(is_win);
    local_out->mid = std::move(mid_out);
    local_out->stop_risk = std::move(stop_risk_out);
}

}  // namespace

SimResult simulate_generic(const double* zone_top, const double* zone_bottom, const double* zone_mid,
                            const uint8_t* zone_is_buy, const int64_t* zone_start_idx,
                            const int64_t* zone_created_ns, int64_t n_zones,
                            const int64_t* times_ns, const double* lows, const double* highs, int64_t n_bars,
                            int64_t min_fill_age_ns, double stop_mult, double target_mult,
                            bool require_bounce, double probe_depth, double exc_mult,
                            int64_t max_touches, int64_t max_window) {
    if (n_zones <= 0) return SimResult{};

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    // Each zone can walk up to max_window bars, so don't split into more chunks than there
    // are zones, and don't bother threading a tiny zone count.
    const int64_t nt = std::max<int64_t>(1, std::min<int64_t>(hw, n_zones));

    std::vector<SimResult> parts(nt);
    std::vector<std::thread> threads;
    threads.reserve(nt);
    const int64_t chunk = (n_zones + nt - 1) / nt;

    for (int64_t t = 0; t < nt; ++t) {
        const int64_t begin = t * chunk;
        const int64_t end = std::min(begin + chunk, n_zones);
        if (begin >= end) break;
        threads.emplace_back(simulate_worker, zone_top, zone_bottom, zone_mid, zone_is_buy,
                              zone_start_idx, zone_created_ns, begin, end, times_ns, lows, highs,
                              n_bars, min_fill_age_ns, stop_mult, target_mult, require_bounce,
                              probe_depth, exc_mult, max_touches, max_window, &parts[t]);
    }
    for (auto& th : threads) th.join();

    // Concatenate in ascending chunk order == ascending zone-index order == the exact same
    // trade order the single-threaded/numba version produces.
    size_t total = 0;
    for (const auto& p : parts) total += p.touch_num.size();
    SimResult out;
    out.touch_num.reserve(total);
    out.entry_idx.reserve(total);
    out.exit_idx.reserve(total);
    out.is_win.reserve(total);
    out.mid.reserve(total);
    out.stop_risk.reserve(total);
    for (auto& p : parts) {
        out.touch_num.insert(out.touch_num.end(), p.touch_num.begin(), p.touch_num.end());
        out.entry_idx.insert(out.entry_idx.end(), p.entry_idx.begin(), p.entry_idx.end());
        out.exit_idx.insert(out.exit_idx.end(), p.exit_idx.begin(), p.exit_idx.end());
        out.is_win.insert(out.is_win.end(), p.is_win.begin(), p.is_win.end());
        out.mid.insert(out.mid.end(), p.mid.begin(), p.mid.end());
        out.stop_risk.insert(out.stop_risk.end(), p.stop_risk.begin(), p.stop_risk.end());
    }
    return out;
}
