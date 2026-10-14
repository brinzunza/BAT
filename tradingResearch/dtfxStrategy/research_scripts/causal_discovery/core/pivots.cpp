#include "pivots.hpp"

std::vector<Pivot> find_pivots_strict(const double* open, const double* high, const double* low,
                                       const double* close, const int64_t* times_ns,
                                       int64_t n, int min_retracement) {
    std::vector<Pivot> pivots;
    if (n < 1) return pivots;

    bool looking_for_high = true;
    double candidate_price = high[0];
    int64_t candidate_i = 0;
    int streak = 0;

    for (int64_t i = 1; i < n; ++i) {
        const bool is_red = close[i] < open[i];
        const bool is_green = close[i] > open[i];

        if (looking_for_high) {
            if (close[i] > candidate_price) {
                candidate_price = high[i];
                candidate_i = i;
                streak = 0;
            } else if (is_red) {
                ++streak;
                if (streak >= min_retracement) {
                    pivots.push_back(Pivot{times_ns[candidate_i], candidate_price, times_ns[i], 'H'});
                    looking_for_high = false;
                    candidate_price = low[i];
                    candidate_i = i;
                    streak = 0;
                }
            } else {
                streak = 0;
            }
        } else {
            if (close[i] < candidate_price) {
                candidate_price = low[i];
                candidate_i = i;
                streak = 0;
            } else if (is_green) {
                ++streak;
                if (streak >= min_retracement) {
                    pivots.push_back(Pivot{times_ns[candidate_i], candidate_price, times_ns[i], 'L'});
                    looking_for_high = true;
                    candidate_price = high[i];
                    candidate_i = i;
                    streak = 0;
                }
            } else {
                streak = 0;
            }
        }
    }
    return pivots;
}
