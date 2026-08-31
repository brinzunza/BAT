// Port of research_scripts/goal10_cana_nt_mc_capped.py::run_mc_capped: block-bootstrap Monte
// Carlo with a position-sizing cap (risk = risk_frac * min(equity, cap)) and drawdown/ruin
// tracking. Account assumptions (start, risk_frac, cap, targets, ruin_floor) are runtime
// arguments, not hardcoded, so a different strategy/account setup never requires touching
// this file. Parallelized over independent simulation paths with std::thread -- each path is
// embarrassingly parallel (no shared state), so this also scales with core count, which the
// single-threaded numba original did not.
//
// NOTE: uses std::mt19937_64, not numba/NumPy's Mersenne Twister -- results will not bit-match
// the Python original path-for-path. Validate statistically (percentile agreement at large
// n_sims), not bit-exact -- see cpp_engine/validate.py.
#pragma once
#include <cstdint>
#include <vector>

struct MCResult {
    std::vector<int64_t> trades_to_1m;
    std::vector<int64_t> trades_to_10m;
    std::vector<double> max_dd;
    std::vector<uint8_t> ruined;
};

MCResult run_mc_capped(const double* net_r, int64_t n, int64_t n_sims, int64_t block,
                        int64_t max_trades, uint32_t seed, double start, double risk_frac,
                        double cap, double target1, double target2, double ruin_floor,
                        int n_threads);
