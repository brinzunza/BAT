#include "monte_carlo.hpp"
#include <thread>
#include <random>
#include <algorithm>

namespace {

void mc_worker(const double* net_r, int64_t n, int64_t sim_begin, int64_t sim_end,
                int64_t block, int64_t max_trades, uint32_t seed, uint32_t thread_idx,
                double start, double risk_frac, double cap, double target1, double target2,
                double ruin_floor, int64_t* out_t1, int64_t* out_t10, double* out_dd,
                uint8_t* out_ruined) {
    std::seed_seq seq{seed, thread_idx};
    std::mt19937_64 rng(seq);
    std::uniform_int_distribution<int64_t> dist(0, n - 1);

    for (int64_t s = sim_begin; s < sim_end; ++s) {
        double eq = start;
        double peak = start;
        double dd_worst = 0.0;
        int64_t trades = 0;
        int64_t t1 = -1;
        int64_t t10 = -1;
        bool is_ruined = false;
        bool done = false;

        while (trades < max_trades && !done) {
            int64_t start_idx = dist(rng);
            for (int64_t k = 0; k < block; ++k) {
                const int64_t idx = (start_idx + k) % n;
                const double r = net_r[idx];
                const double risk_base = (eq < cap) ? eq : cap;
                const double risk = risk_base * risk_frac;
                eq += risk * r;
                if (eq < ruin_floor) eq = ruin_floor;
                ++trades;
                if (eq > peak) peak = eq;
                const double dd = (eq - peak) / peak;
                if (dd < dd_worst) dd_worst = dd;
                if (eq <= ruin_floor) {
                    is_ruined = true;
                    done = true;
                    break;
                }
                if (t1 == -1 && eq >= target1) t1 = trades;
                if (t1 != -1 && t10 == -1 && eq >= target2) {
                    t10 = trades;
                    done = true;
                    break;
                }
                if (trades >= max_trades) {
                    done = true;
                    break;
                }
            }
        }

        out_t1[s] = t1;
        out_t10[s] = t10;
        out_dd[s] = dd_worst;
        out_ruined[s] = is_ruined ? 1 : 0;
    }
}

}  // namespace

MCResult run_mc_capped(const double* net_r, int64_t n, int64_t n_sims, int64_t block,
                        int64_t max_trades, uint32_t seed, double start, double risk_frac,
                        double cap, double target1, double target2, double ruin_floor,
                        int n_threads) {
    MCResult out;
    out.trades_to_1m.assign(n_sims, -1);
    out.trades_to_10m.assign(n_sims, -1);
    out.max_dd.assign(n_sims, 0.0);
    out.ruined.assign(n_sims, 0);

    if (n <= 0 || n_sims <= 0) return out;

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    int64_t nt = (n_threads > 0) ? n_threads : static_cast<int64_t>(hw);
    nt = std::max<int64_t>(1, std::min<int64_t>(nt, n_sims));

    std::vector<std::thread> threads;
    threads.reserve(nt);
    const int64_t chunk = (n_sims + nt - 1) / nt;

    for (int64_t t = 0; t < nt; ++t) {
        const int64_t begin = t * chunk;
        const int64_t end = std::min(begin + chunk, n_sims);
        if (begin >= end) break;
        threads.emplace_back(mc_worker, net_r, n, begin, end, block, max_trades, seed,
                              static_cast<uint32_t>(t), start, risk_frac, cap, target1, target2,
                              ruin_floor, out.trades_to_1m.data(), out.trades_to_10m.data(),
                              out.max_dd.data(), out.ruined.data());
    }
    for (auto& th : threads) th.join();

    return out;
}
