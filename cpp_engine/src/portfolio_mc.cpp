#include "portfolio_mc.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <random>
#include <thread>

namespace {

struct Parent {
    int account;
    double risk_dollars;
    double base_margin;
    double reserved_margin;
    double accepted_r;
    bool active;
};

struct Event {
    int64_t time_ns;
    int priority;  // additions before exits at the same instant
    int64_t sequence;
    int64_t parent;
    int32_t unit;
};

struct EventCmp {
    bool operator()(const Event& a, const Event& b) const {
        if (a.time_ns != b.time_ns) return a.time_ns > b.time_ns;
        if (a.priority != b.priority) return a.priority > b.priority;
        return a.sequence > b.sequence;
    }
};

void worker(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* account_code,
    const double* risk_pct, const double* leverage_risk_cap, const int32_t* unit_count,
    const double* unit_r, const int64_t* unit_offset_ns, const double* unit_price_ratio,
    int64_t n_trades, int32_t max_units, int32_t n_accounts,
    int64_t sim_begin, int64_t sim_end, int64_t block, int64_t max_trades, uint32_t seed,
    uint32_t thread_idx, double start_per_account, double cap, double target,
    double leg_floor, double total_floor, double year_ns, int64_t boundary_gap_ns,
    double* out_target, double* out_dd, uint8_t* out_ruined, int64_t* out_processed,
    int64_t* out_accepted, int64_t* out_skipped) {
    std::seed_seq seed_seq{seed, thread_idx};
    std::mt19937_64 rng(seed_seq);
    const int64_t effective_block = std::max<int64_t>(1, std::min(block, n_trades));
    std::uniform_int_distribution<int64_t> draw(0, n_trades - effective_block);

    for (int64_t sim = sim_begin; sim < sim_end; ++sim) {
        std::vector<double> equity(n_accounts, start_per_account);
        std::vector<double> margin(n_accounts, 0.0);
        std::vector<uint8_t> alive(n_accounts, 1);
        std::vector<Parent> parents;
        parents.reserve(max_trades);
        std::priority_queue<Event, std::vector<Event>, EventCmp> events;
        int64_t sequence = 0;
        int64_t processed = 0, accepted = 0, skipped = 0;
        int64_t clock = 0;
        double target_years = -1.0;
        const double start_total = start_per_account * n_accounts;
        double peak = start_total, worst_dd = 0.0;
        bool ruined = false;

        auto total_equity = [&]() {
            return std::accumulate(equity.begin(), equity.end(), 0.0);
        };

        auto settle_through = [&](int64_t through_ns) {
            while (!events.empty() && events.top().time_ns <= through_ns && !ruined && target_years < 0.0) {
                const Event event = events.top();
                events.pop();
                Parent& p = parents[event.parent];
                if (!p.active) continue;
                const int a = p.account;
                if (event.priority == 0) {
                    // Unit arrays belong to the sampled source row, stored in Event::unit as a
                    // flattened index so no copied per-parent array is needed.
                    const int64_t ui = event.unit;
                    const double add_margin = p.base_margin * unit_price_ratio[ui];
                    if (alive[a] && add_margin <= equity[a] - margin[a]) {
                        margin[a] += add_margin;
                        p.reserved_margin += add_margin;
                        p.accepted_r += unit_r[ui];
                    } else {
                        ++skipped;
                    }
                    continue;
                }
                margin[a] = std::max(0.0, margin[a] - p.reserved_margin);
                equity[a] += p.risk_dollars * p.accepted_r;
                p.active = false;
                if (equity[a] <= leg_floor) alive[a] = 0;
                const double total = total_equity();
                if (total > peak) peak = total;
                const double dd = 100.0 * (total - peak) / peak;
                if (dd < worst_dd) worst_dd = dd;
                if (total >= target) target_years = static_cast<double>(event.time_ns) / year_ns;
                if (total <= total_floor || std::none_of(alive.begin(), alive.end(), [](uint8_t x){return x != 0;}))
                    ruined = true;
            }
        };

        // `clock`/`et`/`xt` are int64_t nanoseconds anchored to real epoch-0 entry timestamps
        // (already ~1.5e18 ns for 2018-2024 dates) and re-based every block via
        // `offset = clock - entry_ns[start]`, which grows without bound as the simulated clock
        // advances -- a strategy needing longer than int64_t's ~292-year range to reach its
        // target (confirmed to actually happen: a real 241-trade small-sample config needed
        // ~572 simulated years) silently wraps to a garbage value. `settle_through` then reads
        // that wrapped `event.time_ns` and can spuriously satisfy `total >= target` at a
        // nonsense "years" result -- this produced impossible sub-1-year years-to-$1M outputs
        // that a from-scratch Python reimplementation and a hard lower bound on elapsed time
        // both proved could not be real (see cpp_engine/README.md's "Fixed incident" note).
        // Detecting
        // the actual signed-overflow condition on every addition (rather than an arbitrary
        // year-count cutoff, which would risk silently truncating a genuinely slow but valid
        // multi-century result) and treating it as "not reached" is the correct fix: it can
        // only ever make an already-overflowing path report -1.0 instead of a fabricated value,
        // never change the outcome of a path that never approaches the boundary.
        bool overflowed = false;
        while (processed < max_trades && target_years < 0.0 && !ruined && !overflowed) {
            const int64_t start = draw(rng);
            const int64_t end = start + effective_block;
            int64_t offset;
            if (__builtin_sub_overflow(clock, entry_ns[start], &offset)) { overflowed = true; break; }
            int64_t last_entry = clock;
            for (int64_t idx = start; idx < end && processed < max_trades; ++idx) {
                int64_t et, xt, dur;
                if (__builtin_add_overflow(entry_ns[idx], offset, &et)) { overflowed = true; break; }
                dur = std::max<int64_t>(exit_ns[idx] - entry_ns[idx], 1);
                if (__builtin_add_overflow(et, dur, &xt)) { overflowed = true; break; }
                last_entry = et;
                settle_through(et);
                if (ruined || target_years >= 0.0) break;
                ++processed;
                const int a = account_code[idx];
                if (a < 0 || a >= n_accounts || !alive[a]) { ++skipped; continue; }
                const double risk_base = std::min(equity[a], cap);
                const double actual_fraction = std::min(risk_pct[idx], leverage_risk_cap[idx]);
                if (risk_base <= 0.0 || actual_fraction <= 0.0 || leverage_risk_cap[idx] <= 0.0) {
                    ++skipped; continue;
                }
                const double risk_dollars = risk_base * actual_fraction;
                const double base_margin = risk_dollars / leverage_risk_cap[idx];
                if (base_margin > equity[a] - margin[a]) { ++skipped; continue; }

                const int64_t parent_id = static_cast<int64_t>(parents.size());
                const int64_t base = idx * static_cast<int64_t>(max_units);
                parents.push_back({a, risk_dollars, base_margin, base_margin, unit_r[base], true});
                margin[a] += base_margin;
                ++accepted;
                const int32_t count = std::max<int32_t>(1, std::min(unit_count[idx], max_units));
                for (int32_t u = 1; u < count; ++u) {
                    const int64_t ui = base + u;
                    events.push({et + std::max<int64_t>(unit_offset_ns[ui], 0), 0,
                                 ++sequence, parent_id, static_cast<int32_t>(ui)});
                }
                events.push({xt, 1, ++sequence, parent_id, -1});
            }
            if (overflowed) break;
            if (__builtin_add_overflow(last_entry, std::max<int64_t>(boundary_gap_ns, 1), &clock)) {
                overflowed = true;
                break;
            }
        }
        settle_through(INT64_MAX);
        out_target[sim] = target_years;
        out_dd[sim] = worst_dd;
        out_ruined[sim] = ruined ? 1 : 0;
        out_processed[sim] = processed;
        out_accepted[sim] = accepted;
        out_skipped[sim] = skipped;
    }
}

}  // namespace

PortfolioMCResult run_portfolio_mc(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* account_code,
    const double* risk_pct, const double* leverage_risk_cap, const int32_t* unit_count,
    const double* unit_r, const int64_t* unit_offset_ns, const double* unit_price_ratio,
    int64_t n_trades, int32_t max_units, int32_t n_accounts,
    int64_t n_sims, int64_t block, int64_t max_trades, uint32_t seed,
    double start_per_account, double cap, double target, double leg_floor,
    double total_floor, double year_ns, int64_t boundary_gap_ns, int n_threads) {
    PortfolioMCResult out;
    out.target_years.assign(n_sims, -1.0);
    out.max_dd_pct.assign(n_sims, 0.0);
    out.ruined.assign(n_sims, 0);
    out.processed.assign(n_sims, 0);
    out.accepted.assign(n_sims, 0);
    out.skipped.assign(n_sims, 0);
    if (n_trades <= 0 || n_sims <= 0 || n_accounts <= 0 || max_units <= 0) return out;

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    int64_t nt = n_threads > 0 ? n_threads : static_cast<int64_t>(hw);
    nt = std::max<int64_t>(1, std::min<int64_t>(nt, n_sims));
    const int64_t chunk = (n_sims + nt - 1) / nt;
    std::vector<std::thread> threads;
    for (int64_t t = 0; t < nt; ++t) {
        const int64_t begin = t * chunk, end = std::min(begin + chunk, n_sims);
        if (begin >= end) break;
        threads.emplace_back(worker, entry_ns, exit_ns, account_code, risk_pct, leverage_risk_cap,
            unit_count, unit_r, unit_offset_ns, unit_price_ratio, n_trades, max_units, n_accounts,
            begin, end, block, max_trades, seed, static_cast<uint32_t>(t), start_per_account,
            cap, target, leg_floor, total_floor, year_ns, boundary_gap_ns,
            out.target_years.data(), out.max_dd_pct.data(), out.ruined.data(),
            out.processed.data(), out.accepted.data(), out.skipped.data());
    }
    for (auto& thread : threads) thread.join();
    return out;
}
