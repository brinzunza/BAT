#include "two_account.hpp"
#include <algorithm>
#include <queue>
#include <random>
#include <thread>

namespace {

// Exact port of goal20_speed_king_margin_report.py::position_size_and_margin.
inline void position_size_and_margin(int32_t instr_code, double equity, double entry_price,
                                      double stop_risk, double risk_pct, double max_leverage,
                                      double cap, double gbp_usd_approx,
                                      double& out_risk_dollars, double& out_margin) {
    const double risk_base = (equity < cap) ? equity : cap;
    const double risk_dollars_budget = risk_base * risk_pct;
    const double price_diff = stop_risk;

    double risk_per_unit_usd, max_notional, leverage_capped_units, eur_usd_approx = 0.0;
    if (instr_code == 1) {  // USDJPY
        risk_per_unit_usd = price_diff / entry_price;
        max_notional = risk_base * max_leverage;
        leverage_capped_units = max_notional;
    } else if (instr_code == 2) {  // EURGBP
        risk_per_unit_usd = price_diff * gbp_usd_approx;
        max_notional = risk_base * max_leverage;
        eur_usd_approx = entry_price * gbp_usd_approx;
        leverage_capped_units = max_notional / eur_usd_approx;
    } else {  // EURUSD/GBPUSD (default)
        risk_per_unit_usd = price_diff;
        max_notional = risk_base * max_leverage;
        leverage_capped_units = max_notional / entry_price;
    }
    const double risk_based_units = risk_dollars_budget / risk_per_unit_usd;

    double units = std::min(risk_based_units, leverage_capped_units);
    if (units < 1.0) units = 1.0;

    const double actual_risk_dollars = units * risk_per_unit_usd;
    double notional;
    if (instr_code == 1) {
        notional = units;
    } else if (instr_code == 2) {
        notional = units * eur_usd_approx;
    } else {
        notional = units * entry_price;
    }
    out_risk_dollars = actual_risk_dollars;
    out_margin = notional / max_leverage;
}

}  // namespace

void position_size_and_margin_debug(int32_t instr_code, double equity, double entry_price,
                                     double stop_risk, double risk_pct, double max_leverage,
                                     double cap, double gbp_usd_approx,
                                     double& out_units, double& out_risk_dollars,
                                     double& out_margin) {
    const double risk_base = (equity < cap) ? equity : cap;
    const double risk_dollars_budget = risk_base * risk_pct;
    const double price_diff = stop_risk;
    double risk_per_unit_usd, max_notional, leverage_capped_units, eur_usd_approx = 0.0;
    if (instr_code == 1) {
        risk_per_unit_usd = price_diff / entry_price;
        max_notional = risk_base * max_leverage;
        leverage_capped_units = max_notional;
    } else if (instr_code == 2) {
        risk_per_unit_usd = price_diff * gbp_usd_approx;
        max_notional = risk_base * max_leverage;
        eur_usd_approx = entry_price * gbp_usd_approx;
        leverage_capped_units = max_notional / eur_usd_approx;
    } else {
        risk_per_unit_usd = price_diff;
        max_notional = risk_base * max_leverage;
        leverage_capped_units = max_notional / entry_price;
    }
    const double risk_based_units = risk_dollars_budget / risk_per_unit_usd;
    double units = std::min(risk_based_units, leverage_capped_units);
    if (units < 1.0) units = 1.0;
    const double actual_risk_dollars = units * risk_per_unit_usd;
    double notional;
    if (instr_code == 1) {
        notional = units;
    } else if (instr_code == 2) {
        notional = units * eur_usd_approx;
    } else {
        notional = units * entry_price;
    }
    out_units = units;
    out_risk_dollars = actual_risk_dollars;
    out_margin = notional / max_leverage;
}

namespace {

struct HeapEntry {
    int64_t exit_ns;
    double risk_dollars;
    double margin;
    double r;
    int side;  // 0 = sell account, 1 = buy account
};
// Matches Python's heapq tie-break exactly: it pushes/compares the 5-tuple
// (exit_ns, risk_dollars, margin, r, side) lexicographically, so simultaneous exits (routine
// at 1-min-bar granularity -- ~11% of exits in a representative BEID trade set share exit_ns
// with another) settle in the SAME order as the Python original, not an arbitrary one. This
// doesn't change final equity (independent per-side accumulators), but does affect the
// intermediate trace and, in principle, dd_worst (computed after each individual pop, so a
// tie processed in a different order can transiently read a different "combined" equity).
struct HeapCmp {
    bool operator()(const HeapEntry& a, const HeapEntry& b) const {
        if (a.exit_ns != b.exit_ns) return a.exit_ns > b.exit_ns;
        if (a.risk_dollars != b.risk_dollars) return a.risk_dollars > b.risk_dollars;
        if (a.margin != b.margin) return a.margin > b.margin;
        if (a.r != b.r) return a.r > b.r;
        return a.side > b.side;  // Python bool: False(0) < True(1)
    }
};
using MinHeap = std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp>;

}  // namespace

TwoAccountBacktestResult simulate_two_account(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, int64_t n,
    double risk_pct, double start_per_side, double max_leverage, double cap,
    double gbp_usd_approx, bool track_detail) {
    TwoAccountBacktestResult out;
    out.accepted.assign(n, 0);

    double equity[2] = {start_per_side, start_per_side};
    double margin_used[2] = {0.0, 0.0};
    int open_count[2] = {0, 0};
    MinHeap heap;
    double peak_combined = start_per_side * 2.0;
    double dd_worst = 0.0;

    auto settle = [&](const HeapEntry& e) {
        equity[e.side] += e.risk_dollars * e.r;
        margin_used[e.side] -= e.margin;
        open_count[e.side] -= 1;
        const double combined = equity[0] + equity[1];
        if (combined > peak_combined) peak_combined = combined;
        const double dd = 100.0 * (combined - peak_combined) / peak_combined;
        if (dd < dd_worst) dd_worst = dd;
        if (track_detail) {
            out.trace_close_ns.push_back(e.exit_ns);
            out.trace_equity_buy.push_back(equity[1]);
            out.trace_equity_sell.push_back(equity[0]);
        }
    };

    for (int64_t i = 0; i < n; ++i) {
        const int64_t et = entry_ns[i];
        while (!heap.empty() && heap.top().exit_ns <= et) {
            HeapEntry e = heap.top();
            heap.pop();
            settle(e);
        }
        if (margin_used[0] < 0.0) margin_used[0] = 0.0;
        if (margin_used[1] < 0.0) margin_used[1] = 0.0;

        const int side = is_buy[i] ? 1 : 0;
        const double sizing_base = std::min(equity[0], equity[1]);
        double risk_dollars, margin;
        position_size_and_margin(instrument_code[i], sizing_base, entry_price[i], stop_risk[i],
                                  risk_pct, max_leverage, cap, gbp_usd_approx, risk_dollars, margin);
        const double avail = std::min(equity[side], cap) - margin_used[side];

        if (margin <= avail && open_count[side] < 1) {
            out.accepted[i] = 1;
            heap.push({exit_ns[i], risk_dollars, margin, net_r[i], side});
            margin_used[side] += margin;
            open_count[side] += 1;
        }
    }
    while (!heap.empty()) {
        HeapEntry e = heap.top();
        heap.pop();
        settle(e);
    }

    out.equity_buy = equity[1];
    out.equity_sell = equity[0];
    out.max_dd_pct = dd_worst;
    return out;
}

TwoAccountTraceResult simulate_two_account_trace(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, int64_t n,
    double risk_pct, double start_per_side, double max_leverage, double cap,
    double gbp_usd_approx) {
    TwoAccountTraceResult out;
    out.accepted.assign(n, 0);
    out.equity_buy_after.assign(n, 0.0);
    out.equity_sell_after.assign(n, 0.0);

    double equity[2] = {start_per_side, start_per_side};
    double margin_used[2] = {0.0, 0.0};
    int open_count[2] = {0, 0};
    MinHeap heap;

    for (int64_t i = 0; i < n; ++i) {
        const int64_t et = entry_ns[i];
        while (!heap.empty() && heap.top().exit_ns <= et) {
            HeapEntry e = heap.top();
            heap.pop();
            equity[e.side] += e.risk_dollars * e.r;
            margin_used[e.side] -= e.margin;
            open_count[e.side] -= 1;
        }
        if (margin_used[0] < 0.0) margin_used[0] = 0.0;
        if (margin_used[1] < 0.0) margin_used[1] = 0.0;

        const int side = is_buy[i] ? 1 : 0;
        const double sizing_base = std::min(equity[0], equity[1]);
        double risk_dollars, margin;
        position_size_and_margin(instrument_code[i], sizing_base, entry_price[i], stop_risk[i],
                                  risk_pct, max_leverage, cap, gbp_usd_approx, risk_dollars, margin);
        const double avail = std::min(equity[side], cap) - margin_used[side];

        if (margin <= avail && open_count[side] < 1) {
            out.accepted[i] = 1;
            heap.push({exit_ns[i], risk_dollars, margin, net_r[i], side});
            margin_used[side] += margin;
            open_count[side] += 1;
        }
        out.equity_buy_after[i] = equity[1];
        out.equity_sell_after[i] = equity[0];
    }
    return out;
}

TwoAccountBacktestResult simulate_two_account_var_risk(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, const double* risk_pct, int64_t n,
    double start_per_side, double max_leverage, double cap,
    double gbp_usd_approx, bool track_detail) {
    TwoAccountBacktestResult out;
    out.accepted.assign(n, 0);

    double equity[2] = {start_per_side, start_per_side};
    double margin_used[2] = {0.0, 0.0};
    int open_count[2] = {0, 0};
    MinHeap heap;
    double peak_combined = start_per_side * 2.0;
    double dd_worst = 0.0;

    auto settle = [&](const HeapEntry& e) {
        equity[e.side] += e.risk_dollars * e.r;
        margin_used[e.side] -= e.margin;
        open_count[e.side] -= 1;
        const double combined = equity[0] + equity[1];
        if (combined > peak_combined) peak_combined = combined;
        const double dd = 100.0 * (combined - peak_combined) / peak_combined;
        if (dd < dd_worst) dd_worst = dd;
        if (track_detail) {
            out.trace_close_ns.push_back(e.exit_ns);
            out.trace_equity_buy.push_back(equity[1]);
            out.trace_equity_sell.push_back(equity[0]);
        }
    };

    for (int64_t i = 0; i < n; ++i) {
        const int64_t et = entry_ns[i];
        while (!heap.empty() && heap.top().exit_ns <= et) {
            HeapEntry e = heap.top();
            heap.pop();
            settle(e);
        }
        if (margin_used[0] < 0.0) margin_used[0] = 0.0;
        if (margin_used[1] < 0.0) margin_used[1] = 0.0;

        const int side = is_buy[i] ? 1 : 0;
        const double sizing_base = std::min(equity[0], equity[1]);
        double risk_dollars, margin;
        position_size_and_margin(instrument_code[i], sizing_base, entry_price[i], stop_risk[i],
                                  risk_pct[i], max_leverage, cap, gbp_usd_approx, risk_dollars, margin);
        const double avail = std::min(equity[side], cap) - margin_used[side];

        if (margin <= avail && open_count[side] < 1) {
            out.accepted[i] = 1;
            heap.push({exit_ns[i], risk_dollars, margin, net_r[i], side});
            margin_used[side] += margin;
            open_count[side] += 1;
        }
    }
    while (!heap.empty()) {
        HeapEntry e = heap.top();
        heap.pop();
        settle(e);
    }

    out.equity_buy = equity[1];
    out.equity_sell = equity[0];
    out.max_dd_pct = dd_worst;
    return out;
}

namespace {

void mc_worker(const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
                const double* entry_price, const double* stop_risk, const double* net_r,
                const uint8_t* is_buy, int64_t n_trades, int64_t sim_begin, int64_t sim_end,
                int64_t block, int64_t max_trades, uint32_t seed, uint32_t thread_idx,
                double risk_pct, double start_per_side, double max_leverage, double cap,
                double target1, double target2, double year_ns, double gbp_usd_approx,
                double ruin_floor, double* out_t1, double* out_t10, double* out_dd, uint8_t* out_ruined,
                int64_t* out_processed, int64_t* out_accepted, int64_t boundary_gap_ns) {
    std::seed_seq seq{seed, thread_idx};
    std::mt19937_64 rng(seq);
    const int64_t effective_block = std::max<int64_t>(1, std::min(block, n_trades));
    std::uniform_int_distribution<int64_t> dist(0, n_trades - effective_block);

    for (int64_t s = sim_begin; s < sim_end; ++s) {
        double equity[2] = {start_per_side, start_per_side};
        double margin_used[2] = {0.0, 0.0};
        int open_count[2] = {0, 0};
        MinHeap heap;
        int64_t clock = 0;
        double t1 = -1.0, t10 = -1.0;
        double peak_combined = start_per_side * 2.0;
        double dd_worst = 0.0;
        bool ruined = false;
        int64_t processed = 0;
        int64_t accepted_n = 0;

        while (processed < max_trades && t10 < 0.0 && !ruined) {
            const int64_t start_idx = dist(rng);
            const int64_t end_idx = start_idx + effective_block;
            const int64_t offset = clock - entry_ns[start_idx];
            int64_t last_entry = clock;

            for (int64_t idx = start_idx; idx < end_idx; ++idx) {
                const int64_t et = entry_ns[idx] + offset;
                const int64_t xt = exit_ns[idx] + offset;
                last_entry = et;

                while (!heap.empty() && heap.top().exit_ns <= et) {
                    HeapEntry e = heap.top();
                    heap.pop();
                    equity[e.side] += e.risk_dollars * e.r;
                    margin_used[e.side] -= e.margin;
                    open_count[e.side] -= 1;
                    const double combined = equity[0] + equity[1];
                    if (combined > peak_combined) peak_combined = combined;
                    const double dd = 100.0 * (combined - peak_combined) / peak_combined;
                    if (dd < dd_worst) dd_worst = dd;
                    if (t1 < 0.0 && combined >= target1)
                        t1 = static_cast<double>(e.exit_ns) / year_ns;
                    if (t10 < 0.0 && combined >= target2)
                        t10 = static_cast<double>(e.exit_ns) / year_ns;
                    if (combined <= ruin_floor) ruined = true;
                }
                if (margin_used[0] < 0.0) margin_used[0] = 0.0;
                if (margin_used[1] < 0.0) margin_used[1] = 0.0;
                if (ruined) break;

                const int side = is_buy[idx] ? 1 : 0;
                const double sizing_base = std::min(equity[0], equity[1]);
                double risk_dollars, margin;
                position_size_and_margin(instrument_code[idx], sizing_base, entry_price[idx],
                                          stop_risk[idx], risk_pct, max_leverage, cap,
                                          gbp_usd_approx, risk_dollars, margin);
                const double avail = std::min(equity[side], cap) - margin_used[side];

                if (margin <= avail && open_count[side] < 1) {
                    heap.push({xt, risk_dollars, margin, net_r[idx], side});
                    margin_used[side] += margin;
                    open_count[side] += 1;
                    ++accepted_n;
                }
                ++processed;
                if (t10 >= 0.0) break;
                if (processed >= max_trades) break;
            }
            if (ruined || t10 >= 0.0) break;
            clock = last_entry + std::max<int64_t>(boundary_gap_ns, 1);
        }

        // The last sampled trades can close after max_trades is reached. Their P&L and target
        // crossings still belong to the path and must not be silently discarded.
        while (!heap.empty() && t10 < 0.0 && !ruined) {
            HeapEntry e = heap.top();
            heap.pop();
            equity[e.side] += e.risk_dollars * e.r;
            margin_used[e.side] -= e.margin;
            open_count[e.side] -= 1;
            const double combined = equity[0] + equity[1];
            if (combined > peak_combined) peak_combined = combined;
            const double dd = 100.0 * (combined - peak_combined) / peak_combined;
            if (dd < dd_worst) dd_worst = dd;
            if (t1 < 0.0 && combined >= target1)
                t1 = static_cast<double>(e.exit_ns) / year_ns;
            if (t10 < 0.0 && combined >= target2)
                t10 = static_cast<double>(e.exit_ns) / year_ns;
            if (combined <= ruin_floor) ruined = true;
        }

        out_t1[s] = t1;
        out_t10[s] = t10;
        out_dd[s] = dd_worst;
        out_ruined[s] = ruined ? 1 : 0;
        out_processed[s] = processed;
        out_accepted[s] = accepted_n;
    }
}

// Same as mc_worker, but risk_pct is a per-trade array -- a resampled trade is sized by its
// OWN risk_pct[idx], so confidence-weighted sizing survives block-bootstrap resampling intact.
void mc_worker_var_risk(const int64_t* entry_ns, const int64_t* exit_ns,
                         const int32_t* instrument_code, const double* entry_price,
                         const double* stop_risk, const double* net_r, const uint8_t* is_buy,
                         const double* risk_pct, int64_t n_trades, int64_t sim_begin,
                         int64_t sim_end, int64_t block, int64_t max_trades, uint32_t seed,
                         uint32_t thread_idx, double start_per_side, double max_leverage,
                         double cap, double target1, double target2, double year_ns,
                         double gbp_usd_approx, double ruin_floor, double* out_t1,
                         double* out_t10, double* out_dd, uint8_t* out_ruined,
                         int64_t* out_processed, int64_t* out_accepted,
                         int64_t boundary_gap_ns) {
    std::seed_seq seq{seed, thread_idx};
    std::mt19937_64 rng(seq);
    const int64_t effective_block = std::max<int64_t>(1, std::min(block, n_trades));
    std::uniform_int_distribution<int64_t> dist(0, n_trades - effective_block);

    for (int64_t s = sim_begin; s < sim_end; ++s) {
        double equity[2] = {start_per_side, start_per_side};
        double margin_used[2] = {0.0, 0.0};
        int open_count[2] = {0, 0};
        MinHeap heap;
        int64_t clock = 0;
        double t1 = -1.0, t10 = -1.0;
        double peak_combined = start_per_side * 2.0;
        double dd_worst = 0.0;
        bool ruined = false;
        int64_t processed = 0;
        int64_t accepted_n = 0;

        while (processed < max_trades && t10 < 0.0 && !ruined) {
            const int64_t start_idx = dist(rng);
            const int64_t end_idx = start_idx + effective_block;
            const int64_t offset = clock - entry_ns[start_idx];
            int64_t last_entry = clock;

            for (int64_t idx = start_idx; idx < end_idx; ++idx) {
                const int64_t et = entry_ns[idx] + offset;
                const int64_t xt = exit_ns[idx] + offset;
                last_entry = et;

                while (!heap.empty() && heap.top().exit_ns <= et) {
                    HeapEntry e = heap.top();
                    heap.pop();
                    equity[e.side] += e.risk_dollars * e.r;
                    margin_used[e.side] -= e.margin;
                    open_count[e.side] -= 1;
                    const double combined = equity[0] + equity[1];
                    if (combined > peak_combined) peak_combined = combined;
                    const double dd = 100.0 * (combined - peak_combined) / peak_combined;
                    if (dd < dd_worst) dd_worst = dd;
                    if (t1 < 0.0 && combined >= target1)
                        t1 = static_cast<double>(e.exit_ns) / year_ns;
                    if (t10 < 0.0 && combined >= target2)
                        t10 = static_cast<double>(e.exit_ns) / year_ns;
                    if (combined <= ruin_floor) ruined = true;
                }
                if (margin_used[0] < 0.0) margin_used[0] = 0.0;
                if (margin_used[1] < 0.0) margin_used[1] = 0.0;
                if (ruined) break;

                const int side = is_buy[idx] ? 1 : 0;
                const double sizing_base = std::min(equity[0], equity[1]);
                double risk_dollars, margin;
                position_size_and_margin(instrument_code[idx], sizing_base, entry_price[idx],
                                          stop_risk[idx], risk_pct[idx], max_leverage, cap,
                                          gbp_usd_approx, risk_dollars, margin);
                const double avail = std::min(equity[side], cap) - margin_used[side];

                if (margin <= avail && open_count[side] < 1) {
                    heap.push({xt, risk_dollars, margin, net_r[idx], side});
                    margin_used[side] += margin;
                    open_count[side] += 1;
                    ++accepted_n;
                }
                ++processed;
                if (t10 >= 0.0) break;
                if (processed >= max_trades) break;
            }
            if (ruined || t10 >= 0.0) break;
            clock = last_entry + std::max<int64_t>(boundary_gap_ns, 1);
        }

        while (!heap.empty() && t10 < 0.0 && !ruined) {
            HeapEntry e = heap.top();
            heap.pop();
            equity[e.side] += e.risk_dollars * e.r;
            margin_used[e.side] -= e.margin;
            open_count[e.side] -= 1;
            const double combined = equity[0] + equity[1];
            if (combined > peak_combined) peak_combined = combined;
            const double dd = 100.0 * (combined - peak_combined) / peak_combined;
            if (dd < dd_worst) dd_worst = dd;
            if (t1 < 0.0 && combined >= target1)
                t1 = static_cast<double>(e.exit_ns) / year_ns;
            if (t10 < 0.0 && combined >= target2)
                t10 = static_cast<double>(e.exit_ns) / year_ns;
            if (combined <= ruin_floor) ruined = true;
        }

        out_t1[s] = t1;
        out_t10[s] = t10;
        out_dd[s] = dd_worst;
        out_ruined[s] = ruined ? 1 : 0;
        out_processed[s] = processed;
        out_accepted[s] = accepted_n;
    }
}

}  // namespace

TwoAccountMCResult run_two_account_mc(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, int64_t n_trades,
    int64_t n_sims, int64_t block, int64_t max_trades, uint32_t seed,
    double risk_pct, double start_per_side, double max_leverage, double cap,
    double target1, double target2, double year_ns, double gbp_usd_approx, double ruin_floor,
    int n_threads, int64_t boundary_gap_ns) {
    TwoAccountMCResult out;
    out.t1_years.assign(n_sims, -1.0);
    out.t10_years.assign(n_sims, -1.0);
    out.max_dd_pct.assign(n_sims, 0.0);
    out.ruined.assign(n_sims, 0);
    out.processed.assign(n_sims, 0);
    out.accepted_n.assign(n_sims, 0);

    if (n_trades <= 0 || n_sims <= 0) return out;

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
        threads.emplace_back(mc_worker, entry_ns, exit_ns, instrument_code, entry_price,
                              stop_risk, net_r, is_buy, n_trades, begin, end, block, max_trades,
                              seed, static_cast<uint32_t>(t), risk_pct, start_per_side,
                              max_leverage, cap, target1, target2, year_ns, gbp_usd_approx,
                              ruin_floor, out.t1_years.data(), out.t10_years.data(), out.max_dd_pct.data(),
                              out.ruined.data(), out.processed.data(), out.accepted_n.data(),
                              boundary_gap_ns);
    }
    for (auto& th : threads) th.join();

    return out;
}

TwoAccountMCResult run_two_account_mc_var_risk(
    const int64_t* entry_ns, const int64_t* exit_ns, const int32_t* instrument_code,
    const double* entry_price, const double* stop_risk, const double* net_r,
    const uint8_t* is_buy, const double* risk_pct, int64_t n_trades,
    int64_t n_sims, int64_t block, int64_t max_trades, uint32_t seed,
    double start_per_side, double max_leverage, double cap,
    double target1, double target2, double year_ns, double gbp_usd_approx, double ruin_floor,
    int n_threads, int64_t boundary_gap_ns) {
    TwoAccountMCResult out;
    out.t1_years.assign(n_sims, -1.0);
    out.t10_years.assign(n_sims, -1.0);
    out.max_dd_pct.assign(n_sims, 0.0);
    out.ruined.assign(n_sims, 0);
    out.processed.assign(n_sims, 0);
    out.accepted_n.assign(n_sims, 0);

    if (n_trades <= 0 || n_sims <= 0) return out;

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
        threads.emplace_back(mc_worker_var_risk, entry_ns, exit_ns, instrument_code, entry_price,
                              stop_risk, net_r, is_buy, risk_pct, n_trades, begin, end, block,
                              max_trades, seed, static_cast<uint32_t>(t), start_per_side,
                              max_leverage, cap, target1, target2, year_ns, gbp_usd_approx,
                              ruin_floor, out.t1_years.data(), out.t10_years.data(), out.max_dd_pct.data(),
                              out.ruined.data(), out.processed.data(), out.accepted_n.data(),
                              boundary_gap_ns);
    }
    for (auto& th : threads) th.join();

    return out;
}
