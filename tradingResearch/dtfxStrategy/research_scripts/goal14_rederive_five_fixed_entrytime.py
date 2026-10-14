"""Re-derives the five session-hour-filtered strategies in STRATEGY_RANKINGS.md
against the corrected entry_time (see the live-readiness audit + the fix in
prod_filter_grid_search.py / cana_faithful_backtest.py): SOMEGA (live), SOMEGA
Robust, DTFX Robust, CANA+, and goal70_final_winner. All five filter their trade
population by hour-of-day or quarter-consistency-voted session blocks, both of
which were computed on the mislabeled (exit-time) timestamp before the fix --
~48% of trades have their hour bucket reassigned once the correct entry hour is
used, so these numbers are expected to move, unlike Hybrid-Ultra (no session
filter -> confirmed unaffected).

Reuses each strategy's own canonical derivation script unchanged (no logic
duplicated here) -- just calls them in one process so the ~3min data load only
happens once instead of five times.
"""
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv", "XAUUSD": "lse_xauusd_full_1min.csv"})

import goal5_rr_sweep as g5
import goal8_unified_holdout as g8
import goal6_cana_plus as g6
import goal4_no_gold as g4
import somega_faithful_backtest as somega


def warm_up():
    for inst in g5.TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, 2)


def main():
    warm_up()

    print("\n" + "#" * 90)
    print("  1. SOMEGA (live, fixed) -- somega_faithful_backtest.py")
    print("#" * 90)
    somega_trades = somega.main()
    r_somega = g5.score_row(somega_trades, "SOMEGA (live, fixed) [POST-FIX]")
    g5.print_row(r_somega)

    print("\n" + "#" * 90)
    print("  2-5. CANA+, DTFX Robust, SOMEGA Robust, CANA -- goal8_unified_holdout.py")
    print("#" * 90)
    g8.main()

    print("\n" + "#" * 90)
    print("  CANA+ full-history vs TRUE holdout pair -- goal6_cana_plus.py build_cana_plus_v2()")
    print("#" * 90)
    g6.build_cana_plus_v2()

    print("\n" + "#" * 90)
    print("  goal70_final_winner (no gold) -- goal4_no_gold.py")
    print("#" * 90)
    goal70_trades = g4.build_goal70_no_gold()
    r_goal70 = g5.score_row(goal70_trades, "goal70_final_winner (no gold) [POST-FIX]")
    g5.print_row(r_goal70)

    print("\n" + "#" * 90)
    print("  RE-DERIVATION COMPLETE")
    print("#" * 90)


if __name__ == "__main__":
    main()
