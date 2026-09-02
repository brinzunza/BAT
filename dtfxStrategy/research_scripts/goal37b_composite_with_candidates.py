"""Adds each candidate strategy to the existing 14-strategy pool (goal12_strategy_scores.csv's
raw values, reused as-is) and recomputes the composite score with the EXACT same
minmax_norm + DEFAULT_WEIGHTS methodology as goal12_strategy_score.py, so the comparison
against Speed-King's own (renormalized) score is apples-to-apples.
"""
import os
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal12_strategy_score import minmax_norm, DEFAULT_WEIGHTS

existing = pd.read_csv(os.path.join(DATA_DIR, "goal12_strategy_scores.csv"))
raw = {}
for _, r in existing.iterrows():
    raw[r["strategy"]] = dict(trades=r["trades"], tpy=r["tpy"], wr=r["wr"], rr=r["rr"],
                               median_yrs_1m=r["median_yrs_1m"], median_dd=r["median_dd"],
                               pct_ruined=r["pct_ruined"], margin_pct=r["margin_pct"])

with open("/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/goal37_candidates.json") as f:
    candidates = json.load(f)


def composite(pool_raw):
    speed_n = minmax_norm({k: v["median_yrs_1m"] for k, v in pool_raw.items()}, higher_is_better=False)
    dd_n = minmax_norm({k: v["median_dd"] for k, v in pool_raw.items()}, higher_is_better=True)
    wr_n = minmax_norm({k: v["wr"] for k, v in pool_raw.items()}, higher_is_better=True)
    freq_n = minmax_norm({k: v["tpy"] for k, v in pool_raw.items()}, higher_is_better=True)
    margin_n = minmax_norm({k: v["margin_pct"] for k, v in pool_raw.items()}, higher_is_better=True)
    rrr_n = minmax_norm({k: v["rr"] for k, v in pool_raw.items()}, higher_is_better=True)
    w = DEFAULT_WEIGHTS
    scores = {}
    for k in pool_raw:
        scores[k] = (w["speed"] * speed_n[k] + w["drawdown"] * dd_n[k] + w["wr"] * wr_n[k] +
                     w["freq"] * freq_n[k] + w["margin"] * margin_n[k] + w["rrr"] * rrr_n[k])
    return scores, dict(speed=speed_n, dd=dd_n, wr=wr_n, freq=freq_n, margin=margin_n, rrr=rrr_n)


print(f"{'Candidate':22s} {'Composite':>10s} {'MarginScore':>12s}  {'vs SK composite':>16s} {'vs SK margin':>13s}")
for label, cand_raw in candidates.items():
    if label.startswith("Speed-King"):
        continue
    pool = dict(raw)  # 14 existing, including the real Speed-King
    pool[label] = cand_raw
    scores, norms = composite(pool)
    sk_composite = scores["Speed-King"]
    sk_margin = norms["margin"]["Speed-King"]
    cand_composite = scores[label]
    cand_margin = norms["margin"][label]
    beat_composite = cand_composite > sk_composite
    beat_margin = cand_margin > sk_margin
    print(f"{label:22s} {cand_composite:10.1f} {cand_margin:12.1f}  "
          f"SK={sk_composite:5.1f} {'BEATS' if beat_composite else 'loses to':>8s}  "
          f"SK={sk_margin:5.1f} {'BEATS' if beat_margin else 'loses to':>8s}")

# Full detail on the single best-looking candidate for a final ranked table
print("\n" + "=" * 100)
best_label = max((l for l in candidates if not l.startswith("Speed-King")),
                  key=lambda l: composite({**raw, l: candidates[l]})[0][l])
pool = dict(raw)
pool[best_label] = candidates[best_label]
scores, norms = composite(pool)
rows = []
for k in pool:
    rows.append(dict(strategy=k, final_score=round(scores[k], 1),
                      speed_score=round(norms["speed"][k], 1), wr_score=round(norms["wr"][k], 1),
                      freq_score=round(norms["freq"][k], 1), margin_score=round(norms["margin"][k], 1),
                      dd_score=round(norms["dd"][k], 1), rrr_score=round(norms["rrr"][k], 1)))
df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
df.insert(0, "rank", range(1, len(df) + 1))
print(f"FULL RANKED TABLE with best candidate ({best_label}) added:")
print(df.to_string(index=False))
df.to_csv(os.path.join(DATA_DIR, "goal37_final_ranking_with_candidate.csv"), index=False)
