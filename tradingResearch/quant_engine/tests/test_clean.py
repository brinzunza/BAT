import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_engine.data.clean import find_bad_days, find_bad_tick_buckets, load_clean


def _make_df():
    idx = pd.date_range("2026-01-01", periods=3 * 1440, freq="min")
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-5, len(idx)))
    df = pd.DataFrame({
        "open": close, "high": close + 1e-5, "low": close - 1e-5, "close": close,
    }, index=idx)
    # Day 2 (index day offset 1) is frozen: high == low all day.
    day2_mask = (df.index.date == idx[1440].date())
    df.loc[day2_mask, ["open", "high", "low", "close"]] = 1.1234
    return df


def test_frozen_day_is_excluded():
    df = _make_df()
    bad = find_bad_days(df, asset_class="fx", bad_day_threshold_frac=200 / 1440)
    assert len(bad) == 1
    cleaned = load_clean(df, asset_class="fx", bad_day_threshold_frac=200 / 1440)
    assert next(iter(bad)) not in set(cleaned.index.date)
    # the other two (real, non-frozen) days survive
    assert len(set(cleaned.index.date)) == 2


def test_thin_day_is_excluded():
    idx = pd.date_range("2026-01-01", periods=100, freq="min")  # well under 200 bars
    df = pd.DataFrame({"open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1}, index=idx)
    cleaned = load_clean(df, asset_class="fx", bad_day_threshold_frac=200 / 1440)
    assert len(cleaned) == 0


def test_bad_tick_sanity_filter_catches_implausible_spike():
    idx = pd.date_range("2026-01-01", periods=5, freq="h")
    bucket = pd.DataFrame({"open": [100, 100, 100, 100, 100], "high": [101, 350, 101, 101, 101],
                            "low": [99, 99, 99, 99, 99]}, index=idx)
    flagged = find_bad_tick_buckets(bucket, sanity_band=3.0)
    assert flagged.tolist() == [False, True, False, False, False]


def test_load_clean_never_mutates_source_file(tmp_path):
    df = _make_df()
    path = tmp_path / "sample.csv"
    df.reset_index(names="date").to_csv(path, index=False)
    original_bytes = path.read_bytes()
    load_clean(path, asset_class="fx", bad_day_threshold_frac=200 / 1440)
    assert path.read_bytes() == original_bytes
