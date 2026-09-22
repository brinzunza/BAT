import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_engine.data.catalog import Catalog

CORE_INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD", "BTCUSD", "ETHUSD"]


def test_resolves_core_instruments_with_no_file_moves():
    cat = Catalog.build()
    for inst in CORE_INSTRUMENTS:
        entry = cat.lookup(inst)
        assert entry.path.exists(), f"{inst} -> {entry.path} does not exist on disk"


def test_eurusd_prefers_the_vendor_reverified_file():
    # research_scripts/dtfx_lse_vendor_check.py specifically re-verified lse_eurusd_1min_full.csv
    # after a prior OANDA-substitution sign-flip incident -- the catalog must prefer it over the
    # unverified "_master" consolidation, not just whichever sorts first/last.
    cat = Catalog.build()
    entry = cat.lookup("EURUSD")
    assert entry.path.name == "lse_eurusd_1min_full.csv"
    assert entry.vendor == "lse_vault"


def test_ethusd_resolves_to_oanda_not_lse():
    cat = Catalog.build()
    entry = cat.lookup("ETHUSD")
    assert entry.vendor == "oanda"
