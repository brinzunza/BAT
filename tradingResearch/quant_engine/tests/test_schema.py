import pandas as pd
import pytest

from quant_engine.schema import SchemaError, validate_trades

REQUIRED = dict(
    instrument="EURUSD", is_buy=True,
    entry_time=pd.Timestamp("2020-01-01"), exit_time=pd.Timestamp("2020-01-02"),
    entry_price=1.10, stop_risk=0.002, net_r=1.5, risk_pct=0.01,
)


def _df(**overrides):
    row = {**REQUIRED, **overrides}
    return pd.DataFrame([row])


def test_plain_single_unit_trade_passes():
    # The common case: no pyramid columns at all. Must not be rejected -- this is what
    # normalize_trades' default `units=1` injection previously broke (see schema.py comment).
    out = validate_trades(_df())
    assert out["units"].iloc[0] == 1


def test_explicit_units_without_pyramid_detail_passes():
    out = validate_trades(_df(units=1))
    assert out["units"].iloc[0] == 1


def test_full_pyramid_detail_passes():
    row = {
        **REQUIRED, "units": 2,
        "unit_rs": [0.5, 1.0],
        "unit_entry_times": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01T01:00")],
        "unit_entry_prices": [1.10, 1.101],
    }
    out = validate_trades(pd.DataFrame([row]))
    assert out["units"].iloc[0] == 2


def test_partial_pyramid_detail_rejected():
    with pytest.raises(SchemaError, match="all-or-nothing"):
        validate_trades(_df(unit_rs=[[0.5]], unit_entry_times=[[pd.Timestamp("2020-01-01")]]))


def test_missing_required_column_rejected():
    df = _df()
    df = df.drop(columns=["stop_risk"])
    with pytest.raises(SchemaError, match="stop_risk"):
        validate_trades(df)


def test_negative_stop_risk_rejected():
    with pytest.raises(SchemaError, match="stop_risk"):
        validate_trades(_df(stop_risk=-1.0))


def test_exit_before_entry_rejected():
    with pytest.raises(SchemaError, match="exit_time"):
        validate_trades(_df(exit_time=pd.Timestamp("2019-12-31")))
