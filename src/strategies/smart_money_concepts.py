"""Causal Smart Money Concepts strategy for BAT's dataframe backtester.

The strategy derives its higher-timeframe structure from a single 1-minute (or
finer) OHLCV stream.  It intentionally confirms fractals only after the right
hand candles exist, avoiding look-ahead bias.
"""

from typing import Dict
import numpy as np
import pandas as pd


class SmartMoneyConceptsStrategy:
    """Multi-timeframe SMC entry state machine.

    A trade is armed after an opposing liquidity sweep overlaps a higher-timeframe
    FVG (the point of interest) in the direction of confirmed structure.  It is
    entered only after a lower-timeframe close breaks a confirmed swing against
    the pullback.  The two output signals work in both BAT long-only and
    long/short modes: bullish setups emit ``Buy Signal`` and bearish setups emit
    ``Sell Signal``.
    """

    def __init__(self, structure_timeframe: str = "15min", swing_window: int = 2,
                 liquidity_lookback: int = 60, poi_expiry_bars: int = 60,
                 timezone: str = "America/New_York", **kwargs):
        if swing_window < 1 or liquidity_lookback < 2 or poi_expiry_bars < 1:
            raise ValueError("swing_window, liquidity_lookback, and poi_expiry_bars must be positive")
        self.name = "Smart Money Concepts (SMC)"
        self.params = {
            "structure_timeframe": structure_timeframe,
            "swing_window": swing_window,
            "liquidity_lookback": liquidity_lookback,
            "poi_expiry_bars": poi_expiry_bars,
            "timezone": timezone,
            **kwargs,
        }
        self.structure_timeframe = structure_timeframe
        self.swing_window = swing_window
        self.liquidity_lookback = liquidity_lookback
        self.poi_expiry_bars = poi_expiry_bars
        self.timezone = timezone

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(df):
            raise ValueError("Invalid data format")
        data = df.copy().sort_values("timestamp").reset_index(drop=True)
        timestamps = pd.to_datetime(data["timestamp"], utc=True)
        data["timestamp"] = timestamps

        macro = self._macro_structure(data)
        data = pd.merge_asof(
            data.sort_values("timestamp"),
            macro.sort_values("timestamp"),
            on="timestamp", direction="backward",
        )
        data = self._lower_timeframe_structure(data)
        data["Prior Liquidity High"] = data["High"].rolling(self.liquidity_lookback).max().shift(1)
        data["Prior Liquidity Low"] = data["Low"].rolling(self.liquidity_lookback).min().shift(1)
        data["Liquidity Sweep High"] = data["High"] >= data["Prior Liquidity High"]
        data["Liquidity Sweep Low"] = data["Low"] <= data["Prior Liquidity Low"]
        data["POI Overlap"] = (
            data["POI Low"].notna() & data["POI High"].notna() &
            (data["High"] >= data["POI Low"]) & (data["Low"] <= data["POI High"])
        )

        data["Buy Signal"] = False
        data["Sell Signal"] = False
        bullish_armed = bearish_armed = 0
        for i, row in data.iterrows():
            bias = row["HTF Bias"]
            if bias == "bullish" and row["Liquidity Sweep Low"] and row["POI Overlap"]:
                bullish_armed = self.poi_expiry_bars
            elif bias != "bullish":
                bullish_armed = 0
            if bias == "bearish" and row["Liquidity Sweep High"] and row["POI Overlap"]:
                bearish_armed = self.poi_expiry_bars
            elif bias != "bearish":
                bearish_armed = 0

            if bullish_armed and row["LTF CHOCH Up"]:
                data.at[i, "Buy Signal"] = True
                bullish_armed = 0
            if bearish_armed and row["LTF CHOCH Down"]:
                data.at[i, "Sell Signal"] = True
                bearish_armed = 0
            bullish_armed = max(0, bullish_armed - 1)
            bearish_armed = max(0, bearish_armed - 1)
        return data

    def _macro_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        indexed = data.set_index("timestamp")
        macro = indexed.resample(self.structure_timeframe, label="right", closed="right").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna().reset_index()
        swing_high, swing_low = self._confirmed_swings(macro["High"], macro["Low"])
        macro["Confirmed Swing High"] = swing_high
        macro["Confirmed Swing Low"] = swing_low
        macro["Bearish FVG"] = macro["High"] < macro["Low"].shift(2)
        macro["Bullish FVG"] = macro["Low"] > macro["High"].shift(2)
        macro["POI Low"] = np.where(macro["Bearish FVG"], macro["High"], np.where(macro["Bullish FVG"], macro["High"].shift(2), np.nan))
        macro["POI High"] = np.where(macro["Bearish FVG"], macro["Low"].shift(2), np.where(macro["Bullish FVG"], macro["Low"], np.nan))
        macro[["POI Low", "POI High"]] = macro[["POI Low", "POI High"]].ffill()

        last_high = last_low = None
        bias = "neutral"
        biases, breaks = [], []
        for _, row in macro.iterrows():
            # The values are already delayed by swing_window, so they are known now.
            if pd.notna(row["Confirmed Swing High"]):
                last_high = row["Confirmed Swing High"]
            if pd.notna(row["Confirmed Swing Low"]):
                last_low = row["Confirmed Swing Low"]
            structural_break = ""
            if last_high is not None and row["Close"] > last_high:
                structural_break = "BOS Up" if bias == "bullish" else "CHOCH Up"
                bias = "bullish"
            elif last_low is not None and row["Close"] < last_low:
                structural_break = "BOS Down" if bias == "bearish" else "CHOCH Down"
                bias = "bearish"
            biases.append(bias)
            breaks.append(structural_break)
        macro["HTF Bias"] = biases
        macro["HTF Structural Break"] = breaks
        return macro[["timestamp", "HTF Bias", "HTF Structural Break", "Confirmed Swing High", "Confirmed Swing Low", "Bearish FVG", "Bullish FVG", "POI Low", "POI High"]]

    def _lower_timeframe_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        swing_high, swing_low = self._confirmed_swings(data["High"], data["Low"])
        last_high = last_low = None
        choch_up, choch_down = [], []
        for close, high, low in zip(data["Close"], swing_high, swing_low):
            if pd.notna(high):
                last_high = high
            if pd.notna(low):
                last_low = low
            # Close-only validation implements the specification's wick rule.
            choch_up.append(last_high is not None and close > last_high)
            choch_down.append(last_low is not None and close < last_low)
        data["LTF Confirmed Swing High"] = swing_high
        data["LTF Confirmed Swing Low"] = swing_low
        data["LTF CHOCH Up"] = choch_up
        data["LTF CHOCH Down"] = choch_down
        return data

    def _confirmed_swings(self, highs: pd.Series, lows: pd.Series) -> tuple[pd.Series, pd.Series]:
        n = self.swing_window
        high_pivot = pd.Series(True, index=highs.index)
        low_pivot = pd.Series(True, index=lows.index)
        for offset in range(1, n + 1):
            high_pivot &= (highs > highs.shift(offset)) & (highs > highs.shift(-offset))
            low_pivot &= (lows < lows.shift(offset)) & (lows < lows.shift(-offset))
        # A pivot at t becomes available at t + n; this is essential for valid backtests.
        return highs.where(high_pivot).shift(n), lows.where(low_pivot).shift(n)

    def get_signal_names(self) -> Dict[str, str]:
        return {"buy": "Buy Signal", "sell": "Sell Signal"}

    def get_indicators(self) -> list:
        return ["HTF Bias", "HTF Structural Break", "POI Low", "POI High", "Liquidity Sweep High", "Liquidity Sweep Low", "LTF CHOCH Up", "LTF CHOCH Down"]

    def get_required_lookback(self) -> int:
        return max(self.liquidity_lookback, self.swing_window * 2 + 1)

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        return {"Open", "High", "Low", "Close", "Volume", "timestamp"}.issubset(df.columns)
