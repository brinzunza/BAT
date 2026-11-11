"""Providers, brokers and the compatibility shim between them and the engines.

Credentials come from the environment (``.env`` is loaded at startup) or from
the in-app form. They are held in memory only and never written to disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.tui.services.demo import DemoFeed


@dataclass(frozen=True)
class ConnSpec:
    key: str
    label: str
    kind: str                        # "data" | "broker" | "both"
    env: Tuple[str, ...] = ()
    fields: Tuple[Tuple[str, str, bool], ...] = ()   # (name, label, secret?)
    note: str = ""


SPECS: Dict[str, ConnSpec] = {s.key: s for s in [
    ConnSpec("polygon", "Polygon.io", "data", ("POLYGON_API_KEY",),
             (("api_key", "API key", True),), "Stocks, crypto (X:BTCUSD) and forex (C:EURUSD) history."),
    ConnSpec("alpaca", "Alpaca", "both", ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
             (("api_key", "API key", True), ("secret_key", "Secret key", True)),
             "Stocks & crypto data + paper/live execution."),
    ConnSpec("oanda", "OANDA", "data", ("OANDA_ACCESS_TOKEN", "OANDA_ACCOUNT_ID"),
             (("access_token", "Access token", True), ("account_id", "Account ID", False)),
             "Forex data (practice environment by default)."),
    ConnSpec("synth", "Synth", "data", ("SYNTH_API_KEY",),
             (("base_url", "Base URL", False), ("api_key", "API key", True)), "Synthetic market data API."),
    ConnSpec("ib", "Interactive Brokers", "broker", (),
             (("host", "Host", False), ("port", "Port (7497 paper / 7496 live)", False),
              ("client_id", "Client ID", False)), "Forex execution via TWS / IB Gateway."),
]}


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


@dataclass
class ConnStatus:
    key: str
    label: str
    configured: bool          # credentials present in env / form
    connected: bool           # live object cached and tested
    detail: str = ""


class ConnectionManager:
    def __init__(self) -> None:
        self.providers: Dict[str, Any] = {}
        self.brokers: Dict[str, Any] = {}
        self.details: Dict[str, str] = {}
        self.meta: Dict[str, Dict[str, Any]] = {}      # e.g. {'alpaca': {'paper': True}}
        self.demo = DemoFeed()

    # ----- status ---------------------------------------------------------
    @staticmethod
    def env_configured(key: str) -> bool:
        spec = SPECS[key]
        return bool(spec.env) and all(os.getenv(e) for e in spec.env)

    def status(self) -> List[ConnStatus]:
        out = []
        for key, spec in SPECS.items():
            connected = key in self.providers or key in self.brokers
            out.append(ConnStatus(key, spec.label, self.env_configured(key) or connected, connected,
                                  self.details.get(key, "from environment" if self.env_configured(key) else "")))
        return out

    def env_defaults(self, key: str) -> Dict[str, str]:
        envmap = {"polygon": {"api_key": "POLYGON_API_KEY"},
                  "alpaca": {"api_key": "ALPACA_API_KEY", "secret_key": "ALPACA_SECRET_KEY"},
                  "oanda": {"access_token": "OANDA_ACCESS_TOKEN", "account_id": "OANDA_ACCOUNT_ID"},
                  "synth": {"api_key": "SYNTH_API_KEY", "base_url": "SYNTH_BASE_URL"},
                  "ib": {}}
        out = {f: os.getenv(e, "") for f, e in envmap.get(key, {}).items()}
        if key == "ib":
            out = {"host": "127.0.0.1", "port": "7497", "client_id": "1"}
        if key == "synth" and not out.get("base_url"):
            out["base_url"] = "http://35.209.219.174:8000"
        return out

    # ----- connect --------------------------------------------------------
    def connect(self, key: str, **creds: str) -> Tuple[bool, str]:
        """Blocking connect + test. Run from a worker thread."""
        creds = {**self.env_defaults(key), **{k: v for k, v in creds.items() if v not in (None, "")}}
        try:
            ok, msg = getattr(self, f"_connect_{key}")(creds)
        except Exception as e:  # noqa: BLE001 - surfaced to the user verbatim
            ok, msg = False, f"{type(e).__name__}: {e}"
        self.details[key] = msg
        return ok, msg

    def _connect_polygon(self, c) -> Tuple[bool, str]:
        from src.data.providers.polygon_provider import PolygonDataProvider
        if not c.get("api_key"):
            return False, "API key required"
        p = PolygonDataProvider(c["api_key"])
        ok, msg = p.test_connection()
        if ok:
            self.providers["polygon"] = p
        return ok, msg

    def _connect_synth(self, c) -> Tuple[bool, str]:
        from src.data.providers.synth_provider import SynthDataProvider
        if not c.get("api_key"):
            return False, "API key required"
        p = SynthDataProvider(c["base_url"], c["api_key"])
        ok, msg = p.test_connection()
        if ok:
            self.providers["synth"] = p
        return ok, msg

    def _connect_alpaca(self, c) -> Tuple[bool, str]:
        from src.data.providers.alpaca_provider import AlpacaBroker, AlpacaDataProvider
        if not c.get("api_key") or not c.get("secret_key"):
            return False, "API key and secret required"
        paper = str(c.get("paper", "true")).lower() != "false"
        data = AlpacaDataProvider(c["api_key"], c["secret_key"])
        broker = AlpacaBroker(c["api_key"], c["secret_key"], paper)
        acct = broker.get_account()
        if not acct:
            return False, "Could not read Alpaca account"
        self.providers["alpaca"], self.brokers["alpaca"] = data, broker
        self.meta["alpaca"] = {"paper": paper}
        bp = acct.get("buying_power")
        return True, f"{'paper' if paper else 'LIVE'} account · buying power ${float(bp):,.2f}" if bp else "connected"

    def _connect_oanda(self, c) -> Tuple[bool, str]:
        from src.data.providers.oanda_provider import OandaProvider
        if not c.get("access_token") or not c.get("account_id"):
            return False, "Access token and account ID required"
        p = OandaProvider(access_token=c["access_token"], account_id=c["account_id"],
                          environment=c.get("environment", "practice"))
        candle = p.get_latest_candle("EURUSD")
        if not candle:
            return False, "No data returned from OANDA"
        self.providers["oanda"] = p
        return True, f"EUR/USD {candle['close']:.5f}"

    def _connect_ib(self, c) -> Tuple[bool, str]:
        from src.execution.ib_broker import IBBroker
        b = IBBroker()
        if not b.connect_to_tws(c["host"], int(c["port"]), int(c["client_id"])):
            return False, "TWS/Gateway not reachable (is the API enabled?)"
        acct = b.get_account()
        self.brokers["ib"] = b
        self.meta["ib"] = {"paper": int(c["port"]) == 7497}
        return True, f"equity ${acct['equity']:,.2f}"

    # ----- factories ------------------------------------------------------
    def data_sources(self) -> List[Tuple[str, str]]:
        """(label, key) of every usable historical data provider."""
        return [(SPECS[k].label, k) for k in self.providers if k in SPECS]

    def broker_for_live(self, which: str, initial_balance: float = 10_000.0) -> Any:
        from src.execution.brokers import SimulatedBroker
        if which == "simulated":
            return SimulatedBroker(initial_balance)
        if which in self.brokers:
            return self.brokers[which]
        raise KeyError(f"broker '{which}' is not connected")
