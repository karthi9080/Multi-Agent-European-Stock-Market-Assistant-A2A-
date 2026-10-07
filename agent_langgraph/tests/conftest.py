import pandas as pd
import pytest

import app.agent as agent_module


@pytest.fixture(autouse=True)
def clear_quote_cache():
    agent_module._quote_cache.clear()
    yield
    agent_module._quote_cache.clear()


class FakeTicker:
    """Stand-in for yfinance.Ticker that serves canned price history."""

    def __init__(self, closes, currency="EUR", raises=None, previous_close=None):
        self._closes, self._currency, self._raises = closes, currency, raises
        self.fast_info = {"currency": currency}
        if previous_close is not None:
            self.fast_info["previousClose"] = previous_close

    def history(self, **kwargs):
        if self._raises:
            raise self._raises
        idx = pd.date_range("2026-09-24", periods=len(self._closes), freq="B")
        return pd.DataFrame({"Close": self._closes}, index=idx)


@pytest.fixture
def fake_yf(monkeypatch):
    """Patch yfinance.Ticker; returns a list that records every ticker symbol requested."""
    calls = []

    def install(closes=(100.0, 101.0), currency="EUR", raises=None, previous_close=None):
        def factory(symbol):
            calls.append(symbol)
            return FakeTicker(list(closes), currency, raises, previous_close)

        monkeypatch.setattr(agent_module.yf, "Ticker", factory)
        return calls

    return install
