import math

import pytest

from app.agent import (
    COUNTRY_INDICES,
    fetch_index_quote,
    get_stock_index_by_country,
    resolve_country,
)


# ---- country name handling (issue: exact-match only) -------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Germany", "Germany"),
        ("germany", "Germany"),
        ("  FRANCE ", "France"),
        ("UK", "Great Britain"),
        ("United Kingdom", "Great Britain"),
        ("England", "Great Britain"),
        ("Holland", "Netherlands"),
        ("Atlantis", None),
        ("Czech Republic", None),  # no working Yahoo symbol, deliberately unsupported
        ("", None),
    ],
)
def test_resolve_country(text, expected):
    assert resolve_country(text) == expected


# ---- quote fetching ----------------------------------------------------------------------------
def test_quote_computes_daily_change(fake_yf):
    fake_yf(closes=[100.0, 110.0])
    q = fetch_index_quote("^GDAXI")
    assert q["last"] == 110.0 and q["previous"] == 100.0
    assert q["change_pct"] == pytest.approx(10.0)
    assert q["currency"] == "EUR"


def test_quote_ignores_trailing_nan_row(fake_yf):
    """Regression: Yahoo returned NaN for ^IBEX's still-open trading day."""
    fake_yf(closes=[100.0, 102.0, math.nan])
    q = fetch_index_quote("^IBEX")
    assert q["last"] == 102.0 and q["previous"] == 100.0


def test_quote_needs_two_valid_closes(fake_yf):
    fake_yf(closes=[100.0, math.nan])
    with pytest.raises(LookupError):
        fetch_index_quote("^X")


def test_quote_single_row_falls_back_to_live_previous_close(fake_yf):
    """Regression: WIG20 / BUX have one price row and no history; previous close comes from fast_info."""
    fake_yf(closes=[110.0], previous_close=100.0, currency="PLN")
    q = fetch_index_quote("WIG20.WA")
    assert q["last"] == 110.0 and q["previous"] == 100.0 and q["currency"] == "PLN"


def test_quote_single_row_without_previous_close_fails(fake_yf):
    fake_yf(closes=[110.0])
    with pytest.raises(LookupError):
        fetch_index_quote("WIG20.WA")


def test_quote_is_cached(fake_yf):
    calls = fake_yf()
    fetch_index_quote("^GDAXI")
    fetch_index_quote("^GDAXI")
    assert len(calls) == 1  # second call served from the cache


# ---- the LangChain tool ------------------------------------------------------------------------
def test_tool_formats_answer(fake_yf):
    fake_yf(closes=[200.0, 210.0], currency="EUR")
    out = get_stock_index_by_country.invoke({"country": "germany"})
    assert "Germany - DAX (^GDAXI)" in out
    assert "210.00 EUR" in out and "+5.00%" in out and "as of" in out


def test_tool_resolves_alias(fake_yf):
    calls = fake_yf()
    out = get_stock_index_by_country.invoke({"country": "UK"})
    assert "FTSE 100" in out and calls[0] == "^FTSE"


def test_tool_rejects_unsupported_country(fake_yf):
    calls = fake_yf()
    out = get_stock_index_by_country.invoke({"country": "Atlantis"})
    assert "not a supported country" in out and "Germany" in out
    assert calls == []  # never hit Yahoo for an unknown country


def test_tool_reports_yahoo_failure_without_raising(fake_yf):
    fake_yf(raises=RuntimeError("yahoo is down"))
    out = get_stock_index_by_country.invoke({"country": "France"})
    assert "Could not fetch market data for France" in out and "yahoo is down" in out


# ---- real Yahoo Finance (run with: uv run pytest -m live) --------------------------------------
@pytest.mark.live
@pytest.mark.parametrize("country", list(COUNTRY_INDICES))
def test_live_every_country_returns_real_data(country):
    out = get_stock_index_by_country.invoke({"country": country})
    assert "Could not fetch" not in out, out
    assert "as of 20" in out
    print(out)
