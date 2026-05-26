"""Symbol universe definitions for strategy research."""

from __future__ import annotations

from typing import Tuple


_NIFTY50_SYMBOLS: Tuple[str, ...] = (
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "ULTRACEMCO", "BAJFINANCE", "WIPRO", "ONGC", "NTPC",
    "POWERGRID", "M&M", "SUNPHARMA", "HCLTECH", "TATAMOTORS",
    "BAJAJFINSV", "JSWSTEEL", "TATASTEEL", "ADANIENT", "COALINDIA",
    "TECHM", "NESTLEIND", "HINDALCO", "GRASIM", "CIPLA",
    "DIVISLAB", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "APOLLOHOSP",
    "BPCL", "TATACONSUM", "BRITANNIA", "INDUSINDBK", "SBILIFE",
    "HDFCLIFE", "ADANIPORTS", "UPL", "BAJAJ-AUTO", "LTI",
)

_MID_CAP_SYMBOLS: Tuple[str, ...] = (
    "MPHASIS", "LTTS", "COFORGE", "PERSISTENT",
    "AUROPHARMA", "LUPIN", "TORNTPHARM", "BIOCON",
    "TVSMOTOR", "ASHOKLEY", "MRF",
    "MARICO", "DABUR", "COLPAL", "PIDILITIND",
    "GAIL", "IOC", "PETRONET",
    "DLF", "GODREJPROP",
    "VEDL", "NMDC", "NATIONALUM",
    "PIIND", "DEEPAKNITR", "AARTIIND",
    "YESBANK", "IDFCFIRSTB", "RBLBANK", "FEDERALBNK",
    "PNB", "BANKBARODA", "CANBK", "UNIONBANK",
)


def nifty50_universe() -> Tuple[str, ...]:
    """Return Nifty 50 component symbols for research."""
    return _NIFTY50_SYMBOLS


def broad_nse_universe() -> Tuple[str, ...]:
    """Return broader NSE universe spanning Nifty 50 and mid-caps."""
    return _NIFTY50_SYMBOLS + _MID_CAP_SYMBOLS


def core_research_universe() -> Tuple[str, ...]:
    """Return a curated 30-symbol universe covering diverse sectors and risk profiles."""
    return (
        # Large-cap trend candidates
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "AXISBANK", "KOTAKBANK", "BHARTIARTL", "LT",
        # Momentum candidates
        "BAJFINANCE", "TITAN", "MARUTI", "ASIANPAINT", "NESTLEIND",
        # Defensive / low-vol
        "HINDUNILVR", "ITC", "BRITANNIA", "MARICO", "DABUR",
        # Cyclical / vol candidates
        "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO", "ONGC",
        # Mid-cap representation
        "MPHASIS", "PERSISTENT", "AUROPHARMA", "PIDILITIND", "PIIND",
    )


def test_fixture_universe() -> Tuple[str, ...]:
    """Return symbols with CSV test fixtures available."""
    return ("INFY_NS_1d", "TCS_NS_1d", "RELIANCE_NS_1d", "HDFCBANK_NS_1d")


# ---------------------------------------------------------------------------
# Cross-asset universe additions (yfinance, exchange_suffix="")
# ---------------------------------------------------------------------------


_GLOBAL_ETF_SYMBOLS: Tuple[str, ...] = (
    # US equity
    "SPY",   # S&P 500
    "QQQ",   # NASDAQ 100
    "IWM",   # Russell 2000
    # International equity
    "EEM",   # Emerging markets
    "EFA",   # EAFE developed ex-US
    # Fixed income
    "TLT",   # 20+ yr US treasuries
    "HYG",   # High yield corporate bonds
    # Commodities
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Oil
)

_COMMODITY_SYMBOLS: Tuple[str, ...] = (
    "GLD",    # Gold
    "SLV",    # Silver
    "USO",    # Oil
    "UNG",    # Natural gas
    "PDBC",   # Diversified commodity
    "DBA",    # Agriculture
    "CPER",   # Copper
)

_FX_SYMBOLS: Tuple[str, ...] = (
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
)

_INTERNATIONAL_EQUITY_SYMBOLS: Tuple[str, ...] = (
    "EWJ",   # Japan
    "EWG",   # Germany
    "EWZ",   # Brazil
    "FXI",   # China large-cap
    "EWY",   # South Korea
    "EWA",   # Australia
    "INDA",  # India (broad, USD-denominated)
)

_DIVERSIFIED_RESEARCH_SYMBOLS: Tuple[str, ...] = (
    # US equities
    "SPY", "QQQ", "IWM",
    # International equities
    "EEM", "EWJ", "EWZ", "FXI",
    # Bonds
    "TLT", "HYG",
    # Commodities
    "GLD", "SLV", "USO",
    # FX
    "EURUSD=X", "GBPUSD=X", "USDJPY=X",
    # Sector ETFs (US)
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLY",   # Consumer discretionary
)


def global_etf_universe() -> Tuple[str, ...]:
    """Return global ETF symbols for cross-asset research.

    Use with YFinanceHistoricalDataSource(exchange_suffix="").
    """
    return _GLOBAL_ETF_SYMBOLS


def commodity_universe() -> Tuple[str, ...]:
    """Return commodity ETF symbols for research.

    Use with YFinanceHistoricalDataSource(exchange_suffix="").
    """
    return _COMMODITY_SYMBOLS


def fx_universe() -> Tuple[str, ...]:
    """Return major FX pairs for research.

    Use with YFinanceHistoricalDataSource(exchange_suffix="").
    Symbols already include the yfinance =X suffix.
    """
    return _FX_SYMBOLS


def international_equity_universe() -> Tuple[str, ...]:
    """Return international equity ETF symbols for research.

    Use with YFinanceHistoricalDataSource(exchange_suffix="").
    """
    return _INTERNATIONAL_EQUITY_SYMBOLS


def diversified_research_universe() -> Tuple[str, ...]:
    """Return a 20-symbol cross-asset basket for hypothesis testing.

    Covers US equities, international equities, bonds, commodities,
    FX, and sector ETFs. Designed for trend-following hypothesis testing
    that requires broad asset diversity to find surviving strategies.

    Use with YFinanceHistoricalDataSource(exchange_suffix="").
    """
    return _DIVERSIFIED_RESEARCH_SYMBOLS
