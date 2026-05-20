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
