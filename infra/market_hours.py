"""
Utilities for NSE market hours awareness.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from config.settings import (
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_TIMEZONE,
)


def current_market_time() -> datetime:
    return datetime.now(ZoneInfo(MARKET_TIMEZONE))


def is_market_closed(at_time: datetime | None = None) -> bool:
    now = at_time or current_market_time()
    current = now.time()
    close_time = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    return current >= close_time


def is_market_open(at_time: datetime | None = None) -> bool:
    now = at_time or current_market_time()
    open_time = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    close_time = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    return open_time <= now.time() < close_time


def market_time_str(at_time: datetime | None = None) -> str:
    return (at_time or current_market_time()).strftime("%Y-%m-%d %H:%M:%S %Z")
