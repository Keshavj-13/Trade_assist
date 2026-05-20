#!/usr/bin/env python3
"""Run predictor locally without persistence or network calls.

This script monkeypatches open-position lookup to avoid Oracle and
injects a synthetic OHLCV fetcher so yfinance isn't called.
"""
from datetime import datetime
import pandas as pd
import numpy as np
import pprint

import service.research as research


def make_ohlcv(rows=60, start=None, trend=0.0):
    if start is None:
        start = datetime.utcnow()
    idx = pd.date_range(end=start, periods=rows, freq='1d')
    rng = np.random.RandomState(0)
    price = 100.0
    opens = []
    highs = []
    lows = []
    closes = []
    vols = []
    for i in range(rows):
        change = rng.normal(loc=trend, scale=0.5)
        open_ = price
        close_ = open_ + change
        high = max(open_, close_) + abs(change) * 0.5
        low = min(open_, close_) - abs(change) * 0.5
        vol = 1000 + i * 10
        opens.append(open_)
        highs.append(high)
        lows.append(low)
        closes.append(close_)
        vols.append(vol)
        price = close_
    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": vols,
    }, index=idx)
    return df


def fake_fetcher(symbol: str) -> pd.DataFrame:
    return make_ohlcv()


# Avoid Oracle DB calls by returning no open positions
research.get_open_positions = lambda *a, **k: []

result = research.perform_scan(scope="whole", symbols=["INFY"], top_n=3, data_fetcher=fake_fetcher)
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(result)
