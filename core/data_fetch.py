"""
Stock data fetching logic (yfinance, NSE).
"""
from infra.logging import log
from config import settings as cfg


def fetch_data(symbol):
    # Moved from market_assistant.py
    import yfinance as yf
    log.debug(f"Fetching data for {symbol}")
    try:
        df = yf.download(
            symbol + ".NS",
            period=cfg.LOOKBACK,
            interval=cfg.INTERVAL,
            progress=False
        )
        log.debug(f"Fetched {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch data for {symbol}: {e}", exc_info=True)
        import pandas as pd
        return pd.DataFrame()
