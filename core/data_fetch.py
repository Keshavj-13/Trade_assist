"""
Stock data fetching logic (yfinance, NSE).
"""
from infra.logging import log
from config import settings as cfg


def _ensure_yfinance_cache():
    import os

    cache_dir = os.path.join(cfg.DATA_DIR, "yfinance_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        import requests_cache

        requests_cache.install_cache(
            cache_name=os.path.join(cache_dir, "yf_cache"),
            backend="sqlite",
            expire_after=300,
            allowable_methods=("GET", "POST"),
        )
    except ModuleNotFoundError:
        log.warning("requests-cache not available; continuing without cached downloads.")
    except Exception as exc:
        log.warning(f"Failed to install requests-cache: {exc}")


_ensure_yfinance_cache()


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
