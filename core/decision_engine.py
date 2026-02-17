"""
Decision engine for BUY / SELL / IGNORE logic.
"""

def decide(symbol, f, sentiment, open_positions=None):
    # Moved from market_assistant.py
    from infra.logging import log
    from config.settings import (
        INTRADAY_HIGH_BUFFER,
        INTRADAY_LOW_BUFFER,
        INTRADAY_VOLUME_MULTIPLIER,
        INTRADAY_SELL_RSI_THRESHOLD,
    )

    if open_positions is None:
        open_positions = set()

    price = f.get("price", 0.0)
    session_low = max(f.get("session_low", price), 0.0)
    session_high = max(f.get("session_high", price), 0.0)
    avg_volume = max(f.get("avg_volume", 0.0), 1.0)
    volume = f.get("volume", 0.0)
    vwap = f.get("vwap", price)
    rsi = f.get("rsi", 50.0)

    buy_close_to_low = session_low and price <= session_low * (1 + INTRADAY_LOW_BUFFER)
    buy_volume_ok = volume >= avg_volume * INTRADAY_VOLUME_MULTIPLIER
    buy_below_vwap = price <= vwap
    buy_momentum = rsi >= 45

    if symbol in open_positions:
        sell_target = session_high and price >= session_high * (1 - INTRADAY_HIGH_BUFFER)
        if sell_target and rsi >= INTRADAY_SELL_RSI_THRESHOLD:
            log.info(f"{symbol}: SELL (profit target near intraday high)")
            return "SELL"
        log.info(f"{symbol}: HOLD (in portfolio, no sell trigger)")
        return "HOLD"

    if buy_close_to_low and buy_volume_ok and buy_below_vwap and buy_momentum:
        log.info(f"{symbol}: BUY intraday support ({session_low}->{price}, vol {volume:.0f})")
        return "BUY"

    log.debug(f"{symbol}: IGNORE (no intraday buy trigger)")
    return "IGNORE"
