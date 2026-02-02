"""
Decision engine for BUY / SELL / IGNORE logic.
"""

def decide(symbol, f, sentiment):
    # Moved from market_assistant.py
    from infra.logging import log
    from config.settings import POSITIONS
    if symbol in POSITIONS:
        if f["ema20"] < f["ema50"] and sentiment == "negative":
            log.info(f"{symbol}: SELL (ema20 < ema50 and negative sentiment)")
            return "SELL"
        log.info(f"{symbol}: HOLD (in portfolio, no sell trigger)")
        return "HOLD"
    else:
        if f["ema20"] > f["ema50"] and f["rsi"] > 55 and sentiment != "negative":
            log.info(f"{symbol}: BUY (ema20 > ema50, rsi > 55, sentiment={sentiment})")
            return "BUY"
        log.info(f"{symbol}: IGNORE (no buy trigger)")
        return "IGNORE"
