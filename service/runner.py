"""
Runner module for starting the service.
"""

from core.data_fetch import fetch_data
from core.indicators import compute_features
from core.decision_engine import decide
from core.news_sentiment import fetch_news, finbert_sentiment
from infra.database import record_trade_decision
from infra.telegram import send_message
from infra.logging import log
from config.settings import SYMBOLS_FILE, MIN_PRICE, MIN_AVG_VOLUME, MIN_ATR_PCT, MAX_ATR_PCT
import pandas as pd
from datetime import datetime

def start_service():
    pass

def run_once():
    log.info("Running market scan (runner.run_once)")
    try:
        symbols = pd.read_csv(SYMBOLS_FILE)["symbol"].tolist()
    except Exception as e:
        log.error(f"Failed to read symbols file: {e}", exc_info=True)
        return
    for sym in symbols:
        log.info(f"Processing symbol: {sym}")
        try:
            df = fetch_data(sym)
            f = compute_features(df)
            if not f:
                log.warning(f"Skipping {sym}: insufficient or missing data")
                continue
            if f["price"] < MIN_PRICE:
                log.warning(f"Skipping {sym}: price {f['price']} < MIN_PRICE")
                continue
            if f["avg_volume"] < MIN_AVG_VOLUME:
                log.warning(f"Skipping {sym}: avg_volume {f['avg_volume']} < MIN_AVG_VOLUME")
                continue
            if not (MIN_ATR_PCT <= f["atr_pct"] <= MAX_ATR_PCT):
                log.warning(f"Skipping {sym}: atr_pct {f['atr_pct']} not in range")
                continue
            news = fetch_news(sym)
            sentiment = finbert_sentiment(news)
            action = decide(sym, f, sentiment)
            if action in ("BUY", "SELL"):
                confidence = f"rsi={f['rsi']}, atr_pct={f['atr_pct']}, sentiment={sentiment}"
                price = round(f['price'], 2)
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
                record_trade_decision(sym, action, price, confidence)
                msg = f"{sym} — {action}\nConfidence: {confidence}\nPrice: {price}\nTime: {timestamp}"
                send_message(msg)
                log.info(f"{sym}: {action} decision sent. Confidence: {confidence}, price={price}")
        except Exception as e:
            log.error(f"Error processing symbol {sym}: {e}", exc_info=True)
    log.info("Market scan complete.")
