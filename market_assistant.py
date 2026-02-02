
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===================== CONFIG =====================

CONFIG_FILE = "config.json"
PORTFOLIO_FILE = "portfolio.json"
SYMBOLS_FILE = "nse_symbols.csv"

INTERVAL = "5m"
LOOKBACK = "5d"
TOP_N = 5

MIN_PRICE = 5
MIN_AVG_VOLUME = 50000
MIN_ATR_PCT = 0.2
MAX_ATR_PCT = 8.0


# ===================== LOGGING =====================

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "market_assistant.log")

def setup_logging():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

log = setup_logging()

# ===================== SETUP =====================


def load_or_create_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)

    log.info("First time setup")
    cfg = {
        "NEWS_API_KEY": input("NewsAPI key (leave blank to disable news): ").strip(),
        "TELEGRAM_BOT_TOKEN": input("Telegram Bot Token: ").strip(),
        "TELEGRAM_CHAT_ID": input("Telegram Chat ID: ").strip()
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

    return cfg

CONFIG = load_or_create_config()


with open(PORTFOLIO_FILE, "r") as f:
    PORTFOLIO = json.load(f)

POSITIONS = PORTFOLIO.get("positions", {})

# ===================== FINBERT =====================


log.info("Loading FinBERT (downloads on first run)...")
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
LABELS = ["negative", "neutral", "positive"]


def finbert_sentiment(headlines):
    if not headlines:
        return "neutral"
    try:
        inputs = TOKENIZER(
            headlines,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            logits = MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=1).mean(dim=0)
        sentiment = LABELS[int(torch.argmax(probs))]
        log.debug(f"FinBERT sentiment: {sentiment} for headlines: {headlines}")
        return sentiment
    except Exception as e:
        log.error(f"FinBERT sentiment analysis failed: {e}", exc_info=True)
        return "neutral"


# ===================== PANDAS SCALAR HELPER =====================

def scalar(x):
    if isinstance(x, pd.Series):
        if x.size == 1:
            return x.iloc[0]
        raise ValueError("Series has more than one element")
    return x

# ===================== MARKET DATA =====================


def fetch_data(symbol):
    log.debug(f"Fetching data for {symbol}")
    try:
        df = yf.download(
            symbol + ".NS",
            period=LOOKBACK,
            interval=INTERVAL,
            progress=False
        )
        log.debug(f"Fetched {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        log.error(f"Failed to fetch data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()


def compute_features(df):
    if df is None or df.empty:
        log.warning("Empty dataframe passed to compute_features")
        return None
    df = df.dropna()
    if len(df) < 30:
        log.warning("Insufficient data (<30 rows) for feature computation")
        return None

    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()

    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )
    df["atr"] = tr.rolling(14).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]
    price = scalar(last["Close"])
    atr = scalar(last["atr"])
    ema20 = scalar(last["ema20"])
    ema50 = scalar(last["ema50"])
    rsi = scalar(last["rsi"])
    avg_volume = scalar(df["Volume"].tail(20).mean())
    last_volume = scalar(last["Volume"])
    atr_pct = (atr / price * 100) if price else 0
    vol_spike = (last_volume / avg_volume) if avg_volume else 0

    log.debug(f"Indicators: price={price}, ema20={ema20}, ema50={ema50}, rsi={rsi}, atr_pct={atr_pct}, avg_volume={avg_volume}, vol_spike={vol_spike}")

    return {
        "price": float(price),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "rsi": float(rsi),
        "atr_pct": float(atr_pct),
        "avg_volume": float(avg_volume),
        "vol_spike": float(vol_spike)
    }

# ===================== STOCK FINDER =====================


def score_stock(f):
    score = 0
    if f["ema20"] > f["ema50"]:
        score += 3
    if f["rsi"] > 50:
        score += 2
    if f["vol_spike"] > 1.5:
        score += 2
    if f["atr_pct"] > 5:
        score -= 1
    if f["price"] < 50:
        score -= 0.5
    log.debug(f"Score for stock: {score} with features: {f}")
    return score


def find_top_stocks():
    try:
        symbols = pd.read_csv(SYMBOLS_FILE)["symbol"].tolist()
    except Exception as e:
        log.error(f"Failed to read symbols file: {e}", exc_info=True)
        return []
    results = []

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

            results.append({
                "symbol": sym,
                "features": f,
                "score": score_stock(f)
            })

        except Exception as e:
            log.error(f"Exception processing {sym}: {e}", exc_info=True)
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_N]


# ===================== NEWS CACHE =====================

import threading
import time
from datetime import timedelta

NEWS_CACHE_FILE = "news_cache.json"
NEWS_TTL_MINUTES = 45
_news_cache_lock = threading.Lock()

def _load_news_cache():
    if not os.path.exists(NEWS_CACHE_FILE):
        return {}
    try:
        with open(NEWS_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Failed to load news cache: {e}", exc_info=True)
        return {}

def _save_news_cache(cache):
    try:
        with _news_cache_lock:
            with open(NEWS_CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save news cache: {e}", exc_info=True)

def fetch_news(symbol):
    key = CONFIG.get("NEWS_API_KEY")
    if not key:
        log.warning("NEWS_API_KEY not set, skipping news fetch.")
        return []

    now = datetime.utcnow()
    cache = _load_news_cache()
    entry = cache.get(symbol)
    headlines = []
    if entry:
        try:
            ts = datetime.strptime(entry.get("timestamp", ""), "%Y-%m-%dT%H:%M:%SZ")
            if now - ts < timedelta(minutes=NEWS_TTL_MINUTES):
                headlines = entry.get("headlines", [])
                log.info(f"Reusing cached news for {symbol} ({len(headlines)} articles)")
                return headlines
        except Exception as e:
            log.error(f"Malformed cache entry for {symbol}: {e}", exc_info=True)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "apiKey": key,
        "pageSize": 5,
        "sortBy": "publishedAt",
        "language": "en"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        headlines = [a.get("title", "") for a in articles if a.get("title")][:5]
        log.info(f"Fetched fresh news for {symbol} ({len(headlines)} articles)")
        cache[symbol] = {
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "headlines": headlines
        }
        _save_news_cache(cache)
        return headlines
    except Exception as e:
        log.error(f"Failed to fetch news for {symbol}: {e}", exc_info=True)
        return []

# ===================== DECISION ENGINE =====================


def decide(symbol, f, sentiment):
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


# ===================== NOTIFIER (TELEGRAM) =====================

def notify(symbol, action, confidence, price, timestamp):
    token = CONFIG.get("TELEGRAM_BOT_TOKEN")
    chat_id = CONFIG.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.error("Telegram bot token or chat id missing in config.")
        return
    if action not in ("BUY", "SELL"):
        return
    msg = (
        f"{symbol} — {action}\n"
        f"Confidence: {confidence}\n"
        f"Price: {price}\n"
        f"Time: {timestamp}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": msg}
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            log.info(f"Telegram notification sent for {symbol} {action}.")
        else:
            log.error(f"Telegram API error {resp.status_code}: {resp.text}")
    except Exception as e:
        log.error(f"Failed to send Telegram notification: {e}", exc_info=True)

# ===================== MAIN =====================




def main():
    log.info("Running market assistant")
    try:
        for item in find_top_stocks():
            sym = item["symbol"]
            f = item["features"]

            action = decide(sym, f, "neutral")
            news = []
            sentiment = "neutral"
            if action in ("BUY", "SELL"):
                news = fetch_news(sym)
                sentiment = finbert_sentiment(news)
                action = decide(sym, f, sentiment)

            if action in ("BUY", "SELL"):
                confidence = f"rsi={f['rsi']}, atr_pct={f['atr_pct']}, sentiment={sentiment}"
                price = round(f['price'], 2)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                log.info(f"{sym}: {action} decision sent. Confidence: {confidence}, price={price}")
                notify(sym, action, confidence, price, timestamp)
        log.info("Market assistant run complete.")
    except Exception as e:
        log.error(f"Fatal error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()
