"""
News fetching and FinBERT sentiment analysis.
"""

import time
import os
from datetime import datetime, timedelta
import requests
from infra.logging import log

# In-memory cache: { symbol: { 'ts': epoch, 'headlines': [...] } }
from config import settings as cfg

_news_cache = {}
_NEWS_TTL_SECONDS = int(os.environ.get("NEWS_TTL_SECONDS", 45 * 60))
_NEWS_PAGE_SIZE = 5

# load persistent cache if present
try:
    if hasattr(cfg, 'NEWS_CACHE_FILE') and os.path.exists(cfg.NEWS_CACHE_FILE):
        with open(cfg.NEWS_CACHE_FILE, 'r') as _f:
            import json
            _news_cache = json.load(_f)
            # normalize ts to float epoch if stored as ISO
            for k, v in list(_news_cache.items()):
                ts = v.get('ts') or v.get('timestamp')
                if isinstance(ts, str):
                    try:
                        # try parse ISO
                        from datetime import datetime as _dt
                        _news_cache[k]['ts'] = _dt.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").timestamp()
                    except Exception:
                        try:
                            _news_cache[k]['ts'] = float(ts)
                        except Exception:
                            _news_cache[k]['ts'] = 0
                else:
                    _news_cache[k]['ts'] = ts or 0
except Exception:
    _news_cache = {}

def fetch_news(symbol):
    from config import settings as cfg
    key = cfg.NEWS_API_KEY
    if not key:
        log.warning("NEWS_API_KEY not set, skipping news fetch.")
        return []

    now = time.time()
    entry = _news_cache.get(symbol)
    if entry:
        age = now - entry.get("ts", 0)
        if age < _NEWS_TTL_SECONDS:
            headlines = entry.get("headlines", [])
            log.info(f"Reusing cached news for {symbol} ({len(headlines)} articles)")
            return headlines

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "apiKey": key,
        "pageSize": _NEWS_PAGE_SIZE,
        "sortBy": "publishedAt",
        "language": "en"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", []) if isinstance(data, dict) else []
        headlines = []
        for a in articles:
            t = a.get("title")
            if t:
                headlines.append(t)
            if len(headlines) >= _NEWS_PAGE_SIZE:
                break
        _news_cache[symbol] = {"ts": now, "headlines": headlines}
        # persist cache
        try:
            import json
            if hasattr(cfg, 'NEWS_CACHE_FILE'):
                with open(cfg.NEWS_CACHE_FILE, 'w') as _f:
                    json.dump(_news_cache, _f, indent=2)
        except Exception:
            log.error("Failed to persist news cache", exc_info=True)
        log.info(f"Fetched fresh news for {symbol} ({len(headlines)} articles)")
        return headlines
    except Exception as e:
        log.error(f"Failed to fetch news for {symbol}: {e}", exc_info=True)
        return []


def finbert_sentiment(headlines):
    if not headlines:
        return "neutral"
    try:
        # lazy load model/tokenizer
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        global _TOKENIZER, _MODEL, _LABELS
        try:
            _TOKENIZER
        except NameError:
            _TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        try:
            _MODEL
        except NameError:
            _MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _LABELS = ["negative", "neutral", "positive"]

        inputs = _TOKENIZER(
            headlines,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            logits = _MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=1).mean(dim=0)
        sentiment = _LABELS[int(torch.argmax(probs))]
        log.debug(f"FinBERT sentiment: {sentiment} for headlines: {headlines}")
        return sentiment
    except Exception as e:
        log.error(f"FinBERT sentiment analysis failed: {e}", exc_info=True)
        return "neutral"
