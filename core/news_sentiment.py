"""
News fetching and FinBERT sentiment analysis.
"""

import time
import os
import requests
from datetime import datetime
from typing import Dict, List

from infra.logging import log
from infra.database import fetch_news_cache, upsert_news_cache
from config import settings as cfg

_news_cache: Dict[str, Dict] = {}
_NEWS_TTL_SECONDS = int(os.environ.get("NEWS_TTL_SECONDS", 45 * 60))
_NEWS_PAGE_SIZE = 5


def _to_epoch(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return float(value.timestamp())
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            return float(datetime.fromisoformat(raw).timestamp())
        except ValueError:
            return 0.0
    return 0.0


def _cache_from_db(symbol: str):
    rows = fetch_news_cache(symbol)
    if not rows:
        return None
    headlines = [row["headline"] for row in rows if row["headline"]]
    fetched_at = max((row["fetched_at"] or 0) for row in rows)
    entry = {"ts": _to_epoch(fetched_at), "headlines": headlines}
    _news_cache[symbol] = entry
    return entry


def _cache_to_memory(symbol: str, headlines: List[str], ts: float):
    entry = {"ts": ts, "headlines": headlines}
    _news_cache[symbol] = entry
    return entry


def fetch_news(symbol):
    key = cfg.NEWS_API_KEY
    if not key:
        log.warning("NEWS_API_KEY not set, skipping news fetch.")
        cached = _cache_from_db(symbol)
        return cached["headlines"] if cached else []

    now = time.time()
    cached = _news_cache.get(symbol) or _cache_from_db(symbol)
    if cached and now - cached.get("ts", 0) < _NEWS_TTL_SECONDS:
        headlines = cached.get("headlines", [])
        log.info(f"Reusing cached news for {symbol} ({len(headlines)} articles)")
        return headlines

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "apiKey": key,
        "pageSize": _NEWS_PAGE_SIZE,
        "sortBy": "publishedAt",
        "language": "en",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", []) if isinstance(data, dict) else []
        headlines = []
        entries = []
        for article in articles:
            title = article.get("title")
            if not title:
                continue
            headlines.append(title)
            entries.append(
                {
                    "title": title,
                    "source": article.get("source", {}).get("name"),
                    "published_at": article.get("publishedAt"),
                    "score": None,
                }
            )
            if len(headlines) >= _NEWS_PAGE_SIZE:
                break
        _cache_to_memory(symbol, headlines, now)
        if entries:
            try:
                upsert_news_cache(symbol, entries)
            except Exception:
                log.error("Failed to persist news cache", exc_info=True)
        log.info(f"Fetched fresh news for {symbol} ({len(headlines)} articles)")
        return headlines
    except requests.exceptions.HTTPError as exc:
        log.error(f"Failed to fetch news for {symbol}: {exc}", exc_info=True)
        if exc.response.status_code == 429:
            log.warning("NewsAPI rate limited; reusing cache if available.")
        cached = _cache_from_db(symbol)
        return cached["headlines"] if cached else []
    except Exception as exc:
        log.error(f"Failed to fetch news for {symbol}: {exc}", exc_info=True)
        cached = _cache_from_db(symbol)
        return cached["headlines"] if cached else []


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
