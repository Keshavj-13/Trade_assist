"""Optional news and sentiment helpers.

News scoring is disabled by default to keep predictor runs fast and deterministic.
Set `FIN_ASSIST_ENABLE_NEWS=1` to enable this path.
"""

from __future__ import annotations

from datetime import datetime
import os
import time
from typing import Any, Dict, List, Optional

import requests

from config import settings as cfg
from infra.logging import log


_NEWS_TTL_SECONDS = int(os.environ.get("NEWS_TTL_SECONDS", 45 * 60))
_NEWS_PAGE_SIZE = 5
_NEWS_ENABLED = os.environ.get("FIN_ASSIST_ENABLE_NEWS", "0") == "1"
_NEWS_CACHE: Dict[str, Dict[str, Any]] = {}
_TOKENIZER = None
_MODEL = None
_LABELS = ["negative", "neutral", "positive"]


def _to_epoch(value: Any) -> float:
    """Normalize timestamp-like values to epoch seconds."""

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


def _cache_backend():
    """Return optional cache backend functions.

    Returns:
        Tuple(fetch_fn, upsert_fn) or `(None, None)` when unavailable.
    """

    try:
        from infra.database import fetch_news_cache, upsert_news_cache
    except ImportError as exc:
        log.debug(f"News cache backend unavailable: {exc}")
        return None, None
    return fetch_news_cache, upsert_news_cache


def _fetch_cached_headlines(symbol: str) -> Optional[List[str]]:
    """Load cached news headlines from memory or database."""

    cached = _NEWS_CACHE.get(symbol)
    now = time.time()
    if cached and now - cached.get("ts", 0.0) < _NEWS_TTL_SECONDS:
        return list(cached.get("headlines", []))

    fetch_cache, _ = _cache_backend()
    if fetch_cache is None:
        return cached.get("headlines", []) if cached else None

    try:
        rows = fetch_cache(symbol)
    except Exception as exc:
        log.debug(f"News cache read failed for {symbol}: {exc}")
        return cached.get("headlines", []) if cached else None

    if not rows:
        return cached.get("headlines", []) if cached else None

    headlines = [row.get("headline") for row in rows if row.get("headline")]
    fetched_at = max((_to_epoch(row.get("fetched_at")) for row in rows), default=0.0)
    _NEWS_CACHE[symbol] = {"ts": fetched_at, "headlines": headlines}
    return headlines


def _persist_headlines(symbol: str, headlines: List[str], now_epoch: float) -> None:
    """Persist fetched headlines to memory and optional DB cache."""

    _NEWS_CACHE[symbol] = {"ts": now_epoch, "headlines": list(headlines)}

    _, upsert_cache = _cache_backend()
    if upsert_cache is None or not headlines:
        return

    payload = [{"title": title} for title in headlines]
    try:
        upsert_cache(symbol, payload)
    except Exception as exc:
        log.debug(f"News cache write failed for {symbol}: {exc}")


def _fetch_news_api(symbol: str, api_key: str) -> List[str]:
    """Fetch headlines from NewsAPI for one symbol."""

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "apiKey": api_key,
        "pageSize": _NEWS_PAGE_SIZE,
        "sortBy": "publishedAt",
        "language": "en",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    articles = data.get("articles", []) if isinstance(data, dict) else []
    headlines: List[str] = []
    for article in articles:
        title = article.get("title")
        if not title:
            continue
        headlines.append(str(title))
        if len(headlines) >= _NEWS_PAGE_SIZE:
            break
    return headlines


def fetch_news(symbol: str) -> List[str]:
    """Return recent headlines for a symbol.

    News path is disabled by default. When disabled, this function returns an
    empty list to avoid external cost and non-deterministic behavior.
    """

    if not _NEWS_ENABLED:
        return []

    key = cfg.NEWS_API_KEY
    if not key:
        log.warning("NEWS_API_KEY not set, skipping news fetch.")
        return _fetch_cached_headlines(symbol) or []

    cached_headlines = _fetch_cached_headlines(symbol)
    if cached_headlines is not None and cached_headlines:
        return cached_headlines

    now = time.time()
    try:
        headlines = _fetch_news_api(symbol, key)
    except requests.exceptions.HTTPError as exc:
        log.error(f"Failed to fetch news for {symbol}: {exc}", exc_info=True)
        if exc.response is not None and exc.response.status_code == 429:
            log.warning("NewsAPI rate limited; reusing cache if available.")
        return _fetch_cached_headlines(symbol) or []
    except requests.exceptions.RequestException as exc:
        log.error(f"Failed to fetch news for {symbol}: {exc}", exc_info=True)
        return _fetch_cached_headlines(symbol) or []

    _persist_headlines(symbol, headlines, now)
    log.info(f"Fetched fresh news for {symbol} ({len(headlines)} articles)")
    return headlines


def _load_finbert_model():
    """Load and cache FinBERT model artifacts."""

    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return _TOKENIZER, _MODEL


def finbert_sentiment(headlines: List[str]) -> str:
    """Classify aggregate headline sentiment.

    Returns `neutral` when news mode is disabled, no headlines are present,
    or the model fails.
    """

    if not _NEWS_ENABLED or not headlines:
        return "neutral"

    try:
        import torch

        tokenizer, model = _load_finbert_model()
        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).mean(dim=0)
        sentiment = _LABELS[int(torch.argmax(probs))]
        log.debug(f"FinBERT sentiment: {sentiment} for headlines: {headlines}")
        return sentiment
    except Exception as exc:
        log.error(f"FinBERT sentiment analysis failed: {exc}", exc_info=True)
        return "neutral"
