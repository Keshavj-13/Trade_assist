"""Tests for optional news/sentiment path gating behavior."""

from __future__ import annotations

import importlib


def _reload_news_module():
    import core.news_sentiment as news_sentiment

    return importlib.reload(news_sentiment)


def test_news_fetch_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_NEWS", raising=False)
    news = _reload_news_module()

    called = {"network": 0}

    def _fail(*args, **kwargs):
        called["network"] += 1
        raise AssertionError("network call should not occur when news is disabled")

    monkeypatch.setattr(news.requests, "get", _fail)
    result = news.fetch_news("INFY")

    assert result == []
    assert called["network"] == 0


def test_finbert_sentiment_is_neutral_when_news_disabled(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_NEWS", raising=False)
    news = _reload_news_module()

    monkeypatch.setattr(
        news,
        "_load_finbert_model",
        lambda: (_ for _ in ()).throw(AssertionError("model load should not happen")),
    )

    assert news.finbert_sentiment(["headline"]) == "neutral"


def test_news_enabled_without_api_key_uses_cache_only(monkeypatch) -> None:
    monkeypatch.setenv("FIN_ASSIST_ENABLE_NEWS", "1")
    news = _reload_news_module()

    monkeypatch.setattr(news.cfg, "NEWS_API_KEY", None)
    monkeypatch.setattr(news, "_fetch_cached_headlines", lambda symbol: ["cached headline"])

    called = {"network": 0}

    def _fail(*args, **kwargs):
        called["network"] += 1
        raise AssertionError("network call should not occur without api key")

    monkeypatch.setattr(news.requests, "get", _fail)
    result = news.fetch_news("INFY")

    assert result == ["cached headline"]
    assert called["network"] == 0
