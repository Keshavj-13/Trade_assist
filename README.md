# Market Assistant

Single-file Python market assistant for NSE intraday analysis.

## What it does
- Finds top 5 new stocks (balanced scanner, penny stocks allowed)
- Numeric chart analysis (EMA, RSI, ATR, volume)
- Portfolio-aware decision engine
- News risk analysis using pretrained FinBERT (local)
- Telegram notification
- No agents, no frameworks, no cloud LLMs

## Setup
1. Create a virtualenv (recommended)
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python market_assistant.py

On first run, you will be asked for API keys.
They are saved locally in config.json.

## Recent refactor (2026-02-02)

- Project split into modules under `core/`, `infra/`, and `service/`.
- Data and config moved into `data/` directory; update paths in `config/settings.py`.
- Added CLI in `market_assistant.py` with modes: `once`, `daemon`, `scheduler`, `telegram`.
- Persistent logging now under `logs/` with rotation.
- News caching persisted to `data/news_cache.json` and in-memory cache used.
- Telegram command handlers and a simple SQLite DB in `data/market.db`.

## Disclaimer
This is not automated trading software.
For research and decision support only.# Trade_assist
