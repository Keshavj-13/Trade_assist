# Market Assistant

Market Assistant is a modular Python tool for supporting NSE intraday research and decisions. It combines technical indicators, portfolio awareness, FinBERT-powered news sentiment, and Telegram notifications without agents, frameworks, or cloud LLMs.

## What it does
- Finds top NSE momentum stocks (balanced scanning with penny names allowed)
- Numeric chart analysis (EMA, RSI, ATR, volume spike)
- Portfolio-aware decision engine that enforces sell logic for tracked positions
- News risk analysis using a local FinBERT model
- Telegram notification sink for alerts command handling
- Lightweight CLI/service layers with no external agents

## Setup
1. Create a dedicated virtualenv (recommended).
2. Install dependencies:
   `pip install -r requirements.txt`
3. Configure API tokens by running any command once: `python market_assistant.py once`.
   Required keys are saved to `data/config.json`.

## CLI Modes
`market_assistant.py` exposes a small CLI to keep running variations simple:

- `once` (default): Run a single market scan, send notifications for filtered BUY/SELL decisions, then exit.
- `daemon`: Poll Telegram commands continuously; use `/research` to trigger manual scans from the chat.
- `scheduler`: Run a simple loop that sleeps for 5 minutes between sequential scans (no Telegram polling).
- `telegram`: Start only the Telegram listener loop, useful if scanning runs elsewhere.

Each mode bootstraps logging (`logs/`), initializes the SQLite store (`data/market.db`), and delegates analysis to `service.runner.run_once`, so fixes to the runner affect every mode.

### Daemon + manual research
When you run `python market_assistant.py daemon`, the process now only polls Telegram for commands—it will not scan the market on its own. Trigger research manually with `/research` in Telegram:

- `/research w [limit]`: “whole” scan across the NSE symbols list; returns BUY/SELL/HOLD signals and records decisions. Append an optional numeric `limit` to override the configured `TOP_N`.
- `/research p`: “portfolio” scan; only evaluates your logged positions so it can only HOLD or SELL.

Each `/research` call persists the recommendations to `data/market.db` and replies with a summary message, so the daemon becomes a manual research assistant you control from Telegram. Other CLI modes (`once`, `scheduler`, `telegram`) still call the scan pipeline automatically, so use them if you want scheduled work instead.

## Monitoring & analysis
- Scan statistics (price, intraday high/low, VWAP, volatility) are appended to CSVs under `data/analysis/{SYMBOL}.csv`, so you can chart readouts across multiple scans.
- Intraday snapshot charts are saved to `logs/graphs/{SYMBOL}_{TIMESTAMP}.png`. When `/research` produces BUY or SELL signals it adds the file path to the Telegram reply so you can open the most recent chart quickly.
- Use `tail -n 20 data/analysis/RELIANCE.csv` or open the PNG in your viewer to review how the intraday range, VWAP, and momentum behaved before a decision.

## Telegram commands
- `/bought SYMBOL QTY [PRICE]`: log a new position (price defaults to latest close if omitted).
- `/sold SYMBOL QTY [PRICE]`: deduct from a position (price defaults to latest close if omitted).
- `/positions`: list open positions currently stored in SQLite.
- `/research`: see “Daemon + manual research” above for how to use this command with `w`/`p` scopes.

## File structure
- `core/`: Algorithms that fetch market data, compute indicators, and decide BUY/SELL/HOLD.
- `infra/`: Supporting infrastructure such as logging, database helpers, and Telegram utilities.
- `service/`: Runners, daemon loop, scheduler, and Telegram bot entry points that wire CLI modes together.
- `data/`: Cached news, SQLite database, API/config state, and the symbol list definitions.
- `logs/`: Rotating logs produced by the CLI service (created at runtime).

## Vision
Keep Market Assistant a transparent research companion that helps you think through trades. The focus is on:
- deterministic, local signal generation (no black-box LLMs or external agents).
- tight coupling between portfolio state and decision logic so sells line up with tracked holdings.
- modular layers for easy experimentation (e.g., swap indicator rules in `core/`, plug in new Telegram handlers in `infra/`, or extend scheduler behavior in `service/` without touching the scan logic).

## Recent refactor (2026-02-02)
- Folder split into `core/`, `infra/`, and `service/` modules.
- Shared data/config under `data/`; update paths via `config/settings.py`.
- CLI now offers `once`, `daemon`, `scheduler`, and `telegram` modes while still running the same scan pipeline.
- Persistent logging under `logs/` with rotation.
- News caching in `data/news_cache.json`; in-memory cache avoids redundant downloads.
- Telegram command handlers backed by a lightweight SQLite store at `data/market.db`.

## Disclaimer
This is not automated trading software.
For research and decision support only.
