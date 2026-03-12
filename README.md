# Market Assistant

Market Assistant is an intraday research companion tailored to the Indian equity market. It merges deterministic technical indicators, sentiment/money-flow guardrails, wallet-aware risk budgets, and reinforcement-learning nudges so that every BUY/SELL/HOLD recommendation is explainable, auditable, and tied to the portfolio you actually manage.

## What it delivers

- **Intraday focus** — EMA/RSI/ATR, volume spikes, VWAP, session range, and stop-loss/profit-target guardrails stay alert to today’s swings, not long-term holdings.
- **Constrained actor-critic RL** — the new masked actor-critic policy tracks portfolio state, wallet budgets, and reward-shaped guardrails so invalid sells are impossible and the agent only acts when it sees a real position.
- **Wallet & logging aware** — the wallet ledger lives in `data/market.db`, `log_transaction` enforces guardrails, and no SELL is permitted unless an open position exists; every decision writes to the `trades` table and the shared wallet simultaneously.
- **News cache resilience** — cached headlines (seeded from `data/news_cache.json`) answer your sentiment questions whenever NewsAPI throttles you, so `/research` always returns explainable context with the latest stored text.
- **Telegram + Node control room** — control the assistant from Telegram commands or a dedicated Node.js UI; research runs are always manual (`/research`) so you stay in charge.
- **Explainability, graphs, audits** — decisions carry a `reason` string, summary text, and optional graph snapshots (`logs/graphs/`), and every scan logs to `logs/market_assistant.log`.

## Vision

Enforce a transparent intraday workflow: deterministic signals, portfolio coupling, explicit stop-losses, and manual triggers (no autopilot). The system helps you think, not blindly trade.

## Setup

1. **Create a Python virtual environment** (strongly recommended) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `requests-cache` is unavailable you will see a warning but the scan will still run (caching just speeds up repeated yfinance downloads).
2. **Configure API keys** by running any mode once; e.g., `python market_assistant.py once`. The CLI will prompt for Telegram and news API tokens and persist them to `data/config.json`.
3. **Install the Node.js UI dependencies**:
   ```bash
   npm install --prefix frontend_node
   ```

## Running modes

| Command | Description |
| --- | --- |
| `python market_assistant.py once` | Run a single intraday scan, log BUY/SELL hits, and exit. |
| `python market_assistant.py daemon` | Poll Telegram and respond to `/research`, `/bought`, `/sold` etc; scans only run when you command them. |
| `python market_assistant.py scheduler` | Loop every 5 minutes to scan the universe without Telegram integration (useful for cron-style execution). |
| `python market_assistant.py telegram` | Just run the Telegram listener loop if scanning runs elsewhere. |

### Manual research

- **Telegram** — use `/research w [limit]` for a whole-universe scan, `/research p` to focus only on your open positions, `/update_wallet AMOUNT` to tune spending limits, and `/positions` to see the stored portfolio. Each `/research` run now writes a `research_runs` record plus a `trades` row via `log_transaction`, so the decision path is auditable.
- **Node UI** — launch the full stack with `./run_full_stack.sh`. The script starts the daemon plus the Node control room (see below). Once signed in you can run “Full universe” or “Portfolio-only” scans, watch summary text, and see BUY/SELL/HOLD reasoning and candidate lists.

## Node control room

1. `./run_full_stack.sh` installs Node deps and launches the Node.js UI (default `http://localhost:3000`). Pass `--with-backend` if you also want the Python daemon to start alongside the UI.
2. The UI talks to two Python helpers:
   - `frontend_user_api.py` (login, signup, wallet updates backed by `infra/user_store`).
   - `frontend_api.py` (runs `service.research` with wallet info, persists decisions, and returns the text summary).
3. The Node UI includes:
   - Login + signup flows with wallet propagation.
   - Wallet update form that writes back to the shared store.
   - Action buttons for “Full universe” and “Portfolio-only” research that report BUY/SELL/HOLD counts plus explicit reasoning pulled from `service.research.format_summary_text`.
   - A decision grid that lists top buys, sells, and holdings with the rationale attached.
   - A log pane so you can monitor what the assistant just did.
4. You can also run the Node app separately via `npm --prefix frontend_node run dev`, or just hit the backend CLI methods directly if you prefer (e.g., `python market_assistant.py once` + `/research` in Telegram).
   All user/position/trade writes happen in `data/market.db` so the UI, Telegram, and CLI share the same ledger.

## Telegram commands

- `/research w [limit]` — manual whole-universe scan (append a positive number to override `TOP_N`).
- `/research p` — only judging the positions you already logged.
- `/bought SYMBOL QTY [PRICE]` — log a new position (price defaults to the last close).
- `/sold SYMBOL QTY [PRICE]` — log an exit (reduces quantity by `qty`, so you can partial-sell).
- `/positions` — list the stored portfolio.
- `/update_wallet AMOUNT` — top up or reduce your risk budget for Telegram-triggered scans.

Every `/research` run explains why each symbol was classified (included via `reason` and the summary text), so Telegram replies are much more than “BUY/SELL”.

## Logging & data

- Logs stream to the console and to `logs/market_assistant.log` with rotation; generated graphs land in `logs/graphs/` and the monitor normalizes `Low`/`High` series before drawing so fill_between never crashes.
- `data/market.db` now hosts `users`, `positions`, `trades`, `snapshots`, `news_cache`, `research_runs`, and `rl_transitions`. Triggers enforce that SELLs must reference an open position, wallets never go negative, and every decision is recorded through `log_transaction` so the ledger stays consistent.
- `news_cache.json` is imported into the new `news_cache` table during initialization, so the assistant still shows cached headlines whenever NewsAPI responds with HTTP 429.
- `infra/user_store` reads/writes against the same SQLite wallet table, so Telegram, the Node UI, and the scanner operate on the same source of truth.
- `data/rl_agent.pt` records the actor-critic weights between scans; the persistent model keeps guardrails improving over time.
- `config/settings.py` holds all strategy knobs (`STOP_LOSS_PCT`, `PROFIT_TARGET_PCT`, `RL_*`, etc.). Tweak them if you want different sensitivity.

## File structure

- `core/` — technical indicators, data fetchers, the decision engine, and ML/RL helpers.
- `infra/` — logging, database wrappers, Telegram helpers, and user stores.
- `service/` — runner entrypoints, research workflow, daemon/scheduler loops, and Telegram bot wiring.
- `frontend_node/` — Node control room, static assets, and server glue for login/research.
- `data/` — NSE symbol list, news cache, SQLite store, RL files, and generated user state.
- `logs/` — rotating log files plus generated graph images.

-## Troubleshooting & gotchas

- When NewsAPI throttles (`429 Client Error`), the scan automatically falls back to the cached headlines stored in `data/market.db` (the table is seeded from `data/news_cache.json` during initialization), so `/research` never fails because of rate limits.
- `requests-cache` is optional but recommended; you can install it via `pip install requests-cache`. If it’s missing you’ll see a warning but the assistant still functions.
- When you get repeated `OperationalError('attempt to write a readonly database')` traces, it means yfinance is trying to cache under `/tmp` which is mounted read-only in this environment. Fix it on your machine by pointing yfinance to a cache directory inside the repo:  
  ```bash
  mkdir -p data/yfinance_cache
  export YFINANCE_CACHE_DIR="$PWD/data/yfinance_cache"
  python market_assistant.py once
  ```  
  The `export` ensures every yfinance download (and `requests-cache`) writes to `data/yfinance_cache` instead of `/tmp`, so the noisy log disappears. You can add the `export` line to your shell profile if you run the assistant often.
- You might see `OMP: Warning #179`; that comes from PyTorch’s OpenMP layer when TensorFlow/transformers load FinBERT in the container; it is audible but harmless.
- Running `python market_assistant.py once` without Telegram credentials will prompt you to enter them; `data/config.json` will hold the values afterward.
- The Node UI relies on `frontend_user_api.py` and `frontend_api.py`. Keep those scripts in sync with your Python path (the Node server runs them with `python` from wherever it is started).
- If you are upgrading from the legacy JSON store and `./run_full_stack.sh` reports “Unknown user” or login never works, delete `data/market.db` and run `python -m infra.migrate_users`; that script re-creates the SQLite schema and pulls every wallet/password from `data/users.json` so the Node UI and Telegram share the same accounts again.

## Next steps

1. Tune `config/settings.py` to match your risk appetite (ATR bands, stop-loss thresholds, etc.).
2. Use `/bought` and `/sold` to keep the portfolio mirror accurate so SELL logic can trigger appropriately.
3. Iterate on the RL actor-critic weights (`data/rl_agent.pt`) by running manual scans and observing whether the assistant’s suggestions align with your expectations.
4. Query `data/market.db` yourself (especially the `trades` and `research_runs` tables) to verify `log_transaction` wrote the expected wallet deltas and reasons.

## Disclaimer

This is research software. Do not use Market Assistant to execute live orders without your own validation. No automated trading or brokerage integrations are provided.
