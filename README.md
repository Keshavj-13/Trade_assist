# Market Assistant (DBMS-Centered)

Market Assistant is an intraday research companion for Indian equities, built as a **DBMS-first system**.  
It combines indicator-driven signal generation, sentiment analysis, and RL-assisted decisioning, while enforcing wallet/position/trade integrity through an Oracle + PL/SQL backend.

## Team Ownership

- **Keshav**: AI/ML and backend integration
- **Satwik**: backend database engineering (Oracle schema + PL/SQL)
- **Anonya**: frontend and API integration

## Project Journey (Why We Migrated)

We built the first full-stack baseline quickly using SQLite to validate:
- user + wallet flows
- positions/trades lifecycle
- scan execution + summary delivery
- frontend control room behavior

After baseline stabilization, we migrated to **Oracle PL/SQL** because the project required stronger DBMS guarantees:
- database-enforced business rules (not only app-layer checks)
- procedural mutation logic (stored procedures/functions)
- better transactional control for wallet/trade operations
- production-style relational integrity and schema governance

## What It Delivers

- **Intraday focus**: EMA/RSI/ATR, VWAP, session range, volume spike filters
- **Explainable outcomes**: BUY/SELL/HOLD decisions include reason text
- **DB-backed guardrails**: invalid SELL paths are blocked at DB layer
- **Wallet-aware logging**: decisions, wallet deltas, and position state are persisted
- **Operator control**: Telegram commands + Node UI for manual research triggers

## Architecture

- `core/`: indicators, data fetch, decision logic, sentiment, RL
- `infra/`: Oracle DB layer, logging, monitoring, user store, migrations
- `service/`: run modes (`once`, `daemon`, `scheduler`, `telegram`), research orchestration
- `frontend_node/`: Node control room that calls Python backend APIs

## DBMS Design

### Core Tables

- `users`
- `positions`
- `trades`
- `snapshots`
- `news_cache`
- `research_runs`

### PL/SQL Objects

- Procedures:
  - `pr_upsert_user`
  - `pr_set_user_wallet`
  - `pr_upsert_position`
- Function:
  - `fn_adjust_user_wallet`
- Trigger:
  - `trg_validate_sell_position`

### DBMS Concepts Applied

- Entity integrity (primary keys / identity columns)
- Referential integrity (foreign keys)
- Domain integrity (CHECK constraints)
- Normalized relational modeling
- Procedural encapsulation in PL/SQL
- Transaction-safe wallet/position/trade updates

## Setup

### 1) Python + Node dependencies

```bash
pip install -r requirements.txt
npm install --prefix frontend_node
```

### 2) Oracle connection env vars

Set these before running the app:

```bash
export ORACLE_DB_USER='finassist'
export ORACLE_DB_PASSWORD='FinAssistPass123!'
export ORACLE_DB_DSN='localhost:1521/XEPDB1'
```

### 3) (Optional) Local Oracle XE via Docker

```bash
docker run -d \
  --name finassist-oracle \
  -p 1521:1521 \
  -e ORACLE_PASSWORD='OracleSysPass123!' \
  -e APP_USER='finassist' \
  -e APP_USER_PASSWORD='FinAssistPass123!' \
  -v finassist-oracle-data:/opt/oracle/oradata \
  gvenzl/oracle-xe:21-slim
```

## Running Modes

| Command | Description |
| --- | --- |
| `python market_assistant.py once` | Run one scan cycle and persist decisions. |
| `python market_assistant.py daemon` | Telegram daemon mode (manual command-driven scans). |
| `python market_assistant.py scheduler` | Continuous scheduled scan loop. |
| `python market_assistant.py telegram` | Telegram listener only. |

## Frontend Control Room

Start the Node frontend:

```bash
npm --prefix frontend_node run dev
```

Or full stack helper:

```bash
./run_full_stack.sh
```

Open: `http://localhost:3000`

## Telegram Commands

- `/research w [limit]` — full-universe manual research
- `/research p` — portfolio-only research
- `/bought SYMBOL QTY [PRICE]` — log entry
- `/sold SYMBOL QTY [PRICE]` — log exit (supports partial)
- `/positions` — view open positions
- `/update_wallet AMOUNT` — adjust wallet budget

## Inspecting Database Entries

If using the local container:

```bash
docker exec -it finassist-oracle sqlplus finassist/FinAssistPass123!@localhost:1521/XEPDB1
```

Useful queries:

```sql
SELECT username, wallet_balance, created_at FROM users ORDER BY created_at DESC;
SELECT id, symbol, side, quantity, price, created_at FROM trades ORDER BY created_at DESC FETCH FIRST 20 ROWS ONLY;
SELECT id, symbol, quantity, entry_price, entry_time FROM positions ORDER BY entry_time DESC;
SELECT id, symbol, created_at, rl_action, composite_signal FROM research_runs ORDER BY created_at DESC FETCH FIRST 20 ROWS ONLY;
```

## Troubleshooting

- **`DPY-4001: no credentials specified`**
  - Set `ORACLE_DB_USER`, `ORACLE_DB_PASSWORD`, `ORACLE_DB_DSN`.

- **Oracle DDL/runtime errors after schema edits**
  - Ensure app user has table/procedure permissions and that stale objects are cleaned before rerun.

- **Matplotlib cache warning**
  - Set a writable cache path:
    ```bash
    export MPLCONFIGDIR="$PWD/data/mplcache"
    ```

- **Slow first scan**
  - FinBERT model warm-up and first news pulls can add startup latency.

- **Rate-limited NewsAPI**
  - Cached headlines in `news_cache` are reused when possible.

## Testing Status

There is currently no formal automated test suite in this repository (`pytest` discovers no tests).  
Validation is performed through runtime smoke checks and DB integration checks.

## Disclaimer

This is research software, not auto-trading infrastructure.  
Do not place live trades without independent validation and risk controls.
