"""
Database logic for fin_assist (SQLite schema + helpers).
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import (
    DATA_DIR,
    DEFAULT_USER_INITIAL_CAPITAL,
    DEFAULT_USER_MAX_ALLOCATION,
    DEFAULT_USER_RISK_PER_TRADE,
    DEFAULT_USER_WALLET,
    NEWS_CACHE_FILE,
    PROFIT_TARGET_PCT,
    STOP_LOSS_PCT,
    SYSTEM_USERNAME,
)
from infra.logging import log

DB_PATH = os.path.join(DATA_DIR, 'market.db')
MAX_SNAPS = 200


def _ensure_data_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _get_conn():
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    conn = _get_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            wallet_balance REAL NOT NULL CHECK(wallet_balance >= 0),
            initial_capital REAL NOT NULL CHECK(initial_capital > 0),
            risk_per_trade REAL NOT NULL CHECK(risk_per_trade > 0 AND risk_per_trade <= 1),
            max_allocation_pct REAL NOT NULL CHECK(max_allocation_pct > 0 AND max_allocation_pct <= 1),
            rl_enabled INTEGER NOT NULL DEFAULT 1 CHECK(rl_enabled IN (0,1)),
            notes TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL CHECK(quantity > 0),
            entry_price REAL NOT NULL CHECK(entry_price > 0),
            entry_time DATETIME NOT NULL,
            stop_loss REAL NOT NULL CHECK(stop_loss > 0),
            target_price REAL NOT NULL CHECK(target_price > 0),
            rl_confidence REAL,
            guardrail_reason TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status = 'OPEN'),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, symbol)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            position_id INTEGER,
            side TEXT NOT NULL CHECK(side IN ('BUY','SELL','HOLD','PARTIAL_EXIT','ADJUST_STOP')),
            quantity INTEGER CHECK(quantity >= 0),
            price REAL CHECK(price >= 0),
            pnl REAL,
            pnl_pct REAL,
            rl_score REAL,
            reward REAL,
            metadata TEXT,
            guardrail_reason TEXT,
            research_alignment INTEGER CHECK(research_alignment IN (0,1)),
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(position_id) REFERENCES positions(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            price REAL NOT NULL,
            vwap REAL,
            rsi REAL,
            atr REAL,
            volume REAL,
            pct_from_low REAL,
            pct_from_high REAL,
            position_flag INTEGER NOT NULL CHECK(position_flag IN (0,1)),
            unrealized_pnl REAL,
            wallet_utilization REAL,
            state_vector TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    c.execute('DROP TABLE IF EXISTS news_cache')
    c.execute('''
        CREATE TABLE news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            headline TEXT NOT NULL,
            sentiment_score REAL,
            source TEXT,
            published_at DATETIME,
            fetched_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, headline)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS research_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            ema_signal INTEGER,
            rsi_signal INTEGER,
            vwap_signal INTEGER,
            volume_signal INTEGER,
            composite_signal REAL,
            rl_action TEXT,
            rl_score REAL,
            guardrail_reason TEXT,
            news_hits INTEGER,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS rl_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            state TEXT NOT NULL,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            next_state TEXT NOT NULL,
            done INTEGER NOT NULL CHECK(done IN (0,1)),
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    def _ensure_column(table, column, definition):
        existing = [row["name"] for row in c.execute(f'PRAGMA table_info({table})')]
        if column not in existing:
            c.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
            log.info(f"Added missing column {column} to {table}")

    _ensure_column("positions", "user_id", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column("trades", "user_id", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column("snapshots", "user_id", "INTEGER NOT NULL DEFAULT 1")

    try:
        c.execute('CREATE INDEX IF NOT EXISTS idx_positions_user_symbol ON positions(user_id, symbol)')
    except sqlite3.OperationalError as exc:
        log.debug(f"Could not create idx_positions_user_symbol: {exc}")
    try:
        c.execute('CREATE INDEX IF NOT EXISTS idx_trades_user_symbol ON trades(user_id, symbol)')
    except sqlite3.OperationalError as exc:
        log.debug(f"Could not create idx_trades_user_symbol: {exc}")
    try:
        c.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_time ON snapshots(symbol, timestamp)')
    except sqlite3.OperationalError as exc:
        log.debug(f"Could not create idx_snapshots_symbol_time: {exc}")
    c.execute('''
        CREATE TRIGGER IF NOT EXISTS validate_sell_position
        BEFORE INSERT ON trades
        WHEN NEW.side = 'SELL' AND NEW.position_id IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'SELL requires position_id');
        END;
    ''')
    conn.commit()
    conn.close()
    ensure_user_profile(SYSTEM_USERNAME)
    _seed_news_cache_from_file()
    log.info("Database initialized.")


def fetch_user(username: str):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    return row


def get_user_id(username: str):
    row = fetch_user(username)
    if not row:
        return None
    return row['id']


def ensure_user_profile(username, wallet=None, initial_capital=None, risk_per_trade=None, max_allocation_pct=None, password=None, notes=""):
    row = fetch_user(username)
    if row:
        return row
    wallet = wallet if wallet is not None else DEFAULT_USER_WALLET
    initial_capital = initial_capital if initial_capital is not None else DEFAULT_USER_INITIAL_CAPITAL
    risk_per_trade = risk_per_trade if risk_per_trade is not None else DEFAULT_USER_RISK_PER_TRADE
    max_allocation_pct = max_allocation_pct if max_allocation_pct is not None else DEFAULT_USER_MAX_ALLOCATION
    upsert_user(username, wallet, initial_capital, risk_per_trade, max_allocation_pct, password=password, notes=notes)
    return fetch_user(username)


def upsert_user(username, wallet, initial_capital, risk_per_trade, max_allocation_pct, password=None, notes=""):
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        '''INSERT INTO users (username, password, wallet_balance, initial_capital, risk_per_trade, max_allocation_pct, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(username) DO UPDATE SET
               password=COALESCE(excluded.password, users.password),
               wallet_balance=excluded.wallet_balance,
               initial_capital=excluded.initial_capital,
               risk_per_trade=excluded.risk_per_trade,
               max_allocation_pct=excluded.max_allocation_pct,
               notes=excluded.notes''',
        (username, password, wallet, initial_capital, risk_per_trade, max_allocation_pct, notes),
    )
    conn.commit()
    conn.close()
    log.info(f"User upserted: {username} wallet={wallet}")


def adjust_user_wallet(username, delta):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT wallet_balance FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    new_wallet = float(row['wallet_balance']) + delta
    c.execute('UPDATE users SET wallet_balance = ? WHERE username = ?', (new_wallet, username))
    conn.commit()
    conn.close()
    log.info(f"Wallet adjusted: {username} delta={delta} -> {new_wallet}")
    return new_wallet


def update_user_wallet(username, wallet):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('UPDATE users SET wallet_balance = ? WHERE username = ?', (wallet, username))
    conn.commit()
    conn.close()
    log.info(f"Wallet set: {username} = {wallet}")


def get_wallet_balance(username):
    row = fetch_user(username)
    if not row:
        return 0.0
    return float(row['wallet_balance'] or 0.0)


def record_trade_decision(symbol, action, price, metadata, username, position_id=None, **extras):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    user_row = c.fetchone()
    if not user_row:
        conn.close()
        raise ValueError(f"Unknown user {username}")
    user_id = user_row['id']
    c.execute(
        '''INSERT INTO trades (user_id, symbol, position_id, side, quantity, price, pnl, pnl_pct, rl_score, reward,
            metadata, guardrail_reason, research_alignment)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            user_id,
            symbol.upper(),
            position_id,
            action,
            extras.get('quantity'),
            price,
            extras.get('pnl'),
            extras.get('pnl_pct'),
            extras.get('rl_score'),
            extras.get('reward'),
            str(metadata),
            extras.get('guardrail'),
            extras.get('aligned', 0),
        ),
    )
    conn.commit()
    conn.close()
    log.info(f"Trade decision recorded: {symbol} {action} price={price} meta={metadata}")


def get_open_positions(user_id=None):
    conn = _get_conn()
    c = conn.cursor()
    query = 'SELECT id, user_id, symbol, quantity as qty, entry_price as price FROM positions WHERE status = "OPEN"'
    params = ()
    if user_id is not None:
        query += ' AND user_id = ?'
        params = (user_id,)
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def upsert_position(user_id, symbol, quantity, entry_price, stop_loss, target_price, guardrail_reason=None, rl_confidence=None):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        '''INSERT INTO positions (user_id, symbol, quantity, entry_price, entry_time, stop_loss, target_price, guardrail_reason, rl_confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(user_id, symbol) DO UPDATE SET
               quantity=excluded.quantity,
               entry_price=excluded.entry_price,
               entry_time=excluded.entry_time,
               stop_loss=excluded.stop_loss,
               target_price=excluded.target_price,
               guardrail_reason=excluded.guardrail_reason,
               rl_confidence=excluded.rl_confidence''',
        (user_id, symbol.upper(), quantity, entry_price, now, stop_loss, target_price, guardrail_reason, rl_confidence),
    )
    conn.commit()
    conn.close()
    log.info(f"Position recorded/updated: {symbol} qty={quantity} price={entry_price}")


def delete_position(user_id, symbol):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM positions WHERE user_id = ? AND symbol = ?', (user_id, symbol.upper()))
    conn.commit()
    conn.close()
    log.info(f"Position removed: {symbol} user={user_id}")


def insert_snapshot(symbol, stats, timestamp):
    conn = _get_conn()
    c = conn.cursor()
    state_vector = stats.get('state_vector', '{}')
    c.execute(
        '''INSERT INTO snapshots (user_id, symbol, timestamp, price, vwap, rsi, atr, volume, pct_from_low,
           pct_from_high, position_flag, unrealized_pnl, wallet_utilization, state_vector)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (stats.get('user_id', 1), symbol.upper(), timestamp, stats.get('price'), stats.get('vwap'), stats.get('rsi'),
         stats.get('atr'), stats.get('avg_volume'), stats.get('pct_from_low'), stats.get('pct_from_high'),
         stats.get('position_flag', 0), stats.get('unrealized_pnl'), stats.get('wallet_util'), state_vector),
    )
    if MAX_SNAPS:
        c.execute(
            'DELETE FROM snapshots WHERE rowid NOT IN '
            '(SELECT rowid FROM snapshots ORDER BY timestamp DESC LIMIT ?)', (MAX_SNAPS,)
        )
    conn.commit()
    conn.close()


def fetch_news_cache(symbol):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT headline, sentiment_score, source, fetched_at FROM news_cache WHERE symbol = ?', (symbol.upper(),))
    rows = c.fetchall()
    conn.close()
    return rows


def upsert_news_cache(symbol, headlines):
    conn = _get_conn()
    c = conn.cursor()
    for entry in headlines:
        c.execute(
            'INSERT OR IGNORE INTO news_cache (symbol, headline, sentiment_score, source, published_at, fetched_at)'
            ' VALUES (?, ?, ?, ?, ?, ?)',
            (symbol.upper(), entry.get('title'), entry.get('score'), entry.get('source'), entry.get('published_at'), datetime.utcnow()),
        )
    conn.commit()
    conn.close()


def record_research_run(user_id, symbol, payload):
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        '''INSERT INTO research_runs (user_id, symbol, ema_signal, rsi_signal, vwap_signal, volume_signal,
           composite_signal, rl_action, rl_score, guardrail_reason, news_hits)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (user_id, symbol.upper(), payload.get('ema_signal'), payload.get('rsi_signal'), payload.get('vwap_signal'),
         payload.get('volume_signal'), payload.get('composite_signal'), payload.get('rl_action'), payload.get('rl_score'),
         payload.get('guardrail_reason'), payload.get('news_hits')),
    )
    conn.commit()
    conn.close()


def _default_stop_target(price: float):
    price = price if price > 0 else 1.0
    stop_loss = max(price * (1 - STOP_LOSS_PCT), 0.01)
    target_price = max(price * (1 + PROFIT_TARGET_PCT), stop_loss + 0.01)
    return stop_loss, target_price


def _system_user_id():
    row = ensure_user_profile(SYSTEM_USERNAME)
    return row['id']


def _resolve_user_id(username: Optional[str] = None) -> int:
    if username:
        row = ensure_user_profile(username)
        return row['id']
    return _system_user_id()


def fetch_position(symbol: str, username: Optional[str] = None):
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        'SELECT * FROM positions WHERE user_id = ? AND symbol = ? AND status = "OPEN"',
        (_resolve_user_id(username), symbol.upper()),
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def record_position(symbol: str, qty: float, price: float, timestamp=None, guardrail_reason=None, rl_confidence=None, username: Optional[str] = None):
    if qty <= 0 or price <= 0:
        return
    user_id = _resolve_user_id(username)
    stop_loss, target_price = _default_stop_target(price)
    existing = fetch_position(symbol, username=username)
    if existing:
        ex_qty = existing['quantity']
        ex_price = existing['entry_price']
        total_qty = ex_qty + qty
        if total_qty <= 0:
            delete_position(user_id, symbol)
            return
        avg_price = (ex_qty * ex_price + qty * price) / total_qty
        stop_loss = min(stop_loss, existing.get('stop_loss') or stop_loss)
        target_price = max(target_price, existing.get('target_price') or target_price)
        upsert_position(
            user_id,
            symbol,
            int(total_qty),
            avg_price,
            stop_loss,
            target_price,
            guardrail_reason or existing.get('guardrail_reason'),
            rl_confidence or existing.get('rl_confidence'),
        )
    else:
        upsert_position(
            user_id,
            symbol,
            int(qty),
            price,
            stop_loss,
            target_price,
            guardrail_reason,
            rl_confidence,
        )


def update_position(symbol: str, qty_delta: float, price: float, timestamp=None, username: Optional[str] = None):
    user_id = _resolve_user_id(username)
    existing = fetch_position(symbol, username=username)
    if not existing:
        return
    new_qty = existing['quantity'] + qty_delta
    if new_qty <= 0:
        delete_position(user_id, symbol)
        return
    stop_loss, target_price = _default_stop_target(price)
    stop_loss = min(stop_loss, existing.get('stop_loss') or stop_loss)
    target_price = max(target_price, existing.get('target_price') or target_price)
    upsert_position(
        user_id,
        symbol,
        int(new_qty),
        existing['entry_price'],
        stop_loss,
        target_price,
        existing.get('guardrail_reason'),
        existing.get('rl_confidence'),
    )


def _seed_news_cache_from_file():
    if not NEWS_CACHE_FILE or not os.path.exists(NEWS_CACHE_FILE):
        return
    try:
        with open(NEWS_CACHE_FILE, 'r') as fh:
            payload = json.load(fh)
    except Exception as exc:
        log.debug(f"Failed to read legacy news cache: {exc}")
        return
    for symbol, entry in payload.items():
        headlines = entry.get('headlines') or []
        if not headlines:
            continue
        upsert_news_cache(symbol, [{'title': item} for item in headlines])
