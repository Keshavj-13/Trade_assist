"""
Database logic for fin_assist (SQLite or placeholder).
"""

import sqlite3
import os
from datetime import datetime
from infra.logging import log

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/market.db')

# --- Helper ---
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    conn = _get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT PRIMARY KEY,
        qty REAL,
        price REAL,
        timestamp TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        action TEXT,
        price REAL,
        metadata TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()
    log.info("Database initialized.")

def record_position(symbol, qty, price, timestamp):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('REPLACE INTO positions (symbol, qty, price, timestamp) VALUES (?, ?, ?, ?)',
              (symbol, qty, price, timestamp))
    conn.commit()
    conn.close()
    log.info(f"Position recorded: {symbol} qty={qty} price={price} @ {timestamp}")

def update_position(symbol, qty_delta, price, timestamp):
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT qty FROM positions WHERE symbol=?', (symbol,))
    row = c.fetchone()
    new_qty = qty_delta
    if row:
        new_qty += row['qty']
    if new_qty == 0:
        c.execute('DELETE FROM positions WHERE symbol=?', (symbol,))
    else:
        c.execute('REPLACE INTO positions (symbol, qty, price, timestamp) VALUES (?, ?, ?, ?)',
                  (symbol, new_qty, price, timestamp))
    conn.commit()
    conn.close()
    log.info(f"Position updated: {symbol} qty_delta={qty_delta} price={price} @ {timestamp}")

def get_open_positions():
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT symbol, qty, price, timestamp FROM positions')
    positions = [dict(row) for row in c.fetchall()]
    conn.close()
    log.info(f"Fetched open positions: {positions}")
    return positions

def record_trade_decision(symbol, action, price, metadata):
    conn = _get_conn()
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute('INSERT INTO trades (symbol, action, price, metadata, timestamp) VALUES (?, ?, ?, ?, ?)',
              (symbol, action, price, str(metadata), ts))
    conn.commit()
    conn.close()
    log.info(f"Trade decision recorded: {symbol} {action} price={price} meta={metadata}")
