"""
Telegram send and receive logic.
"""

import requests
from infra.logging import log
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TOP_N
from infra.database import record_position, update_position, get_open_positions
from datetime import datetime
from core.data_fetch import fetch_data
from service.research import perform_scan, persist_scan_results, format_summary_text

def notify(symbol, action, confidence, price, timestamp):
    # Moved from market_assistant.py
    try:
        text = f"{symbol} — {action}\nConfidence: {confidence}\nPrice: {price}\nTime: {timestamp}"
        send_message(text)
    except Exception as e:
        log.error(f"Failed to notify via Telegram: {e}", exc_info=True)

def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            log.info(f"Telegram message sent: {text}")
        else:
            log.error(f"Telegram API error {resp.status_code}: {resp.text}")
    except Exception as e:
        log.error(f"Failed to send Telegram message: {e}", exc_info=True)

def parse_command(text):
    parts = text.strip().split()
    if not parts:
        return None
    cmd = parts[0].lower()
    if cmd == '/bought' and len(parts) >= 3:
        symbol = parts[1].upper()
        qty = float(parts[2])
        price = float(parts[3]) if len(parts) > 3 else 0.0
        # If price not provided or zero, attempt to fetch latest close price
        if price == 0.0:
            try:
                df = fetch_data(symbol)
                if df is not None and not df.empty:
                    # use last Close value
                    price = float(df['Close'].iloc[-1])
                    log.info(f"Fetched market price for {symbol}: {price}")
                else:
                    log.warning(f"No market data available to derive price for {symbol}")
            except Exception as e:
                log.error(f"Error fetching price for {symbol}: {e}", exc_info=True)
                price = 0.0
        timestamp = datetime.utcnow().isoformat()
        record_position(symbol, qty, price, timestamp)
        return f"Recorded BUY: {symbol} qty={qty} price={price}"
    elif cmd == '/sold' and len(parts) >= 3:
        symbol = parts[1].upper()
        qty = float(parts[2])
        price = float(parts[3]) if len(parts) > 3 else 0.0
        # If price not provided or zero, attempt to fetch latest close price
        if price == 0.0:
            try:
                df = fetch_data(symbol)
                if df is not None and not df.empty:
                    price = float(df['Close'].iloc[-1])
                    log.info(f"Fetched market price for {symbol}: {price}")
                else:
                    log.warning(f"No market data available to derive price for {symbol}")
            except Exception as e:
                log.error(f"Error fetching price for {symbol}: {e}", exc_info=True)
                price = 0.0
        timestamp = datetime.utcnow().isoformat()
        update_position(symbol, -qty, price, timestamp)
        return f"Recorded SELL: {symbol} qty={qty} price={price}"
    elif cmd == '/positions':
        positions = get_open_positions()
        msg = "Open Positions:\n" + "\n".join([f"{p['symbol']}: qty={p['qty']} price={p['price']}" for p in positions])
        return msg
    elif cmd == '/research':
        scope_arg = parts[1].lower() if len(parts) > 1 else "w"
        if scope_arg.startswith("p"):
            scope = "portfolio"
        else:
            scope = "whole"
        limit = None
        if len(parts) > 2:
            try:
                limit = int(parts[2])
            except ValueError:
                log.warning(f"Invalid /research limit {parts[2]}, falling back to TOP_N")
        top_n = limit if limit and limit > 0 else TOP_N
        scan_result = perform_scan(scope=scope, top_n=top_n)
        persist_scan_results(scan_result)
        return format_summary_text(scan_result)
    return "Unknown command."
