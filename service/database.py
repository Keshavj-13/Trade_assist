"""
Database module for position tracking.
"""
from infra.database import initialize_db, get_open_positions as _get_open_positions, record_position as _record_position
from datetime import datetime
from infra.logging import log


def init_db():
    initialize_db()


def record_position(symbol, qty, price, timestamp=None):
    # Recalculate average price when adding to existing position
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    try:
        existing = {p['symbol']: p for p in _get_open_positions()}
        if symbol in existing:
            ex = existing[symbol]
            ex_qty = float(ex.get('qty', 0))
            ex_price = float(ex.get('price', 0))
            total_qty = ex_qty + qty
            if total_qty <= 0:
                # replace directly
                _record_position(symbol, 0, 0.0, timestamp)
                log.info(f"Position zeroed for {symbol}")
                return
            avg_price = (ex_qty * ex_price + qty * price) / total_qty
            _record_position(symbol, total_qty, avg_price, timestamp)
            log.info(f"Updated avg price for {symbol}: qty={total_qty} avg_price={avg_price}")
        else:
            _record_position(symbol, qty, price, timestamp)
            log.info(f"Recorded new position for {symbol}: qty={qty} price={price}")
    except Exception as e:
        log.error(f"Failed to record position {symbol}: {e}", exc_info=True)


def get_open_positions():
    try:
        res = _get_open_positions()
        # filter qty > 0 and ensure dicts
        out = [p for p in res if float(p.get('qty', 0)) > 0]
        return out
    except Exception as e:
        log.error(f"Failed to fetch open positions: {e}", exc_info=True)
        return []
