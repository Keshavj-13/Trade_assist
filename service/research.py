"""
Market scan helper shared by runner and Telegram commands.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from core.data_fetch import fetch_data
from core.decision_engine import decide
from core.indicators import compute_features
from core.news_sentiment import fetch_news, finbert_sentiment
from infra.database import record_trade_decision
from infra.logging import log
from infra.monitor import record_snapshot, save_intraday_graph
from service.database import get_open_positions
from config.settings import (
    SYMBOLS_FILE,
    MIN_PRICE,
    MIN_AVG_VOLUME,
    MIN_ATR_PCT,
    MAX_ATR_PCT,
    TOP_N,
    MONITOR_GRAPH_POINTS,
    MONITOR_MAX_SELL_GRAPHS,
)


def _load_symbol_universe() -> List[str]:
    try:
        df = pd.read_csv(SYMBOLS_FILE)
        symbols = [str(s).upper().strip() for s in df.get("symbol", []) if str(s).strip()]
        return symbols
    except Exception as exc:
        log.error(f"Failed to load symbols file {SYMBOLS_FILE}: {exc}", exc_info=True)
        return []


def perform_scan(
    scope: str = "whole",
    symbols: Optional[List[str]] = None,
    top_n: int = TOP_N,
) -> Dict:
    """
    Run the indicator + sentiment scan across the provided symbol list.

    `scope` can be "whole" (default) to use the full universe or "portfolio"
    to restrict to open positions.
    """
    timestamp = datetime.utcnow()
    active_positions = {p["symbol"] for p in get_open_positions()}
    allow_buy = scope != "portfolio"

    if symbols is not None:
        target_symbols = [s.upper().strip() for s in symbols if s]
    elif scope == "portfolio":
        target_symbols = sorted(active_positions)
    else:
        target_symbols = _load_symbol_universe()

    if not target_symbols:
        log.warning("No symbols available to scan.")

    buy_candidates: List[Dict] = []
    sell_candidates: List[Dict] = []
    hold_candidates: List[str] = []
    processed = 0

    sell_graphs = 0
    now_iso = timestamp.isoformat()
    for symbol in target_symbols:
        processed += 1
        try:
            df = fetch_data(symbol)
            f = compute_features(df)
            if not f:
                log.debug(f"Skipping {symbol}: insufficient data")
                continue
            if f["price"] < MIN_PRICE:
                log.debug(f"Skipping {symbol}: price {f['price']} < MIN_PRICE")
                continue
            if f["avg_volume"] < MIN_AVG_VOLUME:
                log.debug(f"Skipping {symbol}: avg_volume {f['avg_volume']} < MIN_AVG_VOLUME")
                continue
            if not (MIN_ATR_PCT <= f["atr_pct"] <= MAX_ATR_PCT):
                log.debug(f"Skipping {symbol}: atr_pct {f['atr_pct']} not in range")
                continue
            snapshot_stats = {
                "price": f.get("price"),
                "session_low": f.get("session_low"),
                "session_high": f.get("session_high"),
                "vwap": f.get("vwap"),
                "rsi": f.get("rsi"),
                "atr_pct": f.get("atr_pct"),
                "avg_volume": f.get("avg_volume"),
                "vol_spike": f.get("vol_spike"),
                "pct_from_low": f.get("pct_from_low"),
                "pct_from_high": f.get("pct_from_high"),
            }
            record_snapshot(symbol, snapshot_stats, now_iso)
            sentiment = finbert_sentiment(fetch_news(symbol))
            action = decide(symbol, f, sentiment, open_positions=active_positions)
            confidence = f"rsi={f['rsi']}, atr_pct={f['atr_pct']}, sentiment={sentiment}"
            price = round(f["price"], 2)
            df_snapshot = df.tail(MONITOR_GRAPH_POINTS).copy()
            if action == "BUY" and allow_buy:
                score = f.get("vol_spike", 0.0) + (f.get("rsi", 0.0) / 100)
                buy_candidates.append(
                    {
                        "symbol": symbol,
                        "price": price,
                        "confidence": confidence,
                        "score": score,
                        "trace_df": df_snapshot,
                    }
                )
            elif action == "SELL":
                graph_path = None
                if sell_graphs < MONITOR_MAX_SELL_GRAPHS:
                    graph_path = save_intraday_graph(symbol, df_snapshot, now_iso)
                    sell_graphs += 1
                sell_candidates.append(
                    {"symbol": symbol, "price": price, "confidence": confidence, "graph": graph_path}
                )
            elif action == "HOLD":
                hold_candidates.append(symbol)
        except Exception as exc:
            log.error(f"{symbol}: scan error {exc}", exc_info=True)

    buy_candidates.sort(key=lambda entry: entry["score"], reverse=True)
    selected_buys = buy_candidates[:top_n] if allow_buy else []
    filtered_buy_count = max(0, len(buy_candidates) - len(selected_buys))

    for cand in selected_buys:
        trace = cand.pop("trace_df", None)
        if trace is not None:
            cand["graph"] = save_intraday_graph(cand["symbol"], trace, now_iso)

    return {
        "scope": scope,
        "symbols_scanned": processed,
        "buy_candidates": selected_buys,
        "sell_candidates": sell_candidates,
        "hold_candidates": hold_candidates,
        "filtered_buy_count": filtered_buy_count,
        "active_positions": sorted(active_positions),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "timestamp_iso": timestamp.isoformat(),
    }


def persist_scan_results(scan_result: Dict) -> None:
    ts = scan_result.get("timestamp_iso") or datetime.utcnow().isoformat()
    for sell in scan_result.get("sell_candidates", []):
        record_trade_decision(sell["symbol"], "SELL", sell["price"], sell["confidence"])
    for buy in scan_result.get("buy_candidates", []):
        record_trade_decision(buy["symbol"], "BUY", buy["price"], buy["confidence"])
    log.debug(
        f"Persisted {len(scan_result.get('sell_candidates', []))} SELLs and "
        f"{len(scan_result.get('buy_candidates', []))} BUYs at {ts}"
    )


def format_summary_text(scan_result: Dict) -> str:
    lines = []
    scope_label = "Portfolio" if scan_result.get("scope") == "portfolio" else "Full universe"
    lines.append(f"{scope_label} research @ {scan_result.get('timestamp')}")
    lines.append(f"Symbols processed: {scan_result.get('symbols_scanned', 0)}")

    sells = scan_result.get("sell_candidates", [])
    if sells:
        lines.append("SELL:")
        for entry in sells:
            lines.append(f"- {entry['symbol']} @ {entry['price']} ({entry['confidence']})")
            if entry.get("graph"):
                lines.append(f"  Graph: {entry['graph']}")
    else:
        lines.append("SELL: none")

    buys = scan_result.get("buy_candidates", [])
    if buys:
        lines.append("BUY (top candidates):")
        for entry in buys:
            lines.append(f"- {entry['symbol']} @ {entry['price']} ({entry['confidence']})")
            if entry.get("graph"):
                lines.append(f"  Graph: {entry['graph']}")
    else:
        lines.append("BUY: none")

    holds = scan_result.get("hold_candidates", [])
    if holds:
        lines.append("HOLD:")
        lines.extend([f"- {sym}" for sym in holds])
    else:
        lines.append("HOLD: none")

    extra = scan_result.get("filtered_buy_count", 0)
    if extra:
        lines.append(f"{extra} buy candidates filtered after TOP_N cap.")

    return "\n".join(lines)
