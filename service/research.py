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
from infra.database import (
    fetch_position,
    record_position,
    record_trade_decision,
    update_position,
)
from infra.logging import log
from infra.market_hours import current_market_time, is_market_closed, market_time_str
from infra.monitor import record_snapshot, save_intraday_graph
from infra.actor_critic import RLAgent
from infra.user_store import adjust_wallet, ensure_user
STATE_DIM = 12
rl_agent = RLAgent(STATE_DIM)
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
    STOP_LOSS_PCT,
    PROFIT_TARGET_PCT,
    DEFAULT_TRADE_QTY,
)


def _load_symbol_universe() -> List[str]:
    try:
        df = pd.read_csv(SYMBOLS_FILE)
        symbols = [str(s).upper().strip() for s in df.get("symbol", []) if str(s).strip()]
        return symbols
    except Exception as exc:
        log.error(f"Failed to load symbols file {SYMBOLS_FILE}: {exc}", exc_info=True)
        return []


def _calculate_reward(action: str, price: float, f: Dict) -> float:
    session_high = f.get("session_high", price)
    session_low = f.get("session_low", price)
    if action == "BUY" and session_high:
        return max(0.0, (session_high - price) / price)
    if action == "SELL" and session_low:
        return max(0.0, (price - session_low) / price)
    return -0.001


def _resolve_rl_action(base_action: str, rl_label: str, allow_buy: bool, position_qty: float):
    if base_action == "BUY":
        return "BUY", False
    if base_action == "SELL":
        return "SELL", False
    if rl_label == "BUY" and allow_buy:
        return "BUY", True
    if rl_label in ("SELL", "ADJUST_STOP", "PARTIAL_EXIT") and position_qty > 0:
        return "SELL", True
    return "HOLD", False


def _rl_reward(final_action: str, price: float, entry_price: float, position_qty: float, wallet: float, f: Dict) -> float:
    reward = _calculate_reward(final_action, price, f)
    if final_action == "BUY":
        reward -= 0.01
    if final_action == "SELL":
        if position_qty <= 0:
            reward -= 1.0
        if entry_price:
            reward += (price - entry_price) / max(entry_price, 1.0)
            if price <= entry_price * (1 - STOP_LOSS_PCT):
                reward -= 0.3
            if price >= entry_price * (1 + PROFIT_TARGET_PCT):
                reward += 0.25
    reward += 0.001 * f.get("vol_spike", 0.0)
    reward += _safe_wallet_bonus(wallet, price)
    return reward


def _safe_wallet_bonus(wallet: float, price: float) -> float:
    if price <= 0 or wallet <= 0:
        return 0.0
    return min(wallet / (wallet + price), 0.05)


def _budgeted_top_n(wallet: Optional[float]) -> int:
    if wallet is None or wallet <= 0:
        return TOP_N
    factor = min(1.0, wallet / 10000.0)
    return max(1, int(TOP_N * (0.5 + 0.5 * factor)))


def perform_scan(
    scope: str = "whole",
    symbols: Optional[List[str]] = None,
    top_n: int = TOP_N,
    wallet: Optional[float] = None,
) -> Dict:
    """
    Run the indicator + sentiment scan across the provided symbol list.

    `scope` can be "whole" (default) to use the full universe or "portfolio"
    to restrict to open positions.
    """
    scan_time = current_market_time()
    timestamp = scan_time
    market_closed = is_market_closed(scan_time)
    market_time_display = market_time_str(scan_time)
    positions_list = get_open_positions()
    active_positions = {p["symbol"] for p in positions_list}
    position_lookup = {p["symbol"]: p for p in positions_list}
    allow_buy = scope != "portfolio"
    if wallet is not None and wallet < 100:
        allow_buy = False
    effective_top_n = _budgeted_top_n(wallet) if wallet is not None else top_n

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
    now_iso = scan_time.isoformat()
    last_prices: Dict[str, float] = {}
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
            last_prices[symbol] = price
            df_snapshot = df.tail(MONITOR_GRAPH_POINTS).copy()

            position_info = position_lookup.get(symbol, {})
            position_qty = max(position_info.get("qty", 0.0), 0.0)
            wallet_value = wallet or 0.0
            rl_state = rl_agent.build_state(f, position_info, wallet_value)
            rl_mask = rl_agent.build_mask(allow_buy, wallet_value, price, position_qty)
            rl_decision = rl_agent.select_action(rl_state, rl_mask)
            rl_label = rl_agent.action_label(rl_decision.action_id)

            reason = ""
            entry_price = position_info.get("price") or 0.0
            final_action, rl_used = _resolve_rl_action(
                action, rl_label, allow_buy, position_qty
            )

            if position_qty > 0 and entry_price:
                guard_reason = ""
                if price <= entry_price * (1 - STOP_LOSS_PCT):
                    guard_reason = "Stop-loss guardrail triggered"
                elif price >= entry_price * (1 + PROFIT_TARGET_PCT):
                    guard_reason = "Profit target guardrail triggered"
                if guard_reason:
                    final_action = "SELL"
                    reason = guard_reason if not reason else f"{guard_reason} | {reason}"

            if rl_used:
                if reason:
                    if "RL action" not in reason:
                        reason = f"{reason} | RL action {rl_label}"
                else:
                    reason = f"RL action {rl_label}"

            if final_action == "BUY" and not reason:
                reason = (
                    f"vol_spike={f.get('vol_spike', 0):.2f}, "
                    f"RSI={f['rsi']:.1f}, "
                    f"pct_from_low={f.get('pct_from_low',0):.1f}%"
                )

            rl_score_value = float(rl_decision.value.detach().item()) if rl_decision is not None else None
            if final_action == "BUY" and allow_buy:
                score = f.get("vol_spike", 0.0) + (f.get("rsi", 0.0) / 100)
                log.info(f"{symbol}: BUY reason={reason} score={score:.2f}")
                buy_candidates.append(
                    {
                        "symbol": symbol,
                        "price": price,
                        "confidence": confidence,
                        "score": score,
                        "trace_df": df_snapshot,
                        "reason": reason,
                        "rl_used": rl_used,
                        "rl_score": rl_score_value,
                    }
                )
            elif final_action == "SELL":
                graph_path = None
                if sell_graphs < MONITOR_MAX_SELL_GRAPHS:
                    graph_path = save_intraday_graph(symbol, df_snapshot, now_iso)
                    sell_graphs += 1
                log.info(f"{symbol}: SELL reason={reason} confidence={confidence}")
                if not reason:
                    reason = f"Sell signal from rules ({confidence})"
                sell_candidates.append(
                    {
                        "symbol": symbol,
                        "price": price,
                        "confidence": confidence,
                        "graph": graph_path,
                        "reason": reason,
                        "rl_used": rl_used,
                        "rl_score": rl_score_value,
                    }
                )
            elif final_action == "HOLD":
                hold_candidates.append(symbol)

            reward = _rl_reward(final_action, price, entry_price, position_qty, wallet_value, f)
            if rl_used:
                rl_agent.update(rl_decision, reward)
        except Exception as exc:
            log.error(f"{symbol}: scan error {exc}", exc_info=True)

    buy_candidates.sort(key=lambda entry: entry["score"], reverse=True)
    selected_buys = buy_candidates[:effective_top_n] if allow_buy else []
    filtered_buy_count = max(0, len(buy_candidates) - len(selected_buys))

    for cand in selected_buys:
        trace = cand.pop("trace_df", None)
        if trace is not None:
            cand["graph"] = save_intraday_graph(cand["symbol"], trace, now_iso)

    if market_closed:
        sold_symbols = {entry["symbol"] for entry in sell_candidates}
        for symbol in active_positions:
            if symbol in sold_symbols:
                continue
            price = last_prices.get(symbol)
            if price is None:
                continue
            sell_candidates.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "confidence": "Market closed – exit before next session",
                    "graph": None,
                }
            )

    return {
        "scope": scope,
        "symbols_scanned": processed,
        "buy_candidates": selected_buys,
        "sell_candidates": sell_candidates,
        "hold_candidates": hold_candidates,
        "filtered_buy_count": filtered_buy_count,
        "active_positions": sorted(active_positions),
        "market_closed": market_closed,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "timestamp_iso": timestamp.isoformat(),
        "market_time": market_time_display,
    }


def log_transaction(
    username: str,
    symbol: str,
    action: str,
    price: float,
    reason: str,
    quantity: float,
    reward: float,
    guardrail: Optional[str] = None,
    aligned: bool = False,
    rl_score: Optional[float] = None,
) -> None:
    ensure_user(username)
    position = fetch_position(symbol, username=username)
    position_id = position["id"] if position else None
    entry_price = position["entry_price"] if position else None
    if action == "SELL" and not position:
        log.warning(f"Skipping SELL log; no open position for {symbol} and {username}")
        return
    pnl = None
    pnl_pct = None
    if action == "SELL" and entry_price:
        pnl = (price - entry_price) * quantity
        pnl_pct = (price - entry_price) / entry_price
    record_trade_decision(
        symbol,
        action,
        price,
        {"reason": reason},
        username=username,
        position_id=position_id,
        quantity=int(quantity),
        pnl=pnl,
        pnl_pct=pnl_pct,
        reward=reward,
        rl_score=rl_score,
        guardrail=guardrail,
        aligned=int(aligned),
    )
    wallet_delta = price * quantity if action == "SELL" else -price * quantity
    adjust_wallet(username, wallet_delta)
    if action == "BUY":
        record_position(symbol, quantity, price, guardrail_reason=guardrail or reason, rl_confidence=rl_score, username=username)
    elif action == "SELL":
        update_position(symbol, -quantity, price, username=username)


def persist_scan_results(scan_result: Dict, username: str = "system") -> None:
    ensure_user(username)
    ts = scan_result.get("timestamp_iso") or datetime.utcnow().isoformat()
    sell_candidates = scan_result.get("sell_candidates", [])
    buy_candidates = scan_result.get("buy_candidates", [])
    for sell in sell_candidates:
        reason = sell.get("reason") or sell.get("confidence") or "Sell decision"
        log_transaction(
            username,
            sell["symbol"],
            "SELL",
            sell["price"],
            reason,
            DEFAULT_TRADE_QTY,
            reward=0.0,
            guardrail=sell.get("reason"),
            aligned=sell.get("rl_used", False),
            rl_score=sell.get("rl_score"),
        )
    for buy in buy_candidates:
        reason = buy.get("reason") or buy.get("confidence") or "Buy decision"
        log_transaction(
            username,
            buy["symbol"],
            "BUY",
            buy["price"],
            reason,
            DEFAULT_TRADE_QTY,
            reward=0.0,
            guardrail=buy.get("reason"),
            aligned=buy.get("rl_used", False),
            rl_score=buy.get("rl_score"),
        )
    log.debug(
        f"Persisted {len(sell_candidates)} SELLs and "
        f"{len(buy_candidates)} BUYs at {ts}"
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
            line = f"- {entry['symbol']} @ {entry['price']} ({entry['confidence']})"
            if entry.get("reason"):
                line += f" → {entry['reason']}"
            lines.append(line)
            if entry.get("graph"):
                lines.append(f"  Graph: {entry['graph']}")
    else:
        lines.append("SELL: none")

    buys = scan_result.get("buy_candidates", [])
    if buys:
        lines.append("BUY (top candidates):")
        for entry in buys:
            line = f"- {entry['symbol']} @ {entry['price']} ({entry['confidence']})"
            if entry.get("reason"):
                line += f" → {entry['reason']}"
            lines.append(line)
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
    if scan_result.get("market_closed"):
        lines.append(f"Market closed at {scan_result.get('market_time')}; exit before next session.")

    return "\n".join(lines)
