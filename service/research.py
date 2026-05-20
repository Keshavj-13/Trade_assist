"""Predictor-focused research service adapter.

Real runtime path for predictor execution:
`main.py` -> `service.runner.run_once` -> `service.research.perform_scan`
-> `predictor.pipeline.PredictorPipeline.run`.

This module translates predictor outputs into legacy-compatible payloads while
keeping side effects behind explicit opt-in feature gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from config.settings import SYMBOLS_FILE, TOP_N
from infra.logging import log
from infra.market_hours import current_market_time, is_market_closed, market_time_str
from predictor import PredictorConfig, PredictorPipeline
from predictor.data import make_default_data_fetcher
from predictor.types import PipelineRun, Prediction
from service.database import get_open_positions


Candidate = Dict[str, Any]
ScanResult = Dict[str, Any]


@lru_cache(maxsize=1)
def _get_default_data_fetcher() -> Callable[[str], pd.DataFrame]:
    """Build and cache the default market-data fetcher on first use."""

    return make_default_data_fetcher()


@dataclass(frozen=True)
class ScanContext:
    """Resolved execution context for a predictor scan."""

    scope: str
    scan_time: datetime
    market_closed: bool
    market_time_display: str
    active_positions: Sequence[str]
    allow_buy: bool
    target_symbols: Sequence[str]
    effective_top_n: int


def _persistence_enabled() -> bool:
    """Return whether non-predictor persistence side effects are enabled."""

    return os.environ.get("FIN_ASSIST_ENABLE_PERSISTENCE", "0") == "1"


def _validate_scope(scope: str) -> str:
    """Validate scope argument for scan execution."""

    if scope not in {"whole", "portfolio"}:
        raise ValueError(f"Invalid scope '{scope}', expected 'whole' or 'portfolio'")
    return scope


def _validate_top_n(top_n: int) -> int:
    """Validate user-provided buy candidate cap."""

    if not isinstance(top_n, int):
        raise ValueError("top_n must be an integer")
    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    return top_n


def _validate_wallet(wallet: Optional[float]) -> Optional[float]:
    """Validate optional wallet input."""

    if wallet is None:
        return None
    if wallet < 0:
        raise ValueError("wallet must be >= 0")
    return float(wallet)


def _validate_data_fetcher(data_fetcher: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Validate injected data fetch callable."""

    if not callable(data_fetcher):
        raise ValueError("data_fetcher must be callable")
    return data_fetcher


def _load_symbol_universe() -> List[str]:
    """Load symbol universe from configured CSV file."""

    df = pd.read_csv(SYMBOLS_FILE)
    if "symbol" in df.columns:
        source = df["symbol"]
    else:
        source = df.iloc[:, 0] if not df.empty else []

    return [str(value).upper().strip() for value in source if str(value).strip()]


def _budgeted_top_n(base_top_n: int, wallet: Optional[float]) -> int:
    """Scale candidate cap from wallet budget without introducing randomness."""

    if wallet is None:
        return base_top_n
    if wallet <= 0:
        return max(1, int(base_top_n * 0.5))
    factor = min(1.0, wallet / 10_000.0)
    return max(1, int(base_top_n * (0.5 + 0.5 * factor)))


def _load_open_position_symbols() -> List[str]:
    """Load open position symbols from persistence boundary."""

    if not _persistence_enabled():
        return []

    rows = get_open_positions()
    return [str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")]


def _resolve_target_symbols(
    *,
    scope: str,
    symbols: Optional[Iterable[str]],
    active_positions: Iterable[str],
) -> List[str]:
    """Resolve scan universe by request scope and explicit symbol override."""

    if symbols is not None:
        return [str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()]
    if scope == "portfolio":
        return sorted({str(symbol).upper().strip() for symbol in active_positions if symbol})
    return _load_symbol_universe()


def _resolve_scan_context(
    *,
    scope: str,
    symbols: Optional[List[str]],
    top_n: int,
    wallet: Optional[float],
) -> ScanContext:
    """Build fully resolved scan context from inputs and environment."""

    scan_time = current_market_time()
    active_positions = _load_open_position_symbols()
    allow_buy = scope != "portfolio"
    if wallet is not None and wallet < 100:
        allow_buy = False

    target_symbols = _resolve_target_symbols(
        scope=scope,
        symbols=symbols,
        active_positions=active_positions,
    )
    if not target_symbols:
        log.warning("No symbols available to scan.")

    return ScanContext(
        scope=scope,
        scan_time=scan_time,
        market_closed=is_market_closed(scan_time),
        market_time_display=market_time_str(scan_time),
        active_positions=sorted(active_positions),
        allow_buy=allow_buy,
        target_symbols=target_symbols,
        effective_top_n=_budgeted_top_n(top_n, wallet),
    )


def _prediction_to_candidate(prediction: Prediction) -> Candidate:
    """Convert predictor output into legacy candidate response payload."""

    confidence = (
        f"p_buy={prediction.buy_probability:.3f}, "
        f"p_sell={prediction.sell_probability:.3f}, "
        f"p_hold={prediction.hold_probability:.3f}, "
        f"u={prediction.uncertainty:.3f}, "
        f"regime={prediction.regime.value}"
    )
    return {
        "symbol": prediction.symbol,
        "price": round(prediction.price, 2),
        "confidence": confidence,
        "reason": " | ".join(prediction.reasons),
        "uncertainty": prediction.uncertainty,
        "regime": prediction.regime.value,
        "probabilities": {
            "buy": prediction.buy_probability,
            "sell": prediction.sell_probability,
            "hold": prediction.hold_probability,
        },
    }


def _map_predictions(run: PipelineRun) -> tuple[list[Candidate], list[Candidate], list[str]]:
    """Map pipeline predictions to legacy candidate structures."""

    buy_candidates = [_prediction_to_candidate(pred) for pred in run.buy_candidates]
    sell_candidates = [_prediction_to_candidate(pred) for pred in run.sell_candidates]
    hold_candidates = [pred.symbol for pred in run.hold_candidates]
    return buy_candidates, sell_candidates, hold_candidates


def _enforce_buy_policy(allow_buy: bool, buy_candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Apply buy policy and report number of filtered candidates."""

    if allow_buy:
        return buy_candidates, 0
    return [], len(buy_candidates)


def _market_close_exit_candidates(
    market_closed: bool,
    active_positions: Sequence[str],
    existing_sell_candidates: Sequence[Candidate],
) -> list[Candidate]:
    """Create market-close exit candidates for open positions not already sold."""

    if not market_closed:
        return []

    sold_symbols = {entry["symbol"] for entry in existing_sell_candidates}
    return [
        {
            "symbol": symbol,
            "price": None,
            "confidence": "Market closed - flattening position",
            "reason": "Market closed risk control",
            "uncertainty": 0.0,
            "regime": "unknown",
            "probabilities": {"buy": 0.0, "sell": 1.0, "hold": 0.0},
        }
        for symbol in active_positions
        if symbol not in sold_symbols
    ]


def _empty_run_payload(
    *,
    context: ScanContext,
    skipped_symbols: Optional[Mapping[str, str]] = None,
) -> ScanResult:
    """Create baseline payload when no symbols are available for inference."""

    return {
        "scope": context.scope,
        "symbols_scanned": len(context.target_symbols),
        "buy_candidates": [],
        "sell_candidates": _market_close_exit_candidates(
            context.market_closed,
            context.active_positions,
            [],
        ),
        "hold_candidates": [],
        "filtered_buy_count": 0,
        "active_positions": list(context.active_positions),
        "market_closed": context.market_closed,
        "timestamp": context.scan_time.strftime("%Y-%m-%d %H:%M"),
        "timestamp_iso": context.scan_time.isoformat(),
        "market_time": context.market_time_display,
        "skipped_symbols": dict(skipped_symbols or {}),
    }


def _build_scan_result(context: ScanContext, run: PipelineRun) -> ScanResult:
    """Assemble final service scan payload from context and pipeline output."""

    buy_candidates, sell_candidates, hold_candidates = _map_predictions(run)
    buy_candidates, filtered_buy_count = _enforce_buy_policy(context.allow_buy, buy_candidates)

    sell_candidates = sell_candidates + _market_close_exit_candidates(
        context.market_closed,
        context.active_positions,
        sell_candidates,
    )

    return {
        "scope": context.scope,
        "symbols_scanned": len(context.target_symbols),
        "buy_candidates": buy_candidates,
        "sell_candidates": sell_candidates,
        "hold_candidates": hold_candidates,
        "filtered_buy_count": filtered_buy_count,
        "active_positions": list(context.active_positions),
        "market_closed": context.market_closed,
        "timestamp": context.scan_time.strftime("%Y-%m-%d %H:%M"),
        "timestamp_iso": context.scan_time.isoformat(),
        "market_time": context.market_time_display,
        "skipped_symbols": run.skipped,
    }


def perform_scan(
    scope: str = "whole",
    symbols: Optional[List[str]] = None,
    top_n: int = TOP_N,
    wallet: Optional[float] = None,
    *,
    data_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    cross_asset_consensus: Optional[Mapping[str, float]] = None,
) -> ScanResult:
    """Run predictor pipeline over the requested symbol universe.

    Args:
        scope: Scan scope (`whole` or `portfolio`).
        symbols: Optional explicit symbol list.
        top_n: Maximum number of BUY candidates.
        wallet: Optional wallet amount for top-N budgeting and buy enablement.
        data_fetcher: Injectable OHLCV data provider for tests.
        cross_asset_consensus: Optional symbol->consensus signal in [-1, 1].

    Returns:
        Legacy-compatible dictionary result for service callers.
    """

    scope = _validate_scope(scope)
    top_n = _validate_top_n(top_n)
    wallet = _validate_wallet(wallet)
    data_fetcher = _validate_data_fetcher(data_fetcher or _get_default_data_fetcher())

    context = _resolve_scan_context(
        scope=scope,
        symbols=symbols,
        top_n=top_n,
        wallet=wallet,
    )
    if not context.target_symbols:
        return _empty_run_payload(context=context)

    pipeline = PredictorPipeline(
        data_fetcher=data_fetcher,
        config=PredictorConfig.from_legacy_settings(),
    )
    run = pipeline.run(
        context.target_symbols,
        open_positions=context.active_positions,
        top_n=context.effective_top_n,
        cross_asset_consensus=cross_asset_consensus,
    )
    return _build_scan_result(context, run)


def _persist_candidate_action(
    *,
    username: str,
    action: str,
    candidate: Candidate,
) -> None:
    """Persist one BUY/SELL candidate as a transaction."""

    price = candidate.get("price")
    if price is None:
        return
    reason = candidate.get("reason") or candidate.get("confidence") or f"{action} decision"

    log_transaction(
        username,
        str(candidate["symbol"]),
        action,
        float(price),
        str(reason),
        quantity=1.0,
        reward=0.0,
        guardrail=str(candidate.get("reason")) if candidate.get("reason") is not None else None,
        aligned=False,
        rl_score=None,
    )


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
    """Persist a decision transaction when persistence is explicitly enabled."""

    if not _persistence_enabled():
        log.info("Persistence disabled; skipping log_transaction side effect")
        return

    from infra.database import (
        fetch_position,
        record_position,
        record_trade_decision,
        update_position,
    )
    from infra.user_store import adjust_wallet, ensure_user

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
        record_position(
            symbol,
            quantity,
            price,
            guardrail_reason=guardrail or reason,
            rl_confidence=rl_score,
            username=username,
        )
    elif action == "SELL":
        update_position(symbol, -quantity, price, username=username)


def persist_scan_results(scan_result: ScanResult, username: str = "system") -> None:
    """Persist scan decisions if side effects are enabled."""

    if not _persistence_enabled():
        log.info("Persistence disabled; skipping persist_scan_results")
        return

    sell_candidates = scan_result.get("sell_candidates", [])
    buy_candidates = scan_result.get("buy_candidates", [])

    for sell_candidate in sell_candidates:
        _persist_candidate_action(username=username, action="SELL", candidate=sell_candidate)
    for buy_candidate in buy_candidates:
        _persist_candidate_action(username=username, action="BUY", candidate=buy_candidate)

    timestamp_iso = scan_result.get("timestamp_iso") or datetime.utcnow().isoformat()
    log.debug(
        f"Persisted {len(sell_candidates)} SELLs and "
        f"{len(buy_candidates)} BUYs at {timestamp_iso}"
    )


def format_summary_text(scan_result: ScanResult) -> str:
    """Format a concise human-readable summary of scan output."""

    lines = []
    scope_label = "Portfolio" if scan_result.get("scope") == "portfolio" else "Full universe"
    lines.append(f"{scope_label} predictor scan @ {scan_result.get('timestamp')}")
    lines.append(f"Symbols processed: {scan_result.get('symbols_scanned', 0)}")

    sells = scan_result.get("sell_candidates", [])
    if sells:
        lines.append("SELL:")
        for entry in sells:
            lines.append(
                f"- {entry['symbol']} @ {entry.get('price')} "
                f"({entry.get('confidence', 'n/a')})"
            )
    else:
        lines.append("SELL: none")

    buys = scan_result.get("buy_candidates", [])
    if buys:
        lines.append("BUY (top candidates):")
        for entry in buys:
            lines.append(
                f"- {entry['symbol']} @ {entry['price']} "
                f"({entry.get('confidence', 'n/a')})"
            )
    else:
        lines.append("BUY: none")

    holds = scan_result.get("hold_candidates", [])
    lines.append("HOLD:" if holds else "HOLD: none")
    for symbol in holds:
        lines.append(f"- {symbol}")

    extra = scan_result.get("filtered_buy_count", 0)
    if extra:
        lines.append(f"{extra} buy candidates filtered after TOP_N cap.")
    if scan_result.get("market_closed"):
        lines.append(
            f"Market closed at {scan_result.get('market_time')}; "
            "positions are marked for exit control."
        )

    skipped = scan_result.get("skipped_symbols") or {}
    if skipped:
        lines.append(f"Skipped symbols: {len(skipped)}")

    return "\n".join(lines)
