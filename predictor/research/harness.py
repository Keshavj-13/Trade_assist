"""Multi-symbol research harness for fair strategy-family comparison."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Sequence, Tuple
import time

logger = logging.getLogger(__name__)

import numpy as np

from predictor.research.data import HistoricalDataSource
from predictor.research.errors import ResearchDataError, ResearchInputError
from predictor.research.metrics import finite_profit_factor
from predictor.research.reporting import build_symbol_robustness_rows
from predictor.research.strategies import TradingStrategy, to_strategy_tuple
from predictor.research.types import (
    DataAvailabilityReport,
    MultiSymbolComparisonResult,
    MultiSymbolStrategyRow,
    PruningResult,
    ResearchRunResult,
    StrategyValidationReport,
    SymbolRobustnessRow,
)
from predictor.research.validation import ResearchValidationConfig, validate_strategy


def _validate_symbols(symbols: Iterable[str]) -> Tuple[str, ...]:
    """Validate and deduplicate symbol universe for research runs."""
    if symbols is None:
        raise ResearchInputError("symbols must not be None")
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in symbols:
        if not isinstance(raw, str) or not raw.strip():
            continue
        symbol = raw.strip().upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    if not normalized:
        raise ResearchInputError("symbols must contain at least one non-empty symbol")
    return tuple(normalized)


def _aggregate_score(row: MultiSymbolStrategyRow) -> float:
    """Compute a stable objective score for multi-symbol ranking."""
    sharpe_component = np.tanh(row.avg_sharpe_ratio / 2.0)
    return_component = np.tanh(row.avg_total_return * 4.0)
    drawdown_component = 1.0 - min(1.0, abs(row.avg_max_drawdown))
    profit_component = np.tanh(finite_profit_factor(row.avg_profit_factor) / 4.0)
    robustness = 1.0 - ((row.avg_in_sample_p_value + row.avg_walk_forward_p_value) / 2.0)
    stability = row.avg_walk_forward_stability
    pass_rate = row.validation_pass_rate
    return float(
        0.18 * sharpe_component
        + 0.16 * return_component
        + 0.12 * drawdown_component
        + 0.10 * profit_component
        + 0.18 * robustness
        + 0.14 * stability
        + 0.12 * pass_rate
    )


def _build_strategy_row(
    strategy: TradingStrategy,
    reports: List[StrategyValidationReport],
    symbol_count: int,
) -> MultiSymbolStrategyRow:
    """Aggregate per-symbol validation reports into one ranked row."""
    raw = [r.resolved_raw_metrics for r in reports]
    total_returns = np.asarray([m.total_return for m in raw])
    sharpes = np.asarray([m.sharpe_ratio for m in raw])
    drawdowns = np.asarray([m.max_drawdown for m in raw])
    win_rates = np.asarray([m.win_rate for m in raw])
    profit_factors = np.asarray([m.profit_factor for m in raw])
    is_p = np.asarray([r.in_sample_permutation.p_value for r in reports])
    wf_p = np.asarray([r.walk_forward_permutation.p_value for r in reports])
    stabilities = np.asarray([r.walk_forward_stability for r in reports])
    pass_rate = float(np.mean([1.0 if r.is_valid else 0.0 for r in reports]))

    t_counts = np.asarray([m.trade_count for m in raw])
    t_holds = np.asarray([m.avg_holding_period for m in raw])
    t_exps = np.asarray([m.expectancy for m in raw])

    return MultiSymbolStrategyRow(
        strategy_name=strategy.name,
        symbols_tested=symbol_count,
        symbols_validated=len(reports),
        avg_total_return=float(np.mean(total_returns)),
        avg_sharpe_ratio=float(np.mean(sharpes)),
        avg_max_drawdown=float(np.mean(drawdowns)),
        avg_win_rate=float(np.mean(win_rates)),
        avg_profit_factor=float(np.mean(profit_factors)),
        avg_in_sample_p_value=float(np.mean(is_p)),
        avg_walk_forward_p_value=float(np.mean(wf_p)),
        avg_walk_forward_stability=float(np.mean(stabilities)),
        avg_trade_count=float(np.mean(t_counts)),
        avg_holding_period=float(np.mean(t_holds)),
        avg_expectancy=float(np.mean(t_exps)),
        validation_pass_rate=pass_rate,
        composite_score=0.0,
    )


def _rank_rows(rows: List[MultiSymbolStrategyRow]) -> Tuple[MultiSymbolStrategyRow, ...]:
    """Score and rank strategy rows by composite objective."""
    scored = [
        row.__class__(**{**row.__dict__, "composite_score": _aggregate_score(row)})
        for row in rows
    ]
    return tuple(sorted(scored, key=lambda r: r.composite_score, reverse=True))


def compare_strategies_across_symbols(
    *,
    symbols: Iterable[str],
    data_source: HistoricalDataSource,
    strategies: Sequence[TradingStrategy],
    config: ResearchValidationConfig,
) -> MultiSymbolComparisonResult:
    """Run the four-stage framework across symbols and aggregate strategy ranks.

    Sequential reference implementation — identical to the original API.
    Prefer run_research() for parallel execution and richer output.
    """
    symbol_tuple = _validate_symbols(symbols)
    strategy_tuple = to_strategy_tuple(strategies)
    rows: List[MultiSymbolStrategyRow] = []
    all_reports: List[Tuple[str, str, StrategyValidationReport]] = []

    for strategy in strategy_tuple:
        reports: List[StrategyValidationReport] = []
        for symbol in symbol_tuple:
            frame = data_source.fetch_ohlcv(symbol)
            report = validate_strategy(frame, strategy, config=config)
            reports.append(report)
            all_reports.append((strategy.name, symbol, report))
        rows.append(_build_strategy_row(strategy, reports, len(symbol_tuple)))

    return MultiSymbolComparisonResult(rows=_rank_rows(rows), reports=tuple(all_reports))


def _fetch_all_symbols(
    symbols: Tuple[str, ...],
    data_source: HistoricalDataSource,
    *,
    max_workers: int,
) -> Tuple[Dict[str, object], Tuple[DataAvailabilityReport, ...]]:
    """Fetch all symbols in parallel and return (frames_dict, availability_reports).

    frames_dict maps symbol -> pd.DataFrame for successful fetches.
    Failed symbols appear only in availability_reports with available=False.
    """
    import pandas as pd

    frames: Dict[str, object] = {}
    availability: List[DataAvailabilityReport] = []

    def _fetch(sym: str) -> Tuple[str, object, DataAvailabilityReport]:
        try:
            frame = data_source.fetch_ohlcv(sym)
            date_range = (
                f"{frame.index[0].date()} to {frame.index[-1].date()}"
                if not frame.empty
                else ""
            )
            report = DataAvailabilityReport(
                symbol=sym,
                available=True,
                row_count=len(frame),
                date_range=date_range,
                failure_reason=None,
            )
            print(f"[DATA] {sym} loaded:\nrows={len(frame)}\nstart={frame.index[0].date() if not frame.empty else 'N/A'}\nend={frame.index[-1].date() if not frame.empty else 'N/A'}\ntimezone={frame.index.tz if not frame.empty and hasattr(frame.index, 'tz') else 'N/A'}\n")
            return sym, frame, report
        except Exception as exc:
            report = DataAvailabilityReport(
                symbol=sym,
                available=False,
                row_count=0,
                date_range="",
                failure_reason=str(exc),
            )
            print(f"[DATA] {sym} failed:\nreason={str(exc)}\n")
            return sym, None, report

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sym, frame, report in executor.map(_fetch, symbols):
            availability.append(report)
            if frame is not None:
                frames[sym] = frame

    return frames, tuple(availability)


def _validate_task(
    frame: object,
    strategy: TradingStrategy,
    config: ResearchValidationConfig,
) -> StrategyValidationReport | None:
    """Run validate_strategy and return None on exception (not a code bug)."""
    try:
        return validate_strategy(frame, strategy, config=config)  # type: ignore[arg-type]
    except Exception:
        return None


def run_research(
    *,
    symbols: Iterable[str],
    data_source: HistoricalDataSource,
    strategies: Sequence[TradingStrategy],
    config: ResearchValidationConfig,
    max_workers: int = 4,
) -> ResearchRunResult:
    """Parallel research run with data caching, robustness reporting, and pruning.

    Steps:
    1. Pre-fetch all symbols in parallel (max_workers threads).
       Symbols that fail are recorded in DataAvailabilityReport and skipped.
    2. Validate each (strategy, symbol) pair in parallel.
       Each validation task is independent: frozen strategy, seeded RNG,
       read-only frame. Results are deterministic regardless of task order.
    3. Aggregate per-strategy rows and rank by composite score.
    4. Apply lightweight dominance pruning on the final ranked rows.

    Parallelism is safe because:
    - Strategy instances are frozen dataclasses with no shared state.
    - Each validate_strategy call creates its own seeded RNG instances.
    - DataFrames are only read (never mutated) inside validation.
    - Aggregation happens after all futures complete, in strategy order.
    """
    from predictor.research.pruning import prune_dominated_strategies

    symbol_tuple = _validate_symbols(symbols)
    strategy_tuple = to_strategy_tuple(strategies)

    # Step 1: parallel data fetch
    frames, availability = _fetch_all_symbols(
        symbol_tuple, data_source, max_workers=max_workers
    )
    available_symbols = [sym for sym in symbol_tuple if sym in frames]

    if not available_symbols:
        empty_comparison = MultiSymbolComparisonResult(rows=())
        empty_pruning = PruningResult(kept=(), eliminated=())
        return ResearchRunResult(
            comparison=empty_comparison,
            symbol_robustness=(),
            data_availability=availability,
            pruning=empty_pruning,
        )

    # Step 2: parallel (strategy, symbol) validation
    tasks: List[Tuple[TradingStrategy, str]] = [
        (strategy, sym)
        for strategy in strategy_tuple
        for sym in available_symbols
    ]

    # Collect per-strategy report lists (order within each list is non-deterministic
    # but that is fine — aggregation uses np.mean which is order-independent).
    reports_by_strategy: Dict[str, List[StrategyValidationReport]] = {
        st.name: [] for st in strategy_tuple
    }
    raw_robustness: List[Tuple[str, str, StrategyValidationReport]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_validate_task, frames[sym], strategy, config): (strategy, sym)
            for strategy, sym in tasks
        }
        
        total_tasks = len(tasks)
        completed_tasks = 0
        start_time = time.time()
        
        for future in as_completed(future_to_task):
            completed_tasks += 1
            strategy, sym = future_to_task[future]
            report = future.result()
            if report is not None:
                reports_by_strategy[strategy.name].append(report)
                raw_robustness.append((strategy.name, sym, report))
                
                # Log folds if not valid
                if not report.is_valid:
                    print(f"[REJECTION] {strategy.name} on {sym} failed: {', '.join(report.fail_reasons)}")
                
                # Fold level logging
                for fold in report.walk_forward_folds:
                    print(f"[WF] {strategy.name} {sym} Fold {fold.split_index + 1}:")
                    print(f"IS Sharpe={fold.run.metrics.sharpe_ratio:.2f}")
                    # Estimate WF Sharpe
                    wf_sharpe = report.walk_forward_aggregate.metrics.sharpe_ratio # Simple overall
                    print(f"Trades={fold.run.metrics.trade_count}")
                    print(f"P_IS={fold.permutation.p_value:.2f}")
                    print("")
            
            elapsed = time.time() - start_time
            avg_time = elapsed / completed_tasks
            eta = avg_time * (total_tasks - completed_tasks)
            pct = int((completed_tasks / total_tasks) * 100)
            print(f"[{pct:02d}%] Strategy {strategy.name} | Symbol {sym} | ETA={eta/60:.1f}m")

    # Step 3: aggregate and rank (strategy order is deterministic)
    rows: List[MultiSymbolStrategyRow] = []
    for strategy in strategy_tuple:
        reports = reports_by_strategy[strategy.name]
        if not reports:
            continue
        rows.append(_build_strategy_row(strategy, reports, len(symbol_tuple)))

    all_reports = tuple(raw_robustness)
    ranked = _rank_rows(rows)
    comparison = MultiSymbolComparisonResult(rows=ranked, reports=all_reports)

    # Step 4: dominance pruning on ranked rows
    pruning = prune_dominated_strategies(ranked)

    robustness_rows = build_symbol_robustness_rows(raw_robustness)

    return ResearchRunResult(
        comparison=comparison,
        symbol_robustness=robustness_rows,
        data_availability=availability,
        pruning=pruning,
        reports=all_reports,
    )
