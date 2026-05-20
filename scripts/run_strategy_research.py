"""Run strategy research with four-stage validation and optional parallel execution."""

from __future__ import annotations

import argparse
import datetime
import subprocess
import time
from typing import Iterable

from predictor.research import (
    CSVHistoricalDataSource,
    ResearchValidationConfig,
    YFinanceHistoricalDataSource,
    build_donchian_variant_universe,
    build_literature_strategy_universe,
    build_strict_donchian_validation_config,
    compare_strategies_across_symbols,
    select_stable_donchian_rows,
)
from predictor.research.data import CachedDataSource
from predictor.research.harness import run_research
from predictor.research.library import build_core_strategy_universe
from predictor.research.reporting import (
    compute_failure_rates,
    compute_global_fail_reason_rates,
    summarise_data_availability,
)


def _parse_symbols(raw: str) -> tuple[str, ...]:
    symbols = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not symbols:
        raise ValueError("symbols must contain at least one symbol")
    return symbols


def _build_source(args: argparse.Namespace):
    source: object
    if args.source == "csv":
        source = CSVHistoricalDataSource(directory=args.data_dir)
    else:
        source = YFinanceHistoricalDataSource(
            interval=args.interval,
            lookback=args.lookback,
            exchange_suffix=args.exchange_suffix,
        )
    return CachedDataSource(source)


def _print_comparison_rows(rows: Iterable, label: str = "") -> None:
    if label:
        print(f"\n=== {label} ===")
    print(
        "rank,strategy,pass_rate,avg_return,avg_sharpe,avg_drawdown,avg_pf,"
        "avg_trade_count,avg_holding,avg_expectancy,avg_stability,score"
    )
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank},{row.strategy_name},{row.validation_pass_rate:.2f},"
            f"{row.avg_total_return:.4f},{row.avg_sharpe_ratio:.4f},"
            f"{row.avg_max_drawdown:.4f},{row.avg_profit_factor:.4f},"
            f"{row.avg_trade_count:.1f},{row.avg_holding_period:.1f},"
            f"{row.avg_expectancy:.4f},"
            f"{row.avg_walk_forward_stability:.4f},{row.composite_score:.4f}"
        )


def _resolve_config(args: argparse.Namespace) -> ResearchValidationConfig:
    if args.universe == "donchian" and args.strict:
        base = build_strict_donchian_validation_config()
    else:
        base = ResearchValidationConfig()

    return ResearchValidationConfig(
        bars_per_year=args.bars_per_year if args.bars_per_year is not None else base.bars_per_year,
        transaction_cost_bps=base.transaction_cost_bps,
        permutation_iterations=(
            args.permutation_iterations
            if args.permutation_iterations is not None
            else base.permutation_iterations
        ),
        permutation_block_size=base.permutation_block_size,
        p_value_threshold=base.p_value_threshold,
        train_window=args.train_window if args.train_window is not None else base.train_window,
        test_window=args.test_window if args.test_window is not None else base.test_window,
        walk_forward_step=args.walk_step if args.walk_step is not None else base.walk_forward_step,
        minimum_walk_forward_stability=base.minimum_walk_forward_stability,
        minimum_walk_forward_fold_pass_rate=base.minimum_walk_forward_fold_pass_rate,
        require_positive_walk_forward_return=base.require_positive_walk_forward_return,
        minimum_walk_forward_sharpe=base.minimum_walk_forward_sharpe,
        random_seed=args.random_seed if args.random_seed is not None else base.random_seed,
    )


def _build_strategies(args: argparse.Namespace):
    if args.universe == "donchian":
        return build_donchian_variant_universe()
    if args.universe == "core":
        return build_core_strategy_universe()
    return build_literature_strategy_universe()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-strategy validation and ranking research")
    parser.add_argument(
        "--universe",
        choices=("literature", "donchian", "core"),
        default="literature",
        help="Strategy set. 'core'=one per family (fast), 'literature'=full set, 'donchian'=variants only",
    )
    parser.add_argument("--strict", action="store_true", help="Use strict Donchian validation thresholds")
    parser.add_argument("--parallel", action="store_true", help="Parallel execution (run_research)")
    parser.add_argument("--workers", type=int, default=4, help="Thread count for parallel mode")
    parser.add_argument("--source", choices=("csv", "yfinance"), default="yfinance")
    parser.add_argument("--data-dir", default="data/research_samples", help="CSV data directory")
    parser.add_argument(
        "--symbols",
        default="RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,BHARTIARTL,LT,AXISBANK,KOTAKBANK",
        help="Comma-separated NSE symbols (without .NS suffix)",
    )
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--lookback", default="5y")
    parser.add_argument("--exchange-suffix", default=".NS")
    parser.add_argument("--permutation-iterations", type=int, default=None)
    parser.add_argument("--train-window", type=int, default=None)
    parser.add_argument("--test-window", type=int, default=None)
    parser.add_argument("--walk-step", type=int, default=None)
    parser.add_argument("--bars-per-year", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    data_source = _build_source(args)
    config = _resolve_config(args)
    strategies = _build_strategies(args)

# --- EXPERIMENT METADATA COLLECTION ---
    start_time = time.time()
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        git_hash = "unknown"
        
    print("=== EXPERIMENT METADATA ===")
    print(f"Timestamp    : {datetime.datetime.now().isoformat()}")
    print(f"Git Hash     : {git_hash}")
    print(f"Universe     : {args.universe} ({len(strategies)} strategies)")
    print(f"Symbols      : {len(symbols)} items")
    print(f"Source       : {args.source}")
    print(f"Random Seed  : {config.random_seed}")
    print(f"Permutations : {config.permutation_iterations}")
    print(f"Windows      : Train={config.train_window}, Test={config.test_window}, Step={config.walk_forward_step}")
    print("===========================\n")

    if args.parallel:
        result = run_research(
            symbols=symbols,
            data_source=data_source,
            strategies=strategies,
            config=config,
            max_workers=args.workers,
        )

        avail_summary = summarise_data_availability(result.data_availability)
        print(f"\ndata_availability: {avail_summary}")

        _print_comparison_rows(result.comparison.rows, label="STRATEGY RANKING")

        failure_rates = compute_failure_rates(result.symbol_robustness)
        print("\nfailure_rates_by_strategy:")
        for name, rate in sorted(failure_rates.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name}: {rate:.2%}")
            
        reason_rates = compute_global_fail_reason_rates(result.symbol_robustness)
        if reason_rates:
            print("\nglobal_fail_reasons:")
            for reason, rate in sorted(reason_rates.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {reason}: {rate:.2%}")

        if result.pruning.eliminated:
            print("\ndominated_strategies_eliminated:")
            for e in result.pruning.eliminated:
                print(f"  {e.strategy_name}: {e.reason}")

        comparison_rows = result.comparison.rows
    else:
        comparison = compare_strategies_across_symbols(
            symbols=symbols,
            data_source=data_source,
            strategies=strategies,
            config=config,
        )
        _print_comparison_rows(comparison.rows, label="STRATEGY RANKING")
        comparison_rows = comparison.rows

    if args.universe == "donchian" and args.strict:
        stable = select_stable_donchian_rows(comparison_rows)
        print("\nstrict_stable_candidates:")
        if not stable:
            print("none")
        else:
            for row in stable:
                print(
                    f"  {row.strategy_name} pass_rate={row.validation_pass_rate:.2f} "
                    f"avg_sharpe={row.avg_sharpe_ratio:.4f} "
                    f"stability={row.avg_walk_forward_stability:.4f}"
                )

    elapsed = time.time() - start_time
    print(f"\n[Finished in {elapsed:.2f} seconds]")


if __name__ == "__main__":
    main()
