"""Run strategy research with four-stage validation and optional parallel execution."""

from __future__ import annotations

import argparse
import datetime
import subprocess
import time
from pathlib import Path
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
    build_baseline_universe,
    build_hypothesis_universe,
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
        strategies = build_donchian_variant_universe()
    elif args.universe == "core":
        strategies = build_core_strategy_universe()
    elif args.universe == "baseline":
        strategies = build_baseline_universe()
    elif args.universe == "hypothesis":
        strategies = build_hypothesis_universe()
    else:
        strategies = build_literature_strategy_universe()

    if getattr(args, "baseline", False) and args.universe != "baseline":
        baselines = build_baseline_universe()
        # Merge lists, keeping them unique
        strategies = baselines + tuple(strategies)

    return strategies


def _run_debug_trace(
    args: argparse.Namespace,
    symbols: tuple,
    data_source: object,
    strategies: tuple,
    config: ResearchValidationConfig,
) -> None:
    """Execute single-strategy trace mode and print full diagnostics.

    Fetches data for one symbol, locates the requested strategy by name,
    and delegates to run_strategy_trace() for exhaustive step-by-step output.
    Optionally exports the IS permutation distribution to JSON.
    """
    from predictor.research.diagnostics import run_strategy_trace, export_permutation_distribution

    target_name: str = args.debug_strategy
    target_symbol: str = args.debug_symbol if args.debug_symbol else symbols[0]

    # Locate strategy by name
    matched = [s for s in strategies if getattr(s, "name", None) == target_name]
    if not matched:
        available = [getattr(s, "name", "?") for s in strategies]
        print(f"[ERROR] Strategy '{target_name}' not found in universe '{args.universe}'.")
        print(f"  Available: {available}")
        return
    strategy = matched[0]

    print(f"[DEBUG] Fetching data for {target_symbol} ...")
    frame = data_source.fetch_ohlcv(target_symbol)
    print(f"[DEBUG] Rows fetched: {len(frame)}")

    report = run_strategy_trace(
        frame,
        strategy,
        config=config,
        symbol=target_symbol,
    )

    # Optional: export permutation distribution
    if args.export_permutation_dist is not None:
        from predictor.research.types import PermutationTestResult
        # Reconstruct a PermutationTestResult from the trace report summary
        # We need to re-run the actual permutation to get raw distribution.
        # The trace already ran it; export via dedicated function.
        print(
            "[INFO] --export-permutation-dist is only available when running "
            "run_strategy_trace via the diagnostics API directly. "
            "Use predictor.research.diagnostics.export_permutation_distribution() "
            "in a script for raw null-distribution export."
        )

    print("\n[DEBUG TRACE SUMMARY]")
    print(f"  strategy          : {report.strategy_name}")
    print(f"  symbol            : {report.symbol}")
    print(f"  bars_processed    : {report.bars_processed}")
    print(f"  positions_nonzero : {report.signal_decomposition.positions_nonzero}")
    print(f"  entries_generated : {report.signal_decomposition.entries_generated}")
    print(f"  exits_generated   : {report.signal_decomposition.exits_generated}")
    print(f"  trade_count       : {report.trade_count}")
    print(f"  is_sharpe         : {report.is_sharpe:.4f}")
    print(f"  perm_mean         : {report.permutation_summary.mean:.4f}")
    print(f"  perm_95pct        : {report.permutation_summary.pct95:.4f}")
    print(f"  p_value           : {report.permutation_summary.p_value:.4f}")
    print(f"  threshold         : {report.permutation_summary.threshold:.4f}")
    print(f"  rejection_reason  : {report.rejection_reason or 'none (passed)'}")
    print(f"  stages_executed   : {[s.label() for s in report.stages_executed]}")

    if report.trade_count == 0 and report.signal_decomposition.entries_generated == 0:
        print(
            "\n[DIAGNOSIS] No entries generated. Likely causes:\n"
            "  1. Frame too short for lookback window (warmup period not reached)\n"
            "  2. Price never breaches Donchian channel levels\n"
            "  3. ATR filter is excluding all entry signals\n"
            "  4. Dependency rule blocking all entries\n"
            "  Run with a longer --lookback or inspect price vs channel levels."
        )
    elif report.trade_count == 0 and report.signal_decomposition.entries_generated > 0:
        print(
            "\n[DIAGNOSIS] Entries generated but zero trades extracted.\n"
            "  This is an IMPOSSIBLE STATE — check assert_impossible_states() output above."
        )
    elif report.is_rejected:
        if report.is_sharpe <= 0.0:
            print(
                "\n[DIAGNOSIS] Strategy has negative or zero IS Sharpe.\n"
                "  The strategy is directionally wrong on this symbol/period.\n"
                "  This is a genuine failure, not a framework bug."
            )
        elif report.is_sharpe < report.permutation_summary.mean:
            print(
                "\n[DIAGNOSIS] IS Sharpe is BELOW the permutation null mean.\n"
                "  The strategy performs WORSE than random shuffles.\n"
                "  This is a genuine failure, not a framework bug."
            )
        else:
            print(
                f"\n[DIAGNOSIS] IS Sharpe ({report.is_sharpe:.4f}) > null mean "
                f"({report.permutation_summary.mean:.4f}) but p_value "
                f"({report.permutation_summary.p_value:.4f}) > threshold "
                f"({report.permutation_summary.threshold:.4f}).\n"
                "  Edge exists but is not statistically significant.\n"
                "  Consider: more data, relaxed alpha (--relaxed-alpha 0.10), "
                "or fewer permutations for speed (--reduced-permutations 100)."
            )


def _apply_relaxed_overrides(
    args: argparse.Namespace,
    config: ResearchValidationConfig,
) -> ResearchValidationConfig:
    """Build a relaxed config from CLI diagnostic flags.

    Returns a modified ResearchValidationConfig. Prints warnings.
    This config is used for the standard validation loop — the IS rejection
    gate is NOT bypassed here (use validate_strategy_relaxed() for that).
    For full disable_is_rejection support in batch mode, use
    predictor.research.relaxed directly.
    """
    from predictor.research.relaxed import RelaxedDiagnosticConfig, apply_relaxed_config

    relaxed = RelaxedDiagnosticConfig(
        base_config=config,
        disable_is_rejection=getattr(args, "disable_is_rejection", False),
        relaxed_alpha=getattr(args, "relaxed_alpha", None),
        reduced_permutation_count=getattr(args, "reduced_permutations", None),
    )
    return apply_relaxed_config(relaxed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-strategy validation and ranking research")
    parser.add_argument(
        "--universe",
        choices=("literature", "donchian", "core", "hypothesis", "baseline"),
        default="literature",
        help="Strategy set. 'core'=one per family (fast), 'literature'=full set, 'donchian'=variants only, 'hypothesis'=economic hypotheses, 'baseline'=benchmark strategies",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Prepend baseline strategies to the strategy universe under test."
    )
    parser.add_argument(
        "--plot",
        default=None,
        metavar="OUTDIR",
        help="Generate visual diagnostic plots into OUTDIR directory."
    )
    parser.add_argument(
        "--export-metadata",
        default=None,
        metavar="PATH",
        help="Export reproducible experiment record (JSON) to PATH."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run permutation engine calibration before the main research run."
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

    # --- Debug / diagnostic flags ---
    parser.add_argument(
        "--debug-strategy",
        default=None,
        metavar="NAME",
        help=(
            "Run single-strategy step-by-step trace mode. "
            "NAME must match a strategy name in the chosen universe "
            "(e.g. donchian_s1_20_10). Prints full diagnostic output and exits."
        ),
    )
    parser.add_argument(
        "--debug-symbol",
        default=None,
        metavar="SYMBOL",
        help="Symbol to use for --debug-strategy trace. Defaults to first symbol.",
    )
    parser.add_argument(
        "--disable-is-rejection",
        action="store_true",
        help=(
            "[DIAGNOSTIC ONLY] Ignore IS permutation failures so all strategies "
            "advance to walk-forward stages. Reveals whether downstream metrics "
            "are meaningful. Results are NOT production-valid."
        ),
    )
    parser.add_argument(
        "--relaxed-alpha",
        type=float,
        default=None,
        metavar="ALPHA",
        help="[DIAGNOSTIC ONLY] Override p_value_threshold (e.g. 0.10 or 0.20).",
    )
    parser.add_argument(
        "--reduced-permutations",
        type=int,
        default=None,
        metavar="N",
        help="[DIAGNOSTIC ONLY] Override permutation_iterations for fast diagnostic runs.",
    )
    parser.add_argument(
        "--export-permutation-dist",
        default=None,
        metavar="PATH",
        help=(
            "Export IS permutation distribution JSON to PATH "
            "(only active with --debug-strategy)."
        ),
    )
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

    # ------------------------------------------------------------------ #
    # Debug trace mode: single strategy, one symbol, full diagnostics     #
    # ------------------------------------------------------------------ #
    if args.debug_strategy is not None:
        _run_debug_trace(args, symbols, data_source, strategies, config)
        return

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Calibration                                                        #
    # ------------------------------------------------------------------ #
    if args.calibrate:
        print("[INFO] Running permutation engine calibration...")
        cal_symbol = symbols[0]
        try:
            cal_frame = data_source.fetch_ohlcv(cal_symbol)
            from predictor.research.calibration import run_permutation_calibration, print_calibration_report
            cal_results = run_permutation_calibration(cal_frame, n_trials=3, config=config)
            print_calibration_report(cal_results)
        except Exception as exc:
            print(f"[WARNING] Permutation calibration failed: {exc}")

    # ------------------------------------------------------------------ #
    # Relaxed diagnostic mode: override config, warn user                 #
    # ------------------------------------------------------------------ #
    if args.disable_is_rejection or args.relaxed_alpha is not None or args.reduced_permutations is not None:
        config = _apply_relaxed_overrides(args, config)

    reports = ()
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
        reports = result.reports
    else:
        comparison = compare_strategies_across_symbols(
            symbols=symbols,
            data_source=data_source,
            strategies=strategies,
            config=config,
        )
        _print_comparison_rows(comparison.rows, label="STRATEGY RANKING")
        comparison_rows = comparison.rows
        reports = comparison.reports

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

    # ------------------------------------------------------------------ #
    # Visualization                                                      #
    # ------------------------------------------------------------------ #
    generated_plots = []
    if args.plot is not None and reports:
        plot_dir = Path(args.plot)
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Generating visual diagnostics into {plot_dir}...")
        
        from predictor.research.visualization import (
            plot_equity_curve,
            plot_drawdown_curve,
            plot_permutation_histogram,
            plot_trade_return_distribution,
            plot_rolling_sharpe,
            plot_regime_overlay,
            export_equity_to_json,
            export_metrics_to_parquet,
        )
        
        # Map buy_and_hold backtest runs by symbol to use as benchmark
        bh_runs = {}
        for strat_name, sym, rep in reports:
            if strat_name == "buy_and_hold":
                bh_runs[sym] = rep.walk_forward_aggregate
        
        for strat_name, sym, rep in reports:
            print(f"  Plotting {strat_name} on {sym}...")
            
            bench = bh_runs.get(sym)
            eq_path = plot_dir / f"{strat_name}_{sym}_equity.png"
            res = plot_equity_curve(rep.walk_forward_aggregate, benchmark_run=bench, output_path=eq_path)
            if res:
                generated_plots.append(res)
                
            dd_path = plot_dir / f"{strat_name}_{sym}_drawdown.png"
            res = plot_drawdown_curve(rep.walk_forward_aggregate, output_path=dd_path)
            if res:
                generated_plots.append(res)
                
            perm_path = plot_dir / f"{strat_name}_{sym}_permutation.png"
            res = plot_permutation_histogram(rep.in_sample_permutation, strategy_name=strat_name, output_path=perm_path)
            if res:
                generated_plots.append(res)
                
            tr_path = plot_dir / f"{strat_name}_{sym}_trade_dist.png"
            res = plot_trade_return_distribution(rep.walk_forward_aggregate, output_path=tr_path)
            if res:
                generated_plots.append(res)
                
            roll_path = plot_dir / f"{strat_name}_{sym}_rolling_sharpe.png"
            res = plot_rolling_sharpe(rep.walk_forward_aggregate, output_path=roll_path)
            if res:
                generated_plots.append(res)
                
            try:
                frame = data_source.fetch_ohlcv(sym)
                reg_path = plot_dir / f"{strat_name}_{sym}_regime.png"
                res = plot_regime_overlay(frame, rep.walk_forward_aggregate, output_path=reg_path)
                if res:
                    generated_plots.append(res)
            except Exception as e:
                print(f"    Failed regime overlay for {sym}: {e}")
                
            json_path = plot_dir / f"{strat_name}_{sym}_equity.json"
            export_equity_to_json(rep.walk_forward_aggregate, output_path=json_path)
            
        parquet_path = plot_dir / "all_metrics.parquet"
        export_metrics_to_parquet([rep for _, _, rep in reports], output_path=parquet_path)

    # ------------------------------------------------------------------ #
    # Export Reproducible Experiment Record                              #
    # ------------------------------------------------------------------ #
    if args.export_metadata is not None:
        from predictor.research.experiment import build_experiment_record, save_experiment
        rejection_counts = {}
        total_runs = len(reports)
        passed_runs = 0
        for _, _, rep in reports:
            if rep.is_valid:
                passed_runs += 1
            else:
                for r in rep.fail_reasons:
                    rejection_counts[r] = rejection_counts.get(r, 0) + 1
                    
        notes = f"Universe: {args.universe}, Source: {args.source}, Parallel: {args.parallel}"
        rec = build_experiment_record(
            runtime_seconds=elapsed,
            random_seed=config.random_seed,
            validation_config=config,
            symbol_universe=symbols,
            strategy_names=tuple(s.name for s in strategies),
            generated_plots=tuple(generated_plots),
            rejection_counts=rejection_counts,
            total_runs=total_runs,
            passed_runs=passed_runs,
            notes=notes,
        )
        metadata_path = Path(args.export_metadata)
        if metadata_path.is_dir() or args.export_metadata.endswith("/"):
            save_path = save_experiment(rec, metadata_path)
        else:
            save_path = metadata_path
            from dataclasses import asdict
            from predictor.research.experiment import _serialise
            import json
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(_serialise(asdict(rec)), indent=2))
        print(f"[INFO] Saved reproducible experiment record to {save_path}")

    print(f"\n[Finished in {elapsed:.2f} seconds]")


if __name__ == "__main__":
    main()
