#!/usr/bin/env python3
"""Dedicated CLI script for Cross-Sectional factor and ranking research.

Enforces statistical skepticism, reproducibility, and rigorous diagnostics.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add workspace root to python path to support predictor imports
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_DIR))

from predictor.research.cross_sectional import (
    CrossSectionalResearchConfig,
    CrossSectionalValidationReport,
    validate_factor,
)
from predictor.research.errors import ResearchInputError, ResearchValidationError
from predictor.research.data import (
    CachedDataSource,
    CSVHistoricalDataSource,
    YFinanceHistoricalDataSource,
)
from predictor.research.factors import build_factor_universe
from predictor.research.ranking import compute_daily_regimes, compute_next_day_returns, evaluate_ranking
from predictor.research.ranking_baselines import build_ranking_baseline_universe
from predictor.research.ranking_calibration import run_ranking_calibration_suite
from predictor.research.factor_diagnostics import (
    compute_factor_dispersion_metrics,
    extract_degeneracy_warnings,
)
from predictor.research.factor_snapshots import (
    build_factor_snapshot_table,
    export_factor_snapshot_bundle,
)
from predictor.research.visualization import (
    plot_factor_cumulative_returns,
    plot_factor_regime_performance,
    plot_ic_distribution,
)
from predictor.research.ranking_diagnostics import (
    export_ranking_diagnostics_json,
    export_ranking_diagnostics_parquet,
    plot_cumulative_top_k_return,
    plot_factor_distributions,
    plot_prediction_vs_realized,
    plot_ranking_distribution,
    plot_ranking_permutation_histogram,
    plot_ranking_regime_overlay,
    plot_rolling_ic,
)


def get_git_revision_hash() -> str:
    """Safely obtain the current git commit hash for metadata tracking."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def _normalise_symbols(raw_symbols: List[str]) -> List[str]:
    """Upper-case, de-duplicate, and preserve first-seen order."""
    deduped: List[str] = []
    seen: set[str] = set()
    for raw in raw_symbols:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if symbol not in seen:
            deduped.append(symbol)
            seen.add(symbol)
    return deduped


def load_symbols_from_file(file_path: Path) -> List[str]:
    """Load symbols from either CSV (`symbol` column) or plain-text lists."""
    if not file_path.exists():
        raise ResearchInputError(f"Symbols file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path)
        if "symbol" in frame.columns:
            return _normalise_symbols(frame["symbol"].dropna().tolist())
        if len(frame.columns) == 1:
            return _normalise_symbols(frame.iloc[:, 0].dropna().tolist())
        raise ResearchInputError(
            f"CSV symbols file must contain `symbol` column or exactly one column: {file_path}"
        )

    lines = file_path.read_text().splitlines()
    parsed: List[str] = []
    for line in lines:
        clean = line.split("#", maxsplit=1)[0].strip()
        if not clean:
            continue
        parsed.extend(token for token in clean.split(",") if token.strip())
    return _normalise_symbols(parsed)


def prefetch_ohlcv_data(
    symbols: List[str],
    source: CSVHistoricalDataSource | YFinanceHistoricalDataSource,
    workers: int = 4,
) -> Dict[str, pd.DataFrame]:
    """Concurrently download/load OHLCV data for all symbols to avoid rate limits and save time."""
    cached_source = CachedDataSource(source)
    
    logger.info("Prefetching OHLCV data for %d symbols using %d workers...", len(symbols), workers)
    start_time = time.time()
    
    data_map: Dict[str, pd.DataFrame] = {}
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(cached_source.fetch_ohlcv, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                df = future.result()
                data_map[sym] = df
            except Exception as exc:
                logger.warning("Failed to fetch data for %s: %s", sym, exc)
                
    elapsed = time.time() - start_time
    logger.info("Prefetched %d/%d symbols successfully in %.2fs", len(data_map), len(symbols), elapsed)
    return data_map


def print_cross_sectional_report(report: CrossSectionalValidationReport) -> None:
    """Print a highly comprehensive and rigorous validation report for a factor."""
    fm = report.full_sample_metrics
    wm = report.walk_forward_metrics
    pr = report.permutation_result

    print("=" * 60)
    print(f"FACTOR REPORT: {report.factor_name.upper()}")
    print("=" * 60)
    print(f"Verdict: {'✓ PASS' if report.is_valid else '✗ REJECT'}")
    if report.fail_reasons:
        print(f"Rejection Reasons: {', '.join(report.fail_reasons)}")
    print("-" * 60)
    print("Full-Sample Portfolio Metrics:")
    print(f"  Mean IC (Information Coeff) : {fm.mean_ic:.4f} (t-stat: {fm.ic_t_stat:.2f})")
    print(f"  Annualised Top-K Return      : {fm.annualised_return:.2%}")
    print(f"  Sharpe Ratio                 : {fm.sharpe_ratio:.3f}")
    print(f"  Max Drawdown                 : {fm.max_drawdown:.2%}")
    print(f"  Precision at K               : {fm.precision_at_k:.2%}")
    print(f"  Top 1 Hit Rate               : {fm.top_1_hit_rate:.2%}")
    print(f"  Mean Daily Turnover          : {fm.mean_turnover:.2%}")
    print(f"  Turnover-Adjusted Return     : {fm.turnover_adjusted_return:.2%}")
    print("-" * 60)
    print("Out-of-Sample Walk-Forward Metrics:")
    print(f"  OOS Mean IC                  : {wm.mean_ic:.4f}")
    print(f"  OOS Sharpe Ratio             : {wm.sharpe_ratio:.3f}")
    print(f"  OOS Annualised Return        : {wm.annualised_return:.2%}")
    print(f"  OOS Max Drawdown             : {wm.max_drawdown:.2%}")
    print("-" * 60)
    print("Monte Carlo Block Permutation Test:")
    print(f"  Observed IC vs Null Mean     : {pr.observed_ic:.4f} vs {np.mean(pr.null_ic):.4f}")
    print(f"  IC Empirical p-value         : {pr.ic_p_value:.4f} (expect <= {report.full_sample_metrics.ic_std:.4f})")
    print(f"  Observed Sharpe vs Null Mean : {pr.observed_sharpe:.3f} vs {np.mean(pr.null_sharpe):.3f}")
    print(f"  Sharpe Empirical p-value     : {pr.sharpe_p_value:.4f}")
    print("=" * 60)
    print()


def save_reproducible_metadata(
    reports: List[CrossSectionalValidationReport],
    symbol_universe: List[str],
    config: CrossSectionalResearchConfig,
    runtime: float,
    output_path: Path,
) -> None:
    """Save full, structured JSON record of the experiment for rigorous auditing."""
    payload = {
        "experiment_id": str(os.urandom(8).hex()),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": get_git_revision_hash(),
        "runtime_seconds": round(runtime, 2),
        "config": {
            "top_k": config.top_k,
            "transaction_cost_bps": config.transaction_cost_bps,
            "slippage_bps": config.slippage_bps,
            "permutation_iterations": config.permutation_iterations,
            "permutation_block_size": config.permutation_block_size,
            "p_value_threshold": config.p_value_threshold,
            "train_window": config.train_window,
            "test_window": config.test_window,
            "walk_forward_step": config.walk_forward_step,
            "minimum_ic": config.minimum_ic,
            "random_seed": config.random_seed,
        },
        "symbol_universe": symbol_universe,
        "factors_tested": [r.factor_name for r in reports],
        "results": {
            r.factor_name: {
                "is_valid": r.is_valid,
                "fail_reasons": list(r.fail_reasons),
                "full_sample": {
                    "mean_ic": round(r.full_sample_metrics.mean_ic, 4),
                    "ic_t_stat": round(r.full_sample_metrics.ic_t_stat, 2),
                    "annualised_return": round(r.full_sample_metrics.annualised_return, 4),
                    "sharpe_ratio": round(r.full_sample_metrics.sharpe_ratio, 3),
                    "max_drawdown": round(r.full_sample_metrics.max_drawdown, 4),
                    "precision_at_k": round(r.full_sample_metrics.precision_at_k, 4),
                    "top_1_hit_rate": round(r.full_sample_metrics.top_1_hit_rate, 4),
                    "mean_turnover": round(r.full_sample_metrics.mean_turnover, 4),
                    "turnover_adjusted_return": round(r.full_sample_metrics.turnover_adjusted_return, 4),
                },
                "permutation": {
                    "ic_p_value": round(r.permutation_result.ic_p_value, 4),
                    "sharpe_p_value": round(r.permutation_result.sharpe_p_value, 4),
                },
                "walk_forward": {
                    "mean_ic": round(r.walk_forward_metrics.mean_ic, 4),
                    "sharpe_ratio": round(r.walk_forward_metrics.sharpe_ratio, 3),
                    "annualised_return": round(r.walk_forward_metrics.annualised_return, 4),
                    "max_drawdown": round(r.walk_forward_metrics.max_drawdown, 4),
                }
            }
            for r in reports
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved reproducible experiment record to %s", output_path)


def generate_factor_plots(
    reports: List[CrossSectionalValidationReport],
    symbol_data: Dict[str, pd.DataFrame],
    config: CrossSectionalResearchConfig,
    plot_dir: Path,
) -> None:
    """Generate high-quality diagnostic charts for all tested factors."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating diagnostic charts into %s...", plot_dir)
    
    # Compute actual returns and regimes once
    targets = compute_next_day_returns(symbol_data)
    regimes = compute_daily_regimes(symbol_data)
    
    factor_lookup = {factor.name: factor for factor in build_factor_universe()}

    for report in reports:
        factor_name = report.factor_name
        logger.info("  Plotting diagnostic charts for %s...", factor_name)
        
        # 1. Retrieve or build the tested factor from the universe
        factor = factor_lookup.get(factor_name)
        if factor is None:
            continue
            
        try:
            scores = factor.compute_scores(symbol_data)
            daily_dispersion = compute_factor_dispersion_metrics(scores, top_k=config.top_k)
            degeneracy_warnings = extract_degeneracy_warnings(
                daily_dispersion,
                factor_name=factor_name,
            )
            if degeneracy_warnings:
                logger.warning(
                    "Factor %s emitted %d degeneracy warnings",
                    factor_name,
                    len(degeneracy_warnings),
                )
                for warning in degeneracy_warnings:
                    logger.warning(warning)

            # Rerun simple backtest to obtain returns series
            factor_rets, _ = evaluate_ranking(
                scores=scores,
                targets=targets,
                top_k=config.top_k,
                transaction_cost_bps=config.transaction_cost_bps,
                slippage_bps=config.slippage_bps,
                regimes=regimes,
            )
            
            # Plot Cumulative Returns vs benchmarks (e.g. equal_weight_market and winner_continuation)
            benchmark_rets = {}
            for bench_name in ("equal_weight_market", "winner_continuation"):
                bench_factor = next((f for f in build_factor_universe() if f.name == bench_name), None)
                if bench_factor is not None:
                    bench_scores = bench_factor.compute_scores(symbol_data)
                    bench_rets, _ = evaluate_ranking(
                        scores=bench_scores,
                        targets=targets,
                        top_k=config.top_k,
                        transaction_cost_bps=config.transaction_cost_bps,
                        slippage_bps=config.slippage_bps,
                        regimes=regimes,
                    )
                    benchmark_rets[bench_name.replace("_", " ").title()] = bench_rets
            
            plot_factor_cumulative_returns(
                factor_rets,
                benchmark_returns=benchmark_rets,
                output_path=plot_dir / f"{factor_name}_cumulative_returns.png",
            )
            plot_cumulative_top_k_return(
                factor_rets,
                output_path=plot_dir / f"{factor_name}_top_k_cumulative.png",
            )
            plot_ranking_distribution(
                scores,
                output_path=plot_dir / f"{factor_name}_ranking_distribution.png",
            )
            plot_factor_distributions(
                scores,
                output_path=plot_dir / f"{factor_name}_factor_distribution.png",
            )
            plot_prediction_vs_realized(
                scores,
                targets,
                output_path=plot_dir / f"{factor_name}_prediction_vs_realized.png",
            )
            plot_ranking_permutation_histogram(
                report.permutation_result,
                output_path=plot_dir / f"{factor_name}_ranking_permutation.png",
            )
            plot_ranking_regime_overlay(
                factor_rets,
                regimes,
                output_path=plot_dir / f"{factor_name}_regime_overlay.png",
            )
            
            # Plot IC Distribution
            common_dates = scores.index.intersection(targets.index)
            if not common_dates.empty:
                aligned_scores = scores.loc[common_dates]
                aligned_targets = targets.loc[common_dates]
                ic_series = aligned_scores.corrwith(aligned_targets, axis=1, method="spearman")
                plot_ic_distribution(ic_series, output_path=plot_dir / f"{factor_name}_ic_distribution.png")
                plot_rolling_ic(ic_series, output_path=plot_dir / f"{factor_name}_rolling_ic.png")

                # Export daily ranking panel: date,symbol,score,rank,target
                daily_ranks = aligned_scores.rank(axis=1, ascending=False, method="first")
                ranking_panel = pd.concat(
                    [
                        aligned_scores.stack().rename("score"),
                        daily_ranks.stack().rename("rank"),
                        aligned_targets.stack().rename("target"),
                    ],
                    axis=1,
                ).reset_index()
                ranking_panel.columns = ["date", "symbol", "score", "rank", "target"]
                ranking_panel.to_csv(plot_dir / f"{factor_name}_daily_rankings.csv", index=False)
                export_ranking_diagnostics_parquet(
                    ranking_panel,
                    output_path=plot_dir / f"{factor_name}_daily_rankings.parquet",
                )

            # Plot Regime IC Breakdown
            if report.full_sample_metrics.regime_ic:
                plot_factor_regime_performance(
                    report.full_sample_metrics.regime_ic,
                    output_path=plot_dir / f"{factor_name}_regime_performance.png",
                )

            diagnostics_payload = {
                "factor_name": factor_name,
                "is_valid": report.is_valid,
                "fail_reasons": list(report.fail_reasons),
                "metrics": {
                    "mean_ic": report.full_sample_metrics.mean_ic,
                    "rank_correlation": report.full_sample_metrics.rank_correlation,
                    "information_coefficient": report.full_sample_metrics.information_coefficient,
                    "top_1_hit_rate": report.full_sample_metrics.top_1_hit_rate,
                    "top_k_mean_return": report.full_sample_metrics.top_k_mean_return,
                    "precision_at_k": report.full_sample_metrics.precision_at_k,
                    "average_selected_return": report.full_sample_metrics.average_selected_return,
                    "intraday_drawdown": report.full_sample_metrics.intraday_drawdown,
                    "turnover_adjusted_return": report.full_sample_metrics.turnover_adjusted_return,
                    "regime_stability": report.full_sample_metrics.regime_stability,
                },
                "permutation": {
                    "ic_p_value": report.permutation_result.ic_p_value,
                    "sharpe_p_value": report.permutation_result.sharpe_p_value,
                    "null_ic": list(report.permutation_result.null_ic),
                    "null_sharpe": list(report.permutation_result.null_sharpe),
                },
                "factor_dispersion": {
                    "daily_rows": int(len(daily_dispersion)),
                    "degeneracy_warning_count": int(len(degeneracy_warnings)),
                    "mean_cross_sectional_std": float(
                        daily_dispersion["cross_sectional_std"].mean()
                    ) if not daily_dispersion.empty else 0.0,
                    "mean_unique_rank_count": float(
                        daily_dispersion["unique_rank_count"].mean()
                    ) if not daily_dispersion.empty else 0.0,
                },
                "daily": {
                    "ic_values": [float(v) for v in (ic_series.dropna().values if not common_dates.empty else [])],
                    "top_k_returns": [float(v) for v in factor_rets.values],
                    "dates": [str(d) for d in factor_rets.index],
                },
            }
            export_ranking_diagnostics_json(
                diagnostics_payload,
                output_path=plot_dir / f"{factor_name}_diagnostics.json",
            )

            daily_diag = pd.DataFrame(
                {
                    "date": factor_rets.index,
                    "top_k_return": factor_rets.values,
                }
            )
            export_ranking_diagnostics_parquet(
                daily_diag,
                output_path=plot_dir / f"{factor_name}_diagnostics.parquet",
            )

            daily_dispersion.to_csv(
                plot_dir / f"{factor_name}_dispersion_daily.csv",
                index=False,
            )
            export_ranking_diagnostics_parquet(
                daily_dispersion,
                output_path=plot_dir / f"{factor_name}_dispersion_daily.parquet",
            )
            export_ranking_diagnostics_json(
                {
                    "factor_name": factor_name,
                    "degeneracy_warning_count": len(degeneracy_warnings),
                    "warnings": degeneracy_warnings,
                },
                output_path=plot_dir / f"{factor_name}_degeneracy_warnings.json",
            )

            snapshot_table = build_factor_snapshot_table(
                factor_name=factor_name,
                scores=scores,
                targets=targets,
                regimes=regimes,
                top_k=config.top_k,
            )
            export_factor_snapshot_bundle(
                snapshot_table,
                output_dir=plot_dir,
                factor_name=factor_name,
                include_json_summary=True,
            )
                
        except (ResearchInputError, ResearchValidationError, ValueError, KeyError) as exc:
            logger.error("Failed to generate plots for %s: %s", factor_name, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Sectional factor and ranking research CLI.")
    parser.add_argument("--universe", choices=["factors", "baselines", "all"], default="all",
                        help="The category of factors to evaluate.")
    parser.add_argument("--factors", type=str,
                        help="Comma-separated factor names to run (overrides --universe filtering).")
    parser.add_argument("--symbols", type=str,
                        help="Comma-separated list of symbols to research.")
    parser.add_argument("--symbols-file", type=str, default="data/nse_symbols.csv",
                        help="Symbols file path (.csv with symbol column, or .txt newline/comma separated).")
    parser.add_argument("--source", choices=["yfinance", "csv"], default="yfinance",
                        help="The historical data provider.")
    parser.add_argument("--csv-dir", type=str, default="data/research_samples",
                        help="Directory for local CSV data.")
    parser.add_argument("--lookback", type=str, default="5y",
                        help="yfinance lookback period (e.g. 5y, 10y).")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Portfolio size (number of top-ranked stocks to buy daily).")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0,
                        help="Estimated transaction cost in basis points (default 5.0).")
    parser.add_argument("--slippage-bps", type=float, default=0.0,
                        help="Estimated execution slippage in basis points (default 0.0).")
    parser.add_argument("--permutation-iterations", type=int, default=300,
                        help="Number of Monte Carlo permutation runs.")
    parser.add_argument("--reduced-permutations", dest="permutation_iterations", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--run-calibration", action="store_true",
                        help="Run synthetic permutation calibration probes before factor validation.")
    parser.add_argument("--plot", type=str,
                        help="Directory to save diagnostic charts.")
    parser.add_argument("--export-diagnostics", type=str,
                        help="Directory to export ranking diagnostics (PNG/JSON/parquet).")
    parser.add_argument("--export-metadata", type=str,
                        help="JSON file path to save reproducible experiment metadata.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers for data loading.")

    args = parser.parse_args()
    start_time = time.time()

    # 1. Parse symbol universe
    if args.symbols:
        symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    else:
        csv_path = Path(args.symbols_file)
        if not csv_path.is_absolute():
            csv_path = WORKSPACE_DIR / csv_path
        try:
            symbols = load_symbols_from_file(csv_path)
        except ResearchInputError as exc:
            logger.error(str(exc))
            sys.exit(1)

    if not symbols:
        logger.error("No valid symbols specified.")
        sys.exit(1)

    # 2. Setup historical source
    if args.source == "yfinance":
        source = YFinanceHistoricalDataSource(lookback=args.lookback)
    else:
        csv_dir = Path(args.csv_dir)
        if not csv_dir.is_absolute():
            csv_dir = WORKSPACE_DIR / csv_dir
        source = CSVHistoricalDataSource(directory=csv_dir)

    # 3. Load OHLCV data concurrently
    symbol_data = prefetch_ohlcv_data(symbols, source, workers=args.workers)
    if not symbol_data:
        logger.error("No historical data could be loaded for any symbols.")
        sys.exit(1)

    # 4. Construct cross-sectional config
    config = CrossSectionalResearchConfig(
        top_k=args.top_k,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        permutation_iterations=args.permutation_iterations,
        random_seed=42,
    )

    # 5. Build selected factor list
    all_factors = build_factor_universe()
    factor_map = {f.name: f for f in all_factors}
    if args.factors:
        requested = [name.strip() for name in args.factors.split(",") if name.strip()]
        unknown = [name for name in requested if name not in factor_map]
        if unknown:
            logger.error("Unknown factor names: %s", ", ".join(unknown))
            logger.error("Available factors: %s", ", ".join(sorted(factor_map.keys())))
            sys.exit(1)
        factors_to_test = [factor_map[name] for name in requested]
    elif args.universe == "baselines":
        baseline_names = {f.name for f in build_ranking_baseline_universe()}
        factors_to_test = [f for f in all_factors if f.name in baseline_names]
    elif args.universe == "factors":
        baseline_names = {f.name for f in build_ranking_baseline_universe()}
        factors_to_test = [f for f in all_factors if f.name not in baseline_names]
    else:
        factors_to_test = list(all_factors)

    logger.info("Evaluating %d ranking factors across %d symbols...", len(factors_to_test), len(symbol_data))

    if args.run_calibration:
        logger.info("Running synthetic permutation calibration suite...")
        synth_targets = compute_next_day_returns(symbol_data).dropna(how="all")
        calibration = run_ranking_calibration_suite(
            synth_targets,
            config=config,
            n_trials=3,
            weak_signal_strength=0.12,
        )
        logger.info(
            "Calibration pass-rates | cheating=%.2f random=%.2f weak=%.2f",
            calibration.cheating.pass_rate,
            calibration.random.pass_rate,
            calibration.weak.pass_rate,
        )

    # 6. Run backtests and permutation tests
    reports: List[CrossSectionalValidationReport] = []
    for factor in factors_to_test:
        logger.info("Running validation harness for factor: %s...", factor.name)
        report = validate_factor(factor, symbol_data, config)
        reports.append(report)
        print_cross_sectional_report(report)

    runtime = time.time() - start_time
    logger.info("Cross-sectional research run finished in %.2fs", runtime)

    # 7. Generate diagnostic charts if requested
    diagnostics_dir = args.export_diagnostics if args.export_diagnostics else args.plot
    if diagnostics_dir:
        plot_path = Path(diagnostics_dir)
        if not plot_path.is_absolute():
            plot_path = WORKSPACE_DIR / plot_path
        generate_factor_plots(reports, symbol_data, config, plot_path)

    # 8. Export metadata if requested
    if args.export_metadata:
        meta_path = Path(args.export_metadata)
        if not meta_path.is_absolute():
            meta_path = WORKSPACE_DIR / meta_path
        save_reproducible_metadata(reports, list(symbol_data.keys()), config, runtime, meta_path)


if __name__ == "__main__":
    main()
