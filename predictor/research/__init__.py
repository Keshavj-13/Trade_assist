"""Predictor-side strategy research and validation framework."""

from predictor.research.comparison import compare_strategies
from predictor.research.data import (
    CSVHistoricalDataSource,
    HistoricalDataSource,
    YFinanceHistoricalDataSource,
    validate_research_frame,
)
from predictor.research.donchian import (
    DonchianVariantStrategy,
    build_donchian_variant_universe,
    build_strict_donchian_validation_config,
    holm_adjust_p_values,
    select_stable_donchian_rows,
)
from predictor.research.errors import (
    ResearchDataError,
    ResearchError,
    ResearchInputError,
    ResearchValidationError,
)
from predictor.research.harness import compare_strategies_across_symbols
from predictor.research.library import build_literature_strategy_universe
from predictor.research.types import (
    BacktestRun,
    MultiSymbolComparisonResult,
    MultiSymbolStrategyRow,
    PerformanceMetrics,
    PermutationTestResult,
    StrategyComparisonResult,
    StrategyComparisonRow,
    StrategyValidationReport,
    WalkForwardFoldResult,
    WalkForwardSplit,
)
from predictor.research.validation import (
    ResearchValidationConfig,
    build_walk_forward_splits,
    validate_strategy,
)

__all__ = [
    "BacktestRun",
    "CSVHistoricalDataSource",
    "DonchianVariantStrategy",
    "HistoricalDataSource",
    "MultiSymbolComparisonResult",
    "MultiSymbolStrategyRow",
    "PerformanceMetrics",
    "PermutationTestResult",
    "ResearchDataError",
    "ResearchError",
    "ResearchInputError",
    "ResearchValidationConfig",
    "ResearchValidationError",
    "StrategyComparisonResult",
    "StrategyComparisonRow",
    "StrategyValidationReport",
    "WalkForwardFoldResult",
    "WalkForwardSplit",
    "YFinanceHistoricalDataSource",
    "build_donchian_variant_universe",
    "build_literature_strategy_universe",
    "build_strict_donchian_validation_config",
    "build_walk_forward_splits",
    "compare_strategies_across_symbols",
    "compare_strategies",
    "holm_adjust_p_values",
    "select_stable_donchian_rows",
    "validate_research_frame",
    "validate_strategy",
]
