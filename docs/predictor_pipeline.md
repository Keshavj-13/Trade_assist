# Predictor Pipeline

## Scope

This repository now treats prediction as the only active core workflow.
The predictor runtime is implemented in `predictor/` and consumed by
`service/research.py`.

## End-to-End Flow

1. Entry points:
- `main.py`
- `market_assistant.py once`
- Both only initialize the DB when `FIN_ASSIST_ENABLE_PERSISTENCE=1`.

2. Service adapter:
- `service.research.perform_scan()`
- Resolves symbol universe and open-position context.
- Runs deterministic predictor inference.
- Validates `scope`, `top_n`, `wallet`, and injected fetcher before inference.
- Loads the default market-data fetcher lazily on first use instead of at import time.

3. Predictor package:
- `predictor.validation`: validates symbols and OHLCV schema.
- `predictor.features`: builds market-structure features.
- `predictor.model`: produces BUY/SELL/HOLD probabilities with uncertainty.
- `predictor.pipeline`: orchestrates symbol scans and ranking.
- `predictor.data`: isolates yfinance/network fetch side effects from model code.

4. Output:
- Legacy-compatible scan dictionary with buy/sell/hold candidates,
  probabilities, regime labels, and skipped-symbol reasons.

## Data and Feature Schema

OHLCV input dataframe must contain:
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Derived feature schema (`predictor.types.FeatureVector`):
- `price`, `ema20`, `ema50`, `rsi`, `atr_pct`
- `avg_volume`, `volume`, `volume_ratio`
- `vwap`, `vwap_distance_pct`
- `session_low`, `session_high`, `pct_from_low`, `pct_from_high`
- `trend_strength`, `short_return`, `long_return`, `realized_volatility`

## Prediction Contract

`predictor.model.predict_probabilities()` returns:
- `buy_probability`
- `sell_probability`
- `hold_probability`
- `regime`
- `action`

Action selection is probability-thresholded and uncertainty-aware:
- Low confidence or high uncertainty yields `HOLD`.
- SELL is blocked without an open position.
- BUY is blocked when already in position (position sizing is out of scope).

## Configuration

`predictor.config.PredictorConfig` controls:
- Liquidity and volatility filters (`min_avg_volume`, `min_atr_pct`, `max_atr_pct`)
- Candidate cap (`top_n`)
- Uncertainty policy (`min_confidence`, `max_uncertainty`)
- Regime threshold (`regime_threshold`)
- Cross-asset weight (`cross_asset_weight`)
- Determinism seed (`random_seed`)

`random_seed` can be overridden with `FIN_ASSIST_RANDOM_SEED`.

## Failure Modes

Explicit failures are captured as typed errors:
- `InputValidationError`: bad symbol/input contract.
- `DataUnavailableError`: missing/insufficient/filtered data.

Pipeline-level scan failures are isolated per symbol and reported in
`skipped_symbols` without crashing the full run.

## Test Surface

The predictor tests are designed as a truth probe, not a structure check.
They verify:
- Input-contract failures (`scope`, `top_n`, `wallet`, symbols, OHLCV schema).
- The pipeline rejects invalid direct inputs instead of silently coercing them.
- Lazy service-boundary initialization stays behind the adapter instead of happening at import time.
- Deterministic feature/probability invariants.
- BUY/SELL policy constraints around open positions and uncertainty.
- Propagation of unexpected runtime bugs (no silent swallowing in predictor core).

The following behavior is inactive by default:
- Telegram notifications (`FIN_ASSIST_ENABLE_TELEGRAM=1` to enable).
- Trade persistence and wallet mutation (`FIN_ASSIST_ENABLE_PERSISTENCE=1` to enable).
- Legacy non-predictor runtime modes (`FIN_ASSIST_ENABLE_NON_PREDICTOR=1` to enable `daemon`, `scheduler`, `telegram` CLI modes).
- News API + FinBERT sentiment (`FIN_ASSIST_ENABLE_NEWS=1` to enable).

Quarantined legacy modules (inactive unless explicitly enabled):
- `service.daemon`
- `service.scheduler`
- `service.telegram_bot`
- `infra.telegram`
- `frontend/*` and `frontend_node/*` integration paths

Additional cost controls:
- Data HTTP cache can be disabled with `FIN_ASSIST_ENABLE_DATA_CACHE=0`.
- News/sentiment path returns neutral/empty outputs when disabled, avoiding network and model load costs.
- If persistence is enabled, database initialization/read failures are raised rather than silently ignored.

This keeps predictor inference isolated from UI, messaging, and persistence
side effects during normal operation.

## Strategy Research Harness

A separate predictor-side strategy validation framework now lives in
`predictor/research/` and is intentionally isolated from live scan inference.

- It evaluates strategy families under a mandatory four-stage framework:
  in-sample, in-sample permutation, walk-forward, and walk-forward permutation.
- It supports real historical OHLCV data via local CSV or yfinance sources.
- It compares strategies, variants, and hybrid combinations under the same
  metrics and significance tests for fair ranking.

See:

- `docs/strategy_validation_framework.md`
- `scripts/run_strategy_research.py`
- `predictor/research/donchian.py` for Donchian-variant strict mode
