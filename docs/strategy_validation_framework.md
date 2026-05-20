# Strategy Validation Framework

## Scope

This repository now includes a predictor-side strategy research harness under `predictor/research/`.
The runtime predictor path remains unchanged. The new framework is for strategy research and validation only.

## Mandatory Four-Stage Validation

Each strategy is evaluated under the same sequence:

1. **In-sample backtest** at input-data granularity.
2. **In-sample Monte Carlo permutation test** using block-permuted OHLCV paths.
3. **Walk-forward testing** on repeated unseen forward windows.
4. **Walk-forward Monte Carlo permutation test** on stitched out-of-sample folds.

If permutation tests fail (`p_value > threshold`), the strategy is marked untrustworthy.

## Strategy Families Included

The literature-guided default universe (`build_literature_strategy_universe`) includes:

- Donchian channel breakout
- Moving average crossover
- Mean reversion (z-score)
- Momentum (time-series)
- Volatility breakout (ATR-buffered)
- Regime switching (trend in low-vol, reversion in high-vol)
- Hybrid weighted combinations
- Variants (parameter changes) for fair family comparison

## Metrics Reported

Per strategy (and per symbol in multi-symbol mode):

- Return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- In-sample permutation p-value
- Walk-forward permutation p-value
- Walk-forward stability score
- Valid/invalid status

## Data Inputs

Two data-source modes are supported:

- `CSVHistoricalDataSource` (offline deterministic research)
- `YFinanceHistoricalDataSource` (live historical pulls)

Real-market offline fixtures are stored in:

- `tests/fixtures/market_data/*.csv`
- `data/research_samples/*.csv`

## How To Run

Single-symbol comparison:

```bash
PYTHONPATH=. python -c "from predictor.research import *; print('import ok')"
```

Multi-symbol research run:

```bash
PYTHONPATH=. python scripts/run_strategy_research.py \
  --source csv \
  --data-dir data/research_samples \
  --symbols INFY_NS_1d,TCS_NS_1d,RELIANCE_NS_1d \
  --universe donchian \
  --strict \
  --permutation-iterations 500
```

Donchian-only strict research (recommended for stable candidate selection):

```bash
PYTHONPATH=. python scripts/run_strategy_research.py \
  --source csv \
  --data-dir data/research_samples \
  --symbols INFY_NS_1d,TCS_NS_1d,RELIANCE_NS_1d,HDFCBANK_NS_1d \
  --universe donchian \
  --strict
```

`strict_stable_candidates` is printed after the ranking table. If it prints `none`,
no variant passes the stricter robustness gates.

## Module Responsibilities

- `predictor/research/data.py`: data loading and OHLCV validation
- `predictor/research/strategies.py`: strategy logic only
- `predictor/research/backtest.py`: deterministic return simulation only
- `predictor/research/metrics.py`: metric calculations only
- `predictor/research/permutation.py`: block permutation and p-value engine
- `predictor/research/validation.py`: four-stage validator
- `predictor/research/comparison.py`: single-symbol strategy ranking
- `predictor/research/harness.py`: multi-symbol aggregation and ranking
- `predictor/research/library.py`: paper-guided strategy set
- `predictor/research/donchian.py`: Donchian variant set + strict stability filters

## Methodological Notes

- Permutation uses **block resampling** to preserve local dependence structure while disrupting predictive sequence alignment.
- Walk-forward uses non-overlapping forward test windows after each training window.
- Ranking emphasizes both performance and robustness (p-values and stability), not raw return alone.

## Limitations

- This is a research harness, not execution/automation.
- No transaction cost model beyond simple turnover-cost bps.
- No corporate-action handling beyond source-adjusted price inputs.
- P-values depend on permutation count and block-size choices.
- Walk-forward here validates fixed strategy rules; parameter re-optimization policies are intentionally not added.

## Research Basis (Primary Sources)

- Brock, Lakonishok, LeBaron (1992), moving averages + trading-range break rules with bootstrap:
  https://ideas.repec.org/a/bla/jfinan/v47y1992i5p1731-64.html
- White (2000), Reality Check for data-snooping:
  https://www.ntuzov.com/Nik_Site/Niks_files/Research/papers/mut_funds/White_2000.pdf
- Sullivan, Timmermann, White (1999), technical-rule performance with bootstrap data-snooping correction:
  https://bashtage.github.io/kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Sullivan_Timmermann_White.pdf
- Hansen (2005), Superior Predictive Ability test:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569
- Moskowitz, Ooi, Pedersen (2012), time-series momentum:
  https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
- Hurst, Ooi, Pedersen, century-scale trend-following evidence:
  https://www.trendfollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf
- Gatev, Goetzmann, Rouwenhorst (2006), relative-value mean-reversion (pairs):
  https://depot.som.yale.edu/icf/papers/fileuploads/2573/original/08-03.pdf
- Kim, Tse, Wald (2016), volatility scaling effects in time-series momentum:
  https://www.sciencedirect.com/science/article/pii/S1386418116301379
- Bailey, Borwein, Lopez de Prado, Zhu (2015), probability of backtest overfitting:
  https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- Ang, Timmermann (2011), regime changes in financial markets:
  https://www.nber.org/papers/w17182
- Blume, Easley, O'Hara (1994), informational role of volume:
  https://ideas.repec.org/a/bla/jfinan/v49y1994i1p153-81.html
- Lee, Swaminathan (1998/2000), momentum and trading volume:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=92589
