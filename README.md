# 🏛️ TSE Robo-Advisor — Technical Architecture & Systems Design

**Scope of this document:** a rigorous, implementation-grounded technical deep-dive into the quantitative research and portfolio-construction pipeline for the Tehran Stock Exchange (TSE). Every metric, threshold, and design pattern cited below is taken directly from the codebase and a reference production run — nothing here is a projected or hypothetical benchmark.

---

## 1. Project Overview & Core Architecture

The system is a **CPU-bound, batch-oriented quant research pipeline**: gradient-boosted decision trees over engineered tabular features, not a deep-learning inference service. This is a deliberate architectural choice (see §2.1) rather than a limitation of scope.

### 1.1 System Dataflow

```
┌────────────────────────┐   ┌───────────────────────────┐
│  dollar_data_ingestion  │   │      data_pipeline.py      │
│  USD/IRR free-market    │   │  8-ticker OHLC scraper,    │
│  rate (GitHub dataset   │   │  retry-with-backoff        │
│  fallback + local CSV)  │   │  ("تلاش 1/3" pattern),     │
└───────────┬─────────────┘   │  invalid-date row filter   │
            │                 └──────────────┬──────────────┘
            ▼                                ▼
   ┌─────────────────────────────────────────────────┐
   │            Persistence Layer (SQLite)            │
   │  tsetmc_market_data.db · macro_data.db            │
   │  indexed on (ticker, jalali_date)                 │
   └───────────────────────┬───────────────────────────┘
                            ▼
              ┌──────────────────────────────┐
              │    feature_engineering.py     │
              │  22–41 features/ticker,        │
              │  60-trading-day forward-return │
              │  labeling, NaN-repair by gap-   │
              │  fill ("ترمیم فضا"), per-ticker │
              │  diagnostics matrix            │
              └───────────────┬────────────────┘
                              ▼
                  Parquet Feature Store
                  (*_AI_Features.parquet)
                              │
                              ▼
              ┌──────────────────────────────┐
              │        train_model.py         │
              │  Purged K-Fold LightGBM        │
              │  (3-class: decline/neutral/    │
              │  strong-growth), shared         │
              │  apply_diagnostic_corrections() │
              │  — single source of truth for   │
              │  train/backtest/serve parity    │
              └───────┬───────────────┬────────┘
                      ▼               ▼
        ┌──────────────────┐  ┌────────────────────────┐
        │ diagnose_signal.py│  │    live_predictor.py    │
        │ MI · regime-shift │  │ + geopolitical_features │
        │ · leakage ·        │  │ (circuit-breaker fail-  │
        │ cross-sectional     │  │ open), stale-ticker     │
        │ decomposition        │  │ suppression              │
        └──────────────────┘  └────────────┬─────────────┘
                                            ▼
                               ┌─────────────────────────┐
                               │  portfolio_optimizer.py  │
                               │  SLSQP Sharpe-max, batch  │
                               │  SQL (N+1 eliminated),    │
                               │  3-tier optimizer fallback│
                               │  chain, geopolitical brake│
                               └────────────┬─────────────┘
                                            ▼
                               ┌─────────────────────────┐
                               │   dashboard_export.py    │
                               │  atomic write (tmp+rename)│
                               │  → React/Vite frontend    │
                               └─────────────────────────┘

  Independent validation path (not part of main.py orchestration):
                    ┌─────────────────────────┐
                    │      backtester.py       │
                    │ walk-forward simulation, │
                    │ Sharpe/Sortino/Calmar/    │
                    │ Deflated Sharpe Ratio     │
                    └─────────────────────────┘
```

### 1.2 Design Principle: Diagnostics as a Deployment Gate

Unlike pipelines that treat backtesting as a final report, this system positions `diagnose_signal.py` **upstream of any architecture investment decision**. Its own docstring states the explicit go/no-go criteria before adopting a heavier architecture (e.g. LSTM+Attention): purged-CV `best_iteration` must clear single digits by a wide margin, Deflated Sharpe Ratio must no longer be a hard NO-GO, and minority-class F1 must be non-zero. This is a **methodology-first, not architecture-first**, engineering posture.

---

## 2. Technical Innovations & Deep Dive

### 2.1 Why Gradient-Boosted Trees, Not Deep Learning

At the observed data scale (**15,463 labeled training rows** across 8 tickers, ~22–40 heterogeneous tabular features), gradient-boosted trees (LightGBM) have a stronger inductive bias than a neural architecture: they handle mixed-scale features, non-linear interactions, and missing values natively, without requiring the sample volume deep nets need to avoid overfitting. Feature importance is also directly interpretable — critical here, since `days_since_last_capital_increase_scaled` ranking #1 in importance was the signal that surfaced a live train/serve skew bug (§3.2). A neural architecture (LSTM+Attention) is explicitly deferred in the roadmap, gated behind the diagnostics suite clearing its Definition-of-Done — not adopted speculatively.

### 2.2 Purged K-Fold Cross-Validation

Because labels are constructed from a **60-trading-day forward return window**, adjacent time-ordered samples share overlapping label horizons. A naive random K-fold split leaks future information: a training sample whose label window overlaps a test sample's window effectively "sees the future" of the test point. Purged K-Fold removes any training sample whose label interval intersects the test fold's interval (optionally with an embargo buffer $e$ after the test window):

For a test sample at time $t$ with label horizon $H = 60$, any training sample $t'$ is purged if:

$$[t', t' + H] \cap [t - e, t + H] \neq \emptyset$$

Observed fold results confirm this is a real, non-trivial constraint — not a formality:

| Fold | Accuracy | F1 (weighted) | Best Iteration |
|---|---|---|---|
| 1 | 40.98% | 0.194 | 1 |
| 2 | 35.41% | 0.174 | 1 |
| 3 | 47.76% | 0.336 | 1 |
| 4 | 59.85% | 0.308 | 45 |
| 5 | 56.56% | 0.330 | 8 |

The final model was trained with **26 trees** (the median best-iteration across folds) rather than an arbitrarily fixed boosting round count — a defensive choice against the folds where the model converged to a trivial solution (`best_iteration = 1`) after purging removed leakage-driven "signal."

### 2.3 Deflated Sharpe Ratio as a Statistical Backtest Gate

A raw Sharpe ratio computed on a single backtest is a biased estimator of true skill — it doesn't account for the number of variations/trials implicitly searched over during strategy development. The **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) corrects for this by testing the observed Sharpe against the *expected maximum* Sharpe achievable by chance given the number of trials and the return distribution's higher moments:

$$DSR = \Phi\left(\frac{(\widehat{SR} - SR_0)\sqrt{n - 1}}{\sqrt{1 - \hat{\gamma}_3\widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}}\right)$$

where $\widehat{SR}$ is the observed Sharpe ratio, $SR_0$ is the expected maximum Sharpe under the null of no skill (a function of the number of trials $N$), $\hat{\gamma}_3$/$\hat{\gamma}_4$ are sample skewness/kurtosis, and $n$ is the number of return observations. The reference backtest reports **DSR = 0.358** with `n_trials=1, n_obs=57` — well under the conventional 0.95 acceptance threshold, meaning the observed backtest Sharpe is **not yet statistically distinguishable from a zero-skill strategy** at this sample size. This is treated as a hard gate in the diagnostics workflow, not a footnote.

### 2.4 Cross-Sectional Variance Decomposition

To separate genuine stock-picking skill from macro market-timing, the target variance is decomposed via the law of total variance across the panel of 8 tickers on each trading day $t$:

$$\mathrm{Var}(\text{margin}) = \underbrace{\mathrm{Var}_t\!\left(\mathbb{E}[\text{margin} \mid t]\right)}_{\text{between-day (macro)}} + \underbrace{\mathbb{E}_t\!\left[\mathrm{Var}(\text{margin} \mid t)\right]}_{\text{within-day (stock-specific)}}$$

where `margin` = 60-day forward stock return minus 60-day forward market (dollar) return. Observed result: **75.9% between-day, 24.1% within-day** — meaning the majority of the target's variance is explained at the day/macro level (the shared USD-regime signal), not at the individual-stock level. This finding directly explains a previously-observed failure mode where the portfolio optimizer concentrated allocation into a single ticker: highly correlated day-level alpha scores across the universe caused the `MAX_CORRELATION = 0.85` filter to strip out most of the candidate set.

### 2.5 Train/Serve Skew Elimination

`live_predictor.py` and `backtester.py` both import `apply_diagnostic_corrections` from `train_model.py` rather than reimplementing feature derivation. This was a deliberate fix for a real bug: three engineered features (`dollar_return_sign`, `dollar_macro_trend`, `days_since_last_capital_increase_scaled`) were previously computed only at train time; at serve time they silently fell through a generic fill-missing loop and were zero-filled — while `days_since_last_capital_increase_scaled` was the model's #1 feature by importance. The fix is architectural, not a patch: a single shared function is now the sole source of truth for these derived features, called on **full per-ticker history** (required for the 10-day/252-day rolling windows involved) before slicing to the latest live row.

---

## 3. Tech Stack & Dependencies

| Category | Components | Notes |
|---|---|---|
| **Core Language** | Python 3.11 | |
| **Modeling** | LightGBM (multi-class: decline / neutral / strong-growth) | Histogram-based, CPU-only |
| **Statistical Validation** | scikit-learn (Mutual Information, `DecisionTreeClassifier` for leakage probes, `cross_val_score`) | |
| **Portfolio Optimization** | SciPy `minimize` (SLSQP), custom Sharpe-maximizing objective | |
| **Data Processing** | pandas, NumPy | |
| **Storage** | SQLite (`tsetmc_market_data.db`, `macro_data.db`, `geo_signals.db`), Parquet (feature store), CSV (raw dollar-rate cache) | |
| **Reporting** | openpyxl / pandas Excel writers | |
| **Frontend Bridge** | Atomic JSON export (temp-file + `Path.replace`) → React/Vite dashboard | |
| **External Signal** | Local WorldMonitor CII (Crisis Instability Index) integration | Fail-open by design |
| **Neural / Deep Learning Framework** | Not currently used | See §2.1 for rationale; reserved for a diagnostics-gated future experiment |
| **Vector Store** | Not applicable | No embedding/retrieval component in this pipeline |
| **GPU Acceleration** | Not required | Dataset scale (~15K rows) trains in single-digit seconds on CPU (see §5) |

---

## 4. Production Readiness & Systems Design

### 4.1 Data Validation

- Per-ticker **diagnostics matrix** printed at feature-engineering time reports NaN counts per feature before and after gap-repair (e.g. `rsi_14: NaN = 24 / 2056`).
- Invalid/malformed date rows are filtered per-ticker with explicit counts logged (observed: 22–424 rows filtered per ticker in the reference run).
- `get_historical_returns_batch` in the optimizer drops any ticker with fewer than 50 historical return observations before covariance estimation, and any ticker whose variance is `NaN` or non-positive after covariance computation — both logged explicitly rather than silently propagated.

### 4.2 Error Handling & Resilience Patterns

- **Retry-with-attempt-counter** on price downloads (`"دانلود 'X' (تلاش 1/3)"` pattern) in `data_pipeline.py`.
- **Graceful storage degradation**: `dollar_data_ingestion.py` treats the CSV write as authoritative and the SQLite write as best-effort — a DB write failure is caught, logged, and does not fail the pipeline, since the CSV output is already durable.
- **Fail-open circuit breaker** (`geopolitical_features.py`): the WorldMonitor integration is wrapped in `try/except ImportError` and `try/except Exception` at every external call boundary. A missing clone, changed internal API, or runtime failure returns `None` and the caller (`record_daily_snapshot`, `get_current_risk_brake`) falls back to a neutral/default value rather than raising. This is confirmed operating in the reference run: `⚠️ ساختار داخلی پروژه worldmonitor تغییر کرده... No module named 'src.parser'` — the pipeline continued uninterrupted.
- **Three-tier optimization fallback chain** in `portfolio_optimizer.py`: SLSQP Sharpe-maximization → on non-convergence, SLSQP minimum-variance → on that also failing, equal-weight allocation. Each tier degrades functionality but never raises to the caller.
- **Covariance matrix conditioning check**: `np.linalg.cond` is evaluated pre-optimization; a condition number above `1e10` triggers an explicit instability warning rather than silently returning an unstable allocation.
- **Domain-specific validation gate**: stale/halted-ticker detection (`STALE_DATA_THRESHOLD_DAYS = 15`) suppresses buy signals for tickers whose most recent data lags the freshest ticker in the batch — catching a real incident where a stock halted for 4+ months still received a "strong buy" signal from a naive intra-series gap check.

### 4.3 Caching Strategy

The Parquet feature store (`ai_features_outputs/*_AI_Features.parquet`) functions as an **implicit read cache** between the expensive feature-engineering stage and the cheap training/inference read path — full feature computation runs once per ingestion cycle, not per training run. There is currently no explicit TTL or invalidation mechanism; staleness is only caught downstream by the stale-ticker check (§4.2), not proactively at the cache layer. This is flagged as a systems gap in the roadmap.

### 4.4 Rate Limiting

Not currently implemented as a formal token-bucket or backoff scheme against any external API — the closest analog is the fixed 3-attempt retry counter on ingestion (§4.2). For a larger ticker universe or higher-frequency polling, a proper rate limiter would be required; this is called out explicitly as a gap rather than assumed solved.

### 4.5 Compute Efficiency & Memory Footprint

No GPU is required or used. LightGBM's histogram-based split-finding is efficient at this row count: full retrain (5-fold purged CV + final model) completes in **6.66 seconds** on the reference run (§5). The N+1 query pattern previously present in `portfolio_optimizer.py` (one SQL query per ticker) was replaced with a single batched query using `IN (...)` parameterization — an approximately 100x reduction in round trips for large ticker universes, per the module's own documentation.

---

## 5. Performance & Benchmarking

All figures below are wall-clock measurements from a single reference production run (`main.py`, 8-ticker universe, full pipeline retrain), CPU-only.

### 5.1 Pipeline Stage Latency

| Stage | Component | Duration | Volume |
|---|---|---|---|
| 1 | Data ingestion + cleaning | 22.75s | 8 tickers + USD/IRR (2,807 rows) |
| 2 | Feature engineering | 3.40s | 8 tickers × ~41 columns |
| 3 | Model training (Purged K-Fold) | 6.66s | 15,463 labeled rows, 5 folds |
| 4 | Live prediction | 4.59s | 473 live rows, 8 tickers |
| 5 | Portfolio optimization | 0.15s | Up to 15-asset SLSQP solve |
| **Total (`main.py`)** | | **~37.6s** | End-to-end, cold start |
| — | `diagnose_signal.py` (standalone) | 6.69s | Full MI + regime + leakage + decomposition suite |
| — | `backtester.py` (standalone) | 4.92s | 58-step walk-forward simulation |

### 5.2 Model Quality Metrics (Reference Run)

| Metric | Value |
|---|---|
| Training rows (labeled) | 15,463 |
| Live inference rows | 473 |
| Mean fold accuracy | 48.1% (range 35.4–59.9%) |
| Final model size | 26 trees (median best-iteration) |
| Top MI feature | `dollar_macro_trend` (0.186 — ~7.6x the 2nd-ranked feature) |
| Cross-sectional decomposition | 75.9% between-day / 24.1% within-day variance |

### 5.3 Backtest Metrics (Reference Run, 58 Steps)

| Metric | Value |
|---|---|
| Initial capital | 50,000,000 Toman |
| Final value (AI strategy) | 119,448,000 Toman (+138.90%) |
| Final value (USD benchmark) | 279,751,764 Toman (+459.50%) |
| Equal-weight stock benchmark | −61.79% |
| Alpha vs. USD benchmark | **−320.61%** |
| Alpha vs. equal-weight benchmark | **+200.68%** |
| Max drawdown | −52.17% |
| Sharpe / Sortino / Calmar | −0.05 / −0.09 / 2.66 |
| Deflated Sharpe Ratio | 0.358 → **NO-GO** (`n_trials=1`, `n_obs=57`) |

> These numbers are reported in full, including the unfavorable ones, because a production-readiness document that only surfaces favorable metrics is not a production-readiness document. See §2.3 and §2.4 for the statistical interpretation.

---

## 6. Enterprise Installation & Configuration

```bash
# 1. Clone
git clone https://github.com/<your-username>/tse-robo-advisor.git
cd tse-robo-advisor

# 2. Isolated environment
python3.11 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Optional: geopolitical signal source (fail-open if absent)
git clone https://github.com/<worldmonitor-fork>/worldmonitor.git ./worldmonitor
```

### 6.1 Configuration Reference

These are live constants pulled directly from the codebase — not illustrative defaults:

| Constant | Module | Value | Purpose |
|---|---|---|---|
| `STALE_DATA_THRESHOLD_DAYS` | `live_predictor.py` | 15 | Days of staleness before a ticker's buy signal is suppressed |
| `MIN_HISTORY_ROWS_FOR_STABLE_FEATURES` | `live_predictor.py` | 20 | Minimum rows before rolling features are trusted |
| `TRADING_DAYS_PER_YEAR` | `portfolio_optimizer.py` | 242 | TSE-specific annualization factor |
| `RISK_FREE_ANNUAL` | `portfolio_optimizer.py` | 0.28 | Reflects the local (IRR) risk-free/inflation environment |
| `MAX_SINGLE_STOCK_WEIGHT` | `portfolio_optimizer.py` | 0.35 | Single-position concentration cap |
| `MAX_CORRELATION` | `portfolio_optimizer.py` | 0.85 | Pairwise correlation filter threshold |
| `MAX_PORTFOLIO_SIZE` | `portfolio_optimizer.py` | 15 | Hard cap on number of held positions |
| `MIN_COV_PERIODS` | `portfolio_optimizer.py` | 30 | Minimum overlapping periods for a valid covariance estimate |
| `COV_SHRINKAGE` | `portfolio_optimizer.py` | 1e-4 | Diagonal shrinkage for covariance matrix conditioning |
| `CII_HIGH_RISK_THRESHOLD` / `CII_CRITICAL_THRESHOLD` | `geopolitical_features.py` | 65.0 / 80.0 | Equity-exposure risk-brake trigger points |
| `REGIME_SHIFT_WARNING_THRESHOLD` | `diagnose_signal.py` | 0.75 | Q1-vs-Q4 shift ratio flagged as non-stationary |

### 6.2 Path Configuration

`dashboard_export.py`'s `FRONTEND_DATA_DIR` and `portfolio_optimizer.py`'s `BASE_DIR`/`DB_NAME` are resolved via `Path(__file__).parent`, **not** the process working directory — a deliberate fix for a prior bug where importing these modules from a different `cwd` (e.g. a `backend/` subprocess) produced silent `FileNotFoundError`s that degraded to stale cached JSON on the frontend without surfacing an error.

---

## 7. Modular API / Usage Guide

### 7.1 Full Pipeline Execution

```bash
python main.py
```

### 7.2 Signal Diagnostics (Standalone)

```python
from diagnose_signal import run_full_diagnosis

results = run_full_diagnosis()
# results: {
#   "mutual_information": pd.DataFrame,
#   "regime_shift": pd.DataFrame,
#   "ticker_leakage": pd.DataFrame,
#   "cross_sectional_decomposition": dict | None,
# }
```

### 7.3 Live Inference

```python
from live_predictor import generate_live_predictions

ranking_table = generate_live_predictions(
    output_path="excel_outputs/live_market_predictions.xlsx"
)
```

### 7.4 Portfolio Construction (Custom Risk Parameters)

```python
from portfolio_optimizer import optimize_portfolio

allocation = optimize_portfolio(
    capital=50_000_000,      # Toman
    risk_appetite="medium",  # 'low' | 'medium' | 'high'
    time_horizon="mid",      # 'short' | 'mid' | 'long'
)
```

### 7.5 Dashboard Export (Batch)

```python
from pathlib import Path
from dashboard_export import (
    write_rankings_json,
    write_equity_curve_json,
    write_portfolio_json,
    build_equity_curve_from_backtest,
)

write_rankings_json(rows=ranking_table.to_dict("records"))
equity_points = build_equity_curve_from_backtest(
    Path("excel_outputs/backtest_equity_curve.xlsx")
)
write_equity_curve_json(equity_points)
```

---

## 8. Known Gaps (Explicit, Not Hidden)

| Area | Status |
|---|---|
| Statistical significance of live trading edge | **Not established** — Deflated Sharpe Ratio is NO-GO at current sample size |
| Signal composition | Dominated by macro/USD-regime exposure (75.9%), not idiosyncratic stock-picking |
| Geopolitical (WorldMonitor) integration | Currently broken upstream (`ImportError` on `src.parser`); fails open, does not block pipeline |
| Rate limiting | Not implemented beyond fixed retry counts |
| Cache invalidation | No explicit TTL on the Parquet feature store |
| Backtest sample size | `n_obs=57` — underpowered for a high-confidence DSR read |

A production-readiness document that omits this section is not describing a production-ready system.
