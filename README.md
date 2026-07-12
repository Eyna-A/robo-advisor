# 🇮🇷 TSE Robo-Advisor — AI-Driven Quant Research & Portfolio Pipeline

> An end-to-end machine learning pipeline for signal research, statistically-validated backtesting, and live portfolio construction on the Tehran Stock Exchange (TSE) — built with a diagnostics-first philosophy: every signal is challenged before it's trusted.

---

## 📑 Table of Contents

- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#-tech-stack)
- [🏗️ Architecture / Pipeline Workflow](#-architecture--pipeline-workflow)
- [🎥 Demo](#-demo)
- [⚙️ Installation & Setup](#-installation--setup)
- [🚀 Usage](#-usage)
- [📊 Current Status & Research Findings](#-current-status--research-findings)
- [🗺️ Future Roadmap / Enhancements](#-future-roadmap--enhancements)
- [📁 Project Structure](#-project-structure)

---

## ✨ Key Features

- **Automated ETL pipeline** — retry-safe daily ingestion of OHLC price data for 8 TSE tickers plus the USD/IRR free-market ("Dollar") exchange rate, persisted to SQLite with optimized indexes and exported to Excel for audit.
- **40+ engineered features per ticker** — RSI, MACD, ATR, moving-average distances, rolling volatility, dollar-correlation, beta proxy, capital-increase event features, and macro regime flags, with forward-looking 60-day return labels stored as Parquet.
- **Purged K-Fold LightGBM training** — cross-validation designed to prevent lookahead bias and information leakage across time-ordered folds, with per-fold accuracy/F1/best-iteration reporting.
- **Built-in signal diagnostics suite** (`diagnose_signal.py`) — a hard gate *before* investing in more complex architectures (e.g. LSTM+Attention):
  - Mutual Information ranking (model-free signal strength)
  - Regime-shift / non-stationarity testing (Q1 vs Q4 distribution drift)
  - Ticker-identity leakage testing (is a feature just a hidden stock ID?)
  - Cross-sectional variance decomposition — separates **macro/day-level** signal from **stock-specific** (idiosyncratic) signal
- **Statistically rigorous backtesting** — Sharpe, Sortino, Calmar, and **Deflated Sharpe Ratio** (corrects for multiple-testing bias), benchmarked against both a USD-holding strategy and a naive equal-weighted stock basket.
- **Train/serve skew elimination** — the exact same diagnostic feature-correction function is shared across training, backtesting, and live inference, closing a bug where the live service silently zero-filled its most important feature.
- **Stale/halted-ticker protection** — live predictions are automatically suppressed for suspended tickers (caught a real incident where a 4-month-halted stock still received a "strong buy" signal).
- **Portfolio optimizer** — Sharpe-maximizing SLSQP allocation with correlation-based diversification filtering (drops pairs with `corr > 0.85`), single-position caps, batched SQL (N+1 query elimination — ~100x speedup for large ticker universes), and a **fail-open geopolitical risk brake** that throttles equity exposure during high-instability periods.
- **Atomic dashboard export** — JSON files are written to a temp file and renamed, so the React/Vite frontend never reads a half-written file.
- **Jalali (Persian) calendar aware** end-to-end, from raw data ingestion through the dashboard.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Core Language** | Python 3.11 |
| **ML / Modeling** | LightGBM, scikit-learn (Mutual Information, Decision Trees, cross-validation) |
| **Optimization** | SciPy (`SLSQP` mean-variance / Sharpe-maximizing optimizer) |
| **Data Processing** | pandas, NumPy |
| **Storage** | SQLite (`tsetmc_market_data.db`, `macro_data.db`, `geo_signals.db`), Parquet (feature store), CSV |
| **Reporting** | openpyxl / pandas Excel export |
| **Frontend Integration** | Atomic JSON export → React + Vite dashboard (`RoboAdvisorDashboard.jsx`) |
| **External Signal** | Local WorldMonitor geopolitical Crisis Instability Index (CII) integration, fail-open design |

---

## 🏗️ Architecture / Pipeline Workflow

### Visual Diagram

![TSE Robo-Advisor Architecture Diagram](./architecture-diagram.svg)

### Detailed Dataflow (Text)

```
┌─────────────────────┐     ┌──────────────────────┐
│ dollar_data_ingestion│     │   data_pipeline.py    │
│  (USD/IRR history)   │     │  (8 TSE tickers OHLC) │
└──────────┬───────────┘     └───────────┬───────────┘
           │                             │
           └────────────┬────────────────┘
                         ▼
              SQLite (macro_data.db,
              tsetmc_market_data.db)
                         │
                         ▼
              ┌─────────────────────────┐
              │  feature_engineering.py │  → per-ticker Parquet
              │  (40+ features, 60d     │    feature store
              │   forward labels)       │
              └────────────┬─────────────┘
                            ▼
              ┌─────────────────────────┐
              │     train_model.py       │  Purged K-Fold LightGBM
              │  (diagnostic corrections)│  → lgb_robo_advisor.txt
              └────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                            ▼
   ┌────────────────────┐      ┌────────────────────────┐
   │ diagnose_signal.py  │      │    live_predictor.py    │
   │ MI / regime-shift /  │      │  + geopolitical_features│
   │ leakage / decomp.    │      │  (stale-ticker filter)  │
   └────────────────────┘      └────────────┬─────────────┘
                                              ▼
                                 ┌─────────────────────────┐
                                 │ portfolio_optimizer.py   │
                                 │ SLSQP Sharpe-max +        │
                                 │ correlation filter +      │
                                 │ geopolitical risk brake   │
                                 └────────────┬─────────────┘
                                              ▼
                                 ┌─────────────────────────┐
                                 │   dashboard_export.py    │
                                 │  atomic JSON → frontend  │
                                 └─────────────────────────┘

   (run independently, not part of main.py)
                 ┌─────────────────────────┐
                 │      backtester.py       │
                 │ Sharpe/Sortino/Calmar/    │
                 │ Deflated Sharpe vs USD &  │
                 │ equal-weight benchmarks   │
                 └─────────────────────────┘
```

---

## 🎥 Demo

> 📌 **Add your project demo here.** A short screen recording or GIF of the live dashboard is the single highest-impact addition you can make to this README — it turns "trust me, it works" into "watch it work."

```markdown
![Dashboard Demo](./docs/demo.gif)
```

<!--
Suggested capture checklist:
  1. Live rankings table (from live_rankings.json) — sortable by Alpha Score
  2. Equity curve vs. USD and equal-weight benchmarks (equity_curve.json)
  3. Portfolio allocation pie chart + KPI boxes (portfolio.json)
  4. A terminal recording of `python main.py` running the full 5-stage pipeline end-to-end
  5. `diagnose_signal.py` output — showing the MI / regime-shift / leakage tables
-->

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/tse-robo-advisor.git
cd tse-robo-advisor

# 2. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Clone the local WorldMonitor geopolitical signal source
#    Pipeline runs fine without this — the geopolitical feature/brake
#    fail-opens to neutral values if the folder is missing.
git clone https://github.com/<worldmonitor-fork>/worldmonitor.git ./worldmonitor

# 5. (Optional) Point the pipeline at custom data paths
export TSE_DB_PATH="./tsetmc_market_data.db"
export MACRO_DB_PATH="./macro_data.db"
```

`requirements.txt` (core packages):
```
pandas
numpy
scikit-learn
lightgbm
scipy
openpyxl
requests
```

---

## 🚀 Usage

Run the full 5-stage pipeline (ingestion → features → training → live prediction → portfolio optimization):

```bash
python main.py
```

Run individual stages independently:

```bash
# Signal diagnostics — run this BEFORE trusting any model output
python diagnose_signal.py

# Historical backtest against USD and equal-weight benchmarks
python backtester.py

# Refresh the frontend dashboard JSON only
python -c "from dashboard_export import write_rankings_json; ..."
```

Programmatic portfolio allocation example:

```python
from portfolio_optimizer import optimize_portfolio

allocation = optimize_portfolio(
    capital=50_000_000,      # Toman
    risk_appetite="medium",  # 'low' | 'medium' | 'high'
    time_horizon="mid",      # 'short' | 'mid' | 'long'
)
```

---

## 📊 Current Status & Research Findings

This project treats **model validation as a first-class citizen**, not an afterthought. The diagnostics suite currently reports:

- **Purged K-Fold accuracy:** 35–60% across folds, with `best_iteration` collapsing to 1 in 3 of 5 folds — a signal the current feature set is not yet consistently generalizable.
- **Deflated Sharpe Ratio:** flagged `NO-GO` in the latest backtest run — current out-of-sample performance is not yet statistically distinguishable from a random strategy at this sample size.
- **Cross-sectional decomposition:** ~76% of return variance is explained by common, day-level (macro/USD-regime) movement, vs. ~24% idiosyncratic (stock-specific) variance — meaning the model currently behaves more like a **dollar-regime timing detector** than a pure stock-picker.
- **Backtest alpha:** negative vs. a simple USD-holding benchmark, but strongly positive (+200%) vs. a naive equal-weighted stock basket — suggesting genuine stock-selection skill exists but is currently dominated by unfavorable macro-timing exposure.

These findings actively drive the roadmap below rather than being hidden behind headline return numbers.

> ⚠️ **Disclaimer:** This is a research and educational project. It does not currently demonstrate statistically validated trading alpha and should not be used to make real investment decisions.

---

## 🗺️ Future Roadmap / Enhancements

- Defer LSTM+Attention architecture work until `diagnose_signal.py`'s Definition-of-Done is met (best-iteration reliably >100, Deflated Sharpe clears NO-GO, non-zero F1 on minority classes).
- Explicitly decompose the "macro/dollar regime" signal into a separate hedging/timing overlay, isolated from the stock-ranking model, to unlock the idiosyncratic alpha shown in the equal-weight benchmark comparison.
- Re-evaluate the 60-day label horizon — potentially too long for a regime-heavy, high-volatility market like TSE.
- Replace the currently broken WorldMonitor geopolitical integration (silently fails on `src.parser` import) with a maintained, versioned data source.
- Expand backtesting to rolling-origin / walk-forward windows to increase the statistical power of the Deflated Sharpe estimate (`n_obs=57` is currently small).
- Containerize the pipeline (Docker) and add CI (GitHub Actions) for scheduled nightly runs and regression testing on the diagnostics suite.

---

## 📁 Project Structure

```
.
├── data_pipeline.py            # Raw price ingestion & cleaning
├── dollar_data_ingestion.py    # USD/IRR historical rate sync
├── feature_engineering.py      # Feature + label construction
├── train_model.py              # Purged K-Fold LightGBM training
├── diagnose_signal.py          # MI / regime-shift / leakage / decomposition
├── live_predictor.py           # Live inference + stale-ticker filtering
├── geopolitical_features.py    # WorldMonitor CII integration (fail-open)
├── portfolio_optimizer.py      # SLSQP allocation + risk brake
├── backtester.py                # Statistically-aware backtesting
├── dashboard_export.py         # Atomic JSON export for frontend
└── main.py                      # Orchestrates the 5-stage pipeline
```
