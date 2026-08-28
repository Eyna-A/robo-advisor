from __future__ import annotations

import logging
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from train_model import FEATURE_COLS, load_and_combine_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("diagnose_signal")

REGIME_SHIFT_WARNING_THRESHOLD = 0.75


def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Computes Mutual Information (MI) between each feature and the target label.
    Unlike tree feature importance, this metric is less susceptible to overfitting
    as no model is fitted, measuring raw statistical dependence.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_classif(
            X.fillna(0.0).to_numpy(), y.to_numpy(),
            discrete_features=False, random_state=42, n_neighbors=5,
        )
    return (
        pd.DataFrame({"feature": X.columns, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


def compute_regime_shift(df: pd.DataFrame, feature_cols: list[str], date_col: str = "jalali_date") -> pd.DataFrame:
    """
    Compares the feature distribution between the first quarter (Q1) and last quarter (Q4) of the timeframe.
    """
    unique_dates = np.sort(df[date_col].unique())
    n = len(unique_dates)
    if n < 8:
        raise ValueError("Insufficient data for regime-shift test (fewer than 8 unique dates).")

    q1_dates = set(unique_dates[: n // 4].tolist())
    q4_dates = set(unique_dates[3 * n // 4:].tolist())

    rows = []
    for feat in feature_cols:
        q1_vals = df.loc[df[date_col].isin(q1_dates), feat].dropna()
        q4_vals = df.loc[df[date_col].isin(q4_dates), feat].dropna()
        if len(q1_vals) < 5 or len(q4_vals) < 5:
            continue
        std_q1 = q1_vals.std()
        shift_ratio = abs(q4_vals.mean() - q1_vals.mean()) / (std_q1 + 1e-9)
        rows.append({
            "feature": feat,
            "mean_q1": q1_vals.mean(),
            "mean_q4": q4_vals.mean(),
            "std_q1": std_q1,
            "shift_ratio": shift_ratio,
            "verdict": "🚨 non-stationary" if shift_ratio > REGIME_SHIFT_WARNING_THRESHOLD else "✅ Stable",
        })

    return pd.DataFrame(rows).sort_values("shift_ratio", ascending=False).reset_index(drop=True)


def compute_ticker_leakage(df: pd.DataFrame, feature_cols: list[str], ticker_col: str = "ticker_code") -> pd.DataFrame:
    """
    Evaluates how accurately a shallow decision tree can infer ticker identity from a single feature.
    High accuracy relative to baseline indicates potential ticker-leakage rather than a generalizable signal.
    """
    tickers = df[ticker_col].astype("category")
    n_tickers = tickers.nunique()
    baseline_acc = 1.0 / n_tickers

    rows = []
    for feat in feature_cols:
        X = df[[feat]].fillna(0.0).to_numpy()
        y = tickers.cat.codes.to_numpy()
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        try:
            scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
            acc = float(np.mean(scores))
        except Exception:
            acc = np.nan
        rows.append({
            "feature": feat,
            "ticker_id_accuracy": acc,
            "baseline_random": baseline_acc,
            "leakage_ratio": acc / baseline_acc if baseline_acc > 0 else np.nan,
            "verdict": "🚨 Suspected ticker-leakage" if (acc / baseline_acc) > 2.0 else "✅ Acceptable",
        })

    return pd.DataFrame(rows).sort_values("leakage_ratio", ascending=False).reset_index(drop=True)


def compute_cross_sectional_decomposition(df: pd.DataFrame, date_col: str = "jalali_date") -> Dict[str, float] | None:
    """
    Decomposes the variance of raw return margin (stock return minus USD return over 60 days)
    into macro/common variance and stock-specific (alpha) variance.
    """
    required = {"future_stock_return_60d", "future_market_return_60d"}
    if not required.issubset(df.columns):
        logger.warning(
            "⚠️ Raw columns future_stock_return_60d/future_market_return_60d "
            "not found in combined data; skipping Part D test."
        )
        return None

    margin = (df["future_stock_return_60d"] - df["future_market_return_60d"]).dropna()
    if margin.empty:
        logger.warning("⚠️ Raw margin is empty; skipping Part D test.")
        return None

    valid_dates = df.loc[margin.index, date_col]
    day_means = margin.groupby(valid_dates).transform("mean")

    total_var = float(margin.var())
    between_day_var = float(day_means.var())
    within_day_var = float((margin - day_means).var())
    between_day_share = between_day_var / total_var if total_var > 0 else float("nan")

    logger.info(f"Total Margin Variance (Stock - USD, 60d): {total_var:.6f}")
    logger.info(f"  ├─ Between-day Variance (Macro/USD shared across tickers): "
                f"{between_day_var:.6f} ({between_day_share:.1%})")
    logger.info(f"  └─ Within-day Variance (Stock-specific / True usable alpha): "
                f"{within_day_var:.6f} ({(1 - between_day_share):.1%})")

    if between_day_share > 0.5:
        logger.warning(
            "🚨 Over 50% of target variance is explained at the daily/macro level rather than the stock level. "
            "The model is primarily learning market-timing rather than stock-picking."
        )
    else:
        logger.info("✅ Within-day variance (stock-specific) dominates -- signal is primarily stock-specific.")

    return {
        "total_var": total_var,
        "between_day_var": between_day_var,
        "within_day_var": within_day_var,
        "between_day_share": between_day_share,
    }


def run_full_diagnosis() -> Dict[str, pd.DataFrame]:
    """
    Runs the full diagnostics pipeline including Mutual Information, Regime Shift, Ticker Leakage,
    and Cross-Sectional Decomposition analysis.
    """
    logger.info("📂 Loading combined feature data...")
    df = load_and_combine_features()
    train_val = df[df["label"].notna()].reset_index(drop=True)

    X = train_val[FEATURE_COLS]
    y = train_val["label"].astype(int)

    logger.info(f"📊 {len(train_val)} labeled rows available for diagnosis.")

    logger.info("\n" + "=" * 70)
    logger.info("Part A: Mutual Information (Raw Signal, Independent of Overfitting)")
    logger.info("=" * 70)
    mi_df = compute_mutual_information(X, y)
    logger.info("\n" + mi_df.to_string(index=False))

    logger.info("\n" + "=" * 70)
    logger.info("Part B: Regime-Shift Test (Feature Stability between Q1 and Q4)")
    logger.info("=" * 70)
    shift_df = compute_regime_shift(train_val, FEATURE_COLS)
    logger.info("\n" + shift_df.to_string(index=False))
    n_unstable = (shift_df["verdict"].str.contains("🚨")).sum()
    if n_unstable > 0:
        logger.warning(f"⚠️ {n_unstable} out of {len(shift_df)} features were detected as non-stationary.")

    logger.info("\n" + "=" * 70)
    logger.info("Part C: Ticker-Leakage Test (Does Feature Leak Ticker Identity?)")
    logger.info("=" * 70)
    if "ticker_code" in train_val.columns:
        leak_df = compute_ticker_leakage(train_val, FEATURE_COLS)
        logger.info("\n" + leak_df.to_string(index=False))
        n_leaky = (leak_df["verdict"].str.contains("🚨")).sum()
        if n_leaky > 0:
            logger.warning(f"⚠️ {n_leaky} out of {len(leak_df)} features are suspected of ticker-leakage.")
    else:
        logger.warning("⚠️ ticker_code column not found; skipping Part C test.")
        leak_df = pd.DataFrame()

    logger.info("\n" + "=" * 70)
    logger.info("Part D: Cross-Sectional Decomposition (Market-Timing vs Stock-Picking)")
    logger.info("=" * 70)
    decomposition = compute_cross_sectional_decomposition(train_val)

    return {
        "mutual_information": mi_df,
        "regime_shift": shift_df,
        "ticker_leakage": leak_df,
        "cross_sectional_decomposition": decomposition,
    }


if __name__ == "__main__":
    run_full_diagnosis()