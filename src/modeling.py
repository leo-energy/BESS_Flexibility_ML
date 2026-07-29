#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BESS Flexibility ML Pipeline
============================
Random Forest + SHAP pipeline for day-ahead electricity price forecasting
and BESS arbitrage optimization.

Key features:
- ENTSO-E data loading with UTC alignment
- Feature engineering (lags + temporal)
- Random Forest training and evaluation
- SHAP-based model interpretability

Author: Leonardo Xi
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import shap


def load_entsoe_data(market_files: dict) -> pd.DataFrame:
    """
    Load and synchronize ENTSO-E hourly price data from multiple markets.

    Aligns all market data to UTC standard and handles telemetry gaps
    via linear interpolation.

    Args:
        market_files: Dictionary mapping market codes to file paths.
                      Example: {'NL': 'data/NL_prices.csv', ...}

    Returns:
        DataFrame with synchronized hourly price data for all markets.
    """
    df_list = []
    for market, path in market_files.items():
        temp_df = pd.read_csv(path)
        temp_df.iloc[:, 0] = pd.to_datetime(temp_df.iloc[:, 0])
        temp_df['Actual_Time'] = temp_df.iloc[:, 0] + pd.to_timedelta(
            temp_df['Position'] - 1, unit='h'
        )
        temp_df.set_index('Actual_Time', inplace=True)
        temp_df = temp_df[['Price']]
        temp_df.columns = [f'Price_{market}']
        df_list.append(temp_df)

    df = pd.concat(df_list, axis=1)
    df = df.interpolate(method='linear')
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct lagged price features and temporal indicators for ML.

    Creates:
        - Lag1: 1-hour price inertia
        - Lag24: 24-hour daily seasonality
        - hour: Hour of day (0-23)
        - dayofweek: Day of week (0-6)

    Args:
        df: DataFrame with raw price columns (Price_NL, Price_ES, Price_PT)

    Returns:
        DataFrame with additional feature columns, NaN rows dropped.
    """
    model_df = df.copy()

    # Lagged features (1-hour and 24-hour)
    for m in ['NL', 'ES', 'PT']:
        model_df[f'Lag1_{m}'] = model_df[f'Price_{m}'].shift(1)
        model_df[f'Lag24_{m}'] = model_df[f'Price_{m}'].shift(24)

    # Temporal features
    model_df['hour'] = model_df.index.hour
    model_df['dayofweek'] = model_df.index.dayofweek

    return model_df.dropna()


def train_random_forest(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100):
    """
    Train Random Forest regressor and evaluate on test set.

    Args:
        X: Feature DataFrame
        y: Target Series (Price_NL)
        n_estimators: Number of trees in the forest

    Returns:
        Tuple of (trained model, X_test, y_test, y_pred)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} EUR/MWh")

    return rf, X_test, y_test, y_pred


def run_shap_analysis(model, X_sample: pd.DataFrame):
    """
    Run SHAP analysis for model interpretability.

    Uses TreeExplainer optimized for Random Forest models.
    Generates a summary plot showing feature contributions.

    Args:
        model: Trained Random Forest model
        X_sample: Feature DataFrame subset for SHAP computation

    Returns:
        SHAP values array
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)
    return shap_values


def main():
    """Main execution pipeline."""
    # Configure data paths
    files = {
        'NL': 'data/NL_prices.csv',
        'ES': 'data/ES_prices.csv',
        'PT': 'data/PT_prices.csv'
    }

    # Load and align data
    df = load_entsoe_data(files)

    # Engineer features
    df_feat = create_features(df)

    # Prepare features and target
    X = df_feat.drop(['Price_NL', 'Price_ES', 'Price_PT'], axis=1)
    y = df_feat['Price_NL']

    # Train and evaluate
    rf_model, X_test, y_test, y_pred = train_random_forest(X, y)

    # SHAP interpretability (sample 500 rows for efficiency)
    run_shap_analysis(rf_model, X_test.iloc[:500])


if __name__ == "__main__":
    main()
