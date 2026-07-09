"""Train the retention-time model from the original spreadsheet.

Usage:
    python train_model.py --data data/retention_dataset.xlsx --out models/retention_model.joblib

The deployed app does not need the spreadsheet. It only needs the trained model bundle.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model_utils import (
    KNOWN_CATEGORICAL_COLS,
    PAPER_COL,
    RAW_INPUT_COLS,
    TARGET_COL,
    add_engineered_features,
    build_best_model,
    sanitize_dataframe,
)


def build_schema(df_raw: pd.DataFrame, input_columns: list[str]) -> dict:
    df = sanitize_dataframe(df_raw, require_tau=True)
    categorical_cols = [c for c in input_columns if c in KNOWN_CATEGORICAL_COLS and c != PAPER_COL]
    numeric_cols = [c for c in input_columns if c not in categorical_cols]

    categorical_options = {}
    for col in categorical_cols:
        categorical_options[col] = df[col].dropna().astype(str).value_counts().index.tolist()[:200]

    numeric_stats = {}
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            numeric_stats[col] = {"min": None, "q25": None, "median": None, "q75": None, "max": None}
        else:
            numeric_stats[col] = {
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
    return {
        "input_columns": input_columns,
        "input_categorical_cols": categorical_cols,
        "input_numeric_cols": numeric_cols,
        "categorical_options": categorical_options,
        "numeric_stats": numeric_stats,
    }


def train(data_path: str, out_path: str) -> dict:
    df_raw = pd.read_excel(data_path)
    df = sanitize_dataframe(df_raw, require_tau=True)
    df = add_engineered_features(df)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()
    tau_floor = float(df[TARGET_COL].quantile(0.01))
    df = df[df[TARGET_COL] >= tau_floor].copy()

    y = np.log1p(df[TARGET_COL].values.astype(float))
    X = df.drop(columns=[TARGET_COL])
    if PAPER_COL in X.columns:
        X = X.drop(columns=[PAPER_COL])
    X = X.replace([np.inf, -np.inf], np.nan)

    feature_columns = X.columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    target_encoded_cols = [f"{col}_te" for col in categorical_cols]
    numeric_cols = [col for col in X.columns if col not in categorical_cols] + target_encoded_cols

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_best_model(categorical_cols, numeric_cols, n_jobs=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    input_columns = [c for c in RAW_INPUT_COLS if c in df_raw.columns]
    schema = build_schema(df_raw, input_columns)
    performance = {
        "R2_log": float(r2_score(y_test, pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE_log": float(mean_absolute_error(y_test, pred)),
    }

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target_encoded_cols": target_encoded_cols,
        "input_columns": input_columns,
        "input_categorical_cols": schema["input_categorical_cols"],
        "input_numeric_cols": schema["input_numeric_cols"],
        "schema": schema,
        "tau_floor": tau_floor,
        "target_transform": "log1p",
        "inverse_transform": "expm1",
        "training_data_shape": {
            "raw_rows": int(df_raw.shape[0]),
            "raw_cols": int(df_raw.shape[1]),
            "rows_after_tau_filter": int(df.shape[0]),
            "feature_count_after_engineering": int(len(feature_columns)),
        },
        "test_performance": performance,
        "model_type": "StackingRegressor(ExtraTrees + RandomForest + XGBoost, RidgeCV final estimator)",
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)
    with open(out_path.parent / "input_schema.json", "w", encoding="utf-8") as f:
        json.dump({k: bundle[k] for k in ["input_columns", "input_categorical_cols", "input_numeric_cols", "schema", "training_data_shape", "test_performance", "model_type"]}, f, ensure_ascii=False, indent=2)
    return performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/retention_dataset.xlsx")
    parser.add_argument("--out", default="models/retention_model.joblib")
    args = parser.parse_args()
    result = train(args.data, args.out)
    print(json.dumps(result, indent=2))
