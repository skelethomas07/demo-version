from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from model_utils import (
    RAW_INPUT_COLS,
    TARGET_COL,
    add_engineered_features,
    build_model_pipeline_countfreq,
    sanitize_dataframe,
)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_schema(df_clean: pd.DataFrame, X: pd.DataFrame, categorical_cols, numeric_cols, performance: Dict[str, float]) -> Dict[str, Any]:
    numeric_stats: Dict[str, Dict[str, Any]] = {}
    categorical_options: Dict[str, list] = {}

    for col in RAW_INPUT_COLS:
        if col not in df_clean.columns:
            continue
        if col in categorical_cols or df_clean[col].dtype == "object" or str(df_clean[col].dtype) == "category":
            vals = df_clean[col].dropna().astype(str).sort_values().unique().tolist()
            categorical_options[col] = vals
        else:
            s = pd.to_numeric(df_clean[col], errors="coerce").dropna()
            if len(s):
                numeric_stats[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                    "mean": float(s.mean()),
                }

    return {
        "model_family": "QuantileTarget + ExtraTrees + TargetEncoding + CountFrequencyEncoding",
        "target": "log1p(Tau_ms)",
        "raw_input_cols": RAW_INPUT_COLS,
        "feature_columns": X.columns.tolist(),
        "input_categorical_cols": [c for c in RAW_INPUT_COLS if c in categorical_options],
        "categorical_cols": list(categorical_cols),
        "numeric_cols": list(numeric_cols),
        "numeric_stats": numeric_stats,
        "categorical_options": categorical_options,
        "test_performance": performance,
    }


def train(data_path: str, output_dir: str = "models") -> None:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_path.suffix.lower() in [".xlsx", ".xls"]:
        df_raw = pd.read_excel(data_path)
    elif data_path.suffix.lower() == ".csv":
        df_raw = pd.read_csv(data_path)
    else:
        raise ValueError("지원 형식: .csv, .xlsx, .xls")

    df = sanitize_dataframe(df_raw, require_tau=True)
    df = add_engineered_features(df)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()

    tau_floor = df[TARGET_COL].quantile(0.01)
    df = df[df[TARGET_COL] >= tau_floor].copy()

    y = np.log1p(df[TARGET_COL].values.astype(float))
    X = df.drop(columns=[TARGET_COL])
    X = X.drop(columns=["Paper_ID", "paper_year"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    target_encoded_cols = [f"{c}_te" for c in categorical_cols]
    count_freq_cols = []
    for c in categorical_cols:
        count_freq_cols.append(f"{c}_count")
        count_freq_cols.append(f"{c}_freq")
    numeric_model_cols = numeric_cols + target_encoded_cols + count_freq_cols

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    extra_final = ExtraTreesRegressor(
        n_estimators=1500,
        max_features=0.65,
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=False,
        random_state=100,
        n_jobs=-1,
    )
    base_pipe = build_model_pipeline_countfreq(categorical_cols, numeric_model_cols, extra_final, smoothing=5)
    final_model = TransformedTargetRegressor(
        regressor=base_pipe,
        transformer=QuantileTransformer(
            n_quantiles=min(200, len(y_train)),
            output_distribution="normal",
            random_state=100,
        ),
    )
    final_model.fit(X_train, y_train)

    pred = final_model.predict(X_test)
    performance = {
        "R2_log": float(r2_score(y_test, pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE_log": float(mean_absolute_error(y_test, pred)),
        "tau_floor_bottom_1pct_ms": float(tau_floor),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "raw_rows": int(len(df_raw)),
        "used_rows_after_cleaning": int(len(df)),
    }

    schema = build_schema(df, X, categorical_cols, numeric_cols, performance)
    bundle = {
        "model": final_model,
        "schema": schema,
        "feature_columns": X.columns.tolist(),
        "input_categorical_cols": schema["input_categorical_cols"],
        "model_type": schema["model_family"],
        "test_performance": performance,
    }

    joblib.dump(bundle, output_dir / "retention_model.joblib")
    with open(output_dir / "input_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/retention_dataset.xlsx")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    train(args.data, args.out)
