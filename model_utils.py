"""Core preprocessing and prediction utilities for Synapse Retention Engine.

This module mirrors the uploaded notebook pipeline:
- sanitize raw tabular input
- generate physics-informed engineered features
- align columns to the trained feature schema
- predict log1p(Tau_ms) with a Stacking Ensemble model
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TARGET_COL = "Tau_ms"
PAPER_COL = "Paper_ID"

KNOWN_CATEGORICAL_COLS = [
    "Paper_ID",
    "Channel",
    "Solvent",
    "Process",
    "Ion_type",
    "wt",
    "polymer",
    "Cation",
    "Anion",
    "Electrode_type",
]

RAW_INPUT_COLS = [
    "Channel",
    "Solvent",
    "Concentration_mg_ml",
    "Process",
    "Spin_RPM",
    "Annealing_temp_C",
    "Annealing_time_h",
    "Ion_type",
    "wt",
    "polymer",
    "Ion_diffusion",
    "Ion_viscosity",
    "Anion_radius",
    "Cation_radius",
    "Cation",
    "Anion",
    "Gate_voltage_V",
    "Drain_voltage_V",
    "Gate_pulse_width_ms",
    "Pulse_number",
    "Electrode_type",
    "Vth_V",
    "On_off_ratio",
    "Vth_window_V",
    "Operating_temp_C",
]

NUMERIC_LIKE_COLS = [
    "Concentration_mg_ml",
    "Spin_RPM",
    "Annealing_temp_C",
    "Annealing_time_h",
    "Ion_diffusion",
    "Ion_viscosity",
    "Anion_radius",
    "Cation_radius",
    "Gate_voltage_V",
    "Drain_voltage_V",
    "Gate_pulse_width_ms",
    "Pulse_number",
    "Vth_V",
    "On_off_ratio",
    "Vth_window_V",
    "Operating_temp_C",
]


def force_string_keep_nan(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: np.nan if pd.isna(x) else str(x))


def sanitize_dataframe(df: pd.DataFrame, require_tau: bool = True) -> pd.DataFrame:
    df = df.copy()

    if PAPER_COL not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": PAPER_COL})

    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    elif require_tau:
        raise ValueError("Training data requires a Tau_ms column.")

    for col in df.columns:
        if col == TARGET_COL:
            continue

        if col in KNOWN_CATEGORICAL_COLS:
            df[col] = force_string_keep_nan(df[col])
            continue

        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            original_notna = df[col].notna().sum()
            converted = pd.to_numeric(df[col], errors="coerce")
            converted_notna = converted.notna().sum()

            if original_notna > 0 and converted_notna / original_notna >= 0.70:
                df[col] = converted
            else:
                df[col] = force_string_keep_nan(df[col])

    return df.replace([np.inf, -np.inf], np.nan)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if PAPER_COL in df.columns:
        df["paper_year"] = pd.to_numeric(
            df[PAPER_COL].astype(str).str.extract(r"(20\d{2})")[0],
            errors="coerce",
        )

    for col in NUMERIC_LIKE_COLS:
        if col in df.columns:
            value = pd.to_numeric(df[col], errors="coerce")
            df[f"abs_{col}"] = value.abs()
            df[f"log1p_abs_{col}"] = np.log1p(value.abs())

    if {"Gate_voltage_V", "Gate_pulse_width_ms"}.issubset(df.columns):
        gate_v = pd.to_numeric(df["Gate_voltage_V"], errors="coerce").abs()
        pulse_w = pd.to_numeric(df["Gate_pulse_width_ms"], errors="coerce").abs()
        df["gate_pulse_dose"] = gate_v * pulse_w
        df["log1p_gate_pulse_dose"] = np.log1p(df["gate_pulse_dose"].clip(lower=0))

        if "Pulse_number" in df.columns:
            pulse_n = pd.to_numeric(df["Pulse_number"], errors="coerce").abs()
            df["total_gate_dose"] = gate_v * pulse_w * pulse_n
            df["log1p_total_gate_dose"] = np.log1p(df["total_gate_dose"].clip(lower=0))

    if {"Drain_voltage_V", "Gate_pulse_width_ms"}.issubset(df.columns):
        drain_v = pd.to_numeric(df["Drain_voltage_V"], errors="coerce").abs()
        pulse_w = pd.to_numeric(df["Gate_pulse_width_ms"], errors="coerce").abs()
        df["drain_pulse_dose"] = drain_v * pulse_w
        df["log1p_drain_pulse_dose"] = np.log1p(df["drain_pulse_dose"].clip(lower=0))

    if {"Gate_voltage_V", "Drain_voltage_V"}.issubset(df.columns):
        gate_v = pd.to_numeric(df["Gate_voltage_V"], errors="coerce").abs()
        drain_v = pd.to_numeric(df["Drain_voltage_V"], errors="coerce").abs()
        df["voltage_ratio"] = gate_v / (drain_v + 1e-9)

    if {"Ion_diffusion", "Ion_viscosity"}.issubset(df.columns):
        diff = pd.to_numeric(df["Ion_diffusion"], errors="coerce")
        visc = pd.to_numeric(df["Ion_viscosity"], errors="coerce")
        df["ion_mobility_proxy"] = diff / (visc + 1e-9)
        df["log1p_ion_diffusion"] = np.log1p(diff.clip(lower=0))
        df["log1p_ion_viscosity"] = np.log1p(visc.clip(lower=0))

    if {"Anion_radius", "Cation_radius"}.issubset(df.columns):
        anion_r = pd.to_numeric(df["Anion_radius"], errors="coerce")
        cation_r = pd.to_numeric(df["Cation_radius"], errors="coerce")
        df["radius_sum"] = anion_r + cation_r
        df["radius_diff_abs"] = (anion_r - cation_r).abs()
        df["radius_ratio"] = anion_r / (cation_r + 1e-9)

    if {"Concentration_mg_ml", "Gate_voltage_V", "Gate_pulse_width_ms"}.issubset(df.columns):
        conc = pd.to_numeric(df["Concentration_mg_ml"], errors="coerce")
        gate_v = pd.to_numeric(df["Gate_voltage_V"], errors="coerce").abs()
        pulse_w = pd.to_numeric(df["Gate_pulse_width_ms"], errors="coerce").abs()
        df["concentration_gate_dose"] = conc * gate_v * pulse_w
        df["log1p_concentration_gate_dose"] = np.log1p(df["concentration_gate_dose"].clip(lower=0))

    if {"Annealing_temp_C", "Annealing_time_h"}.issubset(df.columns):
        temp = pd.to_numeric(df["Annealing_temp_C"], errors="coerce")
        time_h = pd.to_numeric(df["Annealing_time_h"], errors="coerce")
        df["annealing_thermal_budget"] = temp * time_h
        df["log1p_annealing_thermal_budget"] = np.log1p(df["annealing_thermal_budget"].clip(lower=0))

    return df.replace([np.inf, -np.inf], np.nan)


class KFoldTargetEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=20, n_splits=5, random_state=42):
        self.cols = cols
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = X.copy()
        y = np.asarray(y)
        if self.cols is None:
            self.cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.cols_ = list(self.cols)
        self.global_mean_ = float(np.mean(y))
        self.maps_ = {}

        for col in self.cols_:
            tmp = pd.DataFrame({"cat": X[col].astype(str).fillna("Missing"), "target": y})
            stats = tmp.groupby("cat")["target"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.maps_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_:
            X[f"{col}_te"] = (
                X[col]
                .astype(str)
                .fillna("Missing")
                .map(self.maps_.get(col, {}))
                .fillna(self.global_mean_)
            )
        return X

    def fit_transform(self, X, y=None, **fit_params):
        X = X.copy()
        y = np.asarray(y)
        if self.cols is None:
            self.cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.cols_ = list(self.cols)
        self.global_mean_ = float(np.mean(y))
        self.maps_ = {}

        for col in self.cols_:
            new_col = f"{col}_te"
            X[new_col] = self.global_mean_
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for train_idx, valid_idx in kf.split(X):
                train_part = X.iloc[train_idx]
                tmp = pd.DataFrame({"cat": train_part[col].astype(str).fillna("Missing"), "target": y[train_idx]})
                stats = tmp.groupby("cat")["target"].agg(["mean", "count"])
                smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                    stats["count"] + self.smoothing
                )
                X.loc[X.index[valid_idx], new_col] = (
                    X.iloc[valid_idx][col].astype(str).fillna("Missing").map(smooth).fillna(self.global_mean_).values
                )

            tmp_all = pd.DataFrame({"cat": X[col].astype(str).fillna("Missing"), "target": y})
            stats_all = tmp_all.groupby("cat")["target"].agg(["mean", "count"])
            smooth_all = (stats_all["mean"] * stats_all["count"] + self.global_mean_ * self.smoothing) / (
                stats_all["count"] + self.smoothing
            )
            self.maps_[col] = smooth_all.to_dict()
        return X


def build_best_model(categorical_cols: List[str], numeric_cols: List[str], n_jobs: int = 1) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    stack_model = StackingRegressor(
        estimators=[
            (
                "extra",
                ExtraTreesRegressor(
                    n_estimators=500,
                    random_state=42,
                    max_features=0.6,
                    min_samples_leaf=1,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    max_features=0.8,
                    min_samples_leaf=1,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=3,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=n_jobs,
                    verbosity=0,
                ),
            ),
        ],
        final_estimator=RidgeCV(),
        cv=3,
        n_jobs=n_jobs,
    )

    return Pipeline(
        [
            (
                "target_encoder",
                KFoldTargetEncoderDF(cols=categorical_cols, smoothing=20, n_splits=5, random_state=42),
            ),
            ("preprocess", preprocessor),
            ("model", stack_model),
        ]
    )


def prepare_input_for_model(user_input: Dict[str, Any] | pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    if isinstance(user_input, dict):
        input_df = pd.DataFrame([user_input])
    else:
        input_df = user_input.copy()

    input_df = sanitize_dataframe(input_df, require_tau=False)
    input_df = add_engineered_features(input_df)

    for col in [TARGET_COL, PAPER_COL]:
        if col in input_df.columns:
            input_df = input_df.drop(columns=[col])

    for col in bundle["feature_columns"]:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[bundle["feature_columns"]]
    return input_df.replace([np.inf, -np.inf], np.nan)


def predict_retention_time(user_input: Dict[str, Any] | pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, float]:
    input_df = prepare_input_for_model(user_input, bundle)
    pred_log = float(bundle["model"].predict(input_df)[0])
    pred_tau_ms = float(np.expm1(pred_log))
    return {"pred_log1p_tau_ms": pred_log, "pred_tau_ms": pred_tau_ms}


def load_model_bundle(model_path: str = "models/retention_model.joblib") -> Dict[str, Any]:
    return joblib.load(model_path)


def get_unit_breakdown(tau_ms: float) -> Dict[str, float]:
    seconds = tau_ms / 1000.0
    return {
        "milliseconds": float(tau_ms),
        "seconds": float(seconds),
        "minutes": float(seconds / 60.0),
        "hours": float(seconds / 3600.0),
        "days": float(seconds / 86400.0),
    }
