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
import re
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
    "wt",
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


# -----------------------------------------------------------------------------
# Semantic cleaning helpers
# -----------------------------------------------------------------------------
# The literature table was assembled from many papers, so the raw categorical
# columns contain a few common artifacts: case-only duplicates (Au/au), spelling
# variants (Drop_casting/Drop-casting), and a swapped pair between wt and polymer
# in a subset of rows. These functions make the training and inference schema
# consistent before encoding.

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "missing", "not specified", "-"}


def _is_missing_value(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().casefold() in MISSING_TOKENS


def _key(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _clean_text(value: Any) -> Any:
    if _is_missing_value(value):
        return np.nan
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


PROCESS_MAP = {
    "spincoating": "Spin-coating",
    "spincoat": "Spin-coating",
    "spincoatinguvcrosslinking": "Spin-coating + UV crosslinking",
    "dropcasting": "Drop-casting",
    "dipcoating": "Dip-coating",
    "inkjetprinting": "Inkjet printing",
    "ehdprinting": "EHD printing",
    "ehdnanowireprinting": "EHD nanowire printing",
    "sputtering": "Sputtering",
    "transfer": "Transfer",
    "interfacialassemblytransfer": "Interfacial assembly + transfer",
    "mocvdtransfer": "MOCVD + transfer",
    "mechanicalexfoliation": "Mechanical exfoliation",
    "selfassemblylaseretching": "Self-assembly + laser etching",
    "solventexchangegelation": "Solvent exchange gelation",
    "solutionshearing": "Solution shearing",
}

ELECTRODE_MAP = {
    "au": "Au",
    "gold": "Au",
    "singleaugate": "single Au gate",
    "dualaugate": "dual Au gate",
    "crau": "Cr/Au",
    "aucr": "Au/Cr",
    "tiau": "Ti/Au",
    "agau": "Ag/Au",
    "agagcl": "Ag/AgCl",
    "agnps": "Ag NPs",
    "agnws": "Ag NWs",
    "aunpsagnwspdms": "AuNPs/AgNWs/PDMS",
    "aunpagnwpdms": "AuNPs/AgNWs/PDMS",
    "aunpsagnwspdms": "AuNPs/AgNWs/PDMS",
    "pt": "Pt",
    "al": "Al",
    "cu": "Cu",
    "ito": "ITO",
    "moito": "Mo/ITO",
    "itau": "ITO/Au",
    "itomoo3": "ITO/MoO3",
    "psi": "p-Si",
    "tial": "Ti/Al",
    "mwcnt": "MWCNT",
    "cntpdms": "CNT/PDMS",
    "pedotpss": "PEDOT:PSS",
    "pedotpsssorbitol": "PEDOT:PSS/sorbitol",
    "carbonnanotube": "carbon nanotube",
    "carbonpaper": "carbon paper",
    "activatedcarbon": "activated carbon",
    "eutecticgalliumindium": "eutectic gallium indium",
    "stainlesssteelmesh": "stainless steel mesh",
    "graphenenanowall": "graphene nanowall",
    "tungsten": "W",
}

ION_TYPE_MAP = {
    "iongel": "ion gel",
    "iongel": "ion gel",
    "electrolyte": "electrolyte",
    "solidstateelectrolyte": "solid-state electrolyte",
    "aqueouselectrolyte": "aqueous electrolyte",
    "biopolymer": "biopolymer electrolyte",
}

CATION_MAP = {
    "bmim": "BMIM", "emim": "EMIM", "tma": "TMA", "pyr14": "PYR14", "deme": "DEME",
    "vbimthma": "VBIm_THMA", "hli": "H_Li", "pdadmac": "pDADMAC",
    "li": "Li", "na": "Na", "h": "H", "ag": "Ag", "k": "K", "rb": "Rb", "cs": "Cs",
}

ANION_MAP = {
    "tfsi": "TFSI", "bf4": "BF4", "pf6": "PF6", "meso4": "MeSO4", "otf": "OTf",
    "bf4meso4": "BF4/MeSO4", "clo4": "ClO4", "cl": "Cl", "so3": "SO3",
    "i": "I", "acetate": "acetate", "tfsif4tcnq": "TFSI/F4TCNQ", "f4tcnq": "F4TCNQ",
}

CHANNEL_MAP = {
    "p3ht": "P3HT", "sno2": "SnO2", "pedotpss": "PEDOT:PSS", "pedotpsspam": "PEDOT:PSS/PAM",
    "igzo": "IGZO", "zno": "ZnO", "wo3": "WO3", "ito": "ITO", "ino": "InO", "in2o3": "In2O3",
    "mos2": "MoS2", "pcdtbt": "PCDT-BT", "cntvt": "CNTVT",
}

POLYMER_MAP = {
    "pvdfhfp": "PVDF-HFP", "peo": "PEO", "pegda": "PEGDA", "pdms": "PDMS", "tpu": "TPU",
    "pvdf": "PVDF", "pvdftrfe": "PVDF-TrFE", "pva": "PVA", "pmma": "PMMA", "pvp": "PVP",
    "pssa": "PSSA", "kgm": "KGM", "gelatin": "gelatin", "chitosan": "chitosan",
}

SOLVENT_MAP = {
    "h2o": "H2O", "water": "water", "deionizedwater": "deionized water", "thf": "THF",
    "pgmea": "PGMEA", "dmf": "DMF", "chloroform": "chloroform", "trichloromethane": "chloroform",
    "chlorobenzene": "chlorobenzene", "toluene": "toluene", "acetonitrile": "acetonitrile",
    "ethanol": "ethanol", "acetone": "acetone", "mxylene": "m-xylene", "pxylene": "p-xylene",
    "odichlorobenzene": "o-dichlorobenzene", "12dichlorobenzene": "1,2-dichlorobenzene",
}


def canonicalize_value(col: str, value: Any) -> Any:
    cleaned = _clean_text(value)
    if pd.isna(cleaned):
        return np.nan
    k = _key(cleaned)
    if col == "Process":
        return PROCESS_MAP.get(k, str(cleaned).replace("_", " "))
    if col == "Electrode_type":
        return ELECTRODE_MAP.get(k, str(cleaned).replace("_", "/"))
    if col == "Ion_type":
        return ION_TYPE_MAP.get(k, str(cleaned).replace("_", " "))
    if col == "Cation":
        return CATION_MAP.get(k, str(cleaned))
    if col == "Anion":
        return ANION_MAP.get(k, str(cleaned))
    if col == "Channel":
        return CHANNEL_MAP.get(k, str(cleaned).replace("_", "_"))
    if col == "polymer":
        return POLYMER_MAP.get(k, str(cleaned))
    if col == "Solvent":
        return SOLVENT_MAP.get(k, str(cleaned).replace("_", "/"))
    return cleaned


def is_spin_process_value(value: Any) -> bool:
    if _is_missing_value(value):
        return False
    return "spin" in str(value).strip().casefold()


def clean_semantic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Some rows have wt and polymer matrix swapped: wt=PVDF-HFP, polymer=0.8.
    # Treat wt as a numeric ratio and polymer as a categorical matrix descriptor.
    if "wt" in df.columns and "polymer" in df.columns:
        wt_num = pd.to_numeric(df["wt"], errors="coerce")
        polymer_num = pd.to_numeric(df["polymer"], errors="coerce")
        swap_mask = df["wt"].notna() & wt_num.isna() & df["polymer"].notna() & polymer_num.notna()
        if swap_mask.any():
            old_wt = df.loc[swap_mask, "wt"].copy()
            df.loc[swap_mask, "wt"] = polymer_num.loc[swap_mask]
            df.loc[swap_mask, "polymer"] = old_wt

        # If any numeric value still remains in polymer, move it to wt when wt is missing.
        wt_num = pd.to_numeric(df["wt"], errors="coerce")
        polymer_num = pd.to_numeric(df["polymer"], errors="coerce")
        polymer_numeric_mask = df["polymer"].notna() & polymer_num.notna()
        move_mask = polymer_numeric_mask & wt_num.isna()
        df.loc[move_mask, "wt"] = polymer_num.loc[move_mask]
        df.loc[polymer_numeric_mask, "polymer"] = np.nan
        df["wt"] = pd.to_numeric(df["wt"], errors="coerce")

    for col in KNOWN_CATEGORICAL_COLS:
        if col in df.columns and col != PAPER_COL:
            df[col] = df[col].apply(lambda v, c=col: canonicalize_value(c, v))

    # Spin speed is physically meaningful only for spin-based processes.
    # For transfer, sputtering, drop-casting, etc., keep it as missing rather than 0.
    if "Process" in df.columns and "Spin_RPM" in df.columns:
        non_spin_mask = ~df["Process"].apply(is_spin_process_value)
        df.loc[non_spin_mask, "Spin_RPM"] = np.nan

    return df


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

    df = clean_semantic_dataframe(df)

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

    if "Process" in df.columns:
        df["is_spin_process"] = df["Process"].apply(lambda v: 1.0 if is_spin_process_value(v) else 0.0)
        if "Spin_RPM" in df.columns:
            rpm = pd.to_numeric(df["Spin_RPM"], errors="coerce")
            df["spin_rpm_effective"] = rpm.fillna(0.0) * df["is_spin_process"]
            df["log1p_spin_rpm_effective"] = np.log1p(df["spin_rpm_effective"].clip(lower=0))

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
