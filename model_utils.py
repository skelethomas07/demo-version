"""Model utilities for Synapse Retention Engine.

Updated for the platform model:
- log1p(Tau_ms) target
- QuantileTransformer target normalization
- ExtraTreesRegressor
- KFold target encoding + count/frequency encoding for categoricals
- physics-informed feature engineering
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

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

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "missing", "not specified", "-"}


def _is_missing_value(value: Any) -> bool:
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().casefold() in MISSING_TOKENS


def _key(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _clean_text(value: Any) -> Any:
    if _is_missing_value(value):
        return np.nan
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


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
    "pt": "Pt",
    "al": "Al",
    "cu": "Cu",
    "ito": "ITO",
    "moito": "Mo/ITO",
    "itoau": "ITO/Au",
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
        return CHANNEL_MAP.get(k, str(cleaned))
    if col == "polymer":
        return POLYMER_MAP.get(k, str(cleaned))
    if col == "Solvent":
        return SOLVENT_MAP.get(k, str(cleaned).replace("_", "/"))
    return cleaned


def is_spin_process_value(value: Any) -> bool:
    if _is_missing_value(value):
        return False
    return "spin" in str(value).strip().casefold()


def clean_column_name(col: Any) -> str:
    col = str(col).strip()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("/", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("[", "")
    col = col.replace("]", "")
    col = col.replace("%", "percent")
    col = re.sub(r"_+", "_", col)
    return col


def clean_semantic_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wt" in df.columns and "polymer" in df.columns:
        wt_num = pd.to_numeric(df["wt"], errors="coerce")
        polymer_num = pd.to_numeric(df["polymer"], errors="coerce")
        swap_mask = df["wt"].notna() & wt_num.isna() & df["polymer"].notna() & polymer_num.notna()
        if swap_mask.any():
            old_wt = df.loc[swap_mask, "wt"].copy()
            df.loc[swap_mask, "wt"] = polymer_num.loc[swap_mask]
            df.loc[swap_mask, "polymer"] = old_wt
        wt_num = pd.to_numeric(df["wt"], errors="coerce")
        polymer_num = pd.to_numeric(df["polymer"], errors="coerce")
        move_mask = df["polymer"].notna() & polymer_num.notna() & wt_num.isna()
        df.loc[move_mask, "wt"] = polymer_num.loc[move_mask]
        df.loc[df["polymer"].notna() & polymer_num.notna(), "polymer"] = np.nan
        df["wt"] = pd.to_numeric(df["wt"], errors="coerce")

    for col in KNOWN_CATEGORICAL_COLS:
        if col in df.columns and col != PAPER_COL:
            df[col] = df[col].apply(lambda v, c=col: canonicalize_value(c, v))

    if "Process" in df.columns and "Spin_RPM" in df.columns:
        non_spin_mask = ~df["Process"].apply(is_spin_process_value)
        df.loc[non_spin_mask, "Spin_RPM"] = np.nan
    return df


def force_string_keep_nan(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: np.nan if pd.isna(x) else str(x))


def sanitize_dataframe(df: pd.DataFrame, require_tau: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    df.columns = [clean_column_name(c) for c in df.columns]

    rename_map = {
        "Unnamed:_0": PAPER_COL,
        "Unnamed_0": PAPER_COL,
        "Concentration_mg_mL": "Concentration_mg_ml",
        "Concentration": "Concentration_mg_ml",
        "Spin_rpm": "Spin_RPM",
        "spin_RPM": "Spin_RPM",
        "Annealing_temp": "Annealing_temp_C",
        "Annealing_temperature_C": "Annealing_temp_C",
        "Annealing_time": "Annealing_time_h",
        "Gate_voltage": "Gate_voltage_V",
        "Drain_voltage": "Drain_voltage_V",
        "Gate_pulse_width": "Gate_pulse_width_ms",
        "Pulse_width_ms": "Gate_pulse_width_ms",
        "pulse_number": "Pulse_number",
        "Operating_temp": "Operating_temp_C",
        "Cation_radius_A": "Cation_radius",
        "Anion_radius_A": "Anion_radius",
        "Tau": "Tau_ms",
        "tau_ms": "Tau_ms",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if require_tau:
        if TARGET_COL not in df.columns:
            raise KeyError("Tau_ms column is required for training.")
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df[df[TARGET_COL].notna()]
        df = df[df[TARGET_COL] > 0]
    elif TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

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


def get_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9

    gate_v = get_num(df, "Gate_voltage_V")
    drain_v = get_num(df, "Drain_voltage_V")
    pulse_w = get_num(df, "Gate_pulse_width_ms")
    pulse_n = get_num(df, "Pulse_number")

    conc = get_num(df, "Concentration_mg_ml")
    spin = get_num(df, "Spin_RPM")
    ann_temp = get_num(df, "Annealing_temp_C")
    ann_time = get_num(df, "Annealing_time_h")
    op_temp = get_num(df, "Operating_temp_C")

    cation_r = get_num(df, "Cation_radius")
    anion_r = get_num(df, "Anion_radius")
    ion_diff = get_num(df, "Ion_diffusion")
    ion_visc = get_num(df, "Ion_viscosity")

    df["abs_Gate_voltage_V"] = gate_v.abs()
    df["abs_Drain_voltage_V"] = drain_v.abs()
    df["voltage_difference_V"] = (gate_v - drain_v).abs()
    df["voltage_ratio"] = gate_v.abs() / (drain_v.abs() + eps)

    df["gate_pulse_dose"] = gate_v.abs() * pulse_w
    df["total_gate_dose"] = gate_v.abs() * pulse_w * pulse_n
    df["log1p_total_gate_dose"] = np.log1p(df["total_gate_dose"].clip(lower=0))

    df["concentration_gate_dose"] = conc * df["total_gate_dose"]
    df["log1p_concentration"] = np.log1p(conc.clip(lower=0))
    df["log1p_concentration_gate_dose"] = np.log1p(df["concentration_gate_dose"].clip(lower=0))

    df["radius_diff_abs"] = (cation_r - anion_r).abs()
    df["radius_sum"] = cation_r + anion_r
    df["radius_ratio"] = cation_r / (anion_r + eps)

    df["diffusion_viscosity_ratio"] = ion_diff / (ion_visc + eps)
    df["ion_mobility_proxy"] = ion_diff / (ion_visc + eps)

    df["annealing_thermal_budget"] = ann_temp * ann_time
    df["log1p_annealing_time_h"] = np.log1p(ann_time.clip(lower=0))

    df["Operating_temp_K"] = op_temp + 273.15
    df["Annealing_temp_K"] = ann_temp + 273.15
    df["inv_Operating_temp_K"] = 1 / (df["Operating_temp_K"] + eps)
    df["inv_Annealing_temp_K"] = 1 / (df["Annealing_temp_K"] + eps)

    df["spin_x_concentration"] = spin * conc
    df["annealing_temp_x_time"] = ann_temp * ann_time
    df["gate_voltage_x_pulse_number"] = gate_v.abs() * pulse_n
    df["pulse_width_x_pulse_number"] = pulse_w * pulse_n
    df["voltage_diff_x_pulse_number"] = df["voltage_difference_V"] * pulse_n

    if "Process" in df.columns:
        df["is_spin_process"] = df["Process"].apply(lambda v: 1.0 if is_spin_process_value(v) else 0.0)
        df["spin_rpm_effective"] = spin.fillna(0.0) * df["is_spin_process"]
        df["log1p_spin_rpm_effective"] = np.log1p(df["spin_rpm_effective"].clip(lower=0))

    return df.replace([np.inf, -np.inf], np.nan)


class CategoricalStringCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()
            X_clean = X_clean.where(pd.notna(X_clean), "Missing")
            return X_clean.astype(str)
        X_arr = np.asarray(X, dtype=object)
        X_arr[pd.isna(X_arr)] = "Missing"
        return X_arr.astype(str)


class KFoldTargetEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=5, n_splits=5, random_state=42):
        self.cols = cols
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, index=X.index)
        self.cols_ = list(self.cols) if self.cols is not None else X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.global_mean_ = float(y.mean())
        self.maps_ = {}
        for col in self.cols_:
            s = X[col].where(pd.notna(X[col]), "Missing").astype(str)
            stats = pd.DataFrame({"category": s, "target": y}).groupby("category")["target"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (stats["count"] + self.smoothing)
            self.maps_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_:
            s = X[col].where(pd.notna(X[col]), "Missing").astype(str)
            X[f"{col}_te"] = s.map(self.maps_[col]).fillna(self.global_mean_).astype(float)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, y).transform(X)
        X = X.copy()
        y = pd.Series(y, index=X.index)
        self.cols_ = list(self.cols) if self.cols is not None else X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.global_mean_ = float(y.mean())
        for col in self.cols_:
            X[f"{col}_te"] = np.nan
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, valid_idx in kf.split(X):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X.iloc[valid_idx]
            for col in self.cols_:
                s_tr = X_tr[col].where(pd.notna(X_tr[col]), "Missing").astype(str)
                s_val = X_val[col].where(pd.notna(X_val[col]), "Missing").astype(str)
                stats = pd.DataFrame({"category": s_tr, "target": y_tr}).groupby("category")["target"].agg(["mean", "count"])
                smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (stats["count"] + self.smoothing)
                X.loc[X.index[valid_idx], f"{col}_te"] = s_val.map(smooth.to_dict()).fillna(self.global_mean_).values
        self.fit(X.drop(columns=[f"{c}_te" for c in self.cols_], errors="ignore"), y)
        for col in self.cols_:
            X[f"{col}_te"] = X[f"{col}_te"].fillna(self.global_mean_).astype(float)
        return X


class CountFrequencyEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        X = X.copy()
        self.cols_ = list(self.cols) if self.cols is not None else X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.count_maps_ = {}
        self.freq_maps_ = {}
        n = len(X)
        for col in self.cols_:
            s = X[col].where(pd.notna(X[col]), "Missing").astype(str)
            counts = s.value_counts(dropna=False)
            self.count_maps_[col] = counts.to_dict()
            self.freq_maps_[col] = (counts / max(n, 1)).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_:
            s = X[col].where(pd.notna(X[col]), "Missing").astype(str)
            X[f"{col}_count"] = s.map(self.count_maps_[col]).fillna(0).astype(float)
            X[f"{col}_freq"] = s.map(self.freq_maps_[col]).fillna(0).astype(float)
        return X


def build_preprocessor_countfreq(categorical_cols: List[str], numeric_model_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric_model_cols),
            (
                "cat",
                Pipeline([
                    ("string_cleaner", CategoricalStringCleaner()),
                    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def build_model_pipeline_countfreq(
    categorical_cols: List[str],
    numeric_model_cols: List[str],
    model: Optional[ExtraTreesRegressor] = None,
    smoothing: int = 5,
) -> Pipeline:
    if model is None:
        model = ExtraTreesRegressor(
            n_estimators=1500,
            max_features=0.65,
            min_samples_leaf=1,
            min_samples_split=2,
            bootstrap=False,
            random_state=100,
            n_jobs=-1,
        )
    return Pipeline([
        ("target_encoder", KFoldTargetEncoderDF(cols=categorical_cols, smoothing=smoothing, n_splits=5, random_state=42)),
        ("count_freq_encoder", CountFrequencyEncoderDF(cols=categorical_cols)),
        ("preprocess", build_preprocessor_countfreq(categorical_cols, numeric_model_cols)),
        ("model", model),
    ])


def prepare_input_for_model(user_input: Dict[str, Any] | pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    input_df = pd.DataFrame([user_input]) if isinstance(user_input, dict) else user_input.copy()
    input_df = sanitize_dataframe(input_df, require_tau=False)
    input_df = add_engineered_features(input_df)
    for col in [TARGET_COL, PAPER_COL, "paper_year"]:
        if col in input_df.columns:
            input_df = input_df.drop(columns=[col])
    for col in bundle["feature_columns"]:
        if col not in input_df.columns:
            input_df[col] = np.nan
    return input_df[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan)


def predict_retention_time(user_input: Dict[str, Any] | pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, float]:
    input_df = prepare_input_for_model(user_input, bundle)
    pred_log = float(bundle["model"].predict(input_df)[0])
    pred_tau_ms = float(np.expm1(pred_log))
    return {"pred_log1p_tau_ms": pred_log, "pred_tau_ms": pred_tau_ms}


def explain_prediction_local(user_input: Dict[str, Any], bundle: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    current_input = dict(user_input)
    current_df = prepare_input_for_model(current_input, bundle)
    current_log = float(bundle["model"].predict(current_df)[0])

    hidden_rows = []
    features = []
    for col in RAW_INPUT_COLS:
        hidden = dict(current_input)
        hidden[col] = np.nan
        hidden_rows.append(hidden)
        features.append(col)

    hidden_prepared = prepare_input_for_model(pd.DataFrame(hidden_rows), bundle)
    hidden_logs = np.asarray(bundle["model"].predict(hidden_prepared), dtype=float)

    rows = []
    for feature, hidden_log in zip(features, hidden_logs):
        delta_log = current_log - float(hidden_log)
        rows.append({
            "feature": feature,
            "current_value": None if _is_missing_value(current_input.get(feature, np.nan)) else current_input.get(feature),
            "pred_log_without_feature": float(hidden_log),
            "delta_log": float(delta_log),
            "impact_log_abs": float(abs(delta_log)),
            "direction": "increases" if delta_log >= 0 else "decreases",
        })
    df = pd.DataFrame(rows).sort_values("impact_log_abs", ascending=False).reset_index(drop=True)
    all_total = float(df["impact_log_abs"].sum())
    top_df = df.head(max(int(top_n), 1)).copy()
    top_total = float(top_df["impact_log_abs"].sum())
    top_df["share_top_percent"] = top_df["impact_log_abs"] / top_total * 100.0 if top_total > 0 else 0.0
    top_df["share_all_percent"] = top_df["impact_log_abs"] / all_total * 100.0 if all_total > 0 else 0.0
    coverage = top_total / all_total * 100.0 if all_total > 0 else 0.0
    return {
        "current_log1p_tau_ms": current_log,
        "top_drivers": top_df.to_dict("records"),
        "top_driver_count": int(len(top_df)),
        "top_impact_coverage_percent": float(coverage),
        "total_impact_log_abs": all_total,
        "method": "local_feature_hiding_sensitivity",
    }


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
