"""Paper-grouped P3HT validation for the retention-time model.

The module deliberately separates model selection from external evaluation.
Historical grouped cross-validation chooses the model; external retention
targets are used only after that choice has been made.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import QuantileTransformer

from model_utils import (
    TARGET_COL,
    add_engineered_features,
    build_model_pipeline_countfreq,
    sanitize_dataframe,
)


EXTERNAL_START_DATE = pd.Timestamp("2026-06-01")

SIMILARITY_CATEGORICAL_COLS = [
    "Channel",
    "Solvent",
    "Process",
    "Ion_type",
    "polymer",
    "Cation",
    "Anion",
    "Electrode_type",
]

SIMILARITY_NUMERIC_COLS = [
    "Concentration_mg_ml",
    "Spin_RPM",
    "Annealing_temp_C",
    "Annealing_time_h",
    "wt",
    "Ion_diffusion",
    "Ion_viscosity",
    "Anion_radius",
    "Cation_radius",
    "Operating_temp_C",
]

PHYSICS_DERIVED_FEATURES = {
    "Ion_diffusion",
    "Ion_viscosity",
    "diffusion_viscosity_ratio",
    "ion_mobility_proxy",
    "log_ion_diffusion",
    "log_ion_viscosity",
    "log1p_ion_diffusion",
    "log1p_ion_viscosity",
    "diffusion_length_proxy",
    "radius_diffusion_time_proxy",
}

POST_FABRICATION_FEATURES = {"Vth_V", "On_off_ratio", "Vth_window_V"}
MEASUREMENT_PROTOCOL_FEATURES = {
    "Gate_voltage_V",
    "Drain_voltage_V",
    "Gate_pulse_width_ms",
    "Pulse_number",
    "abs_Gate_voltage_V",
    "abs_Drain_voltage_V",
    "voltage_difference_V",
    "voltage_ratio",
    "gate_pulse_dose",
    "total_gate_dose",
    "log1p_total_gate_dose",
    "concentration_gate_dose",
    "log1p_concentration_gate_dose",
    "gate_voltage_x_pulse_number",
    "pulse_width_x_pulse_number",
    "voltage_diff_x_pulse_number",
    "diffusion_length_proxy",
}
PHYSICS_RAW_COLS = ["Ion_diffusion", "Ion_viscosity", "Anion_radius", "Cation_radius"]


def filter_external_p3ht(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every target-aligned P3HT row in the fixed publication window."""
    result = frame.copy()
    dates = pd.to_datetime(result["publication_date"], errors="coerce")
    channel = result["Channel"].fillna("").astype(str).str.strip().str.casefold()
    tier = result["inclusion_tier"].fillna("").astype(str).str.strip().str.casefold()
    alignment = result["target_alignment"].fillna("").astype(str).str.strip().str.casefold()
    keep = (
        dates.ge(EXTERNAL_START_DATE)
        & channel.eq("p3ht")
        & tier.eq("strict")
        & alignment.str.startswith("direct")
    )
    return result.loc[keep].reset_index(drop=True)


def select_similarity_inputs() -> list[str]:
    """Return pre-fabrication and operating inputs used for domain similarity."""
    return list(SIMILARITY_CATEGORICAL_COLS + SIMILARITY_NUMERIC_COLS)


def _present(value: object) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() not in {"", "nan", "none", "null", "na", "n/a", "-"}


def calculate_similarity_score(query: pd.Series, historical: pd.DataFrame) -> pd.Series:
    """Calculate a target-blind Gower-style similarity against historical rows.

    Missing query inputs are skipped. Numeric distances are scaled by the
    historical interquartile range, while categoricals receive exact-match
    scores. The returned value is between zero and one.
    """
    score_sum = pd.Series(0.0, index=historical.index)
    weight_sum = pd.Series(0.0, index=historical.index)

    for col in SIMILARITY_CATEGORICAL_COLS:
        if col not in historical.columns or col not in query or not _present(query[col]):
            continue
        values = historical[col]
        available = values.map(_present)
        normalized = values.astype(str).str.strip().str.casefold()
        query_value = str(query[col]).strip().casefold()
        score_sum.loc[available] += normalized.loc[available].eq(query_value).astype(float)
        weight_sum.loc[available] += 1.0

    for col in SIMILARITY_NUMERIC_COLS:
        if col not in historical.columns or col not in query or not _present(query[col]):
            continue
        values = pd.to_numeric(historical[col], errors="coerce")
        available = values.notna()
        if not available.any():
            continue
        query_value = float(query[col])
        q1, q3 = values.loc[available].quantile([0.25, 0.75])
        scale = float(q3 - q1)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(values.loc[available].max() - values.loc[available].min())
        if not np.isfinite(scale) or scale <= 0:
            scale = max(abs(query_value), 1.0)
        distance = (values.loc[available] - query_value).abs() / scale
        score_sum.loc[available] += 1.0 / (1.0 + distance)
        weight_sum.loc[available] += 1.0

    return score_sum.div(weight_sum.replace(0.0, np.nan)).fillna(0.0)


def summarize_external_predictions(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize errors without suppressing or replacing any eligible row."""
    actual = pd.to_numeric(frame["Tau_ms"], errors="raise").astype(float)
    predicted = pd.to_numeric(frame["pred_tau_ms"], errors="raise").astype(float)
    if (actual <= 0).any() or (predicted <= 0).any():
        raise ValueError("Actual and predicted retention times must be positive.")
    ape = (predicted - actual).abs() / actual * 100.0
    factors = np.maximum(predicted / actual, actual / predicted)
    log_error = np.log1p(predicted) - np.log1p(actual)
    return {
        "n_rows": int(len(frame)),
        "n_papers": int(frame["Paper_ID"].nunique()),
        "mape_percent": float(ape.mean()),
        "median_ape_percent": float(ape.median()),
        "median_factor_error": float(np.median(factors)),
        "mae_log1p": float(np.mean(np.abs(log_error))),
        "rmse_log1p": float(np.sqrt(np.mean(log_error**2))),
    }


def drop_physics_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the diffusivity and viscosity feature family for ablation."""
    return frame.drop(columns=sorted(PHYSICS_DERIVED_FEATURES), errors="ignore")


def make_group_splits(groups: Iterable[object], n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create paper-disjoint folds and fail if too few papers are available."""
    group_series = pd.Series(list(groups)).reset_index(drop=True)
    unique_groups = int(group_series.nunique())
    if unique_groups < 2:
        raise ValueError("At least two unique Paper_ID values are required.")
    actual_splits = min(int(n_splits), unique_groups)
    splitter = GroupKFold(n_splits=actual_splits)
    positions = np.arange(len(group_series))
    return list(splitter.split(positions, groups=group_series))


def prepare_historical_p3ht(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the historical dataset and retain valid P3HT observations."""
    result = sanitize_dataframe(frame, require_tau=True)
    if "Paper_ID" not in result.columns:
        raise ValueError("Historical data must contain Paper_ID.")
    channel = result["Channel"].fillna("").astype(str).str.strip().str.casefold()
    paper_present = result["Paper_ID"].map(_present)
    return result.loc[channel.eq("p3ht") & paper_present].reset_index(drop=True)


def fit_tau_floor(training_tau: pd.Series, quantile: float = 0.01) -> float:
    """Fit the optional low-target trimming threshold on training targets only."""
    values = pd.to_numeric(training_tau, errors="coerce").dropna()
    values = values.loc[values > 0]
    if values.empty:
        raise ValueError("At least one positive training Tau_ms value is required.")
    return float(values.quantile(float(quantile)))


def _feature_frame(frame: pd.DataFrame, use_physics: bool) -> pd.DataFrame:
    features = add_engineered_features(frame)
    features = features.drop(
        columns=[
            TARGET_COL,
            "Paper_ID",
            "paper_year",
            *sorted(POST_FABRICATION_FEATURES),
            *sorted(MEASUREMENT_PROTOCOL_FEATURES),
        ],
        errors="ignore",
    )
    if not use_physics:
        features = drop_physics_derived_features(features)
    return features.replace([np.inf, -np.inf], np.nan)


def _build_candidate_model(
    X_train: pd.DataFrame,
    candidate: dict[str, Any],
    train_rows: int,
):
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [column for column in X_train.columns if column not in categorical_cols]
    numeric_model_cols = list(numeric_cols)
    for column in categorical_cols:
        numeric_model_cols.extend([f"{column}_te", f"{column}_count", f"{column}_freq"])
    regressor = ExtraTreesRegressor(
        n_estimators=int(candidate.get("n_estimators", 300)),
        max_features=candidate.get("max_features", 0.65),
        min_samples_leaf=int(candidate.get("min_samples_leaf", 1)),
        min_samples_split=int(candidate.get("min_samples_split", 2)),
        bootstrap=bool(candidate.get("bootstrap", False)),
        random_state=int(candidate.get("random_state", 100)),
        n_jobs=-1,
    )
    pipeline = build_model_pipeline_countfreq(
        categorical_cols,
        numeric_model_cols,
        model=regressor,
        smoothing=int(candidate.get("smoothing", 5)),
    )
    if candidate.get("target_transform", "none") == "quantile":
        return TransformedTargetRegressor(
            regressor=pipeline,
            transformer=QuantileTransformer(
                n_quantiles=min(100, int(train_rows)),
                output_distribution="normal",
                random_state=100,
            ),
        )
    return pipeline


def _factor_metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> tuple[float, float]:
    actual = np.expm1(actual_log)
    predicted = np.maximum(np.expm1(predicted_log), 1e-12)
    factor = np.maximum(predicted / actual, actual / predicted)
    ape = np.abs(predicted - actual) / actual * 100.0
    return float(np.median(factor)), float(np.mean(ape))


def evaluate_candidates_grouped(
    historical: pd.DataFrame,
    candidates: Sequence[dict[str, Any]],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate model candidates with identical paper-disjoint folds.

    The returned table contains fold-level rows for both the physics-on and
    physics-off variants. No external targets are accepted by this function.
    """
    frame = prepare_historical_p3ht(historical)
    splits = make_group_splits(frame["Paper_ID"], n_splits=n_splits)
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_name = str(candidate["candidate"])
        for use_physics, variant_name in [(True, "physics_on"), (False, "physics_off")]:
            for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
                train_frame = frame.iloc[train_idx].copy()
                valid_frame = frame.iloc[valid_idx].copy()
                tau_floor = fit_tau_floor(train_frame[TARGET_COL])
                train_frame = train_frame.loc[train_frame[TARGET_COL] >= tau_floor].copy()

                X_train = _feature_frame(train_frame, use_physics=use_physics)
                X_valid = _feature_frame(valid_frame, use_physics=use_physics)
                X_valid = X_valid.reindex(columns=X_train.columns)
                y_train = np.log1p(train_frame[TARGET_COL].to_numpy(dtype=float))
                y_valid = np.log1p(valid_frame[TARGET_COL].to_numpy(dtype=float))

                model = _build_candidate_model(X_train, candidate, len(X_train))
                model.fit(X_train, y_train)
                predicted_log = np.asarray(model.predict(X_valid), dtype=float)
                median_factor, mape = _factor_metrics(y_valid, predicted_log)
                rows.append(
                    {
                        "candidate": candidate_name,
                        "physics_variant": variant_name,
                        "fold": fold_number,
                        "train_rows": int(len(train_frame)),
                        "valid_rows": int(len(valid_frame)),
                        "train_papers": int(train_frame["Paper_ID"].nunique()),
                        "valid_papers": int(valid_frame["Paper_ID"].nunique()),
                        "tau_floor_ms": tau_floor,
                        "mae_log1p": float(mean_absolute_error(y_valid, predicted_log)),
                        "rmse_log1p": float(np.sqrt(mean_squared_error(y_valid, predicted_log))),
                        "median_factor_error": median_factor,
                        "mape_percent": mape,
                    }
                )
    return pd.DataFrame(rows)


def select_best_candidate(
    cv_results: pd.DataFrame,
    external_results: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Select by grouped historical MAE only; external results are ignored."""
    del external_results
    required = {"candidate", "physics_variant", "mae_log1p"}
    missing = required - set(cv_results.columns)
    if missing:
        raise ValueError(f"Missing grouped CV columns: {sorted(missing)}")
    summary = (
        cv_results.groupby(["candidate", "physics_variant"], as_index=False)
        .agg(
            mean_mae_log1p=("mae_log1p", "mean"),
            std_mae_log1p=("mae_log1p", "std"),
            mean_mape_percent=("mape_percent", "mean") if "mape_percent" in cv_results.columns else ("mae_log1p", "mean"),
        )
        .sort_values(["mean_mae_log1p", "candidate", "physics_variant"], kind="stable")
        .reset_index(drop=True)
    )
    return summary.iloc[0].to_dict()


def fill_external_physics_from_historical(
    external: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing ion properties from target-blind historical ion medians.

    The most specific available lookup is used in this order: cation and
    anion, anion, cation, then the historical global median. Existing external
    values are never overwritten.
    """
    result = external.copy()
    source = historical.copy()
    for column in PHYSICS_RAW_COLS:
        if column not in result.columns:
            result[column] = np.nan
        if column not in source.columns:
            source[column] = np.nan
        source[column] = pd.to_numeric(source[column], errors="coerce")

    levels: list[str] = []
    for row_index, row in result.iterrows():
        row_levels: list[str] = []
        cation = str(row.get("Cation", "")).strip().casefold()
        anion = str(row.get("Anion", "")).strip().casefold()
        source_cation = source.get("Cation", pd.Series(index=source.index, dtype=object)).fillna("").astype(str).str.strip().str.casefold()
        source_anion = source.get("Anion", pd.Series(index=source.index, dtype=object)).fillna("").astype(str).str.strip().str.casefold()

        masks = []
        if cation and anion:
            masks.append(("cation+anion", source_cation.eq(cation) & source_anion.eq(anion)))
        if anion:
            masks.append(("anion", source_anion.eq(anion)))
        if cation:
            masks.append(("cation", source_cation.eq(cation)))
        masks.append(("global", pd.Series(True, index=source.index)))

        for column in PHYSICS_RAW_COLS:
            if _present(result.at[row_index, column]):
                continue
            for level, mask in masks:
                values = source.loc[mask, column].dropna()
                if not values.empty:
                    result.at[row_index, column] = float(values.median())
                    row_levels.append(level)
                    break
        levels.append(row_levels[0] if row_levels and len(set(row_levels)) == 1 else ";".join(sorted(set(row_levels))))
    result["physics_imputation_level"] = levels
    return result


def select_nearest_domain_subset(
    frame: pd.DataFrame,
    fraction: float = 1 / 3,
) -> pd.DataFrame:
    """Select the highest input-similarity fraction without using errors."""
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must be in the interval (0, 1].")
    if "input_similarity_score" not in frame.columns:
        raise ValueError("input_similarity_score is required.")
    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    return (
        frame.sort_values(
            ["input_similarity_score", "validation_id"],
            ascending=[False, True],
            kind="stable",
        )
        .head(count)
        .reset_index(drop=True)
    )


def fit_final_model(
    historical: pd.DataFrame,
    candidate: dict[str, Any],
    use_physics: bool,
) -> dict[str, Any]:
    """Fit one selected candidate on every eligible historical P3HT row."""
    frame = prepare_historical_p3ht(historical)
    tau_floor = fit_tau_floor(frame[TARGET_COL])
    frame = frame.loc[frame[TARGET_COL] >= tau_floor].copy()
    X_train = _feature_frame(frame, use_physics=use_physics)
    y_train = np.log1p(frame[TARGET_COL].to_numpy(dtype=float))
    model = _build_candidate_model(X_train, candidate, len(X_train))
    model.fit(X_train, y_train)
    return {
        "model": model,
        "feature_columns": X_train.columns.tolist(),
        "use_physics": bool(use_physics),
        "tau_floor_ms": tau_floor,
        "training_rows": int(len(frame)),
        "training_papers": int(frame["Paper_ID"].nunique()),
        "candidate": dict(candidate),
    }


def predict_with_trained_model(bundle: dict[str, Any], external: pd.DataFrame) -> np.ndarray:
    """Predict positive retention times from a fitted validation bundle."""
    frame = sanitize_dataframe(external, require_tau=False)
    X_external = _feature_frame(frame, use_physics=bool(bundle["use_physics"]))
    X_external = X_external.reindex(columns=bundle["feature_columns"])
    predicted_log = np.asarray(bundle["model"].predict(X_external), dtype=float)
    predicted_tau = np.expm1(predicted_log)
    if not np.isfinite(predicted_tau).all() or (predicted_tau <= 0).any():
        raise ValueError("Model produced a non-finite or non-positive retention time.")
    return predicted_tau


def add_input_similarity_scores(
    external: pd.DataFrame,
    historical: pd.DataFrame,
    neighbors: int = 5,
) -> pd.DataFrame:
    """Add target-blind mean top-neighbor input similarity to each external row."""
    if int(neighbors) < 1:
        raise ValueError("neighbors must be at least one.")
    result = external.copy()
    scores: list[float] = []
    for _, query in result.iterrows():
        row_scores = calculate_similarity_score(query, historical)
        top = row_scores.nlargest(min(int(neighbors), len(row_scores)))
        scores.append(float(top.mean()) if len(top) else 0.0)
    result["input_similarity_score"] = scores
    return result


def restrict_historical_to_external_domain(
    historical: pd.DataFrame,
    external: pd.DataFrame,
    min_papers: int = 5,
) -> pd.DataFrame:
    """Narrow historical rows by external inputs while preserving paper count.

    Filters are applied sequentially only when the remaining set still has at
    least ``min_papers`` unique papers. Retention targets are never inspected.
    """
    result = prepare_historical_p3ht(historical)
    applied: list[str] = []

    def maybe_apply(mask: pd.Series, description: str) -> None:
        nonlocal result
        candidate = result.loc[mask.fillna(False)].copy()
        if candidate["Paper_ID"].nunique() >= int(min_papers):
            result = candidate
            applied.append(description)

    for column in ["Process", "Cation"]:
        if column not in external.columns or column not in result.columns:
            continue
        values = external[column].dropna().astype(str).str.strip().str.casefold().unique()
        if len(values) == 1 and values[0]:
            normalized = result[column].fillna("").astype(str).str.strip().str.casefold()
            maybe_apply(normalized.eq(values[0]), f"{column}={values[0]}")

    if "polymer" in external.columns and "polymer" in result.columns:
        polymers = external["polymer"].dropna().astype(str).str.strip().str.casefold().unique()
        if len(polymers) == 1 and polymers[0]:
            polymer_key = polymers[0]
            normalized = result["polymer"].fillna("").astype(str).str.strip().str.casefold()
            maybe_apply(normalized.str.contains(polymer_key, regex=False), f"polymer contains {polymer_key}")

    result = result.reset_index(drop=True)
    result.attrs["domain_filters"] = applied
    return result


def default_candidates() -> list[dict[str, Any]]:
    """Return a compact candidate grid fixed before external evaluation."""
    return [
        {
            "candidate": "et_leaf1_mf035_s3",
            "n_estimators": 250,
            "max_features": 0.35,
            "min_samples_leaf": 1,
            "smoothing": 3,
            "target_transform": "none",
        },
        {
            "candidate": "et_leaf1_mf065_s5",
            "n_estimators": 250,
            "max_features": 0.65,
            "min_samples_leaf": 1,
            "smoothing": 5,
            "target_transform": "none",
        },
        {
            "candidate": "et_leaf1_mf100_s10",
            "n_estimators": 250,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "smoothing": 10,
            "target_transform": "none",
        },
        {
            "candidate": "et_leaf2_mf050_s3",
            "n_estimators": 250,
            "max_features": 0.5,
            "min_samples_leaf": 2,
            "smoothing": 3,
            "target_transform": "none",
        },
        {
            "candidate": "et_leaf2_mf080_s5",
            "n_estimators": 250,
            "max_features": 0.8,
            "min_samples_leaf": 2,
            "smoothing": 5,
            "target_transform": "none",
        },
        {
            "candidate": "et_leaf4_mf065_s10",
            "n_estimators": 250,
            "max_features": 0.65,
            "min_samples_leaf": 4,
            "smoothing": 10,
            "target_transform": "none",
        },
        {
            "candidate": "et_leaf1_mf065_quantile",
            "n_estimators": 250,
            "max_features": 0.65,
            "min_samples_leaf": 1,
            "smoothing": 5,
            "target_transform": "quantile",
        },
        {
            "candidate": "et_leaf2_mf065_quantile",
            "n_estimators": 250,
            "max_features": 0.65,
            "min_samples_leaf": 2,
            "smoothing": 5,
            "target_transform": "quantile",
        },
    ]


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file: {path}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _add_prediction_metrics(frame: pd.DataFrame, prediction_col: str, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    actual = pd.to_numeric(result[TARGET_COL], errors="raise").astype(float)
    predicted = pd.to_numeric(result[prediction_col], errors="raise").astype(float)
    absolute = (predicted - actual).abs()
    result[f"{prefix}_ape_percent"] = absolute / actual * 100.0
    result[f"{prefix}_factor_error"] = np.maximum(predicted / actual, actual / predicted)
    return result


def _write_report(summary: dict[str, Any], predictions: pd.DataFrame, path: Path) -> None:
    overall = summary["external_physics_on_all"]
    nearest = summary["external_physics_on_nearest_third"]
    ablation = summary["physics_ablation_grouped_cv"]
    table_cols = [
        "validation_id",
        "Anion",
        "Tau_ms",
        "pred_tau_ms",
        "ape_percent",
        "pred_tau_ms_physics_off",
        "physics_off_ape_percent",
        "input_similarity_score",
        "physics_imputation_level",
    ]
    table = predictions[table_cols].copy()
    for column in [
        "Tau_ms",
        "pred_tau_ms",
        "ape_percent",
        "pred_tau_ms_physics_off",
        "physics_off_ape_percent",
        "input_similarity_score",
    ]:
        table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    headers = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    lines = [
        "# P3HT paper-grouped external validation",
        "",
        "The model candidate was selected only by paper-grouped cross-validation on historical P3HT data. External targets were used after model selection.",
        "",
        "## Historical grouped validation",
        "",
        f"Physics-on best candidate: {summary['selected_physics_on']['candidate']}",
        f"Physics-off best candidate: {summary['selected_physics_off']['candidate']}",
        f"Physics-on mean MAE in log1p space: {ablation['physics_on_mean_mae_log1p']:.4f}",
        f"Physics-off mean MAE in log1p space: {ablation['physics_off_mean_mae_log1p']:.4f}",
        "",
        "## External P3HT result",
        "",
        f"All eligible rows: {overall['n_rows']} from {overall['n_papers']} paper",
        f"All-row MAPE: {overall['mape_percent']:.2f}%",
        f"All-row median factor error: {overall['median_factor_error']:.3f}x",
        f"Highest input-similarity third MAPE: {nearest['mape_percent']:.2f}%",
        f"Highest input-similarity third rows: {', '.join(summary['nearest_validation_ids'])}",
        "",
        "The highest-similarity subset is selected from materials and process inputs only. It is a scope-limited case result, not the overall external error.",
        "",
        "## Row results",
        "",
        headers,
        separator,
        *body,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(
    historical_data: str | Path,
    external_data: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run grouped model selection, physics ablation, and external evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    historical_raw = _load_table(historical_data)
    historical = prepare_historical_p3ht(historical_raw)
    external_raw = _load_table(external_data)
    external = filter_external_p3ht(external_raw)
    if external.empty:
        raise ValueError("No eligible post-May-2026 strict P3HT external rows were found.")
    if external["validation_id"].duplicated().any():
        raise ValueError("External validation_id values must be unique.")
    if pd.to_numeric(external[TARGET_COL], errors="coerce").isna().any() or (pd.to_numeric(external[TARGET_COL], errors="coerce") <= 0).any():
        raise ValueError("Every eligible external Tau_ms value must be positive.")

    historical_domain = restrict_historical_to_external_domain(
        historical,
        external,
        min_papers=8,
    )
    domain_filters = list(historical_domain.attrs.get("domain_filters", []))

    candidates = default_candidates()
    cv_results = evaluate_candidates_grouped(historical_domain, candidates, n_splits=5)
    cv_results.to_csv(output_dir / "p3ht_grouped_cv.csv", index=False)

    selected_by_variant: dict[str, dict[str, Any]] = {}
    candidate_map = {candidate["candidate"]: candidate for candidate in candidates}
    predictions = add_input_similarity_scores(external, historical, neighbors=5)
    predictions = fill_external_physics_from_historical(predictions, historical)

    for variant, use_physics in [("physics_on", True), ("physics_off", False)]:
        variant_cv = cv_results.loc[cv_results["physics_variant"] == variant]
        selection = select_best_candidate(variant_cv)
        selected_by_variant[variant] = selection
        candidate = dict(candidate_map[str(selection["candidate"])])
        candidate["n_estimators"] = 1000
        bundle = fit_final_model(historical_domain, candidate, use_physics=use_physics)
        prediction_col = "pred_tau_ms" if use_physics else "pred_tau_ms_physics_off"
        predictions[prediction_col] = predict_with_trained_model(bundle, predictions)

    predictions = _add_prediction_metrics(predictions, "pred_tau_ms", "physics_on")
    predictions = _add_prediction_metrics(predictions, "pred_tau_ms_physics_off", "physics_off")
    predictions["ape_percent"] = predictions["physics_on_ape_percent"]
    predictions["factor_error"] = predictions["physics_on_factor_error"]
    predictions.to_csv(output_dir / "p3ht_external_predictions.csv", index=False)

    nearest = select_nearest_domain_subset(predictions, fraction=1 / 3)
    overall_summary = summarize_external_predictions(predictions)
    nearest_summary = summarize_external_predictions(nearest)
    physics_off_summary = summarize_external_predictions(
        predictions.rename(columns={"pred_tau_ms": "pred_tau_ms_physics_on_original", "pred_tau_ms_physics_off": "pred_tau_ms"})
    )
    summary = {
        "historical_p3ht_rows": int(len(historical)),
        "historical_p3ht_papers": int(historical["Paper_ID"].nunique()),
        "domain_training_rows": int(len(historical_domain)),
        "domain_training_papers": int(historical_domain["Paper_ID"].nunique()),
        "domain_filters": domain_filters,
        "external_publication_start": str(EXTERNAL_START_DATE.date()),
        "selected_physics_on": selected_by_variant["physics_on"],
        "selected_physics_off": selected_by_variant["physics_off"],
        "physics_ablation_grouped_cv": {
            "physics_on_mean_mae_log1p": float(selected_by_variant["physics_on"]["mean_mae_log1p"]),
            "physics_off_mean_mae_log1p": float(selected_by_variant["physics_off"]["mean_mae_log1p"]),
        },
        "external_physics_on_all": overall_summary,
        "external_physics_on_nearest_third": nearest_summary,
        "external_physics_off_all": physics_off_summary,
        "nearest_validation_ids": nearest["validation_id"].astype(str).tolist(),
        "selection_rule": "candidate chosen by mean paper-grouped historical MAE_log1p; external targets excluded",
        "nearest_subset_rule": "top one-third by mean five-neighbor input similarity; Tau_ms and prediction errors excluded",
    }
    (output_dir / "p3ht_external_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_report(summary, predictions, output_dir / "p3ht_external_report.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-data", required=True)
    parser.add_argument("--external-data", required=True)
    parser.add_argument("--out", default="validation_results")
    args = parser.parse_args()
    summary = run_validation(args.historical_data, args.external_data, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
