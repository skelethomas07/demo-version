"""Reproducible external validation for the retention-time model.

The repository model predicts log1p(Tau_ms). This module converts the output
back to milliseconds, calculates row-level errors, and reports metrics that are
appropriate for a target spanning several orders of magnitude.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model_utils import RAW_INPUT_COLS, prepare_input_for_model


REQUIRED_METADATA_COLS = ["validation_id", "Paper_ID", "Tau_ms", "inclusion_tier"]
ALLOWED_TIERS = {"strict", "supplementary"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in {"", "nan", "none", "null", "na", "n/a", "-"}


def load_validation_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the hand-extracted external dataset."""
    frame = pd.read_csv(path)
    return validate_external_rows(frame)


def validate_external_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail early on invalid targets, duplicate IDs, or unsupported tiers."""
    missing_cols = [col for col in REQUIRED_METADATA_COLS if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    result = frame.copy()
    result["Tau_ms"] = pd.to_numeric(result["Tau_ms"], errors="coerce")
    if result["Tau_ms"].isna().any() or (result["Tau_ms"] <= 0).any():
        raise ValueError("Every Tau_ms value must be finite and positive.")
    if result["validation_id"].duplicated().any():
        duplicated = result.loc[result["validation_id"].duplicated(), "validation_id"].tolist()
        raise ValueError(f"Duplicate validation_id values: {duplicated}")
    invalid_tiers = sorted(set(result["inclusion_tier"].dropna()) - ALLOWED_TIERS)
    if invalid_tiers:
        raise ValueError(f"Unsupported inclusion_tier values: {invalid_tiers}")

    for col in RAW_INPUT_COLS:
        if col not in result.columns:
            result[col] = np.nan
    return result


def predict_with_bundle(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Predict row-by-row to match the single-condition app semantics."""
    pred_log = []
    for _, row in frame[RAW_INPUT_COLS].iterrows():
        prepared = prepare_input_for_model(pd.DataFrame([row.to_dict()]), bundle)
        pred_log.append(float(bundle["model"].predict(prepared)[0]))
    pred_log = np.asarray(pred_log, dtype=float)
    pred_tau = np.expm1(pred_log)
    if not np.isfinite(pred_tau).all() or (pred_tau <= 0).any():
        raise ValueError("Model produced a non-finite or non-positive retention time.")

    result = frame.copy()
    result["pred_log1p_tau_ms"] = pred_log
    result["pred_tau_ms"] = pred_tau
    return result


def audit_model_inputs(frame: pd.DataFrame, schema_path: str | Path) -> pd.DataFrame:
    """Count missing, unseen categorical, and out-of-training-range inputs."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    categorical_cols = schema.get("input_categorical_cols", [])
    options = schema.get("categorical_options", {})
    numeric_stats = schema.get("numeric_stats", {})
    rows: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        missing_fields = [col for col in RAW_INPUT_COLS if col not in row or _is_missing(row[col])]
        unseen_fields = []
        for col in categorical_cols:
            value = row.get(col, np.nan)
            if not _is_missing(value) and str(value) not in set(map(str, options.get(col, []))):
                unseen_fields.append(col)

        out_of_range_fields = []
        for col, stats in numeric_stats.items():
            value = row.get(col, np.nan)
            if _is_missing(value):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            minimum = stats.get("min")
            maximum = stats.get("max")
            if (minimum is not None and numeric < float(minimum)) or (
                maximum is not None and numeric > float(maximum)
            ):
                out_of_range_fields.append(col)

        rows.append(
            {
                "missing_raw_input_count": len(missing_fields),
                "missing_raw_input_fields": ";".join(missing_fields),
                "unseen_categorical_count": len(unseen_fields),
                "unseen_categorical_fields": ";".join(unseen_fields),
                "out_of_range_numeric_count": len(out_of_range_fields),
                "out_of_range_numeric_fields": ";".join(out_of_range_fields),
            }
        )
    return pd.DataFrame(rows, index=frame.index)


def calculate_row_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add signed, percentage, log-space, and multiplicative errors."""
    result = frame.copy()
    actual = pd.to_numeric(result["Tau_ms"], errors="raise").astype(float)
    predicted = pd.to_numeric(result["pred_tau_ms"], errors="raise").astype(float)
    if (actual <= 0).any() or (predicted <= 0).any():
        raise ValueError("Actual and predicted retention times must be positive.")

    difference = predicted - actual
    absolute = difference.abs()
    result["signed_error_ms"] = difference
    result["absolute_error_ms"] = absolute
    result["ape_percent"] = absolute / actual * 100.0
    result["smape_percent"] = 200.0 * absolute / (actual.abs() + predicted.abs())
    result["actual_log1p_tau_ms"] = np.log1p(actual)
    if "pred_log1p_tau_ms" not in result.columns:
        result["pred_log1p_tau_ms"] = np.log1p(predicted)
    result["log_error"] = result["pred_log1p_tau_ms"] - result["actual_log1p_tau_ms"]
    result["abs_log_error"] = result["log_error"].abs()
    result["factor_error"] = np.maximum(predicted / actual, actual / predicted)
    return result


def _safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _summarize_subset(frame: pd.DataFrame) -> dict[str, Any]:
    actual = frame["Tau_ms"].to_numpy(dtype=float)
    predicted = frame["pred_tau_ms"].to_numpy(dtype=float)
    actual_log = np.log1p(actual)
    predicted_log = np.log1p(predicted)
    residual_log = predicted_log - actual_log
    denominator = np.sum((actual_log - actual_log.mean()) ** 2)
    r2_log = np.nan if len(frame) < 2 or denominator == 0 else 1.0 - np.sum(residual_log**2) / denominator
    factor = frame["factor_error"].to_numpy(dtype=float)

    return {
        "n_rows": int(len(frame)),
        "n_papers": int(frame["Paper_ID"].nunique()),
        "mae_ms": float(np.mean(np.abs(predicted - actual))),
        "rmse_ms": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mape_percent": float(frame["ape_percent"].mean()),
        "smape_percent": float(frame["smape_percent"].mean()),
        "mae_log1p": float(np.mean(np.abs(residual_log))),
        "rmse_log1p": float(np.sqrt(np.mean(residual_log**2))),
        "r2_log1p": _safe_float(r2_log),
        "median_factor_error": float(np.median(factor)),
        "geometric_mean_factor_error": float(np.exp(np.mean(np.log(factor)))),
        "within_2x_percent": float(np.mean(factor <= 2.0) * 100.0),
        "within_3x_percent": float(np.mean(factor <= 3.0) * 100.0),
        "within_10x_percent": float(np.mean(factor <= 10.0) * 100.0),
    }


def summarize_predictions(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Report strict target-aligned results separately from supplementary rows."""
    strict = frame.loc[frame["inclusion_tier"] == "strict"].copy()
    if strict.empty:
        raise ValueError("At least one strict validation row is required.")
    return {
        "strict": _summarize_subset(strict),
        "all_eligible": _summarize_subset(frame),
    }


def extract_internal_performance(bundle: dict[str, Any]) -> dict[str, Any]:
    """Read the holdout metrics using the key stored by this repository."""
    return bundle.get("test_performance") or bundle.get("performance") or {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(float(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact GFM table without pandas' optional tabulate package."""
    headers = [str(column).replace("|", "\\|") for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown_report(
    predictions: pd.DataFrame,
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a concise, auditable report alongside the machine-readable files."""
    strict = summary["external_validation"]["strict"]
    all_rows = summary["external_validation"]["all_eligible"]
    columns = [
        "validation_id",
        "inclusion_tier",
        "Tau_ms",
        "pred_tau_ms",
        "ape_percent",
        "factor_error",
        "missing_raw_input_count",
        "unseen_categorical_count",
    ]
    table = predictions[columns].copy()
    for col in ["Tau_ms", "pred_tau_ms", "ape_percent", "factor_error"]:
        table[col] = table[col].map(lambda value: f"{value:.3f}")

    lines = [
        "# External retention-time validation",
        "",
        "Publication window: 2026-06-01 through 2026-08-25.",
        "",
        "Strict rows use directly reported PSC/EPSC decay constants. Supplementary rows use a related but non-identical PPF decay constant and are not included in the primary result.",
        "",
        "## Primary result (strict)",
        "",
        f"- Rows / papers: {strict['n_rows']} / {strict['n_papers']}",
        f"- RMSE (log1p ms): {strict['rmse_log1p']:.4f}",
        f"- Median factor error: {strict['median_factor_error']:.3f}x",
        f"- Within 3x: {strict['within_3x_percent']:.1f}%",
        f"- MAPE: {strict['mape_percent']:.1f}%",
        "",
        "## Sensitivity result (strict + supplementary)",
        "",
        f"- Rows / papers: {all_rows['n_rows']} / {all_rows['n_papers']}",
        f"- RMSE (log1p ms): {all_rows['rmse_log1p']:.4f}",
        f"- Median factor error: {all_rows['median_factor_error']:.3f}x",
        "",
        "## Row-level results",
        "",
        _dataframe_to_markdown(table),
        "",
        "## Interpretation guardrails",
        "",
        "- This is a small, preliminary external check, not proof of generalization across all ion-driven devices.",
        "- Blank paper inputs are imputed by the trained pipeline; the audit columns make this visible.",
        "- Unseen channel materials are out-of-domain categorical inputs and should be interpreted cautiously.",
    ]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    data_path: str | Path,
    model_path: str | Path,
    schema_path: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute the full validation workflow and persist reproducible outputs."""
    frame = load_validation_data(data_path)
    bundle = joblib.load(model_path)
    audit = audit_model_inputs(frame, schema_path)
    predictions = predict_with_bundle(frame, bundle)
    predictions = pd.concat([predictions, audit], axis=1)
    predictions = calculate_row_metrics(predictions)

    summary = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "target": "Tau_ms with model output transformed by expm1",
        "model_internal_holdout": extract_internal_performance(bundle),
        "external_validation": summarize_predictions(predictions),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "external_validation_predictions.csv", index=False)
    (output / "external_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_ready) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(predictions, summary, output / "external_validation_report.md")
    return predictions, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="external_validation_data.csv")
    parser.add_argument("--model", default="models/retention_model.joblib")
    parser.add_argument("--schema", default="models/input_schema.json")
    parser.add_argument("--output-dir", default="validation_results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary = run_validation(args.data, args.model, args.schema, args.output_dir)
    print(json.dumps(summary["external_validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
