"""Target-driven condition search for the retention-time platform.

The recommender keeps material descriptors fixed and searches only process
and operating conditions that an engineer can intentionally adjust. Candidate
values stay inside the numerical ranges recorded in the trained model schema.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from model_utils import is_spin_process_value, prepare_input_for_model


RECOMMENDABLE_FIELDS = [
    "Concentration_mg_ml",
    "Spin_RPM",
    "Annealing_temp_C",
    "Annealing_time_h",
    "wt",
    "Gate_voltage_V",
    "Drain_voltage_V",
    "Gate_pulse_width_ms",
    "Pulse_number",
    "Operating_temp_C",
]

INTEGER_FIELDS = {"Spin_RPM", "Pulse_number"}


def default_search_bounds(
    base_conditions: Dict[str, Any],
    tunable_fields: Sequence[str],
    bundle: Dict[str, Any],
) -> Dict[str, tuple[float, float]]:
    """Build a local default window around the engineer's starting condition."""
    numeric_stats = bundle.get("schema", {}).get("numeric_stats", {})
    bounds: Dict[str, tuple[float, float]] = {}
    for field in tunable_fields:
        stats = numeric_stats[field]
        training_low = float(stats["min"])
        training_high = float(stats["max"])
        try:
            center = float(base_conditions.get(field))
        except (TypeError, ValueError):
            center = float(stats.get("median", (training_low + training_high) / 2.0))
        if not np.isfinite(center):
            center = float(stats.get("median", (training_low + training_high) / 2.0))
        center = float(np.clip(center, training_low, training_high))
        span = training_high - training_low
        half_width = abs(center) * 0.20
        if half_width == 0:
            half_width = span * 0.05
        if field in INTEGER_FIELDS:
            half_width = max(half_width, 1.0)
        low = max(training_low, center - half_width)
        high = min(training_high, center + half_width)
        if low >= high:
            low, high = training_low, training_high
        bounds[field] = (float(low), float(high))
    return bounds


def validate_recommendation_request(
    target_tau_ms: float,
    tunable_fields: Sequence[str],
    bundle: Dict[str, Any],
) -> None:
    """Validate target and search fields before candidate generation."""
    try:
        target = float(target_tau_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target retention time must be a valid number.") from exc
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Target retention time must be greater than 0.")
    if not tunable_fields:
        raise ValueError("Select at least one condition to optimize.")
    if len(set(tunable_fields)) != len(tunable_fields):
        raise ValueError("Tunable conditions must not contain duplicates.")

    numeric_stats = bundle.get("schema", {}).get("numeric_stats", {})
    for field in tunable_fields:
        if field not in RECOMMENDABLE_FIELDS:
            raise ValueError(f"{field} is not recommendable as an independent condition.")
        stats = numeric_stats.get(field, {})
        low = stats.get("min")
        high = stats.get("max")
        if low is None or high is None or not np.isfinite(float(low)) or not np.isfinite(float(high)):
            raise ValueError(f"Training range is unavailable for {field}.")
        if float(low) > float(high):
            raise ValueError(f"Training range is invalid for {field}.")


def _resolve_search_range(
    field: str,
    stats: Dict[str, Any],
    search_bounds: Optional[Mapping[str, Tuple[float, float]]],
) -> tuple[float, float]:
    training_low = float(stats["min"])
    training_high = float(stats["max"])
    if not search_bounds or field not in search_bounds:
        return training_low, training_high

    try:
        low, high = (float(value) for value in search_bounds[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Search range for {field} must contain two numbers.") from exc
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"Search range for {field} must have a lower value smaller than the upper value.")
    if low < training_low or high > training_high:
        raise ValueError(
            f"Search range for {field} must stay inside the training range "
            f"({training_low:g} to {training_high:g})."
        )
    return low, high


def _seed_values(stats: Dict[str, Any], base_value: Any, low: float, high: float) -> list[float]:
    values = [low, high, stats.get("median"), stats.get("mean"), base_value]
    seeded: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and low <= number <= high and number not in seeded:
            seeded.append(number)
    return seeded


def generate_candidate_conditions(
    base_conditions: Dict[str, Any],
    tunable_fields: Sequence[str],
    bundle: Dict[str, Any],
    n_candidates: int = 2000,
    random_state: int = 42,
    search_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Generate deterministic in-range candidates while preserving fixed fields."""
    validate_recommendation_request(1.0, tunable_fields, bundle)
    if int(n_candidates) < 5:
        raise ValueError("n_candidates must be at least 5.")
    if "Spin_RPM" in tunable_fields and not is_spin_process_value(base_conditions.get("Process")):
        raise ValueError("Spin_RPM can only be optimized for a spin-based process.")

    count = int(n_candidates)
    rng = np.random.default_rng(random_state)
    candidates = pd.DataFrame([dict(base_conditions) for _ in range(count)])
    numeric_stats = bundle["schema"]["numeric_stats"]

    for field in tunable_fields:
        stats = numeric_stats[field]
        low, high = _resolve_search_range(field, stats, search_bounds)
        values = rng.uniform(low, high, size=count) if high > low else np.full(count, low)
        seeds = _seed_values(stats, base_conditions.get(field), low, high)
        values[: min(len(seeds), count)] = seeds[:count]
        values = np.clip(values, low, high)
        if field in INTEGER_FIELDS:
            values = np.rint(values).astype(int)
        candidates[field] = values

    return candidates


def recommend_conditions(
    base_conditions: Dict[str, Any],
    target_tau_ms: float,
    tunable_fields: Sequence[str],
    bundle: Dict[str, Any],
    n_candidates: int = 2000,
    top_k: int = 5,
    random_state: int = 42,
    search_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Return candidates ranked by distance from the requested retention time."""
    validate_recommendation_request(target_tau_ms, tunable_fields, bundle)
    if int(top_k) <= 0:
        raise ValueError("top_k must be greater than 0.")

    candidates = generate_candidate_conditions(
        base_conditions=base_conditions,
        tunable_fields=tunable_fields,
        bundle=bundle,
        n_candidates=n_candidates,
        random_state=random_state,
        search_bounds=search_bounds,
    )
    prepared = prepare_input_for_model(candidates, bundle)
    pred_log = np.asarray(bundle["model"].predict(prepared), dtype=float)
    pred_tau = np.maximum(np.expm1(pred_log), 0.0)
    target = float(target_tau_ms)

    ranked = candidates.copy()
    ranked["target_tau_ms"] = target
    ranked["pred_tau_ms"] = pred_tau
    ranked["absolute_log_error"] = np.abs(pred_log - np.log1p(target))
    ranked["target_error_percent"] = np.abs(pred_tau - target) / target * 100.0
    ranked = ranked.sort_values(
        ["absolute_log_error", "target_error_percent"],
        kind="stable",
    )
    ranked = ranked.drop_duplicates(subset=list(tunable_fields), keep="first")
    ranked = ranked.head(min(int(top_k), len(ranked))).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked
