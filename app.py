from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from model_utils import (
    KNOWN_CATEGORICAL_COLS,
    RAW_INPUT_COLS,
    explain_prediction_local,
    get_unit_breakdown,
    is_spin_process_value,
    load_model_bundle,
    predict_retention_time,
)

APP_TITLE = "Synapse Retention Engine"
MODEL_PATH = Path("models/retention_model.joblib")
MISSING_LABEL = "Not specified"

FIELD_LABELS = {
    "Channel": "Channel material",
    "Solvent": "Solvent",
    "Concentration_mg_ml": "Concentration (mg/mL)",
    "Process": "Fabrication process",
    "Spin_RPM": "Spin speed (RPM)",
    "Annealing_temp_C": "Annealing temperature (°C)",
    "Annealing_time_h": "Annealing time (h)",
    "Ion_type": "Ion / electrolyte type",
    "wt": "Weight ratio",
    "polymer": "Polymer matrix",
    "Ion_diffusion": "Ion diffusion",
    "Ion_viscosity": "Ion viscosity",
    "Anion_radius": "Anion radius",
    "Cation_radius": "Cation radius",
    "Cation": "Cation",
    "Anion": "Anion",
    "Gate_voltage_V": "Gate voltage (V)",
    "Drain_voltage_V": "Drain voltage (V)",
    "Gate_pulse_width_ms": "Gate pulse width (ms)",
    "Pulse_number": "Pulse number",
    "Electrode_type": "Electrode type",
    "Vth_V": "Threshold voltage, Vth (V)",
    "On_off_ratio": "On/off ratio",
    "Vth_window_V": "Vth window (V)",
    "Operating_temp_C": "Operating temperature (°C)",
}

HELP_TEXT = {
    "Spin_RPM": "Only applicable to spin-based processes. It is disabled for non-spin processes and passed to the model as missing.",
    "wt": "Numeric matrix/electrolyte weight ratio. Text values are not accepted in this field.",
    "polymer": "Polymer or matrix descriptor. Case-only variants are normalized in the model schema.",
    "Electrode_type": "Case-only duplicates such as Au/au are normalized to canonical material labels.",
    "Vth_V": "Optional. Leave blank if this is not available before device characterization.",
    "On_off_ratio": "Optional. Leave blank if this is not available before device characterization.",
    "Vth_window_V": "Optional. Leave blank if this is not available before device characterization.",
    "Ion_diffusion": "Optional if unknown. The model will impute missing numerical values from the training distribution.",
    "Ion_viscosity": "Optional if unknown. The model will impute missing numerical values from the training distribution.",
    "Anion_radius": "Optional if unknown.",
    "Cation_radius": "Optional if unknown.",
}
TREND_HELP = {
    "Concentration_mg_ml": (
        "Dataset trend: increasing concentration tends to be associated with shorter retention time in the current training dataset. "
        "Interpret this as correlation, not a causal rule, because concentration is also tied to material and process choices."
    ),
    "Pulse_number": (
        "Dataset trend: increasing pulse number tends to be associated with longer retention time. "
        "Repeated pulses can increase the accumulated electrical stimulus applied to the ion-gated channel."
    ),
    "Gate_pulse_width_ms": (
        "Dataset trend: longer gate pulse width tends to be associated with longer retention time. "
        "A longer pulse can increase the effective stimulation time for ion penetration and relaxation."
    ),
    "Spin_RPM": (
        "Dataset trend: within spin-based processes, higher spin speed tends to be associated with shorter retention time. "
        "For non-spin processes, spin speed is not physically applicable and is passed as missing."
    ),
    "Vth_V": (
        "Dataset trend: higher Vth tends to be associated with longer retention time. "
        "This field is an optional post-characterization metric, so leave it blank if it is not known before fabrication."
    ),
    "Vth_window_V": (
        "Dataset trend: wider Vth window tends to be associated with longer retention time. "
        "This trend is based on reported device metrics and should be interpreted with caution when sample coverage is limited."
    ),
    "Annealing_time_h": (
        "Dataset trend: longer annealing time shows a weak positive tendency with retention time. "
        "The effect may depend strongly on channel material, annealing temperature, and solvent system."
    ),
}


def get_field_help(col: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    base = HELP_TEXT.get(col)
    trend = TREND_HELP.get(col)

    if col == "Spin_RPM":
        process = None if context is None else context.get("Process")
        if not is_spin_process_value(process):
            trend = (
                "Spin speed is only meaningful for spin-based fabrication. "
                "Because the selected process is not spin-based, this field is disabled and treated as missing."
            )

    if base and trend:
        return f"{base}\n\n{trend}"
    return base or trend


SECTIONS = {
    "Material & Process": [
        "Channel",
        "Solvent",
        "Concentration_mg_ml",
        "Process",
        "Spin_RPM",
        "Annealing_temp_C",
        "Annealing_time_h",
        "Electrode_type",
    ],
    "Electrolyte & Ion": [
        "Ion_type",
        "wt",
        "polymer",
        "Cation",
        "Anion",
        "Ion_diffusion",
        "Ion_viscosity",
        "Anion_radius",
        "Cation_radius",
    ],
    "Operation": [
        "Gate_voltage_V",
        "Drain_voltage_V",
        "Gate_pulse_width_ms",
        "Pulse_number",
        "Operating_temp_C",
    ],
    "Optional Device Metrics": ["Vth_V", "On_off_ratio", "Vth_window_V"],
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --card-bg: rgba(255,255,255,0.84);
            --muted: #64748b;
            --line: #e5e7eb;
            --ink: #0f172a;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --shadow: 0 18px 50px rgba(15,23,42,.08);
        }
        .stApp { background: linear-gradient(180deg, #f8fbff 0%, #f7f7fb 48%, #ffffff 100%); }
        [data-testid="stHeader"] { background: rgba(248, 251, 255, 0.78); backdrop-filter: blur(10px); }
        .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
        .hero {
            padding: 30px 34px;
            border: 1px solid rgba(226,232,240,.9);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(239,246,255,.82));
            box-shadow: var(--shadow);
            margin-bottom: 22px;
        }
        .eyebrow { font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 800; color: var(--accent); }
        .hero-title { font-size: 2.55rem; font-weight: 850; letter-spacing: -.045em; color: var(--ink); margin: 5px 0 8px 0; line-height: 1.05; }
        .hero-subtitle { color: var(--muted); max-width: 780px; font-size: 1.02rem; line-height: 1.72; margin: 0; }
        .beta-pill { display:inline-block; padding: 7px 11px; border-radius: 999px; background: #111827; color: white; font-size: .78rem; font-weight: 750; margin-left: 8px; vertical-align: middle; }
        .section-card {
            padding: 22px 24px 12px 24px;
            border: 1px solid rgba(226,232,240,.92);
            border-radius: 24px;
            background: rgba(255,255,255,.88);
            box-shadow: 0 10px 35px rgba(15,23,42,.055);
            margin: 14px 0 18px 0;
        }
        .section-title { font-size: 1.08rem; font-weight: 820; color: var(--ink); margin-bottom: 3px; }
        .section-desc { color: var(--muted); font-size: .9rem; margin-bottom: 16px; }
        .result-card {
            position: sticky; top: 86px;
            border-radius: 28px;
            border: 1px solid rgba(37,99,235,.20);
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            box-shadow: var(--shadow);
            padding: 26px;
        }
        .metric-label { color: var(--muted); font-size: .82rem; font-weight: 740; text-transform: uppercase; letter-spacing: .08em; }
        .metric-main { font-size: 2.15rem; font-weight: 880; letter-spacing: -.055em; color: var(--ink); margin-top: 4px; }
        .metric-unit { color: var(--muted); font-size: .95rem; font-weight: 600; margin-left: 4px; }
        .mini-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 18px; }
        .mini-metric { border:1px solid #e5e7eb; border-radius: 18px; padding: 12px 14px; background: rgba(255,255,255,.72); }
        .mini-name { color:#64748b; font-size:.76rem; font-weight:700; }
        .mini-value { color:#0f172a; font-size:1.0rem; font-weight:800; margin-top:2px; }
        .driver-title { font-size: 1.02rem; font-weight: 850; color: #0f172a; margin: 4px 0 4px 0; }
        .driver-method { color:#64748b; font-size:.82rem; line-height:1.55; margin: 0 0 14px 0; }
        .driver-row { margin: 13px 0 15px 0; }
        .driver-top { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
        .driver-name { font-weight: 800; color:#0f172a; font-size:.93rem; }
        .driver-value { color:#64748b; font-size:.78rem; overflow-wrap:anywhere; }
        .driver-percent { font-weight: 850; color:#2563eb; font-size:.95rem; white-space:nowrap; }
        .driver-track { height: 10px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-top:7px; }
        .driver-fill { height:100%; background: linear-gradient(90deg, #2563eb, #60a5fa); border-radius:999px; }
        .driver-caption { color:#64748b; font-size:.76rem; margin-top:5px; }
        .note { color: var(--muted); font-size: .84rem; line-height: 1.55; margin-top: 16px; }
        .stButton > button {
            width: 100%; height: 3.25rem; border-radius: 16px; font-weight: 820; border: 0;
            background: linear-gradient(135deg, #1d4ed8, #2563eb); color: white;
            box-shadow: 0 14px 24px rgba(37,99,235,.22);
        }
        .stButton > button:hover { filter: brightness(.98); border: 0; color: white; }
        div[data-testid="stExpander"] { border: 1px solid rgba(226,232,240,.9); border-radius: 18px; background: rgba(255,255,255,.66); }
        .footer-line { margin-top: 28px; padding-top: 14px; border-top: 1px solid #e5e7eb; color: #94a3b8; font-size: .82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_bundle() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        st.error("Model file not found: models/retention_model.joblib")
        st.stop()
    return load_model_bundle(str(MODEL_PATH))


def parse_float(raw: str) -> Optional[float]:
    if raw is None:
        return None
    value = str(raw).strip().replace(",", "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"'{raw}' is not a valid number.")


def select_field(col: str, bundle: Dict[str, Any], key: str, context: Optional[Dict[str, Any]] = None) -> Any:
    options = bundle["schema"].get("categorical_options", {}).get(col, [])
    values = [MISSING_LABEL] + [v for v in options if str(v).strip() != ""]
    default_index = 0
    # Choose useful defaults for a clean first run.
    suggested = {
        "Channel": "P3HT",
        "Process": "Spin-coating",
        "Ion_type": "ion_gel",
        "polymer": "PVDF-HFP",
        "Cation": "EMIM",
        "Anion": "TFSI",
        "Electrode_type": "Au",
    }
    if col in suggested and suggested[col] in values:
        default_index = values.index(suggested[col])
    selected = st.selectbox(
        FIELD_LABELS.get(col, col),
        values,
        index=default_index,
        key=key,
        help=get_field_help(col, context),
    )
    return np.nan if selected == MISSING_LABEL else selected


def number_field(col: str, bundle: Dict[str, Any], key: str, context: Optional[Dict[str, Any]] = None) -> Any:
    stats = bundle["schema"].get("numeric_stats", {}).get(col, {})
    median = stats.get("median")
    default_map = {
        "Concentration_mg_ml": 10,
        "Spin_RPM": 2000,
        "Annealing_temp_C": 100,
        "Annealing_time_h": 1,
        "Gate_voltage_V": 1.0,
        "Drain_voltage_V": 0.1,
        "Gate_pulse_width_ms": 100,
        "Pulse_number": 10,
        "Operating_temp_C": 25,
    }
    default = default_map.get(col, "")
    if default == "" and median is not None and col not in ["Vth_V", "On_off_ratio", "Vth_window_V"]:
        default = f"{median:g}"
    elif default != "":
        default = f"{float(default):g}"
    raw = st.text_input(
        FIELD_LABELS.get(col, col),
        value=default,
        key=key,
        placeholder="Leave blank if unknown",
        help=get_field_help(col, context),
    )
    return parse_float(raw)


def collect_inputs(bundle: Dict[str, Any], key_prefix: str = "field") -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for section, fields in SECTIONS.items():
        with st.container():
            desc = {
                "Material & Process": "Define the channel, fabrication route, and thermal process.",
                "Electrolyte & Ion": "Add electrolyte, ion, and transport-related descriptors where available.",
                "Operation": "Set the electrical stimulation and operating conditions.",
                "Optional Device Metrics": "Used by the trained model, but can be left blank for pre-characterization screening.",
            }.get(section, "")
            st.markdown(f'<div class="section-card"><div class="section-title">{section}</div><div class="section-desc">{desc}</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for idx, col_name in enumerate(fields):
                with cols[idx % 2]:
                    if col_name == "Spin_RPM" and not is_spin_process_value(data.get("Process")):
                        st.text_input(
                            FIELD_LABELS.get(col_name, col_name),
                            value="Not applicable for selected process",
                            disabled=True,
                            key=f"{key_prefix}_{col_name}_disabled",
                            help=get_field_help(col_name, data),
                        )
                        data[col_name] = None
                    elif col_name in bundle.get("input_categorical_cols", []) or col_name in KNOWN_CATEGORICAL_COLS:
                        data[col_name] = select_field(col_name, bundle, f"{key_prefix}_{col_name}", context=data)
                    else:
                        try:
                            data[col_name] = number_field(col_name, bundle, f"{key_prefix}_{col_name}", context=data)
                        except ValueError as e:
                            st.error(str(e))
                            st.stop()
            st.markdown("</div>", unsafe_allow_html=True)
    return data


def completeness(user_input: Dict[str, Any]) -> tuple[int, int]:
    total = len(user_input)
    filled = 0
    for value in user_input.values():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if str(value).strip() == "":
            continue
        filled += 1
    return filled, total


def range_warnings(user_input: Dict[str, Any], bundle: Dict[str, Any]) -> list[str]:
    warnings = []
    stats = bundle["schema"].get("numeric_stats", {})
    for col, value in user_input.items():
        if value is None or col not in stats:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        min_v = stats[col].get("min")
        max_v = stats[col].get("max")
        if min_v is not None and max_v is not None and (v < min_v or v > max_v):
            warnings.append(f"{FIELD_LABELS.get(col, col)} is outside the training range ({min_v:g} ~ {max_v:g}).")
    return warnings


def _format_driver_value(value: Any) -> str:
    if value is None:
        return "Not specified"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "Not specified"
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):,.4g}"
    text = str(value).strip()
    return text if text else "Not specified"


def render_prediction_panel(result: Optional[Dict[str, float]], user_input: Dict[str, Any], bundle: Dict[str, Any]) -> None:
    st.markdown('<div class="metric-label">Predicted retention time</div>', unsafe_allow_html=True)
    if result is None:
        st.markdown('<div class="metric-main">— <span class="metric-unit">ms</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="note">Enter device conditions and run the engine. The model returns a new prediction for the supplied condition vector.</div>', unsafe_allow_html=True)
        return

    tau = max(float(result["pred_tau_ms"]), 0.0)
    units = get_unit_breakdown(tau)
    st.markdown(f'<div class="metric-main">{tau:,.3g}<span class="metric-unit"> ms</span></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mini-grid">
            <div class="mini-metric"><div class="mini-name">Seconds</div><div class="mini-value">{units['seconds']:,.3g}</div></div>
            <div class="mini-metric"><div class="mini-name">Minutes</div><div class="mini-value">{units['minutes']:,.3g}</div></div>
            <div class="mini-metric"><div class="mini-name">Hours</div><div class="mini-value">{units['hours']:,.3g}</div></div>
            <div class="mini-metric"><div class="mini-name">log1p(Tau_ms)</div><div class="mini-value">{result['pred_log1p_tau_ms']:.3f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filled, total = completeness(user_input)
    st.markdown(f'<div class="note">Input completeness: <b>{filled}/{total}</b> fields. Missing values are handled by the training pipeline imputation step.</div>', unsafe_allow_html=True)
    warns = range_warnings(user_input, bundle)
    if warns:
        with st.expander("Model range checks", expanded=False):
            for w in warns:
                st.warning(w)

    perf = bundle.get("test_performance", {})
    with st.expander("Model details", expanded=False):
        st.write(
            {
                "model": bundle.get("model_type", "StackingRegressor"),
                "target": "log1p(Tau_ms)",
                "R2_log_holdout": perf.get("R2_log"),
                "RMSE_log_holdout": perf.get("RMSE_log"),
                "MAE_log_holdout": perf.get("MAE_log"),
            }
        )


def render_driver_panel(driver_result: Optional[Dict[str, Any]]) -> None:
    st.markdown('<div class="driver-title">Top input drivers</div>', unsafe_allow_html=True)
    if driver_result is None:
        st.markdown(
            '<div class="driver-method">Run an estimate to calculate local feature sensitivity for this specific input condition.</div>',
            unsafe_allow_html=True,
        )
        return

    rows = driver_result.get("top_drivers", [])
    if not rows:
        st.info("No meaningful local sensitivity was detected for the current input.")
        return

    st.markdown(
        f"""
        <div class="driver-method">
        Local sensitivity view. Each field is temporarily hidden, the model is rerun, and the absolute change in predicted log1p(Tau_ms) is measured. The top {len(rows)} effects are normalized to 100%.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for row in rows:
        feature = row.get("feature", "")
        label = FIELD_LABELS.get(feature, feature)
        value = _format_driver_value(row.get("current_value"))
        pct = float(row.get("share_top_percent", 0.0))
        all_pct = float(row.get("share_all_percent", 0.0))
        delta = float(row.get("delta_log", 0.0))
        direction = "higher" if delta >= 0 else "lower"
        width = max(0.0, min(100.0, pct))
        st.markdown(
            f"""
            <div class="driver-row">
                <div class="driver-top">
                    <div>
                        <div class="driver-name">{label}</div>
                        <div class="driver-value">Current value: {value}</div>
                    </div>
                    <div class="driver-percent">{pct:.1f}%</div>
                </div>
                <div class="driver-track"><div class="driver-fill" style="width:{width:.1f}%"></div></div>
                <div class="driver-caption">Compared with hiding this field: prediction moves {direction} by |Δ log1p(Tau)| = {abs(delta):.3f}. Share among all measured input effects: {all_pct:.1f}%.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("How to interpret this", expanded=False):
        st.markdown(
            """
            This is a **local sensitivity explanation**, not a causal proof and not a global feature-importance score. The engine reruns the same trained model while replacing one input field at a time with a missing value. Numeric fields then fall back to the model pipeline's training median, while categorical fields become `Missing`. The reported percentage is normalized within the top drivers for the current prediction.
            """
        )


def render_result(
    result: Optional[Dict[str, float]],
    user_input: Dict[str, Any],
    bundle: Dict[str, Any],
    driver_result: Optional[Dict[str, Any]] = None,
) -> None:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    prediction_tab, driver_tab = st.tabs(["Prediction", "Input drivers"])
    with prediction_tab:
        render_prediction_panel(result, user_input, bundle)
    with driver_tab:
        render_driver_panel(driver_result)
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# External dataset submission backend (Supabase)
# -----------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def get_supabase_settings() -> Optional[Dict[str, str]]:
    """Read Supabase settings from Streamlit secrets.

    Expected Streamlit secrets:
    [supabase]
    url = "https://YOUR_PROJECT.supabase.co"
    anon_key = "YOUR_ANON_PUBLIC_KEY"
    table = "experiment_submissions"
    """
    try:
        cfg = st.secrets.get("supabase", {})
    except Exception:
        return None
    url = str(cfg.get("url", "")).strip()
    anon_key = str(cfg.get("anon_key", "")).strip()
    table = str(cfg.get("table", "experiment_submissions")).strip() or "experiment_submissions"
    if not url or not anon_key:
        return None
    return {"url": url, "anon_key": anon_key, "table": table}


@st.cache_resource(show_spinner=False)
def get_supabase_client_cached(url: str, anon_key: str):
    from supabase import create_client

    return create_client(url, anon_key)


def get_supabase_client():
    settings = get_supabase_settings()
    if settings is None:
        return None, None
    return get_supabase_client_cached(settings["url"], settings["anon_key"]), settings


def insert_experiment_submission(payload: Dict[str, Any]) -> tuple[bool, str]:
    client, settings = get_supabase_client()
    if client is None or settings is None:
        return False, "Supabase is not configured. Add [supabase] settings in Streamlit secrets first."
    try:
        response = client.table(settings["table"]).insert(payload).execute()
        inserted = getattr(response, "data", None)
        if inserted is not None:
            return True, "Submission saved to pending dataset."
        return True, "Submission request completed."
    except Exception as exc:
        return False, f"Submission failed: {exc}"


def render_submission_status_box() -> None:
    settings = get_supabase_settings()
    if settings is None:
        st.warning(
            "External dataset storage is not connected yet. Configure Supabase credentials in Streamlit secrets to enable submissions."
        )
        with st.expander("Required Streamlit secrets", expanded=False):
            st.code(
                '[supabase]\nurl = "https://YOUR_PROJECT.supabase.co"\nanon_key = "YOUR_ANON_PUBLIC_KEY"\ntable = "experiment_submissions"',
                language="toml",
            )
    else:
        st.success("External dataset storage is connected. New experiments will be stored as pending submissions.")


def render_submit_experiment(bundle: Dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Add verified experimental result</div>
            <div class="section-desc">Submit a measured retention time with the exact device, process, electrolyte, and operation conditions. New rows are stored as pending data and should be reviewed before model retraining.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_submission_status_box()

    with st.form("experiment_submission_form", clear_on_submit=False):
        user_input = collect_inputs(bundle, key_prefix="submit")

        st.markdown(
            '<div class="section-card"><div class="section-title">Measured result & source</div><div class="section-desc">Tau_ms is required because the model can only learn from labeled experimental outcomes.</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            tau_raw = st.text_input(
                "Measured retention time, Tau_ms (ms)",
                placeholder="e.g. 12000",
                help="Required. Enter the experimentally measured retention time constant in milliseconds.",
                key="submit_tau_ms",
            )
            submitter_name = st.text_input("Submitter name", placeholder="Optional", key="submitter_name")
            submitter_email = st.text_input("Contact email", placeholder="Optional", key="submitter_email")
        with c2:
            organization = st.text_input("Organization / lab", placeholder="Optional", key="submitter_org")
            source_reference = st.text_input("Source / sample ID / paper DOI", placeholder="Optional", key="submit_reference")
            notes = st.text_area("Notes", placeholder="Optional: fitting method, device remarks, uncertainty, etc.", key="submit_notes")
        confirm = st.checkbox(
            "I confirm that this row is based on an actual experiment and can be stored as pending training data.",
            key="submit_confirm",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Submit experimental result")

    if not submitted:
        return

    try:
        tau_ms = parse_float(tau_raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    if tau_ms is None or tau_ms <= 0:
        st.error("Measured Tau_ms is required and must be greater than 0.")
        return
    if not confirm:
        st.error("Please confirm that the submitted row is based on an actual experiment.")
        return

    try:
        prediction = predict_retention_time(user_input, bundle)
    except Exception:
        prediction = {"pred_tau_ms": None, "pred_log1p_tau_ms": None}

    filled, total = completeness(user_input)
    payload = {
        "status": "pending",
        "source": "user_submission",
        "submitter_name": submitter_name.strip() or None,
        "submitter_email": submitter_email.strip() or None,
        "organization": organization.strip() or None,
        "source_reference": source_reference.strip() or None,
        "notes": notes.strip() or None,
        "tau_ms": float(tau_ms),
        "pred_tau_ms": json_safe(prediction.get("pred_tau_ms")),
        "pred_log1p_tau_ms": json_safe(prediction.get("pred_log1p_tau_ms")),
        "input_data": json_safe(user_input),
        "input_completeness_filled": int(filled),
        "input_completeness_total": int(total),
        "model_type": bundle.get("model_type", "StackingRegressor"),
    }
    ok, message = insert_experiment_submission(payload)
    if ok:
        st.success(message)
        if prediction.get("pred_tau_ms") is not None:
            st.info(f"Current model prediction for this submitted condition: {float(prediction['pred_tau_ms']):,.3g} ms. Measured value submitted: {float(tau_ms):,.3g} ms.")
    else:
        st.error(message)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    bundle = get_bundle()

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Neuromorphic Device Design Intelligence <span class="beta-pill">Beta</span></div>
            <div class="hero-title">{APP_TITLE}</div>
            <p class="hero-subtitle">Estimate retention time and collect verified experimental results to expand the training dataset over time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predict_tab, contribute_tab = st.tabs(["Estimate retention time", "Add experimental result"])

    with predict_tab:
        input_col, result_col = st.columns([1.62, 0.88], gap="large")
        with input_col:
            user_input = collect_inputs(bundle, key_prefix="predict")
            run = st.button("Estimate retention time", type="primary")
        with result_col:
            result = None
            driver_result = None
            if run:
                result = predict_retention_time(user_input, bundle)
                driver_result = explain_prediction_local(user_input, bundle, top_n=5)
            render_result(result, user_input if 'user_input' in locals() else {}, bundle, driver_result)

    with contribute_tab:
        render_submit_experiment(bundle)

    st.markdown(
        '<div class="footer-line">Beta engine for EGST retention-time estimation. User-submitted rows are stored as pending data and should be reviewed before model retraining.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
