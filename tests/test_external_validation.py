from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from external_validation import (
    RAW_INPUT_COLS,
    audit_model_inputs,
    calculate_row_metrics,
    extract_internal_performance,
    predict_with_bundle,
    summarize_predictions,
    validate_external_rows,
    write_markdown_report,
)


class DummyLogModel:
    def predict(self, frame):
        return np.log1p(np.full(len(frame), 100.0))


class ProcessMissingnessModel:
    def predict(self, frame):
        tau = np.where(frame["is_spin_process"].isna(), 100.0, 200.0)
        return np.log1p(tau)


class ExternalValidationTests(unittest.TestCase):
    def test_extract_internal_performance_uses_repository_bundle_key(self):
        bundle = {"test_performance": {"R2_log": 0.89}}
        self.assertEqual(extract_internal_performance(bundle), {"R2_log": 0.89})

    def test_validate_external_rows_rejects_nonpositive_tau(self):
        frame = pd.DataFrame(
            {
                "validation_id": ["bad"],
                "Paper_ID": ["paper-a"],
                "Tau_ms": [0.0],
                "inclusion_tier": ["strict"],
            }
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_external_rows(frame)

    def test_calculate_row_metrics_uses_log_and_factor_errors(self):
        frame = pd.DataFrame({"Tau_ms": [100.0], "pred_tau_ms": [200.0]})
        result = calculate_row_metrics(frame).iloc[0]

        self.assertAlmostEqual(result["signed_error_ms"], 100.0)
        self.assertAlmostEqual(result["absolute_error_ms"], 100.0)
        self.assertAlmostEqual(result["ape_percent"], 100.0)
        self.assertAlmostEqual(result["smape_percent"], 66.6666667, places=6)
        self.assertAlmostEqual(result["factor_error"], 2.0)
        self.assertAlmostEqual(
            result["abs_log_error"], abs(np.log1p(200) - np.log1p(100))
        )

    def test_predict_with_bundle_converts_log1p_output_to_milliseconds(self):
        frame = pd.DataFrame([{col: np.nan for col in RAW_INPUT_COLS} for _ in range(2)])
        frame["Channel"] = ["P3HT", "P3HT"]
        bundle = {
            "model": DummyLogModel(),
            "feature_columns": list(RAW_INPUT_COLS),
        }

        result = predict_with_bundle(frame, bundle)

        np.testing.assert_allclose(result["pred_tau_ms"], [100.0, 100.0])

    def test_prediction_matches_single_row_app_semantics(self):
        row_a = {col: np.nan for col in RAW_INPUT_COLS}
        row_b = {col: np.nan for col in RAW_INPUT_COLS}
        row_a.update({"Channel": "P3HT", "Process": "Spin-coating"})
        row_b.update({"Channel": "CPE-K", "Process": np.nan})
        frame = pd.DataFrame([row_a, row_b])
        bundle = {
            "model": ProcessMissingnessModel(),
            "feature_columns": list(RAW_INPUT_COLS) + ["is_spin_process"],
        }

        result = predict_with_bundle(frame, bundle)

        np.testing.assert_allclose(result["pred_tau_ms"], [200.0, 100.0])

    def test_audit_model_inputs_counts_missing_and_unseen_categories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            schema_path = Path(tmp_dir) / "schema.json"
            schema_path.write_text(
                json.dumps(
                    {
                        "input_categorical_cols": ["Channel", "Anion"],
                        "categorical_options": {
                            "Channel": ["P3HT"],
                            "Anion": ["TFSI"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = {col: np.nan for col in RAW_INPUT_COLS}
            row.update({"Channel": "new-polymer", "Anion": "TFSI"})

            result = audit_model_inputs(pd.DataFrame([row]), schema_path).iloc[0]

        self.assertEqual(result["missing_raw_input_count"], len(RAW_INPUT_COLS) - 2)
        self.assertEqual(result["unseen_categorical_count"], 1)
        self.assertEqual(result["unseen_categorical_fields"], "Channel")

    def test_summary_keeps_strict_and_all_rows_separate(self):
        frame = pd.DataFrame(
            {
                "Paper_ID": ["paper-a", "paper-b"],
                "inclusion_tier": ["strict", "supplementary"],
                "Tau_ms": [100.0, 1000.0],
                "pred_tau_ms": [200.0, 500.0],
            }
        )
        frame = calculate_row_metrics(frame)

        summary = summarize_predictions(frame)

        self.assertEqual(summary["strict"]["n_rows"], 1)
        self.assertEqual(summary["strict"]["n_papers"], 1)
        self.assertEqual(summary["all_eligible"]["n_rows"], 2)
        self.assertAlmostEqual(summary["all_eligible"]["median_factor_error"], 2.0)

    def test_markdown_report_does_not_require_optional_tabulate_dependency(self):
        frame = pd.DataFrame(
            {
                "validation_id": ["row-a"],
                "Paper_ID": ["paper-a"],
                "inclusion_tier": ["strict"],
                "Tau_ms": [100.0],
                "pred_tau_ms": [200.0],
                "missing_raw_input_count": [0],
                "unseen_categorical_count": [0],
            }
        )
        frame = calculate_row_metrics(frame)
        summary = {"external_validation": summarize_predictions(frame)}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "report.md"
            write_markdown_report(frame, summary, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("| validation_id", text)
        self.assertIn("row-a", text)


if __name__ == "__main__":
    unittest.main()
