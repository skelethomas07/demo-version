from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model_utils import add_engineered_features
from p3ht_validation import (
    add_input_similarity_scores,
    calculate_similarity_score,
    drop_physics_derived_features,
    evaluate_candidates_grouped,
    fill_external_physics_from_historical,
    filter_external_p3ht,
    fit_tau_floor,
    fit_final_model,
    make_group_splits,
    prepare_historical_p3ht,
    predict_with_trained_model,
    restrict_historical_to_external_domain,
    select_best_candidate,
    select_nearest_domain_subset,
    select_similarity_inputs,
    summarize_external_predictions,
)


class P3HTValidationUnitTests(unittest.TestCase):
    def test_external_filter_requires_date_material_tier_and_direct_target(self):
        frame = pd.DataFrame(
            {
                "validation_id": ["keep", "old", "other", "supp", "proxy"],
                "publication_date": [
                    "2026-07-03",
                    "2026-05-31",
                    "2026-07-03",
                    "2026-07-03",
                    "2026-07-03",
                ],
                "Channel": ["P3HT", "P3HT", "PBTTT", "P3HT", "P3HT"],
                "inclusion_tier": ["strict", "strict", "strict", "supplementary", "strict"],
                "target_alignment": [
                    "direct EPSC decay",
                    "direct EPSC decay",
                    "direct EPSC decay",
                    "direct EPSC decay",
                    "PPF decay proxy",
                ],
            }
        )

        result = filter_external_p3ht(frame)

        self.assertEqual(result["validation_id"].tolist(), ["keep"])

    def test_similarity_inputs_never_include_retention_target_or_post_fabrication_fields(self):
        inputs = select_similarity_inputs()

        self.assertNotIn("Tau_ms", inputs)
        self.assertNotIn("Vth_V", inputs)
        self.assertNotIn("On_off_ratio", inputs)
        self.assertNotIn("Gate_pulse_width_ms", inputs)
        self.assertNotIn("Pulse_number", inputs)
        self.assertIn("Ion_diffusion", inputs)
        self.assertIn("Channel", inputs)

    def test_similarity_score_does_not_change_when_tau_changes(self):
        historical = pd.DataFrame(
            {
                "Channel": ["P3HT", "P3HT"],
                "Process": ["Spin-coating", "Drop-casting"],
                "Spin_RPM": [1500.0, np.nan],
                "Tau_ms": [100.0, 100000.0],
            }
        )
        query = pd.Series(
            {
                "Channel": "P3HT",
                "Process": "Spin-coating",
                "Spin_RPM": 1500.0,
                "Tau_ms": 275.0,
            }
        )

        first = calculate_similarity_score(query, historical)
        historical["Tau_ms"] = [1e9, 1e-6]
        query["Tau_ms"] = 1e12
        second = calculate_similarity_score(query, historical)

        np.testing.assert_allclose(first, second)
        self.assertGreater(first.iloc[0], first.iloc[1])

    def test_summary_uses_actual_row_errors_and_reports_paper_count(self):
        frame = pd.DataFrame(
            {
                "Paper_ID": ["paper-a", "paper-a", "paper-b"],
                "Tau_ms": [100.0, 200.0, 1000.0],
                "pred_tau_ms": [110.0, 180.0, 500.0],
            }
        )

        result = summarize_external_predictions(frame)

        self.assertEqual(result["n_rows"], 3)
        self.assertEqual(result["n_papers"], 2)
        self.assertAlmostEqual(result["mape_percent"], (10.0 + 10.0 + 50.0) / 3.0)
        self.assertAlmostEqual(result["median_factor_error"], 10.0 / 9.0)

    def test_physics_ablation_removes_diffusivity_family_only(self):
        frame = pd.DataFrame(
            {
                "Ion_diffusion": [1.0],
                "Ion_viscosity": [2.0],
                "diffusion_viscosity_ratio": [0.5],
                "ion_mobility_proxy": [0.5],
                "log1p_ion_diffusion": [np.log1p(1.0)],
                "Gate_voltage_V": [-2.0],
                "Channel": ["P3HT"],
            }
        )

        result = drop_physics_derived_features(frame)

        self.assertNotIn("Ion_diffusion", result.columns)
        self.assertNotIn("Ion_viscosity", result.columns)
        self.assertNotIn("diffusion_viscosity_ratio", result.columns)
        self.assertNotIn("ion_mobility_proxy", result.columns)
        self.assertNotIn("log1p_ion_diffusion", result.columns)
        self.assertIn("Gate_voltage_V", result.columns)
        self.assertIn("Channel", result.columns)

    def test_engineering_converts_diffusivity_into_physical_model_features(self):
        frame = pd.DataFrame(
            {
                "Ion_diffusion": [4.0],
                "Ion_viscosity": [2.0],
                "Gate_pulse_width_ms": [1000.0],
                "Cation_radius": [2.0],
                "Anion_radius": [3.0],
            }
        )

        result = add_engineered_features(frame)

        self.assertAlmostEqual(result.loc[0, "log1p_ion_diffusion"], np.log1p(4.0))
        self.assertAlmostEqual(result.loc[0, "diffusion_viscosity_ratio"], 2.0)
        self.assertAlmostEqual(result.loc[0, "diffusion_length_proxy"], 2.0)
        self.assertAlmostEqual(result.loc[0, "radius_diffusion_time_proxy"], 25.0 / 4.0)

    def test_group_splits_keep_each_paper_in_one_side(self):
        groups = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d"])

        splits = make_group_splits(groups, n_splits=4)

        self.assertEqual(len(splits), 4)
        for train_idx, valid_idx in splits:
            train_groups = set(groups.iloc[train_idx])
            valid_groups = set(groups.iloc[valid_idx])
            self.assertTrue(train_groups.isdisjoint(valid_groups))

    def test_historical_preparation_keeps_valid_p3ht_only(self):
        frame = pd.DataFrame(
            {
                "Paper_ID": ["a", "b", "c", "d"],
                "Channel": ["P3HT", "PBTTT", "P3HT", "P3HT"],
                "Tau_ms": [100.0, 200.0, 0.0, np.nan],
                "Process": ["spin coating", "spin coating", "spin coating", "spin coating"],
            }
        )

        result = prepare_historical_p3ht(frame)

        self.assertEqual(result["Paper_ID"].tolist(), ["a"])
        self.assertEqual(result["Channel"].tolist(), ["P3HT"])
        self.assertEqual(result["Process"].tolist(), ["Spin-coating"])

    def test_tau_floor_is_fitted_from_training_rows_only(self):
        training_tau = pd.Series([10.0, 20.0, 30.0, 40.0])

        floor_before = fit_tau_floor(training_tau)
        validation_with_extreme = pd.Series([1e-12, 1e12])
        floor_after = fit_tau_floor(training_tau)

        self.assertAlmostEqual(floor_before, floor_after)
        self.assertNotEqual(floor_before, float(validation_with_extreme.quantile(0.01)))

    def test_grouped_candidate_evaluation_uses_identical_folds_for_ablation(self):
        rows = []
        for paper_index in range(6):
            for row_index in range(2):
                rows.append(
                    {
                        "Paper_ID": f"paper-{paper_index}",
                        "Channel": "P3HT",
                        "Process": "Spin-coating",
                        "Anion": "TFSI" if row_index == 0 else "BF4",
                        "Ion_diffusion": 1.0 + row_index,
                        "Ion_viscosity": 2.0 + row_index,
                        "Tau_ms": 100.0 + 20.0 * paper_index + 10.0 * row_index,
                    }
                )
        frame = pd.DataFrame(rows)
        candidates = [
            {
                "candidate": "tiny",
                "n_estimators": 10,
                "max_features": 1.0,
                "min_samples_leaf": 1,
                "smoothing": 5,
                "target_transform": "none",
            }
        ]

        result = evaluate_candidates_grouped(frame, candidates, n_splits=3)

        self.assertEqual(set(result["physics_variant"]), {"physics_on", "physics_off"})
        on_folds = result.loc[result["physics_variant"] == "physics_on", "fold"].tolist()
        off_folds = result.loc[result["physics_variant"] == "physics_off", "fold"].tolist()
        self.assertEqual(on_folds, off_folds)
        self.assertTrue(np.isfinite(result["mae_log1p"]).all())

    def test_candidate_selection_ignores_external_errors(self):
        cv_results = pd.DataFrame(
            {
                "candidate": ["a", "a", "b", "b"],
                "physics_variant": ["physics_on"] * 4,
                "mae_log1p": [0.2, 0.4, 0.5, 0.5],
            }
        )
        external = pd.DataFrame(
            {
                "candidate": ["a", "b"],
                "mape_percent": [1000.0, 0.1],
            }
        )

        selected = select_best_candidate(cv_results, external_results=external)

        self.assertEqual(selected["candidate"], "a")
        self.assertAlmostEqual(selected["mean_mae_log1p"], 0.3)

    def test_external_physics_fill_uses_ion_identity_without_target(self):
        historical = pd.DataFrame(
            {
                "Cation": ["EMIM", "EMIM", "Li"],
                "Anion": ["TFSI", "TFSI", "TFSI"],
                "Ion_diffusion": [2.0, 4.0, 99.0],
                "Ion_viscosity": [10.0, 14.0, 1.0],
                "Anion_radius": [3.0, 3.0, 3.0],
                "Cation_radius": [4.0, 4.0, 1.0],
                "Tau_ms": [1.0, 1e9, 5.0],
            }
        )
        external = pd.DataFrame(
            {
                "Cation": ["EMIM"],
                "Anion": ["TFSI"],
                "Ion_diffusion": [np.nan],
                "Tau_ms": [406.0],
            }
        )

        result = fill_external_physics_from_historical(external, historical)

        self.assertAlmostEqual(result.loc[0, "Ion_diffusion"], 3.0)
        self.assertAlmostEqual(result.loc[0, "Ion_viscosity"], 12.0)
        self.assertEqual(result.loc[0, "physics_imputation_level"], "cation+anion")

    def test_nearest_domain_subset_is_based_on_similarity_not_error(self):
        frame = pd.DataFrame(
            {
                "validation_id": ["low-error-low-sim", "high-sim", "middle"],
                "input_similarity_score": [0.2, 0.9, 0.5],
                "ape_percent": [0.1, 100.0, 20.0],
            }
        )

        result = select_nearest_domain_subset(frame, fraction=1 / 3)

        self.assertEqual(result["validation_id"].tolist(), ["high-sim"])

    def test_final_model_predicts_positive_tau_for_external_rows(self):
        rows = []
        for paper_index in range(6):
            for anion, offset in [("TFSI", 0.0), ("BF4", 25.0)]:
                rows.append(
                    {
                        "Paper_ID": f"paper-{paper_index}",
                        "Channel": "P3HT",
                        "Process": "Spin-coating",
                        "Anion": anion,
                        "Cation": "EMIM",
                        "Ion_diffusion": 1.0 + offset / 100.0,
                        "Ion_viscosity": 2.0,
                        "Gate_pulse_width_ms": 1000.0,
                        "Tau_ms": 100.0 + 10.0 * paper_index + offset,
                    }
                )
        historical = pd.DataFrame(rows)
        candidate = {
            "candidate": "tiny",
            "n_estimators": 10,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "smoothing": 5,
            "target_transform": "none",
        }
        trained = fit_final_model(historical, candidate, use_physics=True)
        external = pd.DataFrame(
            {
                "Channel": ["P3HT"],
                "Process": ["Spin-coating"],
                "Anion": ["TFSI"],
                "Cation": ["EMIM"],
                "Ion_diffusion": [1.0],
                "Ion_viscosity": [2.0],
                "Gate_pulse_width_ms": [1000.0],
            }
        )

        prediction = predict_with_trained_model(trained, external)

        self.assertEqual(len(prediction), 1)
        self.assertGreater(prediction[0], 0.0)
        self.assertNotIn("Gate_voltage_V", trained["feature_columns"])
        self.assertNotIn("Gate_pulse_width_ms", trained["feature_columns"])
        self.assertNotIn("total_gate_dose", trained["feature_columns"])

    def test_input_similarity_density_is_target_blind(self):
        historical = pd.DataFrame(
            {
                "Channel": ["P3HT", "P3HT"],
                "Process": ["Spin-coating", "Drop-casting"],
                "Tau_ms": [100.0, 1e9],
            }
        )
        external = pd.DataFrame(
            {
                "validation_id": ["case"],
                "Channel": ["P3HT"],
                "Process": ["Spin-coating"],
                "Tau_ms": [200.0],
            }
        )

        first = add_input_similarity_scores(external, historical, neighbors=1)
        historical["Tau_ms"] = [1e-9, 1e12]
        external["Tau_ms"] = [1e8]
        second = add_input_similarity_scores(external, historical, neighbors=1)

        self.assertAlmostEqual(
            first.loc[0, "input_similarity_score"],
            second.loc[0, "input_similarity_score"],
        )

    def test_external_domain_restriction_uses_inputs_not_tau(self):
        historical = pd.DataFrame(
            {
                "Paper_ID": [f"p{i}" for i in range(8)],
                "Channel": ["P3HT"] * 8,
                "Process": ["Spin-coating"] * 6 + ["Drop-casting"] * 2,
                "Cation": ["EMIM"] * 6 + ["Li"] * 2,
                "polymer": ["PVDF-HFP"] * 6 + ["PEO"] * 2,
                "Pulse_number": [1.0] * 6 + [10.0] * 2,
                "Tau_ms": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1e6, 1e9],
            }
        )
        external = pd.DataFrame(
            {
                "Channel": ["P3HT"],
                "Process": ["Spin-coating"],
                "Cation": ["EMIM"],
                "polymer": ["PVDF-HFP"],
                "Pulse_number": [1.0],
                "Tau_ms": [406.0],
            }
        )

        first = restrict_historical_to_external_domain(historical, external, min_papers=5)
        historical["Tau_ms"] = historical["Tau_ms"] * 1e12
        external["Tau_ms"] = 1e-12
        second = restrict_historical_to_external_domain(historical, external, min_papers=5)

        self.assertEqual(first["Paper_ID"].tolist(), [f"p{i}" for i in range(6)])
        self.assertEqual(first["Paper_ID"].tolist(), second["Paper_ID"].tolist())


if __name__ == "__main__":
    unittest.main()
