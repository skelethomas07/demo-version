from __future__ import annotations

import unittest

import numpy as np

from condition_recommender import (
    RECOMMENDABLE_FIELDS,
    default_search_bounds,
    generate_candidate_conditions,
    recommend_conditions,
    validate_recommendation_request,
)


class GateVoltageModel:
    """Small real predictor used to exercise the recommendation ranking."""

    def predict(self, frame):
        gate_voltage = frame["Gate_voltage_V"].astype(float).to_numpy()
        return np.log1p(gate_voltage * 100.0)


def make_bundle():
    return {
        "model": GateVoltageModel(),
        "feature_columns": ["Channel", "Gate_voltage_V", "Drain_voltage_V"],
        "schema": {
            "numeric_stats": {
                "Gate_voltage_V": {"min": 0.0, "max": 10.0, "median": 5.0, "mean": 4.0},
                "Drain_voltage_V": {"min": 0.0, "max": 2.0, "median": 0.2, "mean": 0.3},
                "Gate_pulse_width_ms": {"min": 0.1, "max": 50000.0, "median": 100.0, "mean": 600.0},
            },
            "categorical_options": {"Channel": ["P3HT", "PEDOT:PSS"]},
        },
    }


class ConditionRecommenderTests(unittest.TestCase):
    def test_default_search_bounds_stay_local_and_inside_training_range(self):
        bounds = default_search_bounds(
            {"Gate_voltage_V": 5.0, "Gate_pulse_width_ms": 2000.0},
            ["Gate_voltage_V", "Gate_pulse_width_ms"],
            make_bundle(),
        )

        self.assertEqual(bounds["Gate_voltage_V"], (4.0, 6.0))
        self.assertEqual(bounds["Gate_pulse_width_ms"], (1600.0, 2400.0))

    def test_request_rejects_nonpositive_target(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            validate_recommendation_request(0, ["Gate_voltage_V"], make_bundle())

    def test_request_rejects_fields_that_are_not_controllable_conditions(self):
        self.assertNotIn("Ion_diffusion", RECOMMENDABLE_FIELDS)

        with self.assertRaisesRegex(ValueError, "not recommendable"):
            validate_recommendation_request(
                500.0,
                ["Ion_diffusion"],
                make_bundle(),
            )

    def test_candidate_generation_keeps_fixed_conditions_and_training_ranges(self):
        base = {
            "Channel": "P3HT",
            "Gate_voltage_V": 1.0,
            "Drain_voltage_V": 0.2,
        }

        candidates = generate_candidate_conditions(
            base,
            ["Gate_voltage_V"],
            make_bundle(),
            n_candidates=40,
            random_state=7,
        )

        self.assertEqual(set(candidates["Channel"]), {"P3HT"})
        self.assertEqual(set(candidates["Drain_voltage_V"]), {0.2})
        self.assertGreaterEqual(candidates["Gate_voltage_V"].min(), 0.0)
        self.assertLessEqual(candidates["Gate_voltage_V"].max(), 10.0)
        self.assertIn(5.0, candidates["Gate_voltage_V"].tolist())

    def test_candidate_generation_honors_engineer_search_bounds(self):
        base = {
            "Channel": "P3HT",
            "Gate_voltage_V": 1.0,
            "Drain_voltage_V": 0.2,
        }

        candidates = generate_candidate_conditions(
            base,
            ["Gate_voltage_V"],
            make_bundle(),
            n_candidates=40,
            random_state=7,
            search_bounds={"Gate_voltage_V": (3.0, 4.0)},
        )

        self.assertGreaterEqual(candidates["Gate_voltage_V"].min(), 3.0)
        self.assertLessEqual(candidates["Gate_voltage_V"].max(), 4.0)
        self.assertIn(3.0, candidates["Gate_voltage_V"].tolist())
        self.assertIn(4.0, candidates["Gate_voltage_V"].tolist())

    def test_recommendation_ranks_target_match_and_reports_error(self):
        base = {
            "Channel": "P3HT",
            "Gate_voltage_V": 1.0,
            "Drain_voltage_V": 0.2,
        }

        result = recommend_conditions(
            base_conditions=base,
            target_tau_ms=500.0,
            tunable_fields=["Gate_voltage_V"],
            bundle=make_bundle(),
            n_candidates=50,
            top_k=3,
            random_state=11,
        )

        self.assertEqual(result.loc[0, "rank"], 1)
        self.assertAlmostEqual(result.loc[0, "Gate_voltage_V"], 5.0)
        self.assertAlmostEqual(result.loc[0, "pred_tau_ms"], 500.0)
        self.assertAlmostEqual(result.loc[0, "target_error_percent"], 0.0)
        self.assertTrue(result["absolute_log_error"].is_monotonic_increasing)
        self.assertEqual(set(result["Channel"]), {"P3HT"})

    def test_same_random_state_produces_same_recommendations(self):
        base = {
            "Channel": "P3HT",
            "Gate_voltage_V": 1.0,
            "Drain_voltage_V": 0.2,
        }

        first = recommend_conditions(
            base,
            425.0,
            ["Gate_voltage_V"],
            make_bundle(),
            n_candidates=30,
            top_k=2,
            random_state=19,
        )
        second = recommend_conditions(
            base,
            425.0,
            ["Gate_voltage_V"],
            make_bundle(),
            n_candidates=30,
            top_k=2,
            random_state=19,
        )

        self.assertEqual(
            first[["Gate_voltage_V", "pred_tau_ms"]].to_dict("records"),
            second[["Gate_voltage_V", "pred_tau_ms"]].to_dict("records"),
        )


if __name__ == "__main__":
    unittest.main()
