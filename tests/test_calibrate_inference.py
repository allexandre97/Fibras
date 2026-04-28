import unittest

from src.inference_calibration import _combined_score, _pareto_front


class CalibrateInferenceTests(unittest.TestCase):
    def test_combined_score_rewards_support_and_penalizes_blank_false_positives(self):
        conservative = {
            "synthetic_f1_median": 0.70,
            "fiber_nonempty_rate": 0.80,
            "fiber_raw_skeleton_contrast_median": 3.0,
            "fiber_streamline_length_median_median": 12.0,
            "blank_nonempty_rate": 0.02,
            "blank_skeleton_fraction_median": 0.00001,
            "fiber_low_support_fraction_median": 0.05,
        }
        permissive = {
            "synthetic_f1_median": 0.72,
            "fiber_nonempty_rate": 0.90,
            "fiber_raw_skeleton_contrast_median": 3.1,
            "fiber_streamline_length_median_median": 12.5,
            "blank_nonempty_rate": 0.40,
            "blank_skeleton_fraction_median": 0.00020,
            "fiber_low_support_fraction_median": 0.10,
        }

        self.assertGreater(_combined_score(conservative), _combined_score(permissive))

    def test_pareto_front_keeps_only_nondominated_settings(self):
        rows = [
            {
                "synthetic_f1_median": 0.60,
                "fiber_raw_skeleton_contrast_median": 2.0,
                "fiber_nonempty_rate": 0.70,
                "blank_nonempty_rate": 0.10,
                "blank_skeleton_fraction_median": 0.00004,
            },
            {
                "synthetic_f1_median": 0.62,
                "fiber_raw_skeleton_contrast_median": 2.1,
                "fiber_nonempty_rate": 0.75,
                "blank_nonempty_rate": 0.08,
                "blank_skeleton_fraction_median": 0.00003,
            },
            {
                "synthetic_f1_median": 0.55,
                "fiber_raw_skeleton_contrast_median": 1.5,
                "fiber_nonempty_rate": 0.60,
                "blank_nonempty_rate": 0.15,
                "blank_skeleton_fraction_median": 0.00006,
            },
        ]

        front = _pareto_front(rows)
        self.assertEqual(front, [1])


if __name__ == "__main__":
    unittest.main()
