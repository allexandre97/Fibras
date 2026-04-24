import os
import tempfile
import unittest

import numpy as np

from generate_dataset import _prepare_2d_sted_scene
from src.sted_calibration import (
    CalibrationSampler,
    build_calibration_profile,
    compute_image_metrics,
    estimate_calibrated_fiber_count,
    load_calibration_profile,
    parse_sted_filename,
)


def _minimal_profile():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[16:48, 31:34] = 80
    image[31:34, 16:48] = 120
    rows = [compute_image_metrics(image, source="PN148_3R_AD_DIV10 (Series 2) [1].tif", row_type="image")]
    return build_calibration_profile(rows, source_dir="/tmp/real", patch_size=64)


class StedCalibrationTests(unittest.TestCase):
    def test_parse_sted_filename_extracts_real_dataset_metadata(self):
        metadata = parse_sted_filename("PN148_4R_PSP_DIV7 (Series 14) [1].tif")

        self.assertEqual(metadata["condition"], "PSP")
        self.assertEqual(metadata["div"], 7)
        self.assertEqual(metadata["replicate"], "4R")
        self.assertEqual(metadata["series"], 14)

    def test_compute_image_metrics_detects_sparse_foreground(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[8:56, 30:34] = 160

        metrics = compute_image_metrics(image)

        self.assertGreater(metrics["foreground_fraction"], 0.0)
        self.assertLess(metrics["foreground_fraction"], 0.20)
        self.assertGreater(metrics["skeleton_fraction"], 0.0)
        self.assertGreater(metrics["width_median"], 0.0)
        self.assertIn("stripe_strength", metrics)

    def test_profile_round_trip_and_sampler(self):
        profile = _minimal_profile()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "profile.json")
            import json

            with open(path, "w", encoding="utf-8") as handle:
                json.dump(profile, handle)

            loaded = load_calibration_profile(path)

        sampler = CalibrationSampler(loaded)
        np.random.seed(0)
        scene_config = sampler.sample_scene_config((64, 64, 16))

        self.assertIn("target_skeleton_fraction", scene_config)
        self.assertIn(scene_config["haze_regime"], {"none", "subtle", "moderate", "strong"})
        self.assertGreater(scene_config["dynamic_range"][1], scene_config["dynamic_range"][0])

    def test_calibrated_fiber_count_uses_skeleton_density_not_area_density(self):
        profile = _minimal_profile()
        sampler = CalibrationSampler(profile)
        np.random.seed(1)
        scene_config = sampler.sample_scene_config((1024, 1024, 16))
        scene_config["target_skeleton_fraction"] = 0.006
        count = estimate_calibrated_fiber_count((1024, 1024, 16), scene_config, depth_of_field=3.0)

        self.assertGreaterEqual(count, 1)
        self.assertLess(count, 100)

    def test_calibrated_scene_records_profile_metadata(self):
        profile = _minimal_profile()
        sampler = CalibrationSampler(profile)

        np.random.seed(4)
        scene = _prepare_2d_sted_scene((48, 48, 12), None, calibration_sampler=sampler)

        self.assertIsNotNone(scene["calibration_scene_config"])
        self.assertIn("target_foreground_fraction", scene["calibration_scene_config"])
        self.assertGreaterEqual(scene["requested_fiber_count"], 0)
        self.assertLessEqual(scene["requested_fiber_count"], 1)


if __name__ == "__main__":
    unittest.main()
