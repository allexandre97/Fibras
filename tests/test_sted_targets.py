import unittest

import numpy as np

from generate_dataset import _build_2d_focus_and_visibility_targets, _project_segments_to_visibility
from src.core import FiberSegment
from src.rasterization import EmpiricalRasterizer
from src.targets import TargetFieldGenerator, WeightedVisibilityTargetGenerator


class StedTargetSplitTests(unittest.TestCase):
    def setUp(self):
        self.rasterizer = EmpiricalRasterizer(
            bounds=(25, 25, 9),
            base_sigma=1.0,
            z_anisotropy=2.0,
            noise_level=0.0,
            debris_count=0,
            gap_prob=0.0,
        )
        self.target_gen = TargetFieldGenerator((25, 25), max_distance=5.0)
        self.visibility_target_gen = WeightedVisibilityTargetGenerator((25, 25), base_sigma=1.0)

    def test_annotation_band_recovers_visible_steep_segment_beyond_focus_slab(self):
        localization_slab = self.rasterizer._sted_optical_section_params()[0]
        steep_segment = FiberSegment(
            start=np.array([0.0, 12.0, 1.0], dtype=np.float64),
            end=np.array([24.0, 12.0, 7.0], dtype=np.float64),
        )

        targets = _build_2d_focus_and_visibility_targets(
            [steep_segment],
            slice_center=4.0,
            localization_slab_thickness=localization_slab,
            rasterizer=self.rasterizer,
            target_gen=self.target_gen,
            visibility_target_gen=self.visibility_target_gen,
            annotation_weight_floor=0.25,
            soft_skeleton_alpha=0.35,
        )

        self.assertEqual(len(targets["focus_segments"]), 1)
        self.assertEqual(len(targets["annotation_segments"]), 1)

        focus_length = np.linalg.norm(targets["focus_segments"][0].end - targets["focus_segments"][0].start)
        annotation_length = np.linalg.norm(targets["annotation_segments"][0].end - targets["annotation_segments"][0].start)

        self.assertLess(focus_length, annotation_length)
        self.assertLess(targets["edt_focus"][0, 12], 0.05)
        self.assertGreater(targets["edt_target"][0, 12], 0.15)
        self.assertGreater(targets["visibility_target"][0, 12], 0.05)

    def test_visibility_projection_clips_partial_segments_to_visible_band(self):
        rasterizer = EmpiricalRasterizer(
            bounds=(21, 21, 9),
            base_sigma=1.0,
            z_anisotropy=0.1,
            noise_level=0.0,
            debris_count=0,
            gap_prob=0.0,
        )
        steep_segment = FiberSegment(
            start=np.array([2.0, 10.0, 0.0], dtype=np.float64),
            end=np.array([18.0, 10.0, 8.0], dtype=np.float64),
        )

        visibility_segments, visibility_weights = _project_segments_to_visibility(
            [steep_segment],
            slice_center=4.0,
            rasterizer=rasterizer,
            min_weight=0.25,
        )

        self.assertEqual(len(visibility_segments), 1)
        self.assertEqual(len(visibility_weights), 1)
        self.assertGreater(visibility_segments[0].start[0], steep_segment.start[0])
        self.assertLess(visibility_segments[0].end[0], steep_segment.end[0])
        self.assertGreater(visibility_weights[0], 0.25)
        self.assertLess(visibility_weights[0], 1.0)


if __name__ == "__main__":
    unittest.main()
