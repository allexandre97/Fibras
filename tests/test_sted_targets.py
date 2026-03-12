import unittest

import numpy as np

from generate_dataset import _build_2d_focus_and_visibility_targets
from src.core import FiberSegment
from src.rasterization import EmpiricalRasterizer
from src.targets import TargetFieldGenerator, WeightedVisibilityTargetGenerator


class StedTargetSplitTests(unittest.TestCase):
    def test_visibility_target_keeps_out_of_focus_fiber_soft(self):
        rasterizer = EmpiricalRasterizer(
            bounds=(17, 17, 9),
            base_sigma=1.0,
            z_anisotropy=2.0,
            noise_level=0.0,
            debris_count=0,
            gap_prob=0.0,
        )
        target_gen = TargetFieldGenerator((17, 17), max_distance=5.0)
        visibility_target_gen = WeightedVisibilityTargetGenerator((17, 17), base_sigma=1.0)
        localization_slab = rasterizer._sted_optical_section_params()[0]

        focus_segment = FiberSegment(
            start=np.array([2.0, 8.0, 4.0], dtype=np.float64),
            end=np.array([14.0, 8.0, 4.0], dtype=np.float64),
        )
        off_focus_segment = FiberSegment(
            start=np.array([2.0, 4.0, 1.0], dtype=np.float64),
            end=np.array([14.0, 4.0, 1.0], dtype=np.float64),
        )

        targets = _build_2d_focus_and_visibility_targets(
            [focus_segment, off_focus_segment],
            slice_center=4.0,
            localization_slab_thickness=localization_slab,
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_target_gen,
        )

        self.assertEqual(len(targets["focus_segments"]), 1)
        self.assertEqual(len(targets["visibility_segments"]), 2)
        self.assertGreater(targets["edt_target"][8, 8], 0.9)
        self.assertLess(targets["edt_target"][8, 4], 0.25)
        self.assertGreater(targets["visibility_target"][8, 8], 0.4)
        self.assertGreater(targets["visibility_target"][8, 4], 0.05)
        self.assertLess(targets["visibility_target"][8, 4], targets["visibility_target"][8, 8])


if __name__ == "__main__":
    unittest.main()
