import unittest

import numpy as np

from src.rasterization import EmpiricalRasterizer


class StedDefocusModelTests(unittest.TestCase):
    def setUp(self):
        self.rasterizer = EmpiricalRasterizer(
            bounds=(17, 17, 9),
            base_sigma=1.0,
            z_anisotropy=2.0,
            noise_level=0.0,
            debris_count=0,
            gap_prob=0.0,
        )

    def test_defocus_response_broadens_and_weakens_with_distance(self):
        focus_sigma, focus_weight = self.rasterizer._sted_defocus_response(0.0)
        far_sigma, far_weight = self.rasterizer._sted_defocus_response(4.0)

        self.assertGreater(far_sigma, focus_sigma)
        self.assertLess(far_weight, focus_weight)

    def test_out_of_focus_planes_contribute_to_optical_section(self):
        volume = np.zeros((17, 17, 9), dtype=np.float64)
        focus_xy = (8, 8)
        off_focus_xy = (3, 12)

        volume[focus_xy[0], focus_xy[1], 4] = 1.0
        volume[off_focus_xy[0], off_focus_xy[1], 1] = 1.0

        components = self.rasterizer._collapse_volume_to_slice_components(volume, slice_center=4.0)

        self.assertGreater(components["weighted_slice"][focus_xy], 0.0)
        self.assertEqual(components["focus_plane"][off_focus_xy], 0.0)
        self.assertGreater(components["weighted_slice"][off_focus_xy], 0.0)
        self.assertGreater(components["defocus_only_slice"][off_focus_xy], 0.0)
        self.assertGreater(components["lateral_sigmas"][1], components["lateral_sigmas"][4])
        self.assertLess(components["axial_weights"][1], components["axial_weights"][4])


if __name__ == "__main__":
    unittest.main()
