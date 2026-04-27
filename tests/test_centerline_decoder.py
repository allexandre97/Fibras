import unittest

import numpy as np

from src.decoder import CenterlineGraphDecoder


class CenterlineDecoderTests(unittest.TestCase):
    def _horizontal_orientation(self, shape):
        orientation = np.zeros((2,) + shape, dtype=np.float32)
        orientation[0] = 1.0
        return orientation

    def test_decoder_is_deterministic(self):
        centerline_prob = np.zeros((16, 16), dtype=np.float32)
        centerline_prob[8, 3:13] = 0.95
        traceability = np.ones((16, 16), dtype=np.float32)
        decoder = CenterlineGraphDecoder(centerline_threshold=0.5)

        first = decoder.decode(centerline_prob, self._horizontal_orientation(centerline_prob.shape), traceability)
        second = decoder.decode(centerline_prob, self._horizontal_orientation(centerline_prob.shape), traceability)

        self.assertTrue(np.array_equal(first.skeleton, second.skeleton))
        self.assertEqual(len(first.component_paths), len(second.component_paths))

    def test_small_threshold_perturbations_preserve_clean_topology(self):
        centerline_prob = np.zeros((32, 32), dtype=np.float32)
        centerline_prob[16, 6:26] = 0.92
        traceability = np.ones((32, 32), dtype=np.float32)
        orientation = self._horizontal_orientation(centerline_prob.shape)

        decoded_lo = CenterlineGraphDecoder(centerline_threshold=0.45).decode(centerline_prob, orientation, traceability)
        decoded_mid = CenterlineGraphDecoder(centerline_threshold=0.50).decode(centerline_prob, orientation, traceability)
        decoded_hi = CenterlineGraphDecoder(centerline_threshold=0.55).decode(centerline_prob, orientation, traceability)

        self.assertTrue(np.array_equal(decoded_lo.skeleton, decoded_mid.skeleton))
        self.assertTrue(np.array_equal(decoded_mid.skeleton, decoded_hi.skeleton))
        self.assertEqual(len(decoded_mid.component_paths), 1)

    def test_decoder_bridges_short_gap_when_traceability_supports_it(self):
        centerline_prob = np.zeros((24, 24), dtype=np.float32)
        centerline_prob[12, 4:10] = 0.95
        centerline_prob[12, 13:19] = 0.95
        traceability = np.zeros((24, 24), dtype=np.float32)
        traceability[12, 4:19] = 1.0

        decoded = CenterlineGraphDecoder(centerline_threshold=0.5, bridge_gap=4).decode(
            centerline_prob,
            self._horizontal_orientation(centerline_prob.shape),
            traceability,
        )

        self.assertEqual(len(decoded.component_paths), 1)
        self.assertTrue(np.any(decoded.skeleton[12, 10:13]))

    def test_default_decoder_bridges_aligned_multpixel_gap(self):
        centerline_prob = np.zeros((28, 28), dtype=np.float32)
        centerline_prob[14, 4:9] = 0.95
        centerline_prob[14, 16:21] = 0.95
        traceability = np.ones((28, 28), dtype=np.float32)

        decoded = CenterlineGraphDecoder(centerline_threshold=0.5).decode(
            centerline_prob,
            self._horizontal_orientation(centerline_prob.shape),
            traceability,
        )

        self.assertEqual(len(decoded.component_paths), 1)
        self.assertTrue(np.any(decoded.skeleton[14, 9:16]))

    def test_decoder_prunes_tiny_components(self):
        centerline_prob = np.zeros((20, 20), dtype=np.float32)
        centerline_prob[10, 3:13] = 0.95
        centerline_prob[3, 3:5] = 0.95
        traceability = np.ones((20, 20), dtype=np.float32)

        decoded = CenterlineGraphDecoder(centerline_threshold=0.5, min_component_pixels=4).decode(
            centerline_prob,
            self._horizontal_orientation(centerline_prob.shape),
            traceability,
        )

        self.assertEqual(len(decoded.component_paths), 1)
        self.assertFalse(np.any(decoded.skeleton[3, 3:5]))

    def test_default_pruning_keeps_short_high_confidence_ridge_fragments(self):
        centerline_prob = np.zeros((24, 24), dtype=np.float32)
        for col in range(4, 20, 4):
            centerline_prob[12, col : col + 2] = 0.95
        traceability = np.ones((24, 24), dtype=np.float32)

        decoded = CenterlineGraphDecoder(centerline_threshold=0.5).decode(
            centerline_prob,
            self._horizontal_orientation(centerline_prob.shape),
            traceability,
        )

        self.assertGreaterEqual(np.count_nonzero(decoded.candidate_mask), 8)
        self.assertGreaterEqual(np.count_nonzero(decoded.skeleton), 8)

    def test_diagonal_skeleton_pixels_are_one_component(self):
        centerline_prob = np.zeros((24, 24), dtype=np.float32)
        for offset in range(12):
            centerline_prob[6 + offset, 6 + offset] = 0.95
        traceability = np.ones((24, 24), dtype=np.float32)
        orientation = np.zeros((2, 24, 24), dtype=np.float32)
        orientation[0] = 1.0 / np.sqrt(2.0)
        orientation[1] = 1.0 / np.sqrt(2.0)

        decoded = CenterlineGraphDecoder(centerline_threshold=0.5).decode(
            centerline_prob,
            orientation,
            traceability,
        )

        self.assertEqual(len(decoded.component_paths), 1)

    def test_hysteresis_discards_isolated_weak_noise(self):
        centerline_prob = np.zeros((24, 24), dtype=np.float32)
        centerline_prob[12, 4:20] = 0.90
        centerline_prob[4, 4] = 0.30
        centerline_prob[5, 18] = 0.33
        traceability = np.ones((24, 24), dtype=np.float32)

        decoded = CenterlineGraphDecoder(
            centerline_threshold=0.5,
            min_component_pixels=1,
        ).decode(
            centerline_prob,
            self._horizontal_orientation(centerline_prob.shape),
            traceability,
        )

        self.assertEqual(len(decoded.component_paths), 1)
        self.assertFalse(decoded.candidate_mask[4, 4])
        self.assertFalse(decoded.candidate_mask[5, 18])
        self.assertTrue(np.any(decoded.strong_seed_mask[12, 4:20]))

    def test_hysteresis_keeps_weak_pixels_that_connect_strong_ridges(self):
        centerline_prob = np.zeros((24, 24), dtype=np.float32)
        centerline_prob[12, 4:9] = 0.92
        centerline_prob[12, 9:15] = 0.28
        centerline_prob[12, 15:20] = 0.92
        traceability = np.ones((24, 24), dtype=np.float32)

        decoded = CenterlineGraphDecoder(
            centerline_threshold=0.5,
            bridge_gap=0,
            min_component_pixels=1,
        ).decode(
            centerline_prob,
            self._horizontal_orientation(centerline_prob.shape),
            traceability,
        )

        self.assertEqual(len(decoded.component_paths), 1)
        self.assertTrue(np.all(decoded.candidate_mask[12, 9:15]))
        self.assertTrue(np.any(decoded.skeleton[12, 9:15]))

    def test_low_orientation_confidence_suppresses_spurious_branch(self):
        centerline_prob = np.zeros((24, 24), dtype=np.float32)
        centerline_prob[12, 4:20] = 0.92
        centerline_prob[8:13, 12] = 0.82
        traceability = np.ones((24, 24), dtype=np.float32)
        orientation = self._horizontal_orientation(centerline_prob.shape)
        orientation_confidence = np.ones((24, 24), dtype=np.float32)
        orientation_confidence[8:12, 12] = 0.0

        decoded = CenterlineGraphDecoder(
            centerline_threshold=0.5,
            min_component_pixels=1,
        ).decode(
            centerline_prob,
            orientation,
            traceability,
            orientation_confidence=orientation_confidence,
        )

        self.assertTrue(np.any(decoded.skeleton[12, 4:20]))
        self.assertFalse(np.any(decoded.candidate_mask[8:12, 12]))


if __name__ == "__main__":
    unittest.main()
