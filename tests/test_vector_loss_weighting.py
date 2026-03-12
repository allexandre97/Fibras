import unittest

import torch

from train import MaskedVectorLoss


class VectorLossWeightingTests(unittest.TestCase):
    def test_vector_loss_uses_confidence_weighting(self):
        criterion = MaskedVectorLoss(dim=2, vector_mask_floor=0.05)

        pred = torch.zeros((1, 4, 1, 2), dtype=torch.float32)
        target = torch.zeros((1, 4, 1, 2), dtype=torch.float32)

        # Confidence map: one strong and one weak supervised pixel.
        target[:, 0, 0, 0] = 1.0
        target[:, 0, 0, 1] = 0.1

        # Target vectors.
        target[:, 1, 0, 0] = 1.0
        target[:, 1, 0, 1] = 1.0

        # Predicted vectors: first pixel has error, second is perfect.
        pred[:, 1, 0, 0] = 0.0
        pred[:, 1, 0, 1] = 1.0

        components = criterion.compute_components(pred, target)
        vector_loss = components["vector"].item()

        expected = (0.5 * 1.0 + 0.0 * 0.1) / (1.0 + 0.1)
        self.assertAlmostEqual(vector_loss, expected, places=6)

    def test_vector_mask_floor_suppresses_low_confidence_tail(self):
        pred = torch.zeros((1, 4, 1, 2), dtype=torch.float32)
        target = torch.zeros((1, 4, 1, 2), dtype=torch.float32)

        target[:, 0, 0, 0] = 1.0
        target[:, 0, 0, 1] = 0.04

        target[:, 1, 0, 0] = 1.0
        target[:, 1, 0, 1] = 1.0

        # First pixel moderate error (0.5), second pixel very large error (~8.0).
        pred[:, 1, 0, 0] = 0.0
        pred[:, 1, 0, 1] = 5.0

        criterion_with_floor = MaskedVectorLoss(dim=2, vector_mask_floor=0.05)
        criterion_no_floor = MaskedVectorLoss(dim=2, vector_mask_floor=0.0)

        vec_with_floor = criterion_with_floor.compute_components(pred, target)["vector"].item()
        vec_no_floor = criterion_no_floor.compute_components(pred, target)["vector"].item()

        self.assertAlmostEqual(vec_with_floor, 0.5, places=6)
        self.assertGreater(vec_no_floor, vec_with_floor)

    def test_negative_vector_mask_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            MaskedVectorLoss(dim=2, vector_mask_floor=-0.1)


if __name__ == "__main__":
    unittest.main()
