import unittest

import numpy as np
import torch

from src.inference_utils import predict_tiled_2d
from src.model import STEDResUNet2D
from src.sted import vector_to_orientation_channels_np
from train import StedFieldLoss2D


class StedOrientationTests(unittest.TestCase):
    def test_double_angle_orientation_is_sign_invariant(self):
        vector = np.zeros((2, 3, 3), dtype=np.float64)
        vector[0] = 1.0
        vector[1] = 0.5

        ori_pos = vector_to_orientation_channels_np(vector)
        ori_neg = vector_to_orientation_channels_np(-vector)

        self.assertTrue(np.allclose(ori_pos, ori_neg))

    def test_fixed_score_is_independent_of_training_weights(self):
        pred = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        target = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        target[:, 0] = 1.0
        target[:, 1] = 1.0
        target[:, 3] = 1.0

        low_weight = StedFieldLoss2D(orientation_weight=0.5, visibility_weight=0.1)
        high_weight = StedFieldLoss2D(orientation_weight=3.0, visibility_weight=0.7)

        low_components = low_weight.compute_components(pred, target)
        high_components = high_weight.compute_components(pred, target)

        self.assertNotAlmostEqual(low_weight(pred, target).item(), high_weight(pred, target).item())
        self.assertAlmostEqual(
            low_weight.fixed_score(low_components).item(),
            high_weight.fixed_score(high_components).item(),
            places=6,
        )

    def test_tiled_prediction_preserves_input_shape(self):
        model = STEDResUNet2D(base_filters=8)
        image = np.zeros((40, 52), dtype=np.float32)
        pred = predict_tiled_2d(
            model,
            image,
            device=torch.device("cpu"),
            tile_size=32,
            overlap=8,
            output_channels=4,
            use_amp=False,
        )

        self.assertEqual(pred.shape, (4, 40, 52))


if __name__ == "__main__":
    unittest.main()
