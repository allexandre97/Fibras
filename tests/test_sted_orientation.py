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

        low_weight = StedFieldLoss2D(
            orientation_weight=0.5,
            visibility_weight=0.1,
            train_centerline_weight=0.05,
            score_centerline_weight=0.25,
        )
        high_weight = StedFieldLoss2D(
            orientation_weight=3.0,
            visibility_weight=0.7,
            train_centerline_weight=0.25,
            score_centerline_weight=0.25,
        )

        low_components = low_weight.compute_components(pred, target)
        high_components = high_weight.compute_components(pred, target)

        self.assertNotAlmostEqual(low_weight(pred, target).item(), high_weight(pred, target).item())
        self.assertAlmostEqual(
            low_weight.fixed_score(low_components).item(),
            high_weight.fixed_score(high_components).item(),
            places=6,
        )

    def test_fixed_score_depends_on_score_centerline_weight_not_train_centerline_weight(self):
        pred = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        target = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        pred[:, 0, 3, 5] = 1.0
        target[:, 0, 3, 4] = 1.0
        pred[:, 1] = 1.0
        target[:, 1] = 1.0
        target[:, 3] = 1.0

        score_low = StedFieldLoss2D(train_centerline_weight=0.15, score_centerline_weight=0.10)
        score_high = StedFieldLoss2D(train_centerline_weight=0.15, score_centerline_weight=0.40)

        components_low = score_low.compute_components(pred, target)
        components_high = score_high.compute_components(pred, target)

        self.assertAlmostEqual(score_low(pred, target).item(), score_high(pred, target).item(), places=6)
        self.assertLess(score_low.fixed_score(components_low).item(), score_high.fixed_score(components_high).item())

    def test_centerline_error_is_zero_for_identical_edt(self):
        criterion = StedFieldLoss2D()
        pred = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        target = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        pred[:, 0, 3, 4] = 1.0
        target[:, 0, 3, 4] = 1.0
        pred[:, 1] = 1.0
        target[:, 1] = 1.0
        target[:, 3] = 1.0

        components = criterion.compute_components(pred, target)

        self.assertAlmostEqual(components["centerline_error"].item(), 0.0, places=6)

    def test_centerline_error_penalizes_shifted_ridge_even_with_same_occupancy(self):
        criterion = StedFieldLoss2D()
        pred = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        target = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        pred[:, 0, 3, 5] = 1.0
        target[:, 0, 3, 4] = 1.0
        pred[:, 1] = 1.0
        target[:, 1] = 1.0
        target[:, 3] = 1.0

        components = criterion.compute_components(pred, target)

        self.assertGreater(components["centerline_error"].item(), 0.9)

    def test_centerline_error_changes_smoothly_near_old_threshold(self):
        criterion = StedFieldLoss2D()
        target = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        target[:, 0, 2, 2] = 1.0
        target[:, 1] = 1.0
        target[:, 3] = 1.0

        pred_lo = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        pred_hi = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        pred_lo[:, 0, 2, 2] = 0.84
        pred_hi[:, 0, 2, 2] = 0.86
        pred_lo[:, 1] = 1.0
        pred_hi[:, 1] = 1.0

        err_lo = criterion.compute_components(pred_lo, target)["centerline_error"].item()
        err_hi = criterion.compute_components(pred_hi, target)["centerline_error"].item()

        self.assertLess(abs(err_lo - err_hi), 0.1)

    def test_forward_includes_train_centerline_weight(self):
        criterion = StedFieldLoss2D(train_centerline_weight=0.2, centerline_warmup_epochs=0)
        pred = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        target = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        pred[:, 0, 3, 5] = 1.0
        target[:, 0, 3, 4] = 1.0
        pred[:, 1] = 1.0
        target[:, 1] = 1.0
        target[:, 3] = 1.0

        components = criterion.compute_components(pred, target)
        expected = (
            components["edt"]
            + criterion.orientation_weight * components["orientation"]
            + criterion.visibility_weight * components["visibility"]
            + criterion.current_train_centerline_weight() * components["centerline_error"]
        )

        self.assertAlmostEqual(criterion(pred, target).item(), expected.item(), places=6)

    def test_centerline_warmup_ramps_train_weight(self):
        criterion = StedFieldLoss2D(
            train_centerline_weight=0.2,
            centerline_warmup_epochs=4,
            centerline_warmup_start_factor=0.25,
        )

        weights = [criterion.set_epoch(epoch) for epoch in range(6)]

        self.assertAlmostEqual(weights[0], 0.05, places=6)
        self.assertAlmostEqual(weights[3], 0.2, places=6)
        self.assertAlmostEqual(weights[4], 0.2, places=6)
        self.assertTrue(all(a <= b for a, b in zip(weights, weights[1:])))

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
