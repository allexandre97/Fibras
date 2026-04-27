import unittest

import numpy as np
import torch

from src.inference_utils import predict_tiled_2d
from src.model import STEDResUNet2D
from src.sted import vector_to_orientation_channels_np
from train import StedFieldLoss2D


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-4, 1.0 - 1e-4))
    return float(np.log(value / (1.0 - value)))


class StedOrientationTests(unittest.TestCase):
    def _make_target(self, width: int = 8, height: int = 8, row: int = 3, cols=range(2, 6)):
        target = torch.zeros((1, 5, height, width), dtype=torch.float32)
        for col in cols:
            target[:, 0, row, col] = 1.0
            target[:, 1, row, col] = 1.0  # cos(2theta)=1, sin(2theta)=0 for a horizontal tangent
            target[:, 3, row, col] = 1.0
            target[:, 4, row, col] = 0.6
        return target

    def _make_prediction(
        self,
        width: int = 8,
        height: int = 8,
        row: int = 3,
        cols=range(2, 6),
        centerline_prob: float = 0.97,
        traceability_prob: float = 0.95,
        radius_value: float = 0.6,
    ):
        pred = torch.zeros((1, 5, height, width), dtype=torch.float32)
        pred[:, 0] = _logit(0.01)
        pred[:, 3] = _logit(0.02)
        pred[:, 4] = _logit(0.05)
        for col in cols:
            pred[:, 0, row, col] = _logit(centerline_prob)
            pred[:, 1, row, col] = 1.0
            pred[:, 3, row, col] = _logit(traceability_prob)
            pred[:, 4, row, col] = _logit(radius_value)
        return pred

    def test_double_angle_orientation_is_sign_invariant(self):
        vector = np.zeros((2, 3, 3), dtype=np.float64)
        vector[0] = 1.0
        vector[1] = 0.5

        ori_pos = vector_to_orientation_channels_np(vector)
        ori_neg = vector_to_orientation_channels_np(-vector)

        self.assertTrue(np.allclose(ori_pos, ori_neg))

    def test_fixed_score_is_independent_of_training_weights(self):
        pred = self._make_prediction(centerline_prob=0.75, traceability_prob=0.70, radius_value=0.4)
        target = self._make_target()

        low_weight = StedFieldLoss2D(
            orientation_weight=0.5,
            visibility_weight=0.1,
            train_centerline_weight=0.25,
            score_centerline_weight=1.0,
        )
        high_weight = StedFieldLoss2D(
            orientation_weight=3.0,
            visibility_weight=0.8,
            train_centerline_weight=1.5,
            score_centerline_weight=1.0,
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
        pred = self._make_prediction(row=3, cols=range(3, 7), centerline_prob=0.70, traceability_prob=0.80)
        target = self._make_target()

        score_low = StedFieldLoss2D(train_centerline_weight=0.25, score_centerline_weight=0.20)
        score_high = StedFieldLoss2D(train_centerline_weight=0.25, score_centerline_weight=0.80)

        components_low = score_low.compute_components(pred, target)
        components_high = score_high.compute_components(pred, target)

        self.assertAlmostEqual(score_low(pred, target).item(), score_high(pred, target).item(), places=6)
        self.assertLess(score_low.fixed_score(components_low).item(), score_high.fixed_score(components_high).item())

    def test_structural_losses_are_low_for_confident_match(self):
        criterion = StedFieldLoss2D()
        pred = self._make_prediction(centerline_prob=0.995, traceability_prob=0.99, radius_value=0.6)
        target = self._make_target()

        components = criterion.compute_components(pred, target)

        self.assertLess(components["centerline"].item(), 0.10)
        self.assertLess(components["orientation"].item(), 1e-4)
        self.assertLess(components["traceability"].item(), 0.05)
        self.assertLess(components["radius"].item(), 0.01)
        self.assertLess(components["threshold_sensitivity"].item(), 0.05)

    def test_centerline_loss_penalizes_shifted_ridge(self):
        criterion = StedFieldLoss2D()
        target = self._make_target()
        pred_good = self._make_prediction(centerline_prob=0.97, traceability_prob=0.95)
        pred_shifted = self._make_prediction(row=3, cols=range(3, 7), centerline_prob=0.97, traceability_prob=0.95)

        good_loss = criterion.compute_components(pred_good, target)["centerline"].item()
        shifted_loss = criterion.compute_components(pred_shifted, target)["centerline"].item()

        self.assertGreater(shifted_loss, good_loss + 0.20)

    def test_threshold_sensitivity_penalizes_ambiguous_centerline_logits(self):
        criterion = StedFieldLoss2D(centerline_threshold=0.5)
        target = self._make_target()
        pred_confident = self._make_prediction(centerline_prob=0.97, traceability_prob=0.95)
        pred_ambiguous = self._make_prediction(centerline_prob=0.52, traceability_prob=0.95)

        sens_confident = criterion.compute_components(pred_confident, target)["threshold_sensitivity"].item()
        sens_ambiguous = criterion.compute_components(pred_ambiguous, target)["threshold_sensitivity"].item()

        self.assertGreater(sens_ambiguous, sens_confident + 0.20)

    def test_forward_includes_new_structural_terms(self):
        criterion = StedFieldLoss2D(
            orientation_weight=0.7,
            visibility_weight=0.3,
            radius_weight=0.4,
            train_centerline_weight=0.2,
            stability_margin_weight=0.5,
            centerline_warmup_epochs=0,
        )
        pred = self._make_prediction(row=3, cols=range(3, 7), centerline_prob=0.70, traceability_prob=0.65, radius_value=0.3)
        target = self._make_target()

        components = criterion.compute_components(pred, target)
        expected = (
            criterion.current_train_centerline_weight() * components["centerline"]
            + criterion.orientation_weight * components["orientation"]
            + criterion.traceability_weight * components["traceability"]
            + criterion.radius_weight * components["radius"]
            + criterion.stability_margin_weight * components["stability_margin"]
        )

        self.assertAlmostEqual(criterion(pred, target).item(), expected.item(), places=6)

    def test_centerline_warmup_ramps_train_weight(self):
        criterion = StedFieldLoss2D(
            train_centerline_weight=0.4,
            centerline_warmup_epochs=4,
            centerline_warmup_start_factor=0.25,
        )

        weights = [criterion.set_epoch(epoch) for epoch in range(6)]

        self.assertAlmostEqual(weights[0], 0.10, places=6)
        self.assertAlmostEqual(weights[3], 0.40, places=6)
        self.assertAlmostEqual(weights[4], 0.40, places=6)
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
            output_channels=5,
            use_amp=False,
        )

        self.assertEqual(pred.shape, (5, 40, 52))


if __name__ == "__main__":
    unittest.main()
