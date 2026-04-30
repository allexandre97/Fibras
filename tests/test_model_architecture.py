import argparse
import unittest

import torch
import torch.nn as nn

import sweep
from src.model import (
    BottleneckPredictionHead2D,
    DEFAULT_HEAD_HIDDEN_CHANNELS,
    DEFAULT_HEAD_TYPE,
    DEFAULT_UNET_DEPTH,
    DEFAULT_USE_HEAD_REFINEMENT,
    FULL_WIDTH_HEAD_TYPE,
    HEAD_TYPES,
    LEGACY_HEAD_TYPE,
    LEGACY_ASPP_DILATIONS,
    LEGACY_UNET_DEPTH,
    PREDICTION_HEAD_TYPE,
    STEDResUNet2D,
    ShallowPredictionHead2D,
)
from train import (
    checkpoint_aspp_dilations,
    checkpoint_head_hidden_channels,
    checkpoint_head_type,
    checkpoint_unet_depth,
    checkpoint_use_head_refinement,
    parse_aspp_dilations,
)


def _aspp_dilated_convs(model):
    aspp = model.bottleneck[1]
    return [branch[0] for branch in aspp.branches[1:]]


class ModelArchitectureTests(unittest.TestCase):
    def test_default_aspp_dilations_are_reduced(self):
        model = STEDResUNet2D(base_filters=8)
        convs = _aspp_dilated_convs(model)

        self.assertEqual(model.unet_depth, DEFAULT_UNET_DEPTH)
        self.assertEqual(model.bottleneck[1].aspp_dilations, (1, 2, 4))
        self.assertEqual(tuple(conv.dilation[0] for conv in convs), (1, 2, 4))
        self.assertEqual(tuple(conv.padding[0] for conv in convs), (1, 2, 4))

        with torch.no_grad():
            pred = model(torch.zeros((1, 1, 41, 53), dtype=torch.float32))
        self.assertEqual(tuple(pred.shape), (1, 6, 41, 53))

    def test_custom_aspp_dilations_are_reflected_in_branches(self):
        model = STEDResUNet2D(base_filters=8, aspp_dilations=(1, 2, 3))
        convs = _aspp_dilated_convs(model)

        self.assertEqual(model.bottleneck[1].aspp_dilations, (1, 2, 3))
        self.assertEqual(tuple(conv.dilation[0] for conv in convs), (1, 2, 3))
        self.assertEqual(tuple(conv.padding[0] for conv in convs), (1, 2, 3))

    def test_unet_depth_controls_encoder_decoder_stage_count(self):
        for depth in (3, 4):
            with self.subTest(depth=depth):
                model = STEDResUNet2D(base_filters=8, unet_depth=depth)
                self.assertEqual(model.unet_depth, depth)
                self.assertEqual(len(model.encoder_blocks), depth)
                self.assertEqual(len(model.upsamplers), depth)
                self.assertEqual(len(model.decoder_blocks), depth)
                with torch.no_grad():
                    pred = model(torch.zeros((1, 1, 41, 53), dtype=torch.float32))
                self.assertEqual(tuple(pred.shape), (1, 6, 41, 53))

    def test_default_prediction_heads_are_bottleneck_spatial_heads(self):
        model = STEDResUNet2D(base_filters=8)
        head_specs = [
            ("centerline_head", 1),
            ("orientation_head", 2),
            ("traceability_head", 1),
            ("radius_head", 1),
            ("bundle_count_head", 1),
        ]

        self.assertEqual(model.head_type, DEFAULT_HEAD_TYPE)
        self.assertEqual(model.head_hidden_channels, DEFAULT_HEAD_HIDDEN_CHANNELS)
        self.assertEqual(model.use_head_refinement, DEFAULT_USE_HEAD_REFINEMENT)
        for head_name, out_channels in head_specs:
            with self.subTest(head_name=head_name):
                head = getattr(model, head_name)
                self.assertIsInstance(head, BottleneckPredictionHead2D)
                self.assertIsInstance(head.reduce, nn.Conv2d)
                self.assertEqual(head.reduce.kernel_size, (1, 1))
                self.assertEqual(head.reduce.out_channels, DEFAULT_HEAD_HIDDEN_CHANNELS)
                self.assertIsInstance(head.spatial, nn.Conv2d)
                self.assertEqual(head.spatial.kernel_size, (3, 3))
                self.assertEqual(head.spatial.padding, (1, 1))
                self.assertEqual(head.spatial.in_channels, DEFAULT_HEAD_HIDDEN_CHANNELS)
                self.assertEqual(head.spatial.out_channels, DEFAULT_HEAD_HIDDEN_CHANNELS)
                self.assertIsInstance(head.norm, nn.GroupNorm)
                self.assertIsInstance(head.act, nn.GELU)
                self.assertIsInstance(head.project, nn.Conv2d)
                self.assertEqual(head.project.kernel_size, (1, 1))
                self.assertEqual(head.project.out_channels, out_channels)
                self.assertIs(list(head.children())[-1], head.project)

    def test_full_width_and_linear_prediction_heads_are_available(self):
        full_model = STEDResUNet2D(base_filters=8, head_type=FULL_WIDTH_HEAD_TYPE)
        self.assertIsInstance(full_model.centerline_head, ShallowPredictionHead2D)
        self.assertTrue(hasattr(full_model, "head_refinement"))

        linear_model = STEDResUNet2D(
            base_filters=8,
            head_type=LEGACY_HEAD_TYPE,
            use_head_refinement=False,
        )
        self.assertIsInstance(linear_model.centerline_head, nn.Conv2d)
        self.assertIsInstance(linear_model.head_refinement, nn.Identity)
        with torch.no_grad():
            pred = linear_model(torch.zeros((1, 1, 41, 53), dtype=torch.float32))
        self.assertEqual(tuple(pred.shape), (1, 6, 41, 53))

    def test_invalid_aspp_dilations_are_rejected(self):
        invalid_values = [
            (),
            (0, 2),
            (-1, 2),
            (1.5, 2),
            "1,2,4",
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    STEDResUNet2D(base_filters=8, aspp_dilations=value)

    def test_invalid_unet_depths_are_rejected(self):
        for value in [0, 2, 5, 3.5, True, "3"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    STEDResUNet2D(base_filters=8, unet_depth=value)

    def test_invalid_head_options_are_rejected(self):
        with self.assertRaises(ValueError):
            STEDResUNet2D(base_filters=8, head_type="wide")
        for value in [0, -1, 1.5, True, "16"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    STEDResUNet2D(base_filters=8, head_hidden_channels=value)

    def test_parse_aspp_dilations_accepts_comma_separated_values(self):
        self.assertEqual(parse_aspp_dilations("1,2,4"), (1, 2, 4))
        self.assertEqual(parse_aspp_dilations("1, 2, 3"), (1, 2, 3))

    def test_parse_aspp_dilations_rejects_invalid_strings(self):
        for value in ["", "1,,2", "1,0,2", "1,-2", "1,2.5"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_aspp_dilations(value)

    def test_checkpoint_aspp_dilations_prefers_config_and_legacy_fallback(self):
        self.assertEqual(
            checkpoint_aspp_dilations({"config": {"aspp_dilations": "1,2,3"}}),
            (1, 2, 3),
        )
        self.assertEqual(checkpoint_aspp_dilations({"model_state_dict": {}}), LEGACY_ASPP_DILATIONS)
        self.assertEqual(checkpoint_aspp_dilations({}, override="1,2,4"), (1, 2, 4))

    def test_checkpoint_unet_depth_prefers_config_and_legacy_fallback(self):
        self.assertEqual(checkpoint_unet_depth({"config": {"unet_depth": 3}}), 3)
        self.assertEqual(checkpoint_unet_depth({"model_state_dict": {}}), LEGACY_UNET_DEPTH)
        self.assertEqual(checkpoint_unet_depth({}, override=4), 4)

    def test_checkpoint_head_settings_prefer_config_and_legacy_fallback(self):
        checkpoint = {
            "config": {
                "head_type": DEFAULT_HEAD_TYPE,
                "head_hidden_channels": 24,
                "use_head_refinement": True,
            }
        }
        self.assertEqual(checkpoint_head_type(checkpoint), DEFAULT_HEAD_TYPE)
        self.assertEqual(checkpoint_head_hidden_channels(checkpoint), 24)
        self.assertTrue(checkpoint_use_head_refinement(checkpoint))

        previous_new_model = {"config": {"prediction_head_type": "shallow_3x3_gn_gelu"}}
        self.assertEqual(checkpoint_head_type(previous_new_model), FULL_WIDTH_HEAD_TYPE)
        self.assertTrue(checkpoint_use_head_refinement(previous_new_model))

        self.assertEqual(checkpoint_head_type({"model_state_dict": {}}), LEGACY_HEAD_TYPE)
        self.assertFalse(checkpoint_use_head_refinement({"model_state_dict": {}}))
        self.assertEqual(checkpoint_head_type({}, override=FULL_WIDTH_HEAD_TYPE), FULL_WIDTH_HEAD_TYPE)
        self.assertEqual(checkpoint_head_hidden_channels({}, override=8), 8)
        self.assertTrue(checkpoint_use_head_refinement({}, override="true"))

    def test_sweep_config_includes_aspp_dilation_values(self):
        args = argparse.Namespace(
            base_filters_values=[32],
            unet_depth_values=[3, 4],
            aspp_dilation_values=["1,2,4", "1,2,3", "2,4,8"],
            head_variant_values=["bottleneck_3x3_8", "linear_1x1"],
            use_head_refinement_values=[True, False],
        )

        config = sweep._build_sweep_config(args)

        self.assertEqual(
            config["parameters"]["aspp_dilations"]["values"],
            ["1,2,4", "1,2,3", "2,4,8"],
        )
        self.assertEqual(config["parameters"]["unet_depth"]["values"], [3, 4])
        self.assertEqual(config["parameters"]["head_variant"]["values"], ["bottleneck_3x3_8", "linear_1x1"])
        self.assertEqual(config["parameters"]["use_head_refinement"]["values"], [True, False])

    def test_sweep_static_config_records_prediction_head_type(self):
        self.assertEqual(PREDICTION_HEAD_TYPE, "bottleneck_3x3_gn_gelu")
        self.assertIn(DEFAULT_HEAD_TYPE, HEAD_TYPES)


if __name__ == "__main__":
    unittest.main()
