import argparse
import unittest

import torch
import torch.nn as nn

import sweep
from src.model import (
    DEFAULT_UNET_DEPTH,
    LEGACY_ASPP_DILATIONS,
    LEGACY_UNET_DEPTH,
    PREDICTION_HEAD_TYPE,
    STEDResUNet2D,
    ShallowPredictionHead2D,
)
from train import checkpoint_aspp_dilations, checkpoint_unet_depth, parse_aspp_dilations


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

    def test_prediction_heads_are_task_specific_shallow_heads(self):
        model = STEDResUNet2D(base_filters=8)
        head_specs = [
            ("centerline_head", 1),
            ("orientation_head", 2),
            ("traceability_head", 1),
            ("radius_head", 1),
            ("bundle_count_head", 1),
        ]

        for head_name, out_channels in head_specs:
            with self.subTest(head_name=head_name):
                head = getattr(model, head_name)
                self.assertIsInstance(head, ShallowPredictionHead2D)
                self.assertIsInstance(head.spatial, nn.Conv2d)
                self.assertEqual(head.spatial.kernel_size, (3, 3))
                self.assertEqual(head.spatial.padding, (1, 1))
                self.assertIsInstance(head.norm, nn.GroupNorm)
                self.assertIsInstance(head.act, nn.GELU)
                self.assertIsInstance(head.project, nn.Conv2d)
                self.assertEqual(head.project.kernel_size, (1, 1))
                self.assertEqual(head.project.out_channels, out_channels)
                self.assertIs(list(head.children())[-1], head.project)

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

    def test_sweep_config_includes_aspp_dilation_values(self):
        args = argparse.Namespace(
            base_filters_values=[32],
            unet_depth_values=[3, 4],
            aspp_dilation_values=["1,2,4", "1,2,3", "2,4,8"],
        )

        config = sweep._build_sweep_config(args)

        self.assertEqual(
            config["parameters"]["aspp_dilations"]["values"],
            ["1,2,4", "1,2,3", "2,4,8"],
        )
        self.assertEqual(config["parameters"]["unet_depth"]["values"], [3, 4])

    def test_sweep_static_config_records_prediction_head_type(self):
        self.assertEqual(PREDICTION_HEAD_TYPE, "shallow_3x3_gn_gelu")


if __name__ == "__main__":
    unittest.main()
