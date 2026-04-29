import argparse
import unittest

import torch

import sweep
from src.model import LEGACY_ASPP_DILATIONS, STEDResUNet2D
from train import checkpoint_aspp_dilations, parse_aspp_dilations


def _aspp_dilated_convs(model):
    aspp = model.bottleneck[1]
    return [branch[0] for branch in aspp.branches[1:]]


class ModelArchitectureTests(unittest.TestCase):
    def test_default_aspp_dilations_are_reduced(self):
        model = STEDResUNet2D(base_filters=8)
        convs = _aspp_dilated_convs(model)

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

    def test_sweep_config_includes_aspp_dilation_values(self):
        args = argparse.Namespace(
            base_filters_values=[32],
            aspp_dilation_values=["1,2,4", "1,2,3", "2,4,8"],
        )

        config = sweep._build_sweep_config(args)

        self.assertEqual(
            config["parameters"]["aspp_dilations"]["values"],
            ["1,2,4", "1,2,3", "2,4,8"],
        )


if __name__ == "__main__":
    unittest.main()
