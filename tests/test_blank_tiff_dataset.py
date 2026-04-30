import os
import json
import sys
import tempfile
import unittest

import numpy as np
import tifffile
import torch

from generate_dataset import main as generate_dataset_main
from generate_dataset import plan_blank_tiff_splits, save_blank_tiff_crops
from train import _validate_dataset


class BlankTiffDatasetTests(unittest.TestCase):
    def test_blank_tiff_crops_are_saved_as_zero_target_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blank_dir = os.path.join(tmpdir, "blanks")
            out_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(blank_dir)
            image_path = os.path.join(blank_dir, "blank.tif")
            image = (np.arange(120, dtype=np.uint16).reshape(10, 12) * 100).astype(np.uint16)
            tifffile.imwrite(image_path, image)

            saved = save_blank_tiff_crops(
                split_name="train",
                base_dir=out_dir,
                blank_tiff_paths=[image_path],
                crop_bounds=(8, 6),
                crops_per_image=2,
                normalization="percentile",
                percentile_low=0.5,
                percentile_high=99.9,
            )

            self.assertEqual(saved, 2)
            sample_paths = sorted(
                os.path.join(out_dir, "train", name)
                for name in os.listdir(os.path.join(out_dir, "train"))
                if name.endswith(".pt")
            )
            self.assertEqual(len(sample_paths), 2)

            sample = torch.load(sample_paths[0], map_location="cpu", weights_only=True)
            self.assertEqual(sample["target_schema"], "structural_v2")
            self.assertEqual(tuple(sample["volume"].shape), (1, 1, 6, 8))
            self.assertEqual(tuple(sample["targets"].shape), (6, 1, 6, 8))
            self.assertTrue(torch.all(sample["targets"] == 0.0))
            self.assertGreaterEqual(float(sample["volume"].min()), 0.0)
            self.assertLessEqual(float(sample["volume"].max()), 1.0)
            self.assertEqual(sample["metadata"]["sample_type"], "blank")
            self.assertEqual(sample["metadata"]["blank_split"], "train")
            self.assertEqual(sample["metadata"]["source_path"], os.path.abspath(image_path))
            self.assertIn("source_file_id", sample["metadata"])
            self.assertIn("source_intensity_summary", sample["metadata"])

    def test_blank_rewrite_removes_stale_blank_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blank_dir = os.path.join(tmpdir, "blanks")
            out_dir = os.path.join(tmpdir, "dataset")
            split_dir = os.path.join(out_dir, "train")
            os.makedirs(blank_dir)
            os.makedirs(split_dir)
            image_path = os.path.join(blank_dir, "blank.tif")
            tifffile.imwrite(image_path, np.ones((8, 8), dtype=np.uint16))
            torch.save({"stale": True}, os.path.join(split_dir, "blank_999999.pt"))
            torch.save({"synthetic": True}, os.path.join(split_dir, "sample_000000.pt"))

            saved = save_blank_tiff_crops(
                split_name="train",
                base_dir=out_dir,
                blank_tiff_paths=[image_path],
                crop_bounds=(8, 8),
                crops_per_image=1,
                normalization="percentile",
                percentile_low=0.5,
                percentile_high=99.9,
            )

            self.assertEqual(saved, 1)
            names = sorted(os.listdir(split_dir))
            self.assertIn("sample_000000.pt", names)
            self.assertEqual([name for name in names if name.startswith("blank_")], ["blank_000000.pt"])

    def test_blank_split_planner_is_deterministic_and_disjoint(self):
        paths = [f"/tmp/blank_{idx}.tif" for idx in range(9)]
        split_sizes = {"train": 8, "val": 2, "test": 2}

        first, unused_first = plan_blank_tiff_splits(
            paths,
            split_names=["train", "val", "test"],
            split_sizes=split_sizes,
            policy="disjoint",
            seed=17,
        )
        second, unused_second = plan_blank_tiff_splits(
            paths,
            split_names=["train", "val", "test"],
            split_sizes=split_sizes,
            policy="disjoint",
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(unused_first, unused_second)
        self.assertEqual(sum(len(values) for values in first.values()), len(paths))
        assigned_sets = [set(values) for values in first.values()]
        self.assertFalse(assigned_sets[0] & assigned_sets[1])
        self.assertFalse(assigned_sets[0] & assigned_sets[2])
        self.assertFalse(assigned_sets[1] & assigned_sets[2])
        self.assertTrue(all(len(values) > 0 for values in first.values()))

    def test_blank_split_planner_rejects_too_few_disjoint_sources(self):
        with self.assertRaisesRegex(ValueError, "Need at least 3 blank TIFFs"):
            plan_blank_tiff_splits(
                ["/tmp/blank_0.tif", "/tmp/blank_1.tif"],
                split_names=["train", "val", "test"],
                split_sizes={"train": 1, "val": 1, "test": 1},
                policy="disjoint",
                seed=1,
            )

    def test_generator_main_writes_disjoint_blank_config_and_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blank_dir = os.path.join(tmpdir, "blanks")
            out_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(blank_dir)
            for index in range(3):
                image = np.full((8, 8), index + 1, dtype=np.uint16)
                tifffile.imwrite(os.path.join(blank_dir, f"blank_{index}.tif"), image)

            old_argv = sys.argv
            sys.argv = [
                "generate_dataset.py",
                "--output_dir",
                out_dir,
                "--bounds",
                "8",
                "8",
                "--scene_bounds",
                "8",
                "8",
                "--synth_depth",
                "4",
                "--train_size",
                "0",
                "--val_size",
                "0",
                "--test_size",
                "0",
                "--blank_tiff_dir",
                blank_dir,
                "--blank_splits",
                "train",
                "val",
                "test",
                "--blank_split_seed",
                "11",
            ]
            try:
                generate_dataset_main()
            finally:
                sys.argv = old_argv

            with open(os.path.join(out_dir, "generation_config.json"), "r", encoding="utf-8") as handle:
                config = json.load(handle)
            self.assertEqual(config["blank_split_policy"], "disjoint")
            self.assertEqual(config["blank_unused_source_count"], 0)
            allocated_paths = []
            for split in ("train", "val", "test"):
                self.assertEqual(config["blank_allocation"][split]["source_count"], 1)
                self.assertEqual(config["split_counts"][split]["blank_count"], 1)
                self.assertEqual(config["split_counts"][split]["synthetic_count"], 0)
                allocated_paths.extend(config["blank_allocation"][split]["source_paths"])
            self.assertEqual(len(set(allocated_paths)), 3)

    def _write_blank_record(self, path, source_path):
        torch.save(
            {
                "volume": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
                "targets": torch.zeros((6, 1, 8, 8), dtype=torch.float32),
                "target_schema": "structural_v2",
                "metadata": {
                    "sample_type": "blank",
                    "source_path": source_path,
                },
            },
            path,
        )

    def _write_sample_record(self, path, target_channels=6):
        torch.save(
            {
                "volume": torch.zeros((1, 1, 8, 8), dtype=torch.float32),
                "targets": torch.zeros((target_channels, 1, 8, 8), dtype=torch.float32),
                "target_schema": "structural_v2",
            },
            path,
        )

    def test_training_validation_rejects_blank_source_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, "train")
            val_dir = os.path.join(tmpdir, "val")
            os.makedirs(train_dir)
            os.makedirs(val_dir)
            shared_source = os.path.abspath(os.path.join(tmpdir, "shared_blank.tif"))
            self._write_blank_record(os.path.join(train_dir, "blank_000000.pt"), shared_source)
            self._write_blank_record(os.path.join(val_dir, "blank_000000.pt"), shared_source)

            with self.assertRaisesRegex(ValueError, "Blank TIFF source leakage"):
                _validate_dataset(tmpdir, check_samples=1)

            with open(os.path.join(tmpdir, "generation_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"blank_split_policy": "reuse"}, handle)
            _validate_dataset(tmpdir, check_samples=1)

    def test_training_validation_checks_sample_and_blank_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, "train")
            val_dir = os.path.join(tmpdir, "val")
            os.makedirs(train_dir)
            os.makedirs(val_dir)
            self._write_blank_record(os.path.join(train_dir, "blank_000000.pt"), "/tmp/train_blank.tif")
            self._write_sample_record(os.path.join(train_dir, "sample_999999.pt"), target_channels=5)
            self._write_sample_record(os.path.join(val_dir, "sample_000000.pt"))

            with self.assertRaisesRegex(ValueError, "expected 6 target channels"):
                _validate_dataset(tmpdir, check_samples=1)


if __name__ == "__main__":
    unittest.main()
