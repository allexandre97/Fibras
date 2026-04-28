import os
import tempfile
import unittest

import numpy as np
import tifffile
import torch

from generate_dataset import save_blank_tiff_crops


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
            self.assertEqual(sample["metadata"]["source_path"], os.path.abspath(image_path))


if __name__ == "__main__":
    unittest.main()
