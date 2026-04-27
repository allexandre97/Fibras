import unittest

from batch_infer_real import _select_image_paths


class BatchInferSelectionTests(unittest.TestCase):
    def setUp(self):
        self.paths = [
            "/tmp/PN148_3R_AD_DIV03 (Series 0) [1].tif",
            "/tmp/PN148_3R_AD_DIV03 (Series 1) [1].tif",
            "/tmp/PN148_3R_PID_DIV05 (Series 0) [1].tif",
            "/tmp/PN148_3R_PID_DIV05 (Series 1) [1].tif",
            "/tmp/PN148_4R_PSP_DIV10 (Series 0) [1].tif",
            "/tmp/PN148_4R_PSP_DIV10 (Series 1) [1].tif",
        ]

    def test_first_sampling_preserves_input_order(self):
        selected = _select_image_paths(self.paths, max_images=3, sample_strategy="first")
        self.assertEqual(selected, self.paths[:3])

    def test_random_sampling_is_reproducible(self):
        first = _select_image_paths(self.paths, max_images=4, sample_strategy="random", sample_seed=7)
        second = _select_image_paths(self.paths, max_images=4, sample_strategy="random", sample_seed=7)
        third = _select_image_paths(self.paths, max_images=4, sample_strategy="random", sample_seed=13)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_stratified_sampling_covers_each_condition_div_group_before_repeating(self):
        selected = _select_image_paths(
            self.paths,
            max_images=3,
            sample_strategy="stratified",
            sample_group="condition_div",
            sample_seed=0,
        )
        self.assertEqual(len(selected), 3)
        self.assertTrue(any("_AD_DIV03" in path for path in selected))
        self.assertTrue(any("_PID_DIV05" in path for path in selected))
        self.assertTrue(any("_PSP_DIV10" in path for path in selected))


if __name__ == "__main__":
    unittest.main()
