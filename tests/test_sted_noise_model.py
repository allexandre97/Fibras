import unittest
from unittest.mock import patch

import numpy as np

from generate_dataset import (
    _build_2d_sample,
    _build_2d_focus_and_visibility_targets,
    _prepare_2d_sted_scene,
    _project_segments_to_label_slab,
    _to_2d_tensors,
    build_sted_debug_sample,
)
from src.rasterization import EmpiricalRasterizer
from src.targets import TargetFieldGenerator, WeightedVisibilityTargetGenerator


class StedNoiseModelTests(unittest.TestCase):
    def test_sted_noise_monotonicity_increases_background_variance(self):
        image = np.zeros((64, 64), dtype=np.float64)
        low_noise = EmpiricalRasterizer(bounds=(64, 64, 8), noise_level=0.005, debris_count=0, gap_prob=0.0)
        high_noise = EmpiricalRasterizer(bounds=(64, 64, 8), noise_level=0.045, debris_count=0, gap_prob=0.0)

        low_var = []
        high_var = []
        for seed in range(24):
            np.random.seed(seed)
            low_var.append(low_noise._add_sted_noise(image).var())
            np.random.seed(seed)
            high_var.append(high_noise._add_sted_noise(image).var())

        self.assertGreater(np.mean(high_var), np.mean(low_var))

    def test_floor_preservation_keeps_zero_clip_low(self):
        zero_fractions = []
        for seed in range(20):
            debug_data = build_sted_debug_sample((64, 64), synth_depth=16, seed=seed)
            zero_fraction = np.mean(debug_data["final_slice"] <= 1e-6)
            zero_fractions.append(zero_fraction)

        self.assertLess(np.mean(zero_fractions), 0.08)
        self.assertLess(np.quantile(zero_fractions, 0.9), 0.08)

    def test_scope_guard_uses_separate_noise_paths(self):
        rasterizer = EmpiricalRasterizer(bounds=(32, 32, 8), noise_level=0.03, debris_count=0, gap_prob=0.0)

        with patch.object(rasterizer, "_add_noise", wraps=rasterizer._add_noise) as legacy_noise, patch.object(
            rasterizer, "_add_sted_noise", wraps=rasterizer._add_sted_noise
        ) as sted_noise, patch.object(
            rasterizer, "_apply_sted_monomer_cloud", wraps=rasterizer._apply_sted_monomer_cloud
        ) as monomer_path:
            rasterizer.render_volume([], dynamic_range=(0.2, 0.8), add_haze=False)
            self.assertTrue(legacy_noise.called)
            self.assertFalse(sted_noise.called)
            self.assertFalse(monomer_path.called)

        with patch.object(rasterizer, "_add_noise", wraps=rasterizer._add_noise) as legacy_noise, patch.object(
            rasterizer, "_add_sted_noise", wraps=rasterizer._add_sted_noise
        ) as sted_noise, patch.object(
            rasterizer, "_apply_sted_monomer_cloud", wraps=rasterizer._apply_sted_monomer_cloud
        ) as monomer_path:
            rasterizer.render_sted_slice([], slice_center=3.0, dynamic_range=(0.2, 0.8))
            self.assertFalse(legacy_noise.called)
            self.assertTrue(sted_noise.called)
            self.assertTrue(monomer_path.called)

    def test_2d_sted_noise_sampling_range_is_tightened(self):
        for seed in range(40):
            np.random.seed(seed)
            scene = _prepare_2d_sted_scene((64, 64, 16), None)
            noise_level = scene["rasterizer"].noise_level
            self.assertGreaterEqual(noise_level, 0.005)
            self.assertLessEqual(noise_level, 0.045)

    def test_default_label_slab_scale_widens_thickness(self):
        np.random.seed(4)
        scene = _prepare_2d_sted_scene((64, 64, 16), None)
        depth_of_field, _, _ = scene["rasterizer"]._sted_optical_section_params()
        self.assertAlmostEqual(scene["label_slab_scale"], 1.3, places=6)
        self.assertAlmostEqual(scene["label_slab_thickness"], depth_of_field * 1.3, places=6)

    def test_explicit_label_slab_thickness_overrides_scale(self):
        explicit_thickness = 2.75
        np.random.seed(8)
        scene = _prepare_2d_sted_scene((64, 64, 16), explicit_thickness, label_slab_scale=1.8)
        self.assertAlmostEqual(scene["label_slab_thickness"], explicit_thickness, places=6)
        self.assertAlmostEqual(scene["label_slab_scale"], 1.8, places=6)

    def test_focus_projection_is_monotonic_with_larger_slab_scale(self):
        np.random.seed(9)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, label_slab_scale=1.0)
        depth_of_field, _, _ = scene["rasterizer"]._sted_optical_section_params()
        core_segments = scene["core_segments"]
        slice_center = scene["slice_center"]
        focus_1x = _project_segments_to_label_slab(core_segments, slice_center, depth_of_field * 1.0)
        focus_13x = _project_segments_to_label_slab(core_segments, slice_center, depth_of_field * 1.3)
        self.assertGreaterEqual(len(focus_13x), len(focus_1x))

    def test_visibility_target_is_independent_of_label_slab_scale(self):
        np.random.seed(12)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, label_slab_scale=1.0)
        rasterizer = scene["rasterizer"]
        core_segments = scene["core_segments"]
        slice_center = scene["slice_center"]
        target_gen = TargetFieldGenerator((64, 64), max_distance=5.0)
        visibility_gen = WeightedVisibilityTargetGenerator((64, 64), base_sigma=rasterizer.base_sigma)

        targets_1x = _build_2d_focus_and_visibility_targets(
            core_segments,
            slice_center,
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
        )
        targets_18x = _build_2d_focus_and_visibility_targets(
            core_segments,
            slice_center,
            localization_slab_thickness=scene["label_slab_thickness"] * 1.8,
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
        )

        self.assertEqual(len(targets_1x["visibility_segments"]), len(targets_18x["visibility_segments"]))
        self.assertTrue(np.allclose(targets_1x["visibility_weights"], targets_18x["visibility_weights"]))
        self.assertTrue(np.allclose(targets_1x["visibility_target"], targets_18x["visibility_target"]))

    def test_visibility_target_is_independent_of_annotation_weight_floor(self):
        np.random.seed(18)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, soft_skeleton_alpha=0.35)
        rasterizer = scene["rasterizer"]
        core_segments = scene["core_segments"]
        slice_center = scene["slice_center"]
        target_gen = TargetFieldGenerator((64, 64), max_distance=5.0)
        visibility_gen = WeightedVisibilityTargetGenerator((64, 64), base_sigma=rasterizer.base_sigma)

        targets_tight = _build_2d_focus_and_visibility_targets(
            core_segments,
            slice_center,
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            annotation_weight_floor=0.50,
            soft_skeleton_alpha=0.35,
        )
        targets_broad = _build_2d_focus_and_visibility_targets(
            core_segments,
            slice_center,
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            annotation_weight_floor=0.25,
            soft_skeleton_alpha=0.35,
        )

        self.assertEqual(len(targets_tight["visibility_segments"]), len(targets_broad["visibility_segments"]))
        self.assertTrue(np.allclose(targets_tight["visibility_weights"], targets_broad["visibility_weights"]))
        self.assertTrue(np.allclose(targets_tight["visibility_target"], targets_broad["visibility_target"]))

    def test_soft_skeleton_alpha_zero_matches_hard_focus_targets(self):
        np.random.seed(21)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, soft_skeleton_alpha=0.0)
        rasterizer = scene["rasterizer"]
        target_gen = TargetFieldGenerator((64, 64), max_distance=5.0)
        visibility_gen = WeightedVisibilityTargetGenerator((64, 64), base_sigma=rasterizer.base_sigma)

        targets = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            soft_skeleton_alpha=0.0,
        )
        edt_focus, vector_focus = target_gen.generate(targets["focus_segments"])

        self.assertTrue(np.allclose(targets["edt_target"], edt_focus))
        self.assertTrue(np.allclose(targets["vector_target"], vector_focus))

    def test_soft_skeleton_alpha_blend_expands_or_preserves_edt(self):
        np.random.seed(22)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, soft_skeleton_alpha=0.35)
        rasterizer = scene["rasterizer"]
        target_gen = TargetFieldGenerator((64, 64), max_distance=5.0)
        visibility_gen = WeightedVisibilityTargetGenerator((64, 64), base_sigma=rasterizer.base_sigma)

        targets_hard = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            soft_skeleton_alpha=0.0,
        )
        targets_soft = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            soft_skeleton_alpha=0.35,
        )
        self.assertTrue(np.all(targets_soft["edt_target"] >= targets_hard["edt_target"]))

    def test_soft_skeleton_alpha_monotonicity(self):
        np.random.seed(23)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, soft_skeleton_alpha=0.0)
        rasterizer = scene["rasterizer"]
        target_gen = TargetFieldGenerator((64, 64), max_distance=5.0)
        visibility_gen = WeightedVisibilityTargetGenerator((64, 64), base_sigma=rasterizer.base_sigma)

        targets_lo = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            soft_skeleton_alpha=0.2,
        )
        targets_hi = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            soft_skeleton_alpha=0.7,
        )
        area_lo = np.mean(targets_lo["edt_target"] > 0.15)
        area_hi = np.mean(targets_hi["edt_target"] > 0.15)
        self.assertGreaterEqual(area_hi, area_lo)

    def test_annotation_weight_floor_monotonicity(self):
        np.random.seed(25)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, soft_skeleton_alpha=0.35)
        rasterizer = scene["rasterizer"]
        target_gen = TargetFieldGenerator((64, 64), max_distance=5.0)
        visibility_gen = WeightedVisibilityTargetGenerator((64, 64), base_sigma=rasterizer.base_sigma)

        targets_tight = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            annotation_weight_floor=0.50,
            soft_skeleton_alpha=0.35,
        )
        targets_broad = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            annotation_weight_floor=0.25,
            soft_skeleton_alpha=0.35,
        )

        area_tight = np.mean(targets_tight["edt_target"] > 0.15)
        area_broad = np.mean(targets_broad["edt_target"] > 0.15)
        self.assertGreaterEqual(area_broad, area_tight)

    def test_visibility_target_is_independent_of_soft_skeleton_alpha(self):
        np.random.seed(24)
        scene = _prepare_2d_sted_scene((64, 64, 16), None, soft_skeleton_alpha=0.0)
        rasterizer = scene["rasterizer"]
        target_gen = TargetFieldGenerator((64, 64), max_distance=5.0)
        visibility_gen = WeightedVisibilityTargetGenerator((64, 64), base_sigma=rasterizer.base_sigma)

        targets_lo = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            soft_skeleton_alpha=0.0,
        )
        targets_hi = _build_2d_focus_and_visibility_targets(
            scene["core_segments"],
            scene["slice_center"],
            localization_slab_thickness=scene["label_slab_thickness"],
            rasterizer=rasterizer,
            target_gen=target_gen,
            visibility_target_gen=visibility_gen,
            soft_skeleton_alpha=0.7,
        )

        self.assertEqual(len(targets_lo["visibility_segments"]), len(targets_hi["visibility_segments"]))
        self.assertTrue(np.allclose(targets_lo["visibility_weights"], targets_hi["visibility_weights"]))
        self.assertTrue(np.allclose(targets_lo["visibility_target"], targets_hi["visibility_target"]))

    def test_negative_soft_skeleton_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            _prepare_2d_sted_scene((64, 64, 16), None, soft_skeleton_alpha=-0.1)

    def test_invalid_annotation_weight_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            _prepare_2d_sted_scene((64, 64, 16), None, annotation_weight_floor=0.0)
        with self.assertRaises(ValueError):
            _prepare_2d_sted_scene((64, 64, 16), None, annotation_weight_floor=1.1)

    def test_sted_debug_includes_noise_metadata(self):
        debug_data = build_sted_debug_sample((64, 64), synth_depth=16, seed=0)
        self.assertIn("noise_level", debug_data)
        self.assertIn("noise_level_normalized", debug_data)
        self.assertGreaterEqual(debug_data["noise_level_normalized"], 0.0)
        self.assertLessEqual(debug_data["noise_level_normalized"], 1.0)

    def test_monomer_cloud_increases_background_mean_in_sted_path(self):
        enabled = EmpiricalRasterizer(
            bounds=(64, 64, 8),
            noise_level=0.02,
            debris_count=0,
            gap_prob=0.0,
            enable_sted_monomer_cloud=True,
        )
        disabled = EmpiricalRasterizer(
            bounds=(64, 64, 8),
            noise_level=0.02,
            debris_count=0,
            gap_prob=0.0,
            enable_sted_monomer_cloud=False,
        )

        mean_deltas = []
        for seed in range(24):
            np.random.seed(seed)
            with_monomers = enabled.render_sted_slice([], slice_center=3.5, dynamic_range=(0.2, 0.8))
            np.random.seed(seed)
            without_monomers = disabled.render_sted_slice([], slice_center=3.5, dynamic_range=(0.2, 0.8))
            mean_deltas.append(float(with_monomers.mean() - without_monomers.mean()))

        self.assertGreater(np.mean(mean_deltas), 0.002)

    def test_monomer_generation_is_independent_of_fiber_signal(self):
        rasterizer = EmpiricalRasterizer(
            bounds=(32, 32, 8),
            noise_level=0.03,
            debris_count=0,
            gap_prob=0.0,
            enable_sted_monomer_cloud=True,
            sted_monomer_mix=(0.70, 0.20, 0.10),
        )
        signal_zeros = np.zeros((32, 32, 8), dtype=np.float64)
        signal_nonzero = np.full((32, 32, 8), 0.25, dtype=np.float64)

        np.random.seed(11)
        _, debug_zero = rasterizer._apply_sted_monomer_cloud(signal_zeros, dynamic_range=(0.2, 0.8), return_debug=True)
        np.random.seed(11)
        _, debug_nonzero = rasterizer._apply_sted_monomer_cloud(signal_nonzero, dynamic_range=(0.2, 0.8), return_debug=True)

        self.assertEqual(debug_zero["monomer_regime"], debug_nonzero["monomer_regime"])
        self.assertAlmostEqual(debug_zero["monomer_amplitude"], debug_nonzero["monomer_amplitude"], places=10)
        self.assertTrue(np.allclose(debug_zero["monomer_volume"], debug_nonzero["monomer_volume"]))

    def test_sted_debug_includes_monomer_metadata(self):
        debug_data = build_sted_debug_sample((64, 64), synth_depth=16, seed=3)
        self.assertIn("fiber_signal_volume", debug_data)
        self.assertIn("signal_volume_with_monomer", debug_data)
        self.assertIn("monomer_volume", debug_data)
        self.assertIn("monomer_regime", debug_data)
        self.assertIn("monomer_amplitude", debug_data)
        self.assertIn(debug_data["monomer_regime"], {"disabled", "subtle", "moderate", "strong"})
        if debug_data["monomer_regime"] == "disabled":
            self.assertAlmostEqual(debug_data["monomer_amplitude"], 0.0, places=10)
        else:
            self.assertGreater(debug_data["monomer_amplitude"], 0.0)
        self.assertEqual(debug_data["signal_volume"].shape, debug_data["monomer_volume"].shape)
        self.assertTrue(np.all(debug_data["monomer_volume"] >= 0.0))
        self.assertTrue(np.all(debug_data["monomer_volume"] <= 1.0))

    def test_sted_debug_includes_label_slab_metadata(self):
        debug_data = build_sted_debug_sample((64, 64), synth_depth=16, label_slab_scale=1.3, seed=5)
        self.assertIn("label_slab_scale", debug_data)
        self.assertIn("label_slab_thickness", debug_data)
        self.assertGreater(debug_data["label_slab_scale"], 0.0)
        self.assertGreater(debug_data["label_slab_thickness"], 0.0)

    def test_sted_debug_includes_annotation_metadata(self):
        debug_data = build_sted_debug_sample((64, 64), synth_depth=16, annotation_weight_floor=0.25, seed=7)
        self.assertIn("annotation_weight_floor", debug_data)
        self.assertIn("annotation_segments", debug_data)
        self.assertIn("annotation_segment_count", debug_data)
        self.assertIn("edt_focus", debug_data)
        self.assertIn("edt_soft", debug_data)
        self.assertGreater(debug_data["annotation_weight_floor"], 0.0)
        self.assertGreaterEqual(debug_data["annotation_segment_count"], debug_data["projected_segment_count"])

    def test_sted_debug_includes_soft_skeleton_alpha(self):
        debug_data = build_sted_debug_sample((64, 64), synth_depth=16, soft_skeleton_alpha=0.35, seed=6)
        self.assertIn("soft_skeleton_alpha", debug_data)
        self.assertGreaterEqual(debug_data["soft_skeleton_alpha"], 0.0)

    def test_no_haze_regime_frequency_is_within_expected_band(self):
        disabled_count = 0
        sample_count = 120
        for seed in range(sample_count):
            np.random.seed(seed)
            scene = _prepare_2d_sted_scene((64, 64, 16), None)
            if scene["haze_regime"] == "none":
                disabled_count += 1

        disabled_fraction = disabled_count / sample_count
        self.assertGreaterEqual(disabled_fraction, 0.20)
        self.assertLessEqual(disabled_fraction, 0.40)

    def test_2d_walk_and_fiber_constraints_match_realism_ranges(self):
        max_xy = 64.0
        lower_step = max(0.8, max_xy * 0.018)
        upper_step = max(0.8, max_xy * 0.030)

        for seed in range(40):
            np.random.seed(seed)
            scene = _prepare_2d_sted_scene((64, 64, 16), None)
            self.assertGreaterEqual(scene["requested_fiber_count"], 10)
            self.assertLessEqual(scene["requested_fiber_count"], 24)
            self.assertTrue(all(size in {1, 2, 3} for size in scene["bundle_sizes"]))

            for walk_cfg in scene["walk_parameter_samples"]:
                self.assertGreaterEqual(walk_cfg["num_steps"], 4)
                self.assertLessEqual(walk_cfg["num_steps"], 14)
                self.assertGreaterEqual(walk_cfg["max_turn_degrees"], 2.0)
                self.assertLessEqual(walk_cfg["max_turn_degrees"], 12.0)
                self.assertGreaterEqual(walk_cfg["step_length"], lower_step)
                self.assertLessEqual(walk_cfg["step_length"], upper_step)

    def test_deterministic_2d_smoke_preserves_target_channels(self):
        seen_haze_regimes = set()

        for seed in range(18):
            np.random.seed(seed)
            scene = _prepare_2d_sted_scene((40, 40, 14), None)
            seen_haze_regimes.add(scene["haze_regime"])
            self.assertGreaterEqual(scene["requested_fiber_count"], 10)
            self.assertLessEqual(scene["requested_fiber_count"], 24)

            image, edt_target, vector_target, visibility_target = _build_2d_sample((40, 40, 14), None)
            _, targets_tensor = _to_2d_tensors(image, edt_target, vector_target, visibility_target)
            self.assertEqual(targets_tensor.shape[0], 4)

        self.assertIn("none", seen_haze_regimes)


if __name__ == "__main__":
    unittest.main()
