import numpy as np
from typing import Any, Dict, List, Tuple

import numba as nb
import scipy.ndimage as ndi

from src.core import FiberSegment


@nb.njit(cache=True)
def _render_3d_numba(density_map, starts, ends, thicknesses, base_sigma):
    num_segs = starts.shape[0]
    shape_x, shape_y, shape_z = density_map.shape

    for s in range(num_segs):
        start = starts[s]
        end = ends[s]
        sigma = base_sigma * thicknesses[s]
        cutoff = 4.0 * sigma

        min_x = int(np.floor(min(start[0], end[0]) - cutoff))
        max_x = int(np.ceil(max(start[0], end[0]) + cutoff))
        min_y = int(np.floor(min(start[1], end[1]) - cutoff))
        max_y = int(np.ceil(max(start[1], end[1]) + cutoff))
        min_z = int(np.floor(min(start[2], end[2]) - cutoff))
        max_z = int(np.ceil(max(start[2], end[2]) + cutoff))

        min_x, max_x = max(0, min_x), min(shape_x, max_x)
        min_y, max_y = max(0, min_y), min(shape_y, max_y)
        min_z, max_z = max(0, min_z), min(shape_z, max_z)

        if min_x >= max_x or min_y >= max_y or min_z >= max_z:
            continue

        ab_x = end[0] - start[0]
        ab_y = end[1] - start[1]
        ab_z = end[2] - start[2]
        ab_dot = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z
        ab_dot_max = max(ab_dot, 1e-8)
        sigma2_inv = -1.0 / (2.0 * sigma * sigma)

        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                for k in range(min_z, max_z):
                    ap_x = i - start[0]
                    ap_y = j - start[1]
                    ap_z = k - start[2]

                    t = (ap_x * ab_x + ap_y * ab_y + ap_z * ab_z) / ab_dot_max
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0

                    c_x = start[0] + t * ab_x
                    c_y = start[1] + t * ab_y
                    c_z = start[2] + t * ab_z

                    d2 = (i - c_x) ** 2 + (j - c_y) ** 2 + (k - c_z) ** 2
                    density = np.exp(d2 * sigma2_inv)

                    if density > density_map[i, j, k]:
                        density_map[i, j, k] = density


class NDimRasterizer:
    def __init__(self, grid_shape: Tuple[int, ...], base_sigma: float):
        self.grid_shape = grid_shape
        self.dims = len(grid_shape)
        self.base_sigma = base_sigma

    def _point_to_segment_distance(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        ap = p - a
        t = np.sum(ap * ab, axis=-1) / np.maximum(np.dot(ab, ab), 1e-8)
        t_clamped = np.clip(t, 0.0, 1.0)
        closest_point = a + t_clamped[..., np.newaxis] * ab
        return np.linalg.norm(p - closest_point, axis=-1)

    def render(self, segments: List[FiberSegment]) -> np.ndarray:
        density_map = np.zeros(self.grid_shape, dtype=np.float64)
        half_grid = np.array(self.grid_shape) / 2.0

        valid_starts, valid_ends, valid_thicks = [], [], []
        for seg in segments:
            if np.any(np.abs(seg.end - seg.start) > half_grid):
                continue
            valid_starts.append(seg.start)
            valid_ends.append(seg.end)
            valid_thicks.append(seg.thickness_mult)

        if not valid_starts:
            return density_map

        if self.dims == 3:
            starts_arr = np.array(valid_starts, dtype=np.float64)
            ends_arr = np.array(valid_ends, dtype=np.float64)
            thicks_arr = np.array(valid_thicks, dtype=np.float64)

            _render_3d_numba(density_map, starts_arr, ends_arr, thicks_arr, self.base_sigma)
            return density_map

        for i in range(len(valid_starts)):
            start, end, thick = valid_starts[i], valid_ends[i], valid_thicks[i]
            sigma = self.base_sigma * thick
            cutoff = 4.0 * sigma

            mins = np.floor(np.minimum(start, end) - cutoff).astype(int)
            maxs = np.ceil(np.maximum(start, end) + cutoff).astype(int)

            mins = np.maximum(0, mins)
            maxs = np.minimum(np.array(self.grid_shape), maxs)

            if np.any(mins >= maxs):
                continue

            slices = tuple(slice(mins[d], maxs[d]) for d in range(self.dims))
            ranges = [np.arange(mins[d], maxs[d]) for d in range(self.dims)]
            local_grid_points = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)

            dist = self._point_to_segment_distance(local_grid_points, start, end)
            density = np.exp(-(dist**2) / (2 * sigma**2))

            density_map[slices] = np.maximum(density_map[slices], density)

        return density_map


class EmpiricalRasterizer:
    def __init__(
        self,
        bounds,
        base_sigma=1.0,
        z_anisotropy=3.0,
        noise_level=0.1,
        debris_count=30,
        gap_prob=0.08,
        enable_sted_monomer_cloud: bool = False,
        sted_monomer_mix: Tuple[float, float, float] = (0.70, 0.20, 0.10),
    ):
        self.bounds = bounds
        self.base_sigma = base_sigma
        self.z_anisotropy = z_anisotropy
        self.noise_level = noise_level
        self.debris_count = debris_count
        self.gap_prob = gap_prob
        self.enable_sted_monomer_cloud = bool(enable_sted_monomer_cloud)

        monomer_mix_arr = np.asarray(sted_monomer_mix, dtype=np.float64)
        if monomer_mix_arr.shape != (3,):
            raise ValueError("sted_monomer_mix must contain exactly 3 values for subtle/moderate/strong.")
        if np.any(monomer_mix_arr < 0.0):
            raise ValueError("sted_monomer_mix cannot contain negative values.")
        mix_sum = float(monomer_mix_arr.sum())
        if mix_sum <= 0.0:
            raise ValueError("sted_monomer_mix must sum to a positive value.")
        self.sted_monomer_mix = tuple((monomer_mix_arr / mix_sum).tolist())

    def _sted_optical_section_params(self) -> Tuple[float, float, float]:
        # Keep the optical section broader than the localization slab so
        # defocused fibers remain visible without becoming the localization target.
        depth_of_field = max(1.0, self.base_sigma * self.z_anisotropy * 1.5)
        focus_sigma = max(0.20, self.base_sigma * 0.35)
        defocus_slope = self.base_sigma / depth_of_field
        return depth_of_field, focus_sigma, defocus_slope

    def _sted_axial_fwhm(self, depth_of_field: float) -> float:
        return 2.0 * depth_of_field

    def _sted_defocus_response(self, defocus_distance: float) -> Tuple[float, float]:
        depth_of_field, focus_sigma, defocus_slope = self._sted_optical_section_params()
        abs_defocus = abs(defocus_distance)
        lateral_sigma = np.hypot(focus_sigma, defocus_slope * abs_defocus)
        axial_weight = 1.0 / (1.0 + (abs_defocus / depth_of_field) ** 2)
        return lateral_sigma, axial_weight

    def _sample_monomer_regime(self) -> Tuple[str, float]:
        regime_names = ("subtle", "moderate", "strong")
        regime_index = int(np.random.choice(np.arange(3), p=np.asarray(self.sted_monomer_mix)))
        regime = regime_names[regime_index]

        amplitude_ranges = {
            "subtle": (0.015, 0.04),
            "moderate": (0.04, 0.09),
            "strong": (0.09, 0.16),
        }
        min_amp, max_amp = amplitude_ranges[regime]
        amplitude_fraction = float(np.random.uniform(min_amp, max_amp))
        return regime, amplitude_fraction

    def _synthesize_monomer_concentration_volume(self, regime: str) -> np.ndarray:
        if len(self.bounds) != 3:
            raise ValueError("Monomer cloud synthesis expects 3D bounds.")

        regime_cfg = {
            "subtle": {"xy_range": (4.0, 8.0), "beta_range": (0.35, 0.55)},
            "moderate": {"xy_range": (2.5, 6.0), "beta_range": (0.45, 0.70)},
            "strong": {"xy_range": (1.8, 4.0), "beta_range": (0.60, 0.90)},
        }
        if regime not in regime_cfg:
            raise ValueError(f"Unsupported monomer regime '{regime}'.")

        cfg = regime_cfg[regime]
        base_xy = float(np.random.uniform(*cfg["xy_range"]))

        macro_sigma = (
            base_xy * np.random.uniform(1.20, 1.70),
            base_xy * np.random.uniform(1.20, 1.70),
            base_xy * np.random.uniform(1.80, 2.60),
        )
        meso_sigma = (
            base_xy * np.random.uniform(0.60, 1.00),
            base_xy * np.random.uniform(0.60, 1.00),
            base_xy * np.random.uniform(1.10, 1.80),
        )
        micro_sigma = (
            max(0.70, base_xy * np.random.uniform(0.25, 0.45)),
            max(0.70, base_xy * np.random.uniform(0.25, 0.45)),
            max(0.80, base_xy * np.random.uniform(0.45, 0.90)),
        )

        macro_component = ndi.gaussian_filter(np.random.normal(size=self.bounds), sigma=macro_sigma)
        meso_component = ndi.gaussian_filter(np.random.normal(size=self.bounds), sigma=meso_sigma)
        micro_component = ndi.gaussian_filter(np.random.normal(size=self.bounds), sigma=micro_sigma)

        correlated_field = (0.55 * macro_component) + (0.30 * meso_component) + (0.15 * micro_component)
        beta = float(np.random.uniform(*cfg["beta_range"]))
        concentration = np.exp(beta * correlated_field)

        lo = float(np.percentile(concentration, 1.0))
        hi = float(np.percentile(concentration, 99.0))
        concentration = np.clip((concentration - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

        # Simulate diffusion-broadened monomer occupancy while preserving cloud topology.
        concentration = ndi.gaussian_filter(concentration, sigma=(0.35, 0.35, 0.55))
        concentration /= max(float(concentration.max()), 1e-8)
        return np.clip(concentration, 0.0, 1.0)

    def _apply_sted_monomer_cloud(
        self,
        signal_volume: np.ndarray,
        dynamic_range: Tuple[float, float],
        return_debug: bool = False,
    ):
        signal_volume = np.clip(signal_volume, 0.0, 1.0)
        if not self.enable_sted_monomer_cloud:
            if return_debug:
                disabled_volume = np.zeros_like(signal_volume, dtype=np.float64)
                return signal_volume, {
                    "monomer_volume": disabled_volume,
                    "signal_volume_with_monomer": signal_volume,
                    "monomer_regime": "disabled",
                    "monomer_amplitude": 0.0,
                }
            return signal_volume

        regime, amplitude_fraction = self._sample_monomer_regime()
        concentration = self._synthesize_monomer_concentration_volume(regime)
        dynamic_span = max(float(dynamic_range[1] - dynamic_range[0]), 1e-8)
        monomer_amplitude = amplitude_fraction * dynamic_span
        monomer_volume = concentration * monomer_amplitude
        signal_with_monomer = np.clip(signal_volume + monomer_volume, 0.0, 1.0)

        if return_debug:
            return signal_with_monomer, {
                "monomer_volume": np.clip(monomer_volume, 0.0, 1.0),
                "signal_volume_with_monomer": signal_with_monomer,
                "monomer_regime": regime,
                "monomer_amplitude": float(monomer_amplitude),
            }
        return signal_with_monomer

    def _rasterize_signal_volume(self, list_of_segment_lists, dynamic_range):
        volume = np.zeros(self.bounds, dtype=np.float64)
        base_rasterizer = NDimRasterizer(self.bounds, self.base_sigma)
        min_intensity, max_intensity = dynamic_range

        punctate_sigma = (
            self.base_sigma * 0.45,
            self.base_sigma * 0.45,
            max(0.35, self.base_sigma * 0.25),
        )

        for segments in list_of_segment_lists:
            if not segments:
                continue

            valid_idx = np.random.rand(len(segments)) > self.gap_prob
            valid_segments = [seg for i, seg in enumerate(segments) if valid_idx[i]]
            if not valid_segments:
                continue

            bundle_vol = base_rasterizer.render(valid_segments)

            punctate_mask = np.random.uniform(0.1, 1.0, size=self.bounds)
            punctate_mask = ndi.gaussian_filter(punctate_mask, sigma=punctate_sigma)
            punctate_mask /= max(punctate_mask.max(), 1e-8)

            emission_intensity = np.random.uniform(min_intensity, max_intensity)
            volume += bundle_vol * punctate_mask * emission_intensity

        psf_sigma = (
            self.base_sigma * 0.12,
            self.base_sigma * 0.12,
            self.base_sigma * self.z_anisotropy,
        )
        volume = ndi.gaussian_filter(volume, sigma=psf_sigma)
        return np.clip(volume, 0, 1)

    def _add_3d_haze(self, volume):
        haze_sigma = (
            self.base_sigma * 5.0,
            self.base_sigma * 5.0,
            max(1.0, self.base_sigma * self.z_anisotropy * 1.5),
        )
        haze = ndi.gaussian_filter(volume, sigma=haze_sigma)
        haze_intensity = np.random.uniform(0.12, 0.22)
        return volume + (haze * haze_intensity)

    def _add_3d_debris(self, volume, dynamic_range):
        debris_sigma = (self.base_sigma * 1.8, self.base_sigma * 1.8, self.base_sigma * 1.2)
        min_intensity, max_intensity = dynamic_range

        for _ in range(self.debris_count):
            x, y, z = [np.random.randint(0, b) for b in self.bounds]
            blob = np.zeros(self.bounds, dtype=np.float64)
            blob[x, y, z] = 1.0
            blob = ndi.gaussian_filter(blob, sigma=debris_sigma)
            blob_max = blob.max()
            if blob_max > 0:
                intensity = np.random.uniform(min_intensity, max_intensity) * 0.6
                volume += (blob / blob_max) * intensity

        return volume

    def _apply_3d_striping(self, volume):
        stripe_axis = np.random.choice([0, 1])
        stripes = np.ones(self.bounds[stripe_axis])
        stripes += np.random.normal(0, 0.02, size=self.bounds[stripe_axis])

        if stripe_axis == 0:
            volume *= stripes[:, None, None]
        else:
            volume *= stripes[None, :, None]

        return volume

    def _apply_3d_vignette(self, volume):
        grids = [np.arange(s) - (s / 2.0) for s in self.bounds]
        x_grid, y_grid, z_grid = np.meshgrid(*grids, indexing="ij")
        vignette = np.exp(
            -(x_grid**2 + y_grid**2 + (0.35 * z_grid) ** 2) / (2 * (max(self.bounds) * 0.8) ** 2)
        )
        edge_brightness = np.random.uniform(0.60, 0.85)
        vignette = edge_brightness + (1.0 - edge_brightness) * (vignette / vignette.max())
        return volume * vignette

    def _collapse_volume_to_slice_components(self, volume, slice_center):
        weighted_slice = np.zeros(self.bounds[:2], dtype=np.float64)
        defocus_only_slice = np.zeros(self.bounds[:2], dtype=np.float64)
        weight_sum = 0.0
        defocus_weight_sum = 0.0
        axial_weights = np.zeros(self.bounds[2], dtype=np.float64)
        axial_signal_profile = np.zeros(self.bounds[2], dtype=np.float64)
        lateral_sigmas = np.zeros(self.bounds[2], dtype=np.float64)
        depth_of_field, focus_sigma, _ = self._sted_optical_section_params()
        axial_fwhm = self._sted_axial_fwhm(depth_of_field)
        focus_index = int(np.clip(round(slice_center), 0, self.bounds[2] - 1))

        for z_index in range(self.bounds[2]):
            dz = z_index - slice_center
            blur_sigma, weight = self._sted_defocus_response(dz)

            plane = volume[:, :, z_index]
            if blur_sigma > 1e-6:
                plane = ndi.gaussian_filter(plane, sigma=blur_sigma)

            contribution = plane * weight
            weighted_slice += contribution
            weight_sum += weight

            if z_index != focus_index:
                defocus_only_slice += contribution
                defocus_weight_sum += weight

            axial_weights[z_index] = weight
            axial_signal_profile[z_index] = volume[:, :, z_index].sum()
            lateral_sigmas[z_index] = blur_sigma

        if weight_sum > 0:
            weighted_slice /= weight_sum

        if defocus_weight_sum > 0:
            defocus_only_slice /= defocus_weight_sum

        focus_plane = np.clip(volume[:, :, focus_index], 0, 1)
        blended_slice = np.clip(weighted_slice, 0, 1)

        return {
            "weighted_slice": np.clip(weighted_slice, 0, 1),
            "defocus_only_slice": np.clip(defocus_only_slice, 0, 1),
            "focus_plane": focus_plane,
            "blended_slice": blended_slice,
            "axial_weights": axial_weights,
            "axial_signal_profile": axial_signal_profile,
            "lateral_sigmas": lateral_sigmas,
            "focus_index": focus_index,
            "slice_center": slice_center,
            "depth_of_field": depth_of_field,
            "axial_fwhm": axial_fwhm,
            "focus_sigma": focus_sigma,
        }

    def _collapse_volume_to_slice(self, volume, slice_center):
        return self._collapse_volume_to_slice_components(volume, slice_center)["blended_slice"]

    def _add_2d_debris(self, image, dynamic_range):
        debris_count = max(1, self.debris_count // 4)
        debris_sigma = self.base_sigma * 1.4
        min_intensity, max_intensity = dynamic_range

        for _ in range(debris_count):
            x = np.random.randint(0, self.bounds[0])
            y = np.random.randint(0, self.bounds[1])
            blob = np.zeros(self.bounds[:2], dtype=np.float64)
            blob[x, y] = 1.0
            blob = ndi.gaussian_filter(blob, sigma=debris_sigma)
            blob_max = blob.max()
            if blob_max > 0:
                intensity = np.random.uniform(min_intensity, max_intensity) * 0.4
                image += (blob / blob_max) * intensity

        return image

    def _apply_2d_striping(self, image):
        stripe_axis = np.random.choice([0, 1])
        stripes = np.ones(image.shape[stripe_axis])
        stripes += np.random.normal(0, 0.015, size=image.shape[stripe_axis])

        if stripe_axis == 0:
            image *= stripes[:, None]
        else:
            image *= stripes[None, :]

        return image

    def _apply_2d_vignette(self, image):
        x_grid = np.arange(self.bounds[0]) - (self.bounds[0] / 2.0)
        y_grid = np.arange(self.bounds[1]) - (self.bounds[1] / 2.0)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
        vignette = np.exp(-(xx**2 + yy**2) / (2 * (max(self.bounds[:2]) * 0.85) ** 2))
        edge_brightness = np.random.uniform(0.72, 0.90)
        vignette = edge_brightness + (1.0 - edge_brightness) * (vignette / vignette.max())
        return image * vignette

    def _sted_noise_params(self) -> Dict[str, float]:
        normalized_level = float(np.clip(self.noise_level / 0.08, 0.0, 1.0))
        photon_scale = 1200.0 - (850.0 * normalized_level)
        background_counts = 0.8 + (2.0 * normalized_level)
        read_sigma = 0.0003 + (0.0012 * normalized_level)
        return {
            "noise_level": float(self.noise_level),
            "noise_level_normalized": normalized_level,
            "photon_scale": photon_scale,
            "background_counts": background_counts,
            "read_sigma": read_sigma,
            "dc_offset": 0.0025,
        }

    def _add_sted_noise(self, image):
        signal = np.clip(image, 0.0, 1.0)
        params = self._sted_noise_params()
        counts = np.random.poisson(signal * params["photon_scale"] + params["background_counts"])
        signal_est = (counts - (0.10 * params["background_counts"])) / params["photon_scale"]
        read_noise = np.random.normal(0.0, params["read_sigma"], size=image.shape)
        return np.clip(signal_est + read_noise + params["dc_offset"], 0.0, 1.0)

    def _add_noise(self, image):
        signal = np.clip(image, 0.0, 1.0)
        photon_budget = max(8.0, 36.0 * (1.0 - self.noise_level))
        shot_noise = np.random.poisson(signal * photon_budget) / photon_budget
        read_noise = np.random.normal(0.0, self.noise_level * 0.15, size=image.shape)
        return np.clip(shot_noise + read_noise, 0, 1)

    def render_volume(self, list_of_segment_lists, dynamic_range=(0.2, 1.0), add_haze=True):
        volume = self._rasterize_signal_volume(list_of_segment_lists, dynamic_range)

        if add_haze:
            volume = self._add_3d_haze(volume)

        volume = self._add_3d_debris(volume, dynamic_range)
        volume = self._apply_3d_striping(volume)
        volume = self._apply_3d_vignette(volume)
        volume = self._add_noise(volume)
        return np.clip(volume, 0, 1)

    def render_sted_slice(self, list_of_segment_lists, slice_center, dynamic_range=(0.2, 1.0)):
        volume = self._rasterize_signal_volume(list_of_segment_lists, dynamic_range)
        volume = self._apply_sted_monomer_cloud(volume, dynamic_range, return_debug=False)
        slice_image = self._collapse_volume_to_slice(volume, slice_center)
        slice_image = self._add_2d_debris(slice_image, dynamic_range)
        slice_image = self._apply_2d_striping(slice_image)
        slice_image = self._apply_2d_vignette(slice_image)
        slice_image = self._add_sted_noise(slice_image)
        return np.clip(slice_image, 0, 1)

    def render_sted_slice_debug(
        self,
        list_of_segment_lists,
        slice_center,
        dynamic_range=(0.2, 1.0),
    ) -> Dict[str, Any]:
        fiber_signal_volume = self._rasterize_signal_volume(list_of_segment_lists, dynamic_range)
        signal_volume, monomer_debug = self._apply_sted_monomer_cloud(
            fiber_signal_volume,
            dynamic_range,
            return_debug=True,
        )
        slice_components = self._collapse_volume_to_slice_components(signal_volume, slice_center)
        final_slice = self._add_2d_debris(slice_components["blended_slice"].copy(), dynamic_range)
        final_slice = self._apply_2d_striping(final_slice)
        final_slice = self._apply_2d_vignette(final_slice)
        final_slice = self._add_sted_noise(final_slice)
        noise_params = self._sted_noise_params()

        debug_data = dict(slice_components)
        debug_data["signal_volume"] = signal_volume
        debug_data["fiber_signal_volume"] = fiber_signal_volume
        debug_data["final_slice"] = np.clip(final_slice, 0, 1)
        debug_data["noise_level"] = noise_params["noise_level"]
        debug_data["noise_level_normalized"] = noise_params["noise_level_normalized"]
        debug_data["noise_photon_scale"] = noise_params["photon_scale"]
        debug_data["noise_background_counts"] = noise_params["background_counts"]
        debug_data["noise_read_sigma"] = noise_params["read_sigma"]
        debug_data.update(monomer_debug)
        return debug_data

    def render(self, list_of_segment_lists, dynamic_range=(0.2, 1.0)):
        return self.render_volume(list_of_segment_lists, dynamic_range=dynamic_range, add_haze=True)
