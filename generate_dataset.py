import argparse
import concurrent.futures
import os
from functools import partial
from typing import Optional

import numpy as np

from src.core import FiberSegment, ReflectiveBoundary
from src.rasterization import EmpiricalRasterizer
from src.synthesis import CompositeGenerator, RandomWalkGenerator, SpaceColonizationGenerator
from src.targets import TargetFieldGenerator, WeightedVisibilityTargetGenerator


PHENOTYPES = ["Highly Branched", "Directional", "Random Tangle", "Cloudy Bundle", "Heterogeneous Mixed"]
SHORT_FIBER_STEPS = (6, 20)
SHORT_TURN_DEGREES = (5.0, 20.0)
TARGET_MAX_DISTANCE = 5.0
DEFAULT_LABEL_SLAB_SCALE = 1.3
DEFAULT_SOFT_SKELETON_ALPHA = 0.35
DEFAULT_ANNOTATION_WEIGHT_FLOOR = 0.25
DEFAULT_VISIBILITY_WEIGHT_FLOOR = 0.03


def apply_optical_jitter(core_segments, bundle_size=3, jitter_amount=1.5, lock_z=False):
    """Spawns jittered strands for rasterization without altering the target geometry."""
    optical_segments = []
    for seg in core_segments:
        optical_segments.append(seg)
        for _ in range(bundle_size - 1):
            jitter_start = np.random.normal(0, jitter_amount, 3)
            jitter_end = np.random.normal(0, jitter_amount, 3)
            if lock_z:
                jitter_start[2] = 0.0
                jitter_end[2] = 0.0

            optical_segments.append(
                FiberSegment(
                    start=seg.start + jitter_start,
                    end=seg.end + jitter_end,
                    thickness_mult=seg.thickness_mult * np.random.uniform(0.45, 0.90),
                )
            )
    return optical_segments


def _generate_phenotype(bounds: tuple):
    x_size, y_size, z_size = bounds
    is_flat_volume = z_size == 1
    max_dim = max(bounds)

    phenotype = np.random.choice(PHENOTYPES)
    scale = np.random.uniform(0.1, 1.0)

    dynamic_step = max(1.0, max_dim * 0.012 * scale)
    dynamic_kill = max(2.0, max_dim * 0.015 * scale)

    root_x = np.random.uniform(x_size * 0.1, x_size * 0.9)
    root_y = np.random.uniform(y_size * 0.1, y_size * 0.9)
    root_z = 0.5 if is_flat_volume else np.random.uniform(z_size * 0.1, z_size * 0.9)
    root = (root_x, root_y, root_z)

    if phenotype == "Highly Branched":
        num_attractors = int((x_size * y_size * max(1, z_size)) * 0.0015 * scale)
        attractors = np.zeros((num_attractors, 3))
        attractors[:, 0] = np.random.uniform(0, x_size, num_attractors)
        attractors[:, 1] = np.random.uniform(0, y_size, num_attractors)
        attractors[:, 2] = 0.5 if is_flat_volume else np.random.uniform(0, z_size, num_attractors)

        gen = SpaceColonizationGenerator(
            attractors,
            root_pos=root,
            step_length=dynamic_step,
            attraction_distance=max(20.0, max_dim * 0.25 * scale),
            kill_distance=dynamic_kill,
            bounds=bounds,
            max_iterations=int(max_dim * 75 * scale),
            thickness_decay=np.random.uniform(0.95, 0.99),
        )
        return gen.generate()

    if phenotype == "Directional":
        num_attractors = int((x_size * y_size * max(1, z_size)) * 0.0005 * scale)
        cyl_radius = max(10, max_dim * 0.12 * scale)
        attractors = np.zeros((num_attractors, 3))
        attractors[:, 0] = np.random.uniform(root_x - cyl_radius, root_x + cyl_radius, num_attractors)
        attractors[:, 1] = np.random.uniform(root_y - cyl_radius, root_y + cyl_radius, num_attractors)
        attractors[:, 2] = 0.5 if is_flat_volume else np.random.uniform(root_z, z_size, num_attractors)

        gen = SpaceColonizationGenerator(
            attractors,
            root_pos=root,
            step_length=dynamic_step,
            attraction_distance=max(30.0, max_dim * 0.35 * scale),
            kill_distance=dynamic_kill,
            bounds=bounds,
            max_iterations=int(max_dim * 75 * scale),
            thickness_decay=np.random.uniform(0.95, 0.99),
        )
        return gen.generate()

    if phenotype == "Random Tangle":
        steps = int(max_dim * np.random.uniform(0.5, 4.0) * scale)
        angle = np.random.uniform(0.1, 2.5)

        boundary = ReflectiveBoundary(bounds)
        gen = RandomWalkGenerator(
            start_pos=root,
            num_steps=steps,
            step_length=dynamic_step,
            max_turn_angle=angle,
            boundary=boundary,
        )
        return gen.generate()

    if phenotype == "Cloudy Bundle":
        segments = []
        num_fragments = int(np.random.uniform(50, 250) * scale)
        macro_dir = np.random.normal(size=3)
        if is_flat_volume:
            macro_dir[2] = 0.0
        macro_dir /= np.linalg.norm(macro_dir)

        bundle_radius = max_dim * np.random.uniform(0.05, 0.15)

        for _ in range(num_fragments):
            frag_root = np.array(root) + np.random.normal(0, bundle_radius, 3)
            if is_flat_volume:
                frag_root[2] = 0.5

            frag_steps = np.random.randint(3, 15)
            current_pos = frag_root.copy()

            for _ in range(frag_steps):
                noise_dir = np.random.normal(size=3)
                if is_flat_volume:
                    noise_dir[2] = 0.0
                noise_dir /= np.linalg.norm(noise_dir)

                step_dir = macro_dir * 0.7 + noise_dir * 0.3
                step_dir /= np.linalg.norm(step_dir)

                next_pos = current_pos + step_dir * dynamic_step
                segments.append(
                    FiberSegment(current_pos.copy(), next_pos.copy(), thickness_mult=np.random.uniform(0.2, 0.6))
                )
                current_pos = next_pos

        return segments

    proportions = np.random.dirichlet(np.ones(3))
    generators = []

    scale_branched = scale * proportions[0]
    if scale_branched > 0.05:
        num_attr = int((x_size * y_size * max(1, z_size)) * 0.0015 * scale_branched)
        attr_b = np.zeros((num_attr, 3))
        attr_b[:, 0] = np.random.uniform(0, x_size, num_attr)
        attr_b[:, 1] = np.random.uniform(0, y_size, num_attr)
        attr_b[:, 2] = 0.5 if is_flat_volume else np.random.uniform(0, z_size, num_attr)

        generators.append(
            SpaceColonizationGenerator(
                attr_b,
                root_pos=root,
                step_length=dynamic_step,
                attraction_distance=max(20.0, max_dim * 0.25 * scale_branched),
                kill_distance=dynamic_kill,
                bounds=bounds,
                max_iterations=int(max_dim * 75 * scale_branched),
                thickness_decay=np.random.uniform(0.95, 0.99),
            )
        )

    scale_directional = scale * proportions[1]
    if scale_directional > 0.05:
        num_attr = int((x_size * y_size * max(1, z_size)) * 0.0005 * scale_directional)
        cyl_radius = max(10, max_dim * 0.12 * scale_directional)
        attr_d = np.zeros((num_attr, 3))
        attr_d[:, 0] = np.random.uniform(root_x - cyl_radius, root_x + cyl_radius, num_attr)
        attr_d[:, 1] = np.random.uniform(root_y - cyl_radius, root_y + cyl_radius, num_attr)
        attr_d[:, 2] = 0.5 if is_flat_volume else np.random.uniform(root_z, z_size, num_attr)

        generators.append(
            SpaceColonizationGenerator(
                attr_d,
                root_pos=root,
                step_length=dynamic_step,
                attraction_distance=max(30.0, max_dim * 0.35 * scale_directional),
                kill_distance=dynamic_kill,
                bounds=bounds,
                max_iterations=int(max_dim * 75 * scale_directional),
                thickness_decay=np.random.uniform(0.95, 0.99),
            )
        )

    scale_tangle = scale * proportions[2]
    if scale_tangle > 0.05:
        steps = int(max_dim * np.random.uniform(0.5, 4.0) * scale_tangle)
        angle = np.random.uniform(0.1, 2.5)
        boundary = ReflectiveBoundary(bounds)

        generators.append(
            RandomWalkGenerator(
                start_pos=root,
                num_steps=steps,
                step_length=dynamic_step,
                max_turn_angle=angle,
                boundary=boundary,
            )
        )

    if not generators:
        return []

    composite = CompositeGenerator(generators)
    return composite.generate()


def _generate_constrained_random_walk(bounds: tuple):
    x_size, y_size, z_size = bounds
    step_length = max(1.0, max(x_size, y_size) * 0.025)
    num_steps = np.random.randint(SHORT_FIBER_STEPS[0], SHORT_FIBER_STEPS[1] + 1)
    max_turn_angle = np.deg2rad(np.random.uniform(SHORT_TURN_DEGREES[0], SHORT_TURN_DEGREES[1]))

    start_pos = (
        np.random.uniform(x_size * 0.1, x_size * 0.9),
        np.random.uniform(y_size * 0.1, y_size * 0.9),
        np.random.uniform(z_size * 0.2, z_size * 0.8),
    )
    initial_direction = np.random.normal(size=3) * np.array([1.0, 1.0, 0.35], dtype=float)
    orthogonal_scale = np.array([1.0, 1.0, 0.45], dtype=float)

    generator = RandomWalkGenerator(
        start_pos=start_pos,
        num_steps=num_steps,
        step_length=step_length,
        max_turn_angle=max_turn_angle,
        boundary=ReflectiveBoundary(bounds),
        initial_direction=initial_direction,
        orthogonal_scale=orthogonal_scale,
    )
    return generator.generate()


def _clip_segment_to_z_slab(segment: FiberSegment, lower_z: float, upper_z: float):
    start = segment.start
    end = segment.end
    delta = end - start
    dz = delta[2]

    if abs(dz) < 1e-8:
        if lower_z <= start[2] <= upper_z:
            return start.copy(), end.copy()
        return None

    t0 = (lower_z - start[2]) / dz
    t1 = (upper_z - start[2]) / dz
    t_enter = max(0.0, min(t0, t1))
    t_exit = min(1.0, max(t0, t1))

    if t_enter > t_exit:
        return None

    clipped_start = start + (delta * t_enter)
    clipped_end = start + (delta * t_exit)
    return clipped_start, clipped_end


def _resolve_axial_weight_band_half_width(depth_of_field: float, weight_floor: float) -> float:
    if weight_floor <= 0.0 or weight_floor > 1.0:
        raise ValueError("weight_floor must be in the interval (0, 1].")
    return float(depth_of_field * np.sqrt((1.0 / float(weight_floor)) - 1.0))


def _project_segments_to_z_band(core_segments, lower_z: float, upper_z: float):
    projected_segments = []
    clipped_segments = []

    for segment in core_segments:
        clipped = _clip_segment_to_z_slab(segment, lower_z, upper_z)
        if clipped is None:
            continue

        clipped_start, clipped_end = clipped
        clipped_segment = FiberSegment(
            start=np.asarray(clipped_start, dtype=np.float64),
            end=np.asarray(clipped_end, dtype=np.float64),
            thickness_mult=segment.thickness_mult,
        )
        projected = _project_segment_to_xy(clipped_segment)
        if projected is None:
            continue

        projected_segments.append(projected)
        clipped_segments.append(clipped_segment)

    return projected_segments, clipped_segments


def _project_segments_to_label_slab(core_segments, slice_center, slab_thickness):
    lower_z = slice_center - (slab_thickness / 2.0)
    upper_z = slice_center + (slab_thickness / 2.0)
    projected_segments, _ = _project_segments_to_z_band(core_segments, lower_z, upper_z)
    return projected_segments


def _project_segment_to_xy(segment: FiberSegment):
    start_xy = np.array([segment.start[0], segment.start[1]], dtype=np.float64)
    end_xy = np.array([segment.end[0], segment.end[1]], dtype=np.float64)
    if np.linalg.norm(end_xy - start_xy) < 1e-6:
        return None
    return FiberSegment(start=start_xy, end=end_xy, thickness_mult=segment.thickness_mult)


def _mean_axial_weight_over_segment(
    clipped_segment: FiberSegment,
    slice_center: float,
    depth_of_field: float,
) -> float:
    z0 = float(clipped_segment.start[2])
    z1 = float(clipped_segment.end[2])
    dz = z1 - z0

    if abs(dz) < 1e-8:
        relative_z = (z0 - slice_center) / max(depth_of_field, 1e-8)
        return float(np.clip(1.0 / (1.0 + (relative_z ** 2)), 0.0, 1.0))

    rel0 = (z0 - slice_center) / max(depth_of_field, 1e-8)
    rel1 = (z1 - slice_center) / max(depth_of_field, 1e-8)
    integral = depth_of_field * (np.arctan(rel1) - np.arctan(rel0))
    mean_weight = abs(integral / dz)
    return float(np.clip(mean_weight, 0.0, 1.0))


def _project_segments_to_annotation(
    core_segments,
    slice_center,
    rasterizer,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
):
    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    half_width = _resolve_axial_weight_band_half_width(depth_of_field, annotation_weight_floor)
    lower_z = slice_center - half_width
    upper_z = slice_center + half_width
    projected_segments, _ = _project_segments_to_z_band(core_segments, lower_z, upper_z)
    return projected_segments


def _project_segments_to_visibility(
    core_segments,
    slice_center,
    rasterizer,
    min_weight: float = DEFAULT_VISIBILITY_WEIGHT_FLOOR,
):
    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    half_width = _resolve_axial_weight_band_half_width(depth_of_field, min_weight)
    lower_z = slice_center - half_width
    upper_z = slice_center + half_width

    projected_segments, clipped_segments = _project_segments_to_z_band(core_segments, lower_z, upper_z)
    visibility_weights = [
        _mean_axial_weight_over_segment(clipped_segment, slice_center, depth_of_field)
        for clipped_segment in clipped_segments
    ]
    return projected_segments, np.asarray(visibility_weights, dtype=np.float64)


def _resolve_localization_slab_thickness(
    rasterizer,
    label_slab_thickness: Optional[float],
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
) -> float:
    if label_slab_scale <= 0.0:
        raise ValueError("label_slab_scale must be greater than 0.")

    if label_slab_thickness is not None:
        return float(label_slab_thickness)

    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    return depth_of_field * float(label_slab_scale)


def _build_2d_focus_and_visibility_targets(
    core_segments,
    slice_center,
    localization_slab_thickness,
    rasterizer,
    target_gen,
    visibility_target_gen,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
):
    if soft_skeleton_alpha < 0.0:
        raise ValueError("soft_skeleton_alpha must be greater than or equal to 0.")
    if annotation_weight_floor <= 0.0 or annotation_weight_floor > 1.0:
        raise ValueError("annotation_weight_floor must be in the interval (0, 1].")

    focus_segments = _project_segments_to_label_slab(core_segments, slice_center, localization_slab_thickness)
    edt_focus, vector_focus = target_gen.generate(focus_segments)
    annotation_segments = _project_segments_to_annotation(
        core_segments,
        slice_center,
        rasterizer,
        annotation_weight_floor=annotation_weight_floor,
    )
    edt_annotation, vector_annotation = target_gen.generate(annotation_segments)

    visibility_segments, visibility_weights = _project_segments_to_visibility(core_segments, slice_center, rasterizer)
    visibility_target = visibility_target_gen.generate(visibility_segments, visibility_weights)

    edt_soft = np.clip(edt_annotation * float(soft_skeleton_alpha), 0.0, 1.0)
    edt_target = np.maximum(edt_focus, edt_soft)
    vector_target = np.array(vector_focus, copy=True)
    soft_overwrite_mask = edt_soft > edt_focus
    if np.any(soft_overwrite_mask):
        vector_target[:, soft_overwrite_mask] = vector_annotation[:, soft_overwrite_mask]

    return {
        "focus_segments": focus_segments,
        "annotation_segments": annotation_segments,
        "edt_focus": edt_focus,
        "vector_focus": vector_focus,
        "edt_annotation": edt_annotation,
        "vector_annotation": vector_annotation,
        "edt_soft": edt_soft,
        "edt_target": edt_target,
        "vector_target": vector_target,
        "visibility_target": visibility_target,
        "visibility_segments": visibility_segments,
        "visibility_weights": visibility_weights,
        "annotation_weight_floor": float(annotation_weight_floor),
    }


def _prepare_2d_sted_scene(
    bounds: tuple,
    label_slab_thickness: Optional[float],
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
):
    if soft_skeleton_alpha < 0.0:
        raise ValueError("soft_skeleton_alpha must be greater than or equal to 0.")
    if annotation_weight_floor <= 0.0 or annotation_weight_floor > 1.0:
        raise ValueError("annotation_weight_floor must be in the interval (0, 1].")

    x_size, y_size, z_size = bounds

    optical_bundle_lists = []
    core_segments_flat = []
    projected_segments = []
    annotation_segments = []
    visibility_segments = []
    visibility_weights = np.zeros(0, dtype=np.float64)
    edt_focus = np.zeros((x_size, y_size), dtype=np.float64)
    edt_soft = np.zeros((x_size, y_size), dtype=np.float64)
    edt_target = np.zeros((x_size, y_size), dtype=np.float64)
    vector_target = np.zeros((2, x_size, y_size), dtype=np.float64)
    visibility_target = np.zeros((x_size, y_size), dtype=np.float64)
    slice_center = np.random.uniform(z_size * 0.2, z_size * 0.8)

    rasterizer = EmpiricalRasterizer(
        bounds=bounds,
        base_sigma=1.0,
        z_anisotropy=np.random.uniform(1.6, 2.8),
        noise_level=np.random.uniform(0.005, 0.045),
        debris_count=np.random.randint(4, 16),
        gap_prob=np.random.uniform(0.0, 0.08),
        enable_sted_monomer_cloud=True,
        sted_monomer_mix=(0.70, 0.20, 0.10),
    )
    target_gen = TargetFieldGenerator((x_size, y_size), max_distance=TARGET_MAX_DISTANCE)
    visibility_target_gen = WeightedVisibilityTargetGenerator((x_size, y_size), base_sigma=rasterizer.base_sigma)
    localization_slab_thickness = _resolve_localization_slab_thickness(
        rasterizer,
        label_slab_thickness,
        label_slab_scale=label_slab_scale,
    )
    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    axial_fwhm = rasterizer._sted_axial_fwhm(depth_of_field)

    for _ in range(6):
        optical_bundle_lists = []
        core_segments_flat = []

        num_fibers = np.random.randint(5, 11)
        for _ in range(num_fibers):
            core_segments = _generate_constrained_random_walk(bounds)
            if not core_segments:
                continue

            core_segments_flat.extend(core_segments)

            bundle_size = np.random.randint(1, 4)
            if bundle_size > 1:
                jitter_amount = np.random.uniform(0.35, 0.85)
                optical_segments = apply_optical_jitter(
                    core_segments,
                    bundle_size=bundle_size,
                    jitter_amount=jitter_amount,
                    lock_z=False,
                )
            else:
                optical_segments = core_segments

            optical_bundle_lists.append(optical_segments)

        slice_center = np.random.uniform(z_size * 0.2, z_size * 0.8)
        target_data = _build_2d_focus_and_visibility_targets(
            core_segments_flat,
            slice_center,
            localization_slab_thickness,
            rasterizer,
            target_gen,
            visibility_target_gen,
            annotation_weight_floor=annotation_weight_floor,
            soft_skeleton_alpha=soft_skeleton_alpha,
        )
        projected_segments = target_data["focus_segments"]
        annotation_segments = target_data["annotation_segments"]
        edt_focus = target_data["edt_focus"]
        edt_soft = target_data["edt_soft"]
        edt_target = target_data["edt_target"]
        vector_target = target_data["vector_target"]
        visibility_target = target_data["visibility_target"]
        visibility_segments = target_data["visibility_segments"]
        visibility_weights = target_data["visibility_weights"]
        if core_segments_flat and projected_segments:
            break

    img_min = np.random.uniform(0.18, 0.45)
    img_max = np.random.uniform(img_min + 0.15, 1.0)

    return {
        "bounds": bounds,
        "optical_bundle_lists": optical_bundle_lists,
        "core_segments": core_segments_flat,
        "projected_segments": projected_segments,
        "annotation_segments": annotation_segments,
        "visibility_segments": visibility_segments,
        "visibility_weights": visibility_weights,
        "slice_center": slice_center,
        "label_slab_scale": float(label_slab_scale),
        "label_slab_thickness": localization_slab_thickness,
        "annotation_weight_floor": float(annotation_weight_floor),
        "soft_skeleton_alpha": float(soft_skeleton_alpha),
        "axial_fwhm": axial_fwhm,
        "dynamic_range": (img_min, img_max),
        "rasterizer": rasterizer,
        "edt_focus": edt_focus,
        "edt_soft": edt_soft,
        "edt_target": edt_target,
        "vector_target": vector_target,
        "visibility_target": visibility_target,
    }


def _build_2d_sample(
    bounds: tuple,
    label_slab_thickness: Optional[float],
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
):
    scene = _prepare_2d_sted_scene(
        bounds,
        label_slab_thickness,
        label_slab_scale=label_slab_scale,
        annotation_weight_floor=annotation_weight_floor,
        soft_skeleton_alpha=soft_skeleton_alpha,
    )
    image = scene["rasterizer"].render_sted_slice(
        scene["optical_bundle_lists"],
        slice_center=scene["slice_center"],
        dynamic_range=scene["dynamic_range"],
    )
    return image, scene["edt_target"], scene["vector_target"], scene["visibility_target"]


def build_sted_debug_sample(
    bounds: tuple,
    synth_depth: int = 16,
    label_slab_thickness: Optional[float] = None,
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
    seed: int = None,
):
    if len(bounds) == 2:
        synth_bounds = (bounds[0], bounds[1], synth_depth)
    elif len(bounds) == 3:
        synth_bounds = bounds
    else:
        raise ValueError("Bounds must contain either 2 or 3 integers.")

    if synth_bounds[2] < 2:
        raise ValueError("STED debug samples require a synth depth of at least 2.")

    if seed is not None:
        np.random.seed(seed)

    scene = _prepare_2d_sted_scene(
        synth_bounds,
        label_slab_thickness,
        label_slab_scale=label_slab_scale,
        annotation_weight_floor=annotation_weight_floor,
        soft_skeleton_alpha=soft_skeleton_alpha,
    )
    debug_render = scene["rasterizer"].render_sted_slice_debug(
        scene["optical_bundle_lists"],
        slice_center=scene["slice_center"],
        dynamic_range=scene["dynamic_range"],
    )

    debug_render["bounds"] = synth_bounds
    debug_render["label_slab_scale"] = scene["label_slab_scale"]
    debug_render["label_slab_thickness"] = scene["label_slab_thickness"]
    debug_render["annotation_weight_floor"] = scene["annotation_weight_floor"]
    debug_render["soft_skeleton_alpha"] = scene["soft_skeleton_alpha"]
    debug_render["axial_fwhm"] = scene["axial_fwhm"]
    debug_render["edt_focus"] = scene["edt_focus"]
    debug_render["edt_soft"] = scene["edt_soft"]
    debug_render["edt_target"] = scene["edt_target"]
    debug_render["vector_target"] = scene["vector_target"]
    debug_render["visibility_target"] = scene["visibility_target"]
    debug_render["projected_segments"] = scene["projected_segments"]
    debug_render["annotation_segments"] = scene["annotation_segments"]
    debug_render["visibility_segments"] = scene["visibility_segments"]
    debug_render["visibility_weights"] = scene["visibility_weights"]
    debug_render["core_segments"] = scene["core_segments"]
    debug_render["fiber_count"] = len(scene["optical_bundle_lists"])
    debug_render["projected_segment_count"] = len(scene["projected_segments"])
    debug_render["annotation_segment_count"] = len(scene["annotation_segments"])
    debug_render["visibility_segment_count"] = len(scene["visibility_segments"])
    return debug_render


def _build_3d_sample(bounds: tuple):
    x_size, y_size, z_size = bounds
    is_flat_volume = z_size == 1

    num_bundles = np.random.randint(3, 12)
    optical_bundle_lists = []
    core_segments_flat = []

    for _ in range(num_bundles):
        core_segments = _generate_phenotype(bounds)
        if not core_segments:
            continue

        core_segments_flat.extend(core_segments)

        bundle_size = np.random.randint(1, 6)
        if bundle_size > 1:
            optical_segments = apply_optical_jitter(
                core_segments,
                bundle_size=bundle_size,
                jitter_amount=2.0,
                lock_z=is_flat_volume,
            )
        else:
            optical_segments = core_segments

        optical_bundle_lists.append(optical_segments)

    img_min = np.random.uniform(0.1, 0.5)
    img_max = np.random.uniform(img_min + 0.2, 1.0)

    rasterizer = EmpiricalRasterizer(
        bounds=bounds,
        base_sigma=1.0,
        z_anisotropy=1.0 if is_flat_volume else np.random.uniform(1.0, 4.0),
        noise_level=np.random.uniform(0.01, 0.15),
        debris_count=np.random.randint(5, 45),
        gap_prob=np.random.uniform(0.0, 0.12),
    )
    target_gen = TargetFieldGenerator(bounds, max_distance=TARGET_MAX_DISTANCE)

    volume = rasterizer.render(optical_bundle_lists, dynamic_range=(img_min, img_max))
    edt_target, vector_target = target_gen.generate(core_segments_flat)
    return volume, edt_target, vector_target


def _to_2d_tensors(image, edt_target, vector_target, visibility_target):
    import torch

    image = image.transpose(1, 0)
    edt_target = edt_target.transpose(1, 0)
    vector_target = vector_target.transpose(0, 2, 1)
    visibility_target = visibility_target.transpose(1, 0)

    volume_tensor = torch.tensor(image[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    targets = np.zeros((4, 1, image.shape[0], image.shape[1]), dtype=np.float32)
    targets[0, 0, :, :] = edt_target.astype(np.float32)
    targets[1, 0, :, :] = vector_target[0].astype(np.float32)
    targets[2, 0, :, :] = vector_target[1].astype(np.float32)
    targets[3, 0, :, :] = visibility_target.astype(np.float32)

    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    return volume_tensor, targets_tensor


def _to_3d_tensors(volume, edt_target, vector_target):
    import torch

    volume = volume.transpose(2, 1, 0)
    edt_target = edt_target.transpose(2, 1, 0)
    vector_target = vector_target.transpose(0, 3, 2, 1)

    volume_tensor = torch.tensor(np.expand_dims(volume, axis=0), dtype=torch.float32)
    edt_tensor = torch.tensor(np.expand_dims(edt_target, axis=0), dtype=torch.float32)
    vec_tensor = torch.tensor(vector_target, dtype=torch.float32)
    targets_tensor = torch.cat([edt_tensor, vec_tensor], dim=0)
    return volume_tensor, targets_tensor


def process_single_sample(
    idx: int,
    file_offset: int,
    bounds: tuple,
    output_dir: str,
    emit_2d: bool,
    label_slab_thickness: Optional[float],
    label_slab_scale: float,
    annotation_weight_floor: float,
    soft_skeleton_alpha: float,
):
    import torch

    np.random.seed(None)

    if emit_2d:
        image, edt_target, vector_target, visibility_target = _build_2d_sample(
            bounds,
            label_slab_thickness,
            label_slab_scale=label_slab_scale,
            annotation_weight_floor=annotation_weight_floor,
            soft_skeleton_alpha=soft_skeleton_alpha,
        )
        volume_tensor, targets_tensor = _to_2d_tensors(image, edt_target, vector_target, visibility_target)
    else:
        volume, edt_target, vector_target = _build_3d_sample(bounds)
        volume_tensor, targets_tensor = _to_3d_tensors(volume, edt_target, vector_target)

    file_id = file_offset + idx
    file_path = os.path.join(output_dir, f"sample_{file_id}.pt")
    torch.save({"volume": volume_tensor, "targets": targets_tensor}, file_path)
    return file_id


def build_dataset_split(
    split_name: str,
    size: int,
    file_offset: int,
    bounds: tuple,
    base_dir: str,
    workers: int,
    emit_2d: bool,
    label_slab_thickness: Optional[float],
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
):
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    if emit_2d:
        descriptor = f"2D slices from 3D synth {bounds}"
    else:
        descriptor = f"3D volumes {bounds}"

    print(f"Building '{split_name}' split ({size} samples | {descriptor}) at {split_dir}...")

    worker_func = partial(
        process_single_sample,
        file_offset=file_offset,
        bounds=bounds,
        output_dir=split_dir,
        emit_2d=emit_2d,
        label_slab_thickness=label_slab_thickness,
        label_slab_scale=label_slab_scale,
        annotation_weight_floor=annotation_weight_floor,
        soft_skeleton_alpha=soft_skeleton_alpha,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_func, i): i for i in range(size)}

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % max(1, (size // 10)) == 0:
                    print(f"  [{completed}/{size}] Samples processed.")
            except Exception as exc:
                print(f"Error generating sample: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline generator for flexible synthetic fiber datasets")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bounds", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--synth_depth", type=int, default=16)
    parser.add_argument(
        "--label_slab_thickness",
        type=float,
        default=None,
        help="Optional override for the narrow focus-localization slab in 2D mode. Defaults to the optical depth of field.",
    )
    parser.add_argument(
        "--label_slab_scale",
        type=float,
        default=DEFAULT_LABEL_SLAB_SCALE,
        help="Scale factor applied to depth of field when --label_slab_thickness is not provided.",
    )
    parser.add_argument(
        "--soft_skeleton_alpha",
        type=float,
        default=DEFAULT_SOFT_SKELETON_ALPHA,
        help="Soft out-of-focus blend strength for EDT/vector targets in 2D STED generation.",
    )
    parser.add_argument(
        "--annotation_weight_floor",
        type=float,
        default=DEFAULT_ANNOTATION_WEIGHT_FLOOR,
        help="Axial-weight floor that defines the broader soft-annotation band in 2D STED generation.",
    )
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=400)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    dims = len(args.bounds)
    if args.label_slab_scale <= 0.0:
        raise ValueError("--label_slab_scale must be greater than 0.")
    if args.annotation_weight_floor <= 0.0 or args.annotation_weight_floor > 1.0:
        raise ValueError("--annotation_weight_floor must be in the interval (0, 1].")
    if args.soft_skeleton_alpha < 0.0:
        raise ValueError("--soft_skeleton_alpha must be greater than or equal to 0.")

    if dims == 2:
        if args.synth_depth < 2:
            raise ValueError("--synth_depth must be at least 2 when generating 2D slices.")
        synth_bounds = (args.bounds[0], args.bounds[1], args.synth_depth)
        emit_2d = True
    elif dims == 3:
        synth_bounds = tuple(args.bounds)
        emit_2d = False
    else:
        raise ValueError("Bounds must be either 2 (X, Y) or 3 (X, Y, Z) integers.")

    os.makedirs(args.output_dir, exist_ok=True)

    build_dataset_split(
        "train",
        args.train_size,
        file_offset=0,
        bounds=synth_bounds,
        base_dir=args.output_dir,
        workers=args.workers,
        emit_2d=emit_2d,
        label_slab_thickness=args.label_slab_thickness,
        label_slab_scale=args.label_slab_scale,
        annotation_weight_floor=args.annotation_weight_floor,
        soft_skeleton_alpha=args.soft_skeleton_alpha,
    )
    build_dataset_split(
        "val",
        args.val_size,
        file_offset=args.train_size,
        bounds=synth_bounds,
        base_dir=args.output_dir,
        workers=args.workers,
        emit_2d=emit_2d,
        label_slab_thickness=args.label_slab_thickness,
        label_slab_scale=args.label_slab_scale,
        annotation_weight_floor=args.annotation_weight_floor,
        soft_skeleton_alpha=args.soft_skeleton_alpha,
    )
    build_dataset_split(
        "test",
        args.test_size,
        file_offset=(args.train_size + args.val_size),
        bounds=synth_bounds,
        base_dir=args.output_dir,
        workers=args.workers,
        emit_2d=emit_2d,
        label_slab_thickness=args.label_slab_thickness,
        label_slab_scale=args.label_slab_scale,
        annotation_weight_floor=args.annotation_weight_floor,
        soft_skeleton_alpha=args.soft_skeleton_alpha,
    )
