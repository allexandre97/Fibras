import numpy as np


class VolumetricVisualizer:
    """Handles true 3D scalar field rendering using Mayavi."""

    def __init__(self, density_volume: np.ndarray):
        if density_volume.ndim != 3:
            raise ValueError(f"Expected 3D array, got {density_volume.ndim}D array.")
        self.volume = density_volume


class AdvancedVisualizer:
    """Advanced tools for spatial resolution of analysis results."""

    @staticmethod
    def show_interactive_napari(volume: np.ndarray, result):
        """Interactive multi-layer visualization using Napari."""
        try:
            import napari
        except ImportError:
            print("Napari not installed. Use 'pip install napari[all]'")
            return

        viewer = napari.Viewer()
        viewer.add_image(volume, name="Raw Density", colormap="magma", blending="additive")
        viewer.add_image(
            result.hfa_map,
            name="Local HFA (Tubularity)",
            colormap="viridis",
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            result.fa_macro_map,
            name="Macro FA (Alignment)",
            colormap="cyan",
            blending="additive",
            visible=False,
        )
        viewer.add_labels(result.skeleton.astype(int), name="Topological Skeleton")
        viewer.add_labels(result.binary_mask.astype(int), name="Binary Mask (Pre-Skeleton)", visible=False)
        napari.run()


class StedSynthesisVisualizer:
    @staticmethod
    def _show_xy(ax, image, title, cmap="magma", vmin=None, vmax=None):
        im = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return im

    @staticmethod
    def _overlay_segments(ax, segments, color, linewidth=1.0, alpha=0.8):
        for segment in segments:
            ax.plot(
                [segment.start[0], segment.end[0]],
                [segment.start[1], segment.end[1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

    @staticmethod
    def _mask_outline(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0:
            return mask

        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        eroded = mask.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                eroded &= padded[
                    1 + dx : 1 + dx + mask.shape[0],
                    1 + dy : 1 + dy + mask.shape[1],
                ]
        return mask & ~eroded

    @staticmethod
    def _normalize_to_uint8(image):
        image = np.asarray(image, dtype=np.float64)
        image = image - image.min()
        scale = image.max()
        if scale > 1e-8:
            image = image / scale
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _make_pil_image_panel(image, title, segments=None, mask=None, mask_overlays=None, tile_size=256):
        from PIL import Image, ImageDraw

        display = np.flipud(np.asarray(image).T)
        rgb = np.repeat(StedSynthesisVisualizer._normalize_to_uint8(display)[..., None], 3, axis=2)

        if mask is not None:
            display_mask = np.flipud(np.asarray(mask).T.astype(bool))
            rgb[display_mask] = np.array([255, 255, 255], dtype=np.uint8)

        if mask_overlays:
            for overlay_mask, color in mask_overlays:
                display_mask = np.flipud(np.asarray(overlay_mask).T.astype(bool))
                outline = StedSynthesisVisualizer._mask_outline(display_mask)
                rgb[outline] = np.asarray(color, dtype=np.uint8)

        panel_image = Image.fromarray(rgb, mode="RGB").resize((tile_size, tile_size), resample=Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (tile_size, tile_size + 28), color=(18, 18, 18))
        canvas.paste(panel_image, (0, 28))

        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), title, fill=(235, 235, 235))

        if segments:
            scale_x = tile_size / max(image.shape[0], 1)
            scale_y = tile_size / max(image.shape[1], 1)
            for segment in segments:
                x0 = float(segment.start[0] * scale_x)
                y0 = float((image.shape[1] - 1 - segment.start[1]) * scale_y) + 28.0
                x1 = float(segment.end[0] * scale_x)
                y1 = float((image.shape[1] - 1 - segment.end[1]) * scale_y) + 28.0
                draw.line((x0, y0, x1, y1), fill=(0, 255, 255), width=2)

        return canvas

    @staticmethod
    def _make_pil_profile_panel(
        axial_signal,
        axial_weights,
        lateral_sigmas,
        slice_center,
        slab_thickness,
        axial_fwhm,
        depth,
        tile_size=256,
    ):
        from PIL import Image, ImageDraw

        panel = Image.new("RGB", (tile_size, tile_size + 28), color=(18, 18, 18))
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), "Axial Profile", fill=(235, 235, 235))

        left, top = 28, 44
        width = tile_size - 44
        height = tile_size - 56
        draw.rectangle((left, top, left + width, top + height), outline=(180, 180, 180), width=1)

        signal_norm = axial_signal / max(axial_signal.max(), 1e-8)
        weight_norm = axial_weights / max(axial_weights.max(), 1e-8)
        blur_norm = lateral_sigmas / max(lateral_sigmas.max(), 1e-8)

        def to_canvas_points(values):
            points = []
            for idx, value in enumerate(values):
                x = left + (idx / max(depth - 1, 1)) * width
                y = top + height - (float(value) * height)
                points.append((x, y))
            return points

        slab_left = left + ((slice_center - (slab_thickness / 2.0)) / max(depth - 1, 1)) * width
        slab_right = left + ((slice_center + (slab_thickness / 2.0)) / max(depth - 1, 1)) * width
        fwhm_left = left + ((slice_center - (axial_fwhm / 2.0)) / max(depth - 1, 1)) * width
        fwhm_right = left + ((slice_center + (axial_fwhm / 2.0)) / max(depth - 1, 1)) * width

        draw.rectangle((fwhm_left, top, fwhm_right, top + height), outline=(220, 200, 90), width=1)
        draw.rectangle((slab_left, top, slab_right, top + height), fill=(40, 90, 40), outline=None)

        draw.line(to_canvas_points(signal_norm), fill=(255, 160, 40), width=2)
        draw.line(to_canvas_points(weight_norm), fill=(80, 160, 255), width=2)
        draw.line(to_canvas_points(blur_norm), fill=(240, 96, 96), width=2)

        slice_x = left + (slice_center / max(depth - 1, 1)) * width
        draw.line((slice_x, top, slice_x, top + height), fill=(255, 255, 255), width=1)

        legend_y = tile_size + 4
        draw.text((6, legend_y), "orange=signal blue=weight red=blur yellow=fwhm green=slab", fill=(235, 235, 235))
        return panel

    @staticmethod
    def _save_pil_contact_sheet(debug_data, save_path):
        from PIL import Image

        signal_volume = debug_data["signal_volume"]
        fiber_signal_volume = debug_data.get("fiber_signal_volume", signal_volume)
        monomer_volume = debug_data.get("monomer_volume", np.zeros_like(signal_volume))
        monomer_regime = debug_data.get("monomer_regime", "n/a")
        monomer_amplitude = float(debug_data.get("monomer_amplitude", 0.0))
        slab_scale = float(debug_data.get("label_slab_scale", 1.0))
        annotation_weight_floor = float(debug_data.get("annotation_weight_floor", 0.25))
        soft_alpha = float(debug_data.get("soft_skeleton_alpha", 0.0))
        edt_focus = debug_data.get("edt_focus", debug_data["edt_target"])
        edt_target = debug_data["edt_target"]
        visibility_target = debug_data["visibility_target"]
        focus_core_mask = edt_focus > 0.85
        annotation_mask = edt_target > 0.15
        panels = [
            StedSynthesisVisualizer._make_pil_image_panel(
                fiber_signal_volume.max(axis=2),
                f"Fiber-Only MIP | fibers={debug_data['fiber_count']}",
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                monomer_volume.max(axis=2),
                f"Monomer MIP | {monomer_regime} | amp={monomer_amplitude:.4f}",
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                debug_data["weighted_slice"],
                "Defocus-Aware Section (Fiber + Monomer)",
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                debug_data["final_slice"],
                "Final STED Slice | cyan=core white=ann",
                mask_overlays=[
                    (annotation_mask, (255, 255, 255)),
                    (focus_core_mask, (0, 255, 255)),
                ],
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                visibility_target,
                (
                    "Visibility Target | "
                    f"slab={debug_data['label_slab_thickness']:.2f} | "
                    f"ann_w>={annotation_weight_floor:.2f} | a={soft_alpha:.2f}"
                ),
                mask_overlays=[
                    (annotation_mask, (255, 255, 255)),
                    (focus_core_mask, (0, 255, 255)),
                ],
            ),
            StedSynthesisVisualizer._make_pil_profile_panel(
                debug_data["axial_signal_profile"],
                debug_data["axial_weights"],
                debug_data["lateral_sigmas"],
                debug_data["slice_center"],
                debug_data["label_slab_thickness"],
                debug_data["axial_fwhm"],
                signal_volume.shape[2],
            ),
        ]

        gap = 12
        tile_w, tile_h = panels[0].size
        sheet = Image.new("RGB", (tile_w * 3 + gap * 4, tile_h * 2 + gap * 3), color=(8, 8, 8))

        for idx, panel in enumerate(panels):
            row = idx // 3
            col = idx % 3
            x0 = gap + col * (tile_w + gap)
            y0 = gap + row * (tile_h + gap)
            sheet.paste(panel, (x0, y0))

        sheet.save(save_path)

    @staticmethod
    def show_sted_debug_summary(debug_data, save_path=None, show=True):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            fallback_path = save_path or "sted_debug_summary.png"
            StedSynthesisVisualizer._save_pil_contact_sheet(debug_data, fallback_path)
            print(f"Matplotlib unavailable ({exc}). Saved static debug summary to: {fallback_path}")
            return

        signal_volume = debug_data["signal_volume"]
        fiber_signal_volume = debug_data.get("fiber_signal_volume", signal_volume)
        monomer_volume = debug_data.get("monomer_volume", np.zeros_like(signal_volume))
        monomer_regime = debug_data.get("monomer_regime", "n/a")
        monomer_amplitude = float(debug_data.get("monomer_amplitude", 0.0))
        slab_scale = float(debug_data.get("label_slab_scale", 1.0))
        annotation_weight_floor = float(debug_data.get("annotation_weight_floor", 0.25))
        soft_alpha = float(debug_data.get("soft_skeleton_alpha", 0.0))
        weighted_slice = debug_data["weighted_slice"]
        final_slice = debug_data["final_slice"]
        edt_focus = debug_data.get("edt_focus", debug_data["edt_target"])
        edt_target = debug_data["edt_target"]
        visibility_target = debug_data["visibility_target"]
        axial_weights = debug_data["axial_weights"]
        axial_signal = debug_data["axial_signal_profile"]
        lateral_sigmas = debug_data["lateral_sigmas"]
        slice_center = debug_data["slice_center"]
        slab_thickness = debug_data["label_slab_thickness"]
        axial_fwhm = debug_data["axial_fwhm"]

        fiber_mip_xy = fiber_signal_volume.max(axis=2)
        monomer_mip_xy = monomer_volume.max(axis=2)
        focus_core_mask = edt_focus > 0.85
        annotation_mask = edt_target > 0.15

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        axes = axes.ravel()

        StedSynthesisVisualizer._show_xy(
            axes[0],
            fiber_mip_xy,
            f"Fiber-Only MIP\n{debug_data['fiber_count']} fibers, z={signal_volume.shape[2]}",
        )

        StedSynthesisVisualizer._show_xy(
            axes[1],
            monomer_mip_xy,
            f"Monomer Cloud MIP\nregime={monomer_regime}, amp={monomer_amplitude:.4f}",
        )

        StedSynthesisVisualizer._show_xy(
            axes[2],
            weighted_slice,
            "Defocus-Aware Section\nFiber + Monomer (Pre-2D Artifacts)",
        )

        StedSynthesisVisualizer._show_xy(
            axes[3],
            final_slice,
            "Final STED Slice\ncyan=focus core, white=visible annotation",
        )
        axes[3].contour(
            annotation_mask.T.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=1.0,
            linestyles="dashed",
        )
        axes[3].contour(focus_core_mask.T.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)

        StedSynthesisVisualizer._show_xy(
            axes[4],
            visibility_target,
            (
                "Visibility Target\n"
                f"slab={slab_thickness:.2f} px (x{slab_scale:.2f}), "
                f"ann_w>={annotation_weight_floor:.2f}, a={soft_alpha:.2f}"
            ),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axes[4].contour(
            annotation_mask.T.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=1.0,
            linestyles="dashed",
        )
        axes[4].contour(focus_core_mask.T.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)

        z_axis = np.arange(signal_volume.shape[2], dtype=float)
        normalized_signal = axial_signal / max(axial_signal.max(), 1e-8)
        normalized_weights = axial_weights / max(axial_weights.max(), 1e-8)
        normalized_blur = lateral_sigmas / max(lateral_sigmas.max(), 1e-8)

        axes[5].plot(z_axis, normalized_signal, label="signal per z", color="tab:orange", linewidth=2.0)
        axes[5].plot(z_axis, normalized_weights, label="axial weight", color="tab:blue", linewidth=2.0)
        axes[5].plot(z_axis, normalized_blur, label="relative blur", color="tab:red", linewidth=2.0)
        axes[5].axvline(slice_center, color="black", linestyle="--", linewidth=1.0, label="slice center")
        axes[5].axvspan(
            slice_center - (axial_fwhm / 2.0),
            slice_center + (axial_fwhm / 2.0),
            color="gold",
            alpha=0.15,
            label="axial FWHM",
        )
        axes[5].axvspan(
            slice_center - (slab_thickness / 2.0),
            slice_center + (slab_thickness / 2.0),
            color="tab:green",
            alpha=0.2,
            label="focus slab",
        )
        axes[5].set_title("Axial Profile")
        axes[5].set_xlabel("z")
        axes[5].set_ylabel("normalized value")
        axes[5].set_ylim(0.0, 1.05)
        axes[5].legend(loc="upper right")

        fig.suptitle("STED Synthesis Debug Summary", fontsize=14)

        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
