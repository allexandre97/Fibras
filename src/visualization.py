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
    def _normalize_to_uint8(image):
        image = np.asarray(image, dtype=np.float64)
        image = image - image.min()
        scale = image.max()
        if scale > 1e-8:
            image = image / scale
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _make_pil_image_panel(image, title, segments=None, mask=None, tile_size=256):
        from PIL import Image, ImageDraw

        display = np.flipud(np.asarray(image).T)
        rgb = np.repeat(StedSynthesisVisualizer._normalize_to_uint8(display)[..., None], 3, axis=2)

        if mask is not None:
            display_mask = np.flipud(np.asarray(mask).T.astype(bool))
            rgb[display_mask] = np.array([255, 255, 255], dtype=np.uint8)

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
    def _make_pil_profile_panel(axial_signal, axial_weights, lateral_sigmas, slice_center, slab_thickness, depth, tile_size=256):
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
        draw.rectangle((slab_left, top, slab_right, top + height), fill=(40, 90, 40), outline=None)

        draw.line(to_canvas_points(signal_norm), fill=(255, 160, 40), width=2)
        draw.line(to_canvas_points(weight_norm), fill=(80, 160, 255), width=2)
        draw.line(to_canvas_points(blur_norm), fill=(240, 96, 96), width=2)

        slice_x = left + (slice_center / max(depth - 1, 1)) * width
        draw.line((slice_x, top, slice_x, top + height), fill=(255, 255, 255), width=1)

        legend_y = tile_size + 4
        draw.text((6, legend_y), "orange=signal  blue=weight  red=blur  white=slice", fill=(235, 235, 235))
        return panel

    @staticmethod
    def _save_pil_contact_sheet(debug_data, save_path):
        from PIL import Image

        signal_volume = debug_data["signal_volume"]
        edt_target = debug_data["edt_target"]
        panels = [
            StedSynthesisVisualizer._make_pil_image_panel(
                signal_volume.max(axis=2),
                f"3D Signal MIP | fibers={debug_data['fiber_count']}",
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                debug_data["focus_plane"],
                f"Focus Plane z={debug_data['focus_index']}",
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                debug_data["weighted_slice"],
                "Defocus-Aware Section",
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                debug_data["final_slice"],
                "Final STED Slice",
                segments=debug_data["projected_segments"],
            ),
            StedSynthesisVisualizer._make_pil_image_panel(
                edt_target,
                f"EDT Target | slab={debug_data['label_slab_thickness']:.2f}",
                mask=edt_target > 0.85,
            ),
            StedSynthesisVisualizer._make_pil_profile_panel(
                debug_data["axial_signal_profile"],
                debug_data["axial_weights"],
                debug_data["lateral_sigmas"],
                debug_data["slice_center"],
                debug_data["label_slab_thickness"],
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
        weighted_slice = debug_data["weighted_slice"]
        focus_plane = debug_data["focus_plane"]
        final_slice = debug_data["final_slice"]
        edt_target = debug_data["edt_target"]
        axial_weights = debug_data["axial_weights"]
        axial_signal = debug_data["axial_signal_profile"]
        lateral_sigmas = debug_data["lateral_sigmas"]
        slice_center = debug_data["slice_center"]
        focus_index = debug_data["focus_index"]
        slab_thickness = debug_data["label_slab_thickness"]
        projected_segments = debug_data["projected_segments"]

        mip_xy = signal_volume.max(axis=2)
        centerline_mask = edt_target > 0.85

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        axes = axes.ravel()

        StedSynthesisVisualizer._show_xy(
            axes[0],
            mip_xy,
            f"3D Signal MIP\n{debug_data['fiber_count']} fibers, z={signal_volume.shape[2]}",
        )

        StedSynthesisVisualizer._show_xy(
            axes[1],
            focus_plane,
            f"Focus Plane z={focus_index}",
        )

        StedSynthesisVisualizer._show_xy(
            axes[2],
            weighted_slice,
            "Defocus-Aware Section\nPre-artifact",
        )

        StedSynthesisVisualizer._show_xy(
            axes[3],
            final_slice,
            "Final STED Slice",
        )
        StedSynthesisVisualizer._overlay_segments(
            axes[3],
            projected_segments,
            color="cyan",
            linewidth=1.2,
            alpha=0.75,
        )

        StedSynthesisVisualizer._show_xy(
            axes[4],
            edt_target,
            f"2D EDT Target\nslab={slab_thickness:.2f} px, projected={debug_data['projected_segment_count']}",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axes[4].contour(centerline_mask.T.astype(float), levels=[0.5], colors="white", linewidths=0.8)

        z_axis = np.arange(signal_volume.shape[2], dtype=float)
        normalized_signal = axial_signal / max(axial_signal.max(), 1e-8)
        normalized_weights = axial_weights / max(axial_weights.max(), 1e-8)
        normalized_blur = lateral_sigmas / max(lateral_sigmas.max(), 1e-8)

        axes[5].plot(z_axis, normalized_signal, label="signal per z", color="tab:orange", linewidth=2.0)
        axes[5].plot(z_axis, normalized_weights, label="axial weight", color="tab:blue", linewidth=2.0)
        axes[5].plot(z_axis, normalized_blur, label="relative blur", color="tab:red", linewidth=2.0)
        axes[5].axvline(slice_center, color="black", linestyle="--", linewidth=1.0, label="slice center")
        axes[5].axvspan(
            slice_center - (slab_thickness / 2.0),
            slice_center + (slab_thickness / 2.0),
            color="tab:green",
            alpha=0.2,
            label="label slab",
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
