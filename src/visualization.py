import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



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
        viewer.add_image(volume, name='Raw Density', colormap='magma', blending='additive')
        viewer.add_image(result.hfa_map, name='Local HFA (Tubularity)', 
                         colormap='viridis', blending='additive', visible=False)
        viewer.add_image(result.fa_macro_map, name='Macro FA (Alignment)', 
                         colormap='cyan', blending='additive', visible=False)
        viewer.add_labels(result.skeleton.astype(int), name='Topological Skeleton')
        viewer.add_labels(result.binary_mask.astype(int), name='Binary Mask (Pre-Skeleton)', visible=False)
        napari.run()