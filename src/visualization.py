import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

try:
    from mayavi import mlab
    from tvtk.util.ctf import ColorTransferFunction
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    print("Warning: Mayavi not installed. Volumetric visualization will be unavailable.")

class VolumetricVisualizer:
    """Handles true 3D scalar field rendering using Mayavi."""
    def __init__(self, density_volume: np.ndarray):
        if density_volume.ndim != 3:
            raise ValueError(f"Expected 3D array, got {density_volume.ndim}D array.")
        self.volume = density_volume

    def show_volume(self, vmin: float = 0.05, vmax: float = 1.0):
        if not MAYAVI_AVAILABLE:
            print("Mayavi not available for visualization.")
            return
            
        mlab.figure(size=(800, 800), bgcolor=(0.0, 0.0, 0.0), fgcolor=(1, 1, 1))
        scalar_field = mlab.pipeline.scalar_field(self.volume)
        vol = mlab.pipeline.volume(scalar_field, vmin=vmin, vmax=vmax)
        
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(vmin, 0.0, 0.0, 0.5)  
        ctf.add_rgb_point(vmax * 0.5, 0.0, 0.8, 0.8) 
        ctf.add_rgb_point(vmax, 1.0, 1.0, 1.0)  
        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True
        
        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
        mlab.outline()
        mlab.show()

class AdvancedVisualizer:
    """Advanced tools for spatial resolution of analysis results."""
    
    @staticmethod
    def show_3d_network(volume: np.ndarray, G: nx.Graph, title="3D Fiber Network"):
        """Overlays the skeletal graph on the density volume in Mayavi."""
        if not MAYAVI_AVAILABLE: return
        mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0))
        
        # Show raw volume
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(volume), vmin=0.1)
        
        # CORRECTED: Access the internal trait '_volume_property' used elsewhere in this file.
        # Use 'scalar_opacity_unit_distance' to control transparency for volumes.
        # A value of 4.0 - 10.0 usually makes the volume transparent enough to see the graph.
        vol._volume_property.scalar_opacity_unit_distance = 4.0
        
        # Plot Edges
        pos = nx.get_node_attributes(G, 'pos')
        for edge in G.edges():
            p1, p2 = pos[edge[0]], pos[edge[1]]
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        color=(1, 1, 1), tube_radius=0.15)
            
        # Color-code Junctions: Green = Bifurcation (Deg 3), Red = Crossing (Deg 4+)
        degrees = dict(G.degree())
        bifurcations = np.array([pos[n] for n, d in degrees.items() if d == 3])
        crossings = np.array([pos[n] for n, d in degrees.items() if d >= 4])
        
        if bifurcations.size: 
            mlab.points3d(bifurcations[:,0], bifurcations[:,1], 
                          bifurcations[:,2], color=(0,1,0), scale_factor=1.2)
        if crossings.size: 
            mlab.points3d(crossings[:,0], crossings[:,1], 
                          crossings[:,2], color=(1,0,0), scale_factor=1.2)
        
        mlab.title(title)
        mlab.show()

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