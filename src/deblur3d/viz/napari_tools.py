# src/deblur3d/viz/napari_tools.py
import napari, numpy as np

__all__ = ["show_pair_2d", "show_pair_3d"]

def show_pair_2d(sharp, blurred, name_sharp="Original", name_blur="Blurred"):
    v = napari.Viewer(ndisplay=2)
    Lb = v.add_image(np.asarray(blurred), name=name_blur, colormap="gray")
    Lo = v.add_image(np.asarray(sharp),   name=name_sharp, colormap="gray")
    Lo.contrast_limits = Lb.contrast_limits
    napari.run()

def show_pair_3d(sharp, blurred, spacing=(1,1,1)):
    v = napari.Viewer(ndisplay=3)
    Lb = v.add_image(np.asarray(blurred), name="Blurred", colormap="gray",
                     rendering="attenuated_mip", scale=spacing, opacity=0.7)
    Lo = v.add_image(np.asarray(sharp),   name="Original", colormap="gray",
                     rendering="attenuated_mip", scale=spacing, opacity=0.7)
    Lo.contrast_limits = Lb.contrast_limits
    napari.run()
