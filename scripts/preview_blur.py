import argparse, tifffile as tiff, torch, numpy as np
from deblur3d.transforms import LorentzLaminoBlurTransform
from deblur3d.viz.napari_tools import show_pair_2d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif_path")
    ap.add_argument("--xy-mode", default="none", choices=["none","gaussian","lorentz"])
    ap.add_argument("--fwhm-xy", type=float, default=0.0)
    ap.add_argument("--fwhmL-z-min", type=float, default=2.0)
    ap.add_argument("--fwhmL-z-max", type=float, default=16.0)
    args = ap.parse_args()

    vol = tiff.imread(args.tif_path)
    if vol.ndim == 2: vol = vol[None,...]
    v = torch.from_numpy(vol.astype(np.float32))
    if np.issubdtype(vol.dtype, np.integer):
        v = v / np.iinfo(vol.dtype).max

    tf = LorentzLaminoBlurTransform(
        device="cpu",
        xy_mode=args.xy_mode,
        fwhm_xy=args.fwhm_xy,
        fwhmL_z_min=args.fwhmL_z_min,
        fwhmL_z_max=args.fwhmL_z_max,
    )
    y = tf(v)
    show_pair_2d(v.numpy(), y.numpy())

if __name__ == "__main__":
    main()
