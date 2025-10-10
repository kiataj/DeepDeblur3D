# src/deblur3d/infer/tiled.py
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["deblur_volume_tiled"]

@torch.no_grad()
def deblur_volume_tiled(
    net: torch.nn.Module,
    vol: Union[np.ndarray, torch.Tensor],
    tile: Tuple[int, int, int] = (96, 128, 128),
    overlap: Tuple[int, int, int] = (24, 32, 32),
    device: str = "cuda",
    use_amp: bool = False,              # PT1.12 + InstanceNorm: keep False unless you use GroupNorm
    pad_mode: str = "reflect",
    clamp01: bool = True,
) -> np.ndarray:
    """
    Tiled 3D inference with Hann blending.

    Args:
        net:   3D residual model expecting (N,1,D,H,W) in [0,1].
        vol:   (D,H,W) float32 numpy or torch tensor in [0,1].
        tile:  (Dz, Dy, Dx) tile size.
        overlap: (Oz, Oy, Ox) overlap for blending.
        device: "cuda" or "cpu".
        use_amp: enable CUDA autocast (set False for InstanceNorm on PT1.12).
        pad_mode: pad mode for edge tiles ("reflect" | "replicate" | "constant").
        clamp01: clamp output to [0,1] before returning.

    Returns:
        (D,H,W) float32 numpy array.
    """
    net.eval()
    device_t = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # normalize input → torch on device
    if isinstance(vol, np.ndarray):
        v = torch.from_numpy(vol)
    else:
        v = vol
    assert v.dim() == 3, "vol must be (D,H,W)"
    v = v.to(device_t, dtype=torch.float32, non_blocking=True).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    D, H, W = int(v.shape[2]), int(v.shape[3]), int(v.shape[4])
    td, th, tw = map(int, tile)
    od, oh, ow = map(int, overlap)

    out = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device=device_t)
    wei = torch.zeros_like(out)

    def _hann(sz: int, ov: int):
        if sz <= 1:
            return torch.ones(1, device=device_t)
        g = torch.hann_window(sz, periodic=False, device=device_t)
        return g.clamp_min(1e-6) if ov > 0 else g

    wz, wy, wx = _hann(td, od), _hann(th, oh), _hann(tw, ow)
    w3 = wz.view(1, 1, td, 1, 1) * wy.view(1, 1, 1, th, 1) * wx.view(1, 1, 1, 1, tw)

    step_z = td - od if td < D else D
    step_y = th - oh if th < H else H
    step_x = tw - ow if tw < W else W

    # Import here to avoid requiring CUDA on CPUs
    from torch.cuda.amp import autocast

    for z in range(0, max(1, D - td + 1), step_z):
        for y in range(0, max(1, H - th + 1), step_y):
            for x in range(0, max(1, W - tw + 1), step_x):
                patch = v[:, :, z:z + td, y:y + th, x:x + tw]
                # pad to tile if at the border
                if patch.shape[2:] != (td, th, tw):
                    padz = td - patch.shape[2]
                    pady = th - patch.shape[3]
                    padx = tw - patch.shape[4]
                    patch = F.pad(patch, (0, padx, 0, pady, 0, padz), mode=pad_mode)

                with autocast(enabled=(use_amp and device_t.type == "cuda")):
                    pred = net(patch)

                # crop back to valid region for this tile
                pd = min(td, D - z); ph = min(th, H - y); pw = min(tw, W - x)
                pred = pred[:, :, :pd, :ph, :pw]
                w = w3[:, :, :pd, :ph, :pw]

                out[:, :, z:z + pd, y:y + ph, x:x + pw] += pred * w
                wei[:, :, z:z + pd, y:y + ph, x:x + pw] += w

    res = (out / (wei + 1e-8)).squeeze(0).squeeze(0)
    if clamp01:
        res = res.clamp(0, 1)
    return res.detach().cpu().numpy().astype(np.float32)
