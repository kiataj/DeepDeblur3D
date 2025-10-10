import torch
from deblur3d.transforms import lamino_blur_lorentz

def test_shapes_and_monotone_tails():
    D,H,W = 64,64,64
    x = torch.zeros(D,H,W, dtype=torch.float32)
    x[D//2,H//2,W//2] = 1.0  # impulse
    y = lamino_blur_lorentz(x,
        xy_mode="none", fwhmL_z_min=2.0, fwhmL_z_max=10.0, p=1.5, K=8)
    assert y.shape == x.shape
    center = y[D//2,H//2,W//2].item()
    off    = y[D//2 + D//4,H//2,W//2].item()
    assert off < center  # peak higher at focal plane
