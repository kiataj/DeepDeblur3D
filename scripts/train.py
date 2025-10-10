# scripts/train.py

from deblur3d.losses import DeblurLoss, ssim3d, freq_l1
from deblur3d.models import UNet3D_Residual
from deblur3d.data import TiffVolumeDataset
from deblur3d.transforms import GaussianIsoBlurCPUTransform

import os, torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# ----- loss -----
criterion = DeblurLoss(
    w_l1=0.8, w_ssim=0.2, w_freq=0.05, id_weight=0.1,
    use_relative_freq=False,  # set True to use the relative spectrum loss
    freq_alpha=1.0
)

# --- config ---
# root = r"U:\users\taki\DeBlur"     # folder with .tif volumes (mask_* ignored)
root = r"U:\users\taki\DeBlur"
patch_size   = (64, 256, 256)
batch_size   = 5
num_workers  = 4                   # safe: blur runs on CPU
val_frac     = 0.15                # 15% of volumes for validation
seed         = 42

# --- CPU-only isotropic Gaussian blur (with optional noise) ---
blur_tf = GaussianIsoBlurCPUTransform(
    fwhm_range=(8, 10),        # FWHM in voxels (sampled each call)
    radius_mult=3,
    add_noise=True,
    poisson_gain_range=(300, 900),
    read_noise_std_range=(0.006, 0.015),
)

# --- dataset over volumes (each __getitem__ returns one random patch) ---
ds_full = TiffVolumeDataset(
    root=root,
    patch_size=patch_size,
    blur_transform=blur_tf,
    augment=True,                 # flips/rot90; keep False if you need strict comparability
)

n = len(ds_full)
if n == 0:
    raise RuntimeError("No .tif volumes found under root (excluding files starting with 'mask_').")

# --- robust split (per-volume; avoids empty splits) ---
g = torch.Generator().manual_seed(seed)
if n == 1:
    ds_train, ds_val = ds_full, None
    print("Only 1 volume: train only (no validation).")
elif n == 2:
    ds_train, ds_val = random_split(ds_full, [1, 1], generator=g)
else:
    n_val = max(1, int(round(val_frac * n)))
    n_train = n - n_val
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=g)

# --- loaders ---
loader_train = DataLoader(
    ds_train,
    batch_size=min(batch_size, len(ds_train)),
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
)

loader_val = None
if ds_val is not None:
    loader_val = DataLoader(
        ds_val,
        batch_size=min(batch_size, len(ds_val)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

print(f"Volumes: total={n}"
      + ("" if ds_val is None else f", train={len(ds_train)}, val={len(ds_val)}"))

# ----- training -----
device = torch.device("cuda")
net = UNet3D_Residual(in_ch=1, base=24, levels=4).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.99))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)  # epochs

use_amp = torch.cuda.is_available()
scaler  = GradScaler(enabled=use_amp)

def to_ch(x): return x.unsqueeze(1)  # (B,1,D,H,W)

best_psnr, best_path = -1, "deblur3d_unet.pt"
for epoch in range(1, 5):
    net.train()
    tr_loss = 0.0
    for sharp, blurred in loader_train:
        sharp   = to_ch(sharp).to(device, non_blocking=True)
        blurred = to_ch(blurred).to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred = net(blurred)
            loss = criterion(pred, sharp, blurred)

        # PT 1.12 AMP sequence
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                                     # <-- instead of inf_check_and_unscale_
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        tr_loss += loss.item() * sharp.size(0)

    sched.step()

    # validation
    if 'loader_val' in globals() and loader_val is not None:
        net.eval(); psnr_sum, nvox = 0.0, 0
        with torch.no_grad():
            for sharp, blurred in loader_val:
                sharp   = to_ch(sharp).to(device, non_blocking=True)
                blurred = to_ch(blurred).to(device, non_blocking=True)
                pred = net(blurred)
                mse  = F.mse_loss(pred, sharp, reduction='none').mean(dim=(1,2,3,4))
                psnr = 10 * torch.log10(1.0 / (mse + 1e-12))
                psnr_sum += psnr.sum().item()
                nvox += sharp.size(0)
        psnr_epoch = psnr_sum / max(nvox, 1)
        print(f"Epoch {epoch:03d} | train {tr_loss/len(loader_train.dataset):.4f} | PSNR {psnr_epoch:.2f} dB")

        if psnr_epoch > best_psnr:
            best_psnr = psnr_epoch
            torch.save({"epoch": epoch, "state_dict": net.state_dict()}, best_path)
            print(f"  ↳ saved: {best_path} (PSNR {best_psnr:.2f} dB)")
    else:
        print(f"Epoch {epoch:03d} | train {tr_loss/len(loader_train.dataset):.4f} | no val set")
