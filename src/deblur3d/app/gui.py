# src/deblur3d/app/gui.py
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Optional
from inspect import signature

import numpy as np
import torch
import tifffile as tiff

from magicgui import magicgui
from napari import Viewer, run
from napari.layers import Image as NapariImage
from napari.utils.notifications import show_info, show_warning, show_error

from ._workers import make_infer_worker

# ---- Your project imports ----
from deblur3d.data.io import read_volume_float01
from deblur3d.infer.tiled import deblur_volume_tiled
from deblur3d.models import UNet3D_Residual, ControlledUNet3D

# Default checkpoint (auto-used; can be overridden if you want)
DEFAULT_WEIGHTS = Path(r"C:\Users\taki\DeepDeBlur3D\checkpoints\deblur3d_unet_best.pt")


# ----------------- I/O + normalization -----------------
def _normalize_float01_like_io(vol: np.ndarray) -> np.ndarray:
    """
    Mirror deblur3d.data.io.read_volume_float01:
      - ensure 3D (Z,Y,X); if 2D, add Z=1
      - convert to float32
      - scale to [0,1] for integer inputs
      - for float inputs outside ~[0,1.5], percentile map 1–99.9% -> [0,1]
      - clamp to [0,1]
    """
    x = np.asarray(vol)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D or 2D array; got shape {x.shape}")
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / max(np.iinfo(x.dtype).max, 1)
    else:
        x = x.astype(np.float32, copy=False)
        vmin, vmax = float(x.min()), float(x.max())
        if vmin < 0 or vmax > 1.5:
            lo, hi = np.percentile(x, [1.0, 99.9])
            x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            x = np.clip(x, 0, 1)
    return x


def _read_dir_tif_stack(dirpath: Path) -> np.ndarray:
    """Read a directory of TIF slices into a (Z,Y,X) volume, sorted by filename, then normalize."""
    files: List[Path] = sorted(
        [p for p in dirpath.iterdir() if p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: p.name,
    )
    if not files:
        raise ValueError(f"No .tif/.tiff files found in: {dirpath}")
    vol = np.stack([tiff.imread(str(p)) for p in files], axis=0)
    return _normalize_float01_like_io(vol)


def read_volume_auto(path: Path) -> np.ndarray:
    """File/folder loader with normalization → (Z,Y,X) float32 in [0,1]."""
    if path.is_dir():
        return _read_dir_tif_stack(path)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return read_volume_float01(str(path))
    if ext == ".npy":
        arr = np.load(str(path))
        return _normalize_float01_like_io(arr)
    raise ValueError(f"Unsupported input: {path}")


# ----------------- Model cache/loader -----------------
def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    k0 = next(iter(state_dict))
    if isinstance(k0, str) and k0.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _build_model(Model, base: int, levels: int):
    """Robustly construct UNet3D_Residual across signature variants."""
    sig = signature(Model)
    kwargs = {}
    # channels
    if "in_ch" in sig.parameters:
        kwargs["in_ch"] = 1
    if "out_ch" in sig.parameters:
        kwargs["out_ch"] = 1
    if "n_channels" in sig.parameters:
        kwargs["n_channels"] = 1
    if "n_classes" in sig.parameters:
        kwargs["n_classes"] = 1
    # base + depth
    if "base_ch" in sig.parameters:
        kwargs["base_ch"] = base
    elif "base" in sig.parameters:
        kwargs["base"] = base
    if "levels" in sig.parameters:
        kwargs["levels"] = levels

    try:
        return Model(**kwargs)
    except TypeError:
        pass
    try:
        return Model(base, levels, 1, 1)
    except TypeError:
        pass
    try:
        return Model(base, levels)
    except TypeError:
        pass
    return Model()


@lru_cache(maxsize=1)
def _cached_base_model(weights_path: str, device: str):
    """Infer base & levels from checkpoint; build model to match; strict load."""
    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    ckpt = torch.load(weights_path, map_location=dev)
    sd = _strip_module_prefix(ckpt.get("state_dict", ckpt))

    # infer base
    base: Optional[int] = None
    if "down.0.weight" in sd and sd["down.0.weight"].ndim >= 2:
        base = int(sd["down.0.weight"].shape[0])
    if base is None and "out.weight" in sd and sd["out.weight"].ndim >= 2:
        base = int(sd["out.weight"].shape[1])
    if base is None:
        candidates = [int(v.shape[0]) for k, v in sd.items() if ".conv1.weight" in k and v.ndim == 5]
        if candidates:
            base = min(candidates)
    if base is None:
        base = 16

    # infer levels
    down_ids = []
    for k in sd.keys():
        if k.startswith("down.") and k.endswith(".weight"):
            try:
                down_ids.append(int(k.split(".")[1]))
            except Exception:
                pass
    levels: Optional[int] = (max(down_ids) + 1) if down_ids else None
    if levels is None:
        enc_ids = []
        for k in sd.keys():
            if k.startswith("enc."):
                try:
                    enc_ids.append(int(k.split(".")[1]))
                except Exception:
                    pass
        if enc_ids:
            levels = max(enc_ids) + 1
    if levels is None:
        levels = 4

    base_model = _build_model(UNet3D_Residual, base, levels)
    base_model.load_state_dict(sd, strict=True)
    base_model.to(dev).eval()
    return base_model, dev


# ----------------- Controlled wrappers (direct vs residual) -----------------
class _ParamNetDirect(torch.nn.Module):
    """Checkpoint outputs y_hat directly; ControlledUNet3D applies residual-style controls."""
    def __init__(self, base: torch.nn.Module, clamp01: bool,
                 strength: float, hp_sigma: float, hp_gain: float, lp_gain: float):
        super().__init__()
        self.ctrl = ControlledUNet3D(base, clamp01=clamp01)
        self.strength = float(strength); self.hp_sigma = float(hp_sigma)
        self.hp_gain = float(hp_gain);    self.lp_gain = float(lp_gain)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctrl(x, strength=self.strength,
                         hp_sigma=self.hp_sigma, hp_gain=self.hp_gain, lp_gain=self.lp_gain)


class _ParamNetResidual(torch.nn.Module):
    """Checkpoint outputs residual r_hat; we form y = x + r_hat with frequency controls."""
    def __init__(self, base: torch.nn.Module, clamp01: bool,
                 strength: float, hp_sigma: float, hp_gain: float, lp_gain: float):
        super().__init__()
        self.base = base.eval()
        self.clamp01 = clamp01
        self.strength = float(strength); self.hp_sigma = float(hp_sigma)
        self.hp_gain = float(hp_gain);    self.lp_gain = float(lp_gain)

    @torch.no_grad()
    def _gauss1d(self, sigma: float, device, dtype=torch.float32, radius_mult: float = 3.0):
        sigma = max(1e-6, float(sigma))
        r = int(np.ceil(radius_mult * sigma)); r = max(r, 1)
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-(x * x) / (2.0 * sigma * sigma)); k = k / (k.sum() + 1e-12)
        return k, r

    @torch.no_grad()
    def _gaussian_blur3d(self, x: torch.Tensor, sigma: float, pad_mode: str = "replicate"):
        if sigma <= 0: return x
        k1d, r = self._gauss1d(sigma, x.device, x.dtype)
        kz = k1d.view(1, 1, -1, 1, 1); ky = k1d.view(1, 1, 1, -1, 1); kx = k1d.view(1, 1, 1, 1, -1)
        F = torch.nn.functional
        y = F.conv3d(F.pad(x, (0,0,0,0,r,r), mode=pad_mode), kz, padding=0, groups=x.shape[1])
        y = F.conv3d(F.pad(y, (0,0,r,r,0,0), mode=pad_mode), ky, padding=0, groups=x.shape[1])
        y = F.conv3d(F.pad(y, (r,r,0,0,0,0), mode=pad_mode), kx, padding=0, groups=x.shape[1])
        return y

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.base(x)
        if self.hp_sigma > 0:
            r_lp = self._gaussian_blur3d(r, sigma=self.hp_sigma)
            r_hp = r - r_lp
            r = self.lp_gain * r_lp + self.hp_gain * r_hp
        y = x + self.strength * r
        return y.clamp(0, 1) if self.clamp01 else y


def _make_param_net(base_model: torch.nn.Module, mode: str,
                    clamp01: bool, strength: float,
                    hp_sigma: float, hp_gain: float, lp_gain: float) -> torch.nn.Module:
    if mode == "residual":
        return _ParamNetResidual(base_model, clamp01, strength, hp_sigma, hp_gain, lp_gain)
    return _ParamNetDirect(base_model, clamp01, strength, hp_sigma, hp_gain, lp_gain)


# ----------------- Inference wrapper -----------------
def run_infer_bound(
    vol_f32_01: np.ndarray,
    *,
    device: str,
    weights_path: str,
    tile: Tuple[int, int, int],
    overlap: Tuple[int, int, int],
    use_amp: bool = False,
    pad_mode: str = "reflect",
    clamp01: bool = True,
    strength: float = 1.0,
    hp_sigma: float = 0.0,
    hp_gain: float = 1.0,
    lp_gain: float = 1.0,
    model_output_mode: str = "direct",  # "direct" or "residual"
) -> np.ndarray:
    base, dev = _cached_base_model(weights_path, device)
    vol_f32_01 = _normalize_float01_like_io(np.asarray(vol_f32_01))
    param_net = _make_param_net(base, model_output_mode, clamp01, strength, hp_sigma, hp_gain, lp_gain)
    param_net = param_net.to(dev).eval()

    pred = deblur_volume_tiled(
        net=param_net,
        vol=vol_f32_01,
        tile=tile, overlap=overlap,
        device=dev, use_amp=use_amp, pad_mode=pad_mode, clamp01=clamp01,
    )
    return pred.astype(np.float32, copy=False)


# ----------------- Napari GUI -----------------
def build_viewer() -> Viewer:
    v = Viewer(title="deblur3d — Inference")
    v.dims.ndisplay = 2       # default 2D
    v.grid.enabled = True

    state = {
        "weights_path": str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.is_file() else "",
        "run_idx": 1,  # unique prediction name per run
    }

    # Normalize & style an input layer (used by both picker and DnD)
    def _prepare_input_layer(layer: NapariImage):
        data = np.asarray(layer.data)
        if data.ndim not in (2, 3):
            return False
        try:
            norm = _normalize_float01_like_io(data)
        except Exception:
            return False
        layer.data = norm.astype(np.float32, copy=False)
        layer.colormap = "gray"
        layer.contrast_limits = (0.0, 1.0)
        v.dims.ndisplay = 2
        v.grid.enabled = True
        return True

    # Use ACTIVE (blue-selected) layer as input. Enable Run if active is valid.
    def _update_run_enabled_from_active():
        active = v.layers.selection.active
        enable = isinstance(active, NapariImage) and getattr(active.data, "ndim", 0) in (2, 3)
        infer_w.enabled = bool(enable)

    # Drag-and-drop: normalize dropped image and set as active
    def _on_layer_added(event):
        layer = event.value
        if isinstance(layer, NapariImage) and _prepare_input_layer(layer):
            v.layers.selection.active = layer
            _update_run_enabled_from_active()
            show_info(f"Input set from drag-and-drop: {layer.name}")

    v.layers.events.inserted.connect(_on_layer_added)
    v.layers.selection.events.active.connect(lambda e: _update_run_enabled_from_active())

    # --- Open volume (file or folder), auto-call on selection ---
    @magicgui(
        auto_call=True,
        path={
            "label": "Input file OR folder",
            "widget_type": "FileEdit",
            "mode": "r",
            "filter": "*.tif;*.tiff;*.npy",
            "nullable": True,
        },
    )
    def open_w(path: Path | None = None):
        if not path:
            return
        p = Path(path)
        try:
            vol = read_volume_auto(p)
        except Exception as e:
            show_error(f"Open failed: {e}")
            return

        vol = _normalize_float01_like_io(vol)
        lyr = v.add_image(vol, name=f"input: {p.name}", colormap="gray", contrast_limits=(0.0, 1.0))
        lyr.grid_position = (0, 0)
        v.layers.selection.active = lyr
        _update_run_enabled_from_active()
        show_info(f"Loaded {p} | shape={vol.shape} dtype={vol.dtype}")

    # --- Optional override for weights (UI shown only if default missing) ---
    @magicgui(
        auto_call=True,
        path={
            "label": "Weights .pt/.pth",
            "widget_type": "FileEdit",
            "mode": "r",
            "filter": "*.pt;*.pth",
            "nullable": True,
        },
    )
    def weights_w(path: Path | None = None):
        if not path:
            return
        if not Path(path).is_file():
            show_warning(f"File not found: {path}")
            return
        state["weights_path"] = str(path)
        show_info(f"Weights set: {path}")

    # --- Run Filter: uses the CURRENT ACTIVE layer as input ---
    @magicgui(
        call_button="Run Filter",
        device={"choices": ["cuda", "cpu"]},
        model_output_mode={"label": "Model output", "choices": ["direct", "residual"], "value": "direct"},
        tile_x={"label": "Tile X", "min": 16, "max": 512, "step": 16, "value": 256},
        tile_y={"label": "Tile Y", "min": 16, "max": 512, "step": 16, "value": 256},
        tile_z={"label": "Tile Z", "min": 8,  "max": 128, "step": 8,  "value": 64},
        ov_x={"label": "Overlap X", "min": 0, "max": 256, "step": 8, "value": 128},
        ov_y={"label": "Overlap Y", "min": 0, "max": 256, "step": 8, "value": 128},
        ov_z={"label": "Overlap Z", "min": 0, "max": 64,  "step": 4, "value": 32},
        use_amp={"label": "Use AMP", "value": False},
        pad_mode={"choices": ["reflect", "replicate", "constant"], "value": "reflect"},
        clamp01={"label": "Clamp [0,1]", "value": True},
        # Controlled params
        strength={"label": "Strength", "min": 0.0, "max": 3.0, "step": 0.1, "value": 1.0},
        hp_sigma={"label": "HP Sigma (vox)", "min": 0.0, "max": 8.0, "step": 0.1, "value": 0.0},
        hp_gain={"label": "HP Gain", "min": 0.0, "max": 4.0, "step": 0.1, "value": 1.0},
        lp_gain={"label": "LP Gain", "min": 0.0, "max": 4.0, "step": 0.1, "value": 1.0},
    )
    def infer_w(
        device: str = "cuda",
        model_output_mode: str = "direct",  # "direct" or "residual"
        tile_x: int = 256, tile_y: int = 256, tile_z: int = 64,
        ov_x: int = 128,  ov_y: int = 128,  ov_z: int = 32,
        use_amp: bool = False,
        pad_mode: str = "reflect",
        clamp01: bool = True,
        strength: float = 1.0,
        hp_sigma: float = 0.0,
        hp_gain: float = 1.0,
        lp_gain: float = 1.0,
    ):
        active = v.layers.selection.active
        if not (isinstance(active, NapariImage) and getattr(active.data, "ndim", 0) in (2, 3)):
            show_warning("Select an image layer (2D/3D) as input.")
            return

        weights_path = state.get("weights_path", str(DEFAULT_WEIGHTS))
        if not weights_path:
            show_warning(f"Default weights not found: {DEFAULT_WEIGHTS}")
            return
        if not os.path.isfile(weights_path):
            show_warning(f"Weights file not found: {weights_path}")
            return

        # Normalize the ACTIVE layer data just before inference
        vol = _normalize_float01_like_io(np.asarray(active.data))
        tile = (tile_z, tile_y, tile_x)   # (Z,Y,X)
        overlap = (ov_z, ov_y, ov_x)

        infer_w.enabled = False
        run_id = state["run_idx"]
        show_info(f"Starting inference on active layer '{active.name}' …")
        start = time.time()

        worker = make_infer_worker(
            lambda v_, device=None, progress=None: run_infer_bound(
                v_,
                device=device,
                weights_path=weights_path,
                tile=tile,
                overlap=overlap,
                use_amp=use_amp,
                pad_mode=pad_mode,
                clamp01=clamp01,
                strength=strength,
                hp_sigma=hp_sigma,
                hp_gain=hp_gain,
                lp_gain=lp_gain,
                model_output_mode=model_output_mode,
            ),
            vol, device=device, extra_kwargs={}
        )

        def on_return(pred: np.ndarray):
            dt = time.time() - start
            layer_name = f"prediction_{run_id}"
            lyr = v.add_image(
                pred, name=layer_name, colormap="magenta",
                blending="additive", opacity=0.7, contrast_limits=(0.0, 1.0)
            )
            lyr.grid_position = (0, 1)
            show_info(f"Inference #{run_id} done in {dt:.2f}s | shape={pred.shape}")
            v.grid.enabled = True
            state["run_idx"] = run_id + 1
            infer_w.enabled = True

        def on_error(e):
            infer_w.enabled = True
            show_error(f"Inference error: {e}")

        worker.returned.connect(on_return)
        worker.errored.connect(on_error)
        worker.start()

    # Initially disabled; will enable when a valid layer is active
    infer_w.enabled = False

    # Dock widgets
    v.window.add_dock_widget(open_w, area="right")
    if not DEFAULT_WEIGHTS.is_file():
        v.window.add_dock_widget(weights_w, area="right")
    v.window.add_dock_widget(infer_w, area="right")
    return v


def main():
    v = build_viewer()
    run()


if __name__ == "__main__":
    main()
