# src/deblur3d/app/_workers.py
from typing import Callable, Dict, Any, Optional
import numpy as np
import torch
from napari.qt.threading import thread_worker

def _pick_device(requested: str) -> str:
    return "cuda" if (requested == "cuda" and torch.cuda.is_available()) else "cpu"

def make_infer_worker(run_infer: Callable,
                      vol_f32_01: np.ndarray,
                      device: str,
                      extra_kwargs: Optional[Dict[str, Any]] = None):
    """
    Wrap a blocking inference function into a napari worker.

    Expected signature of run_infer:
        run_infer(vol_f32_01, device='cuda'|'cpu', **kwargs) -> np.ndarray
    If run_infer accepts a 'progress' kw, you can pass a 0..1 callback here.
    """
    extra_kwargs = extra_kwargs or {}
    dev = _pick_device(device)

    @thread_worker
    def _worker():
        kwargs = dict(device=dev, **extra_kwargs)
        if "progress" in run_infer.__code__.co_varnames:
            kwargs["progress"] = lambda p: p  # noop; wire up if you add a UI progress bar
        return run_infer(vol_f32_01, **kwargs)

    return _worker()
