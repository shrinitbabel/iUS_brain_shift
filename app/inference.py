# app/inference.py
import numpy as np
import torch
import scipy.ndimage
from typing import Tuple

TARGET_SHAPE = (128, 128, 128)

def _normalize_01(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)

def _resize(vol: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    factors = [target_shape[i] / vol.shape[i] for i in range(3)]
    # order=1 (trilinear) like your training pipeline
    return scipy.ndimage.zoom(vol, factors, order=1).astype(np.float32)

def preprocess_pair(pre_vol: np.ndarray, post_vol: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate modules/dataset.py: normalize [0,1], resize to 128^3,
    then return tensors shaped (B=1, C=1, D, H, W)."""
    pre = _resize(_normalize_01(pre_vol))
    post = _resize(_normalize_01(post_vol))
    pre_t = torch.from_numpy(pre)[None, None, ...]   # (1,1,D,H,W)
    post_t = torch.from_numpy(post)[None, None, ...] # (1,1,D,H,W)
    return pre_t, post_t

def summarize_deformation(flow: np.ndarray) -> Tuple[float, float]:
    """flow shape (3, D, H, W) or (B,3,D,H,W). Return mean/max magnitude."""
    if flow.ndim == 5:
        flow = flow[0]
    # (3, D, H, W) -> magnitude (D,H,W)
    mag = np.linalg.norm(flow, axis=0)
    return float(mag.mean()), float(mag.max())

@torch.no_grad()
def run_model(
    pre_vol: np.ndarray,
    post_vol: np.ndarray,
    device: torch.device,
    model
) -> Tuple[float, float, np.ndarray]:
    pre_t, post_t = preprocess_pair(pre_vol, post_vol)
    pre_t = pre_t.to(device)
    post_t = post_t.to(device)
    model.eval()
    # Your UNet3D.forward(self, pre, post) expects two tensors
    flow_t = model(pre_t, post_t)                 # (1,3,128,128,128)
    flow = flow_t.detach().cpu().numpy()
    mean_mm, max_mm = summarize_deformation(flow)
    return mean_mm, max_mm, flow[0]               # return (3,128,128,128)
