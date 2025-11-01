# app/inference.py
import numpy as np, torch, scipy.ndimage
from typing import Tuple

TARGET_SHAPE = (128, 128, 128)

def _normalize_01(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = float(vol.min()), float(vol.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32, copy=False)

def _resize(vol: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    v = np.ascontiguousarray(vol, dtype=np.float32)
    factors = [target_shape[i] / v.shape[i] for i in range(3)]
    out = scipy.ndimage.zoom(v, factors, order=1, mode="nearest", prefilter=False)
    return out.astype(np.float32, copy=False)


def preprocess_pair(pre_vol: np.ndarray, post_vol: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    pre  = _resize(_normalize_01(pre_vol))
    post = _resize(_normalize_01(post_vol))
    pre_t  = torch.from_numpy(pre).unsqueeze(0).unsqueeze(0).float()   # (1,1,D,H,W)
    post_t = torch.from_numpy(post).unsqueeze(0).unsqueeze(0).float()
    return pre_t, post_t

def summarize_deformation(flow: np.ndarray) -> Tuple[float, float]:
    if flow.ndim == 5: flow = flow[0]  # (3,D,H,W)
    mag = np.linalg.norm(flow, axis=0)
    return float(mag.mean()), float(mag.max())

@torch.no_grad()
def run_model(pre_vol: np.ndarray, post_vol: np.ndarray, device: torch.device, model):
    pre_t, post_t = preprocess_pair(pre_vol, post_vol)
    pre_t = pre_t.to(device, non_blocking=True)
    post_t = post_t.to(device, non_blocking=True)
    model.eval()
    flow_t = model(pre_t, post_t)                 # (1,3,128,128,128)
    flow = flow_t.detach().cpu().numpy()          # keep one copy on CPU
    # free GPU/CPU tensors asap
    del flow_t, pre_t, post_t; gc.collect()
    mean_mm, max_mm = summarize_deformation(flow)
    return mean_mm, max_mm, flow[0]               # (3,128,128,128)
