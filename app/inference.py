## app/inference.py

import torch
import numpy as np
from typing import Tuple, Optional

# Preprocess volumes (normalize, resize/crop/pad as your UNet expects)

def preprocess(pre_vol: np.ndarray, post_vol: np.ndarray) -> torch.Tensor:
    # Example: z-score normalize per volume
    def norm(v):
        v = v.astype(np.float32)
        m, s = v.mean(), v.std() + 1e-6
        return (v - m) / s
    pre = norm(pre_vol)
    post = norm(post_vol)
    # Stack as channels: (C=2, Z, Y, X)
    x = np.stack([pre, post], axis=0)
    return torch.from_numpy(x)[None, ...]  # add batch dim

# Postprocess model output into useful stats

def summarize_deformation(flow: np.ndarray) -> Tuple[float, float]:
    """
    flow: (3, Z, Y, X) or (Z,Y,X,3) displacement in mm/voxel-units.
    We compute magnitude per voxel and return mean/max.
    """
    if flow.shape[0] == 3:
        mag = np.linalg.norm(flow, axis=0)
    else:
        mag = np.linalg.norm(flow, axis=-1)
    return float(mag.mean()), float(mag.max())

@torch.no_grad()
def run_model(pre_vol: np.ndarray, post_vol: np.ndarray, device: torch.device, model) -> Tuple[float, float, np.ndarray]:
    x = preprocess(pre_vol, post_vol).to(device)
    model.eval()
    out = model(x)  # expected shape: (B, 3, Z, Y, X) displacement field
    flow = out[0].detach().cpu().numpy()
    mean_mm, max_mm = summarize_deformation(flow)
    return mean_mm, max_mm, flow

# Tag-file ground truth summary

def summarize_tag(pre_pts: np.ndarray, post_pts: np.ndarray) -> Tuple[float, float]:
    shifts = np.linalg.norm(pre_pts - post_pts, axis=1)
    return float(shifts.mean()), float(shifts.max())
