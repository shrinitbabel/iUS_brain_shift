from __future__ import annotations
import io
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from pathlib import Path
from typing import Tuple


from .settings import TARGET_SHAPE




def load_3d_volume(path: Path) -> np.ndarray:
    """Load .mnc, .nii, or .nii.gz into float32 numpy array (D,H,W)."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    # Ensure channel-last 3D shape
    if data.ndim == 4:
        # drop channels if present; keep first
        data = data[..., 0]
    return data



def minmax01(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol - vmin) / (vmax - vmin)
    return out.astype(np.float32)




def resize3d(vol: np.ndarray, target: Tuple[int,int,int]=TARGET_SHAPE, order: int = 1) -> np.ndarray:
    factors = [t / s for t, s in zip(target, vol.shape[:3])]
    return ndi.zoom(vol, factors, order=order).astype(np.float32)




def save_numpy(arr: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)
    return path




def parse_tag_file(tag_path: Path):
    pre_pts, post_pts = [], []
    with open(tag_path, 'r') as f:
        lines = f.readlines()
    points_started = False
    for ln in lines:
        line = ln.strip()
        if line.startswith("Points"):
            points_started = True
            continue
        if not points_started or not line or ";" in line:
            continue
        try:
            vals = [float(x) for x in line.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
            if len(vals) == 6:
                pre_pts.append(vals[:3])
                post_pts.append(vals[3:])
        except Exception:
            continue
    import numpy as np
    return np.array(pre_pts, dtype=np.float32), np.array(post_pts, dtype=np.float32)