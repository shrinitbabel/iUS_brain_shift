from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from .settings import MODEL_PATH, TARGET_SHAPE
from .utils import load_3d_volume, minmax01, resize3d


# Import your model from modules/unet3d.py
from modules.unet3d import UNet3D

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[torch.nn.Module] = None

def get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        model = UNet3D(in_channels=2, out_channels=3, base_filters=32)
        state = torch.load(MODEL_PATH, map_location=_device)
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _model = model
    return _model

def preprocess_pair(pre_path: Path, post_path: Path) -> Dict[str, Any]:
    pre = load_3d_volume(pre_path)
    post = load_3d_volume(post_path)


    pre_rs = resize3d(minmax01(pre), TARGET_SHAPE)
    post_rs = resize3d(minmax01(post), TARGET_SHAPE)


    pre_t = torch.from_numpy(pre_rs).unsqueeze(0).unsqueeze(0).float().to(_device) # (1,1,D,H,W)
    post_t = torch.from_numpy(post_rs).unsqueeze(0).unsqueeze(0).float().to(_device)
    return {"pre_np": pre_rs, "post_np": post_rs, "pre_t": pre_t, "post_t": post_t}




def predict_flow(pre_t: torch.Tensor, post_t: torch.Tensor) -> torch.Tensor:
    model = get_model()
    with torch.no_grad():
        flow = model(pre_t, post_t) # (1,3,D,H,W)
    return flow



def warp_image(pre_t: torch.Tensor, flow: torch.Tensor) -> np.ndarray:
    b, c, d, h, w = pre_t.shape
    zz = torch.linspace(-1, 1, d, device=pre_t.device)
    yy = torch.linspace(-1, 1, h, device=pre_t.device)
    xx = torch.linspace(-1, 1, w, device=pre_t.device)
    grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=-1) # (D,H,W,3)
    disp = (flow.squeeze(0).permute(1, 2, 3, 0)) * 1e-4
    warped_grid = torch.clamp(grid + disp, -1, 1)
    warped = F.grid_sample(pre_t, warped_grid.unsqueeze(0), align_corners=True, mode="bilinear")
    return warped.squeeze().cpu().numpy()

def shift_metrics(flow: torch.Tensor) -> Dict[str, float]:
    mag = torch.norm(flow, dim=1) # (1,D,H,W)
    return {"mean_shift": float(mag.mean().item()), "max_shift": float(mag.max().item())}

class _Hook:
    def __init__(self):
        self.act = None
        self.grad = None
    def fwd(self, m, i, o):
        self.act = o.detach()
    def bwd(self, m, gi, go):
        self.grad = go[0].detach()

def _compute_cam(model: torch.nn.Module, pre_t: torch.Tensor, post_t: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    # attach to bottleneck (adjust if different name)
    hk = _Hook()
    handle_fwd = model.bottleneck.register_forward_hook(hk.fwd)
    handle_bwd = model.bottleneck.register_backward_hook(hk.bwd)
    try:
        # scalar target: mean flow magnitude
        mag = torch.norm(flow, dim=1, keepdim=True)
        target = mag.mean()
        model.zero_grad(set_to_none=True)
        target.backward()
        A = hk.act # (1,C,d,h,w)
        dA = hk.grad
        w = dA.mean(dim=(2,3,4), keepdim=True)
        cam = (w * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=flow.shape[-3:], mode="trilinear", align_corners=False)
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze(0).squeeze(0) # (D,H,W)
    finally:
        handle_fwd.remove(); handle_bwd.remove()



def make_gradcam(pre_t: torch.Tensor, post_t: torch.Tensor, flow: torch.Tensor, run_dir: Path) -> Tuple[Path,Path,Path]:
    model = get_model()
    cam = _compute_cam(model, pre_t, post_t, flow).cpu().numpy() # (D,H,W)
    pre_np = pre_t.squeeze().cpu().numpy()


    d,h,w = cam.shape
    z, y, x = d//2, h//2, w//2


    def _save_overlay(base2d, heat2d, out_path: Path, title: str):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.imshow(base2d, cmap="gray")
        ax.imshow(heat2d, cmap="inferno", alpha=0.45)
        ax.set_title(title); ax.axis('off')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


    ax_png = run_dir / "gradcam_axial.png"
    cor_png = run_dir / "gradcam_coronal.png"
    sag_png = run_dir / "gradcam_sagittal.png"


    _save_overlay(pre_np[z], cam[z], ax_png, "Grad-CAM (axial)")
    _save_overlay(pre_np[:,y], cam[:,y], cor_png, "Grad-CAM (coronal)")
    _save_overlay(pre_np[:,:,x], cam[:,:,x], sag_png, "Grad-CAM (sagittal)")
    return ax_png, cor_png, sag_png

