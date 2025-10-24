import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib


plt.switch_backend("Agg") # headless

def save_center_slices(volume: np.ndarray, out_png: Path, title: str = "") -> Path:
    d, h, w = volume.shape
    z, y, x = d//2, h//2, w//2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[z, :, :], cmap="gray"); axes[0].set_title("Axial")
    axes[1].imshow(volume[:, y, :], cmap="gray"); axes[1].set_title("Coronal")
    axes[2].imshow(volume[:, :, x], cmap="gray"); axes[2].set_title("Sagittal")
    for ax in axes: ax.axis('off')
    if title: fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

def save_mip(volume: np.ndarray, out_png: Path, axis: int = 0, title: str = "MIP") -> Path:
    mip = volume.max(axis=axis)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(mip, cmap="inferno"); ax.axis('off'); ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

def save_quiver(run_dir: Path, tag_path: Path, flow_torch, pre_path: Path) -> Path:
    """Make a 3D quiver comparing GT vs Pred at landmarks. Saves a PNG."""
    import torch
    import torch.nn.functional as F
    from .utils import parse_tag_file


    pre_pts, post_pts = parse_tag_file(tag_path)
    img = nib.load(str(pre_path))
    aff = img.affine


    # spacing/origin
    hdr = img.header
    if "pixdim" in hdr:
        spacing = np.array(hdr["pixdim"][1:4], dtype=np.float32)
    else:
        spacing = np.array([np.linalg.norm(aff[:3,0]), np.linalg.norm(aff[:3,1]), np.linalg.norm(aff[:3,2])], dtype=np.float32)
    origin = aff[:3,3]
    orig_shape = img.shape[:3]


    # world->voxel (orig), then to resized (128^3)
    def world_to_voxel(world_xyz):
        return (world_xyz - origin) / spacing
    vox_orig = world_to_voxel(pre_pts)
    target = np.array(flow_torch.shape[-3:])
    scale = target / np.array(orig_shape)
    vox_resz = vox_orig * scale


    # in-bounds mask
    mask = (
        (vox_resz[:,0] >= 0) & (vox_resz[:,0] < target[0]) &
        (vox_resz[:,1] >= 0) & (vox_resz[:,1] < target[1]) &
        (vox_resz[:,2] >= 0) & (vox_resz[:,2] < target[2])
    )
    vox_resz = vox_resz[mask]
    gt_vecs = (post_pts - pre_pts)[mask]


    # normalize coords for grid_sample (z,y,x)
    D,H,W = target
    Z = (vox_resz[:,2] / (W-1)) * 2 - 1
    Y = (vox_resz[:,1] / (H-1)) * 2 - 1
    X = (vox_resz[:,0] / (D-1)) * 2 - 1
    pts = np.stack([X,Y,Z], axis=1).astype(np.float32)


    grid = torch.from_numpy(pts).to(flow_torch.device).view(1,1,-1,1,3)
    sampled = F.grid_sample(flow_torch, grid, mode='bilinear', align_corners=True) # (1,3,1,N,1)
    pred = sampled.view(3, -1).T.cpu().numpy() # (N,3)


    # scale to mm approximately (based on resizing)
    voxel_mm = spacing * (np.array(orig_shape)/np.array([D,H,W]))
    pred_mm = pred * voxel_mm


    # plot
    from mpl_toolkits.mplot3d import Axes3D # noqa
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(vox_resz[:,0], vox_resz[:,1], vox_resz[:,2], gt_vecs[:,0], gt_vecs[:,1], gt_vecs[:,2],
        length=4, normalize=False, linewidth=1.0, label='GT', color='tab:blue')
    ax.quiver(vox_resz[:,0], vox_resz[:,1], vox_resz[:,2], pred_mm[:,0], pred_mm[:,1], pred_mm[:,2],
        length=4, normalize=False, linewidth=1.0, label='Pred', color='tab:orange')
    ax.set_title("Landmark vectors (GT vs Pred)")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    out_png = run_dir / "landmark_quiver.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png

