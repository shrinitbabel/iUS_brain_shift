# app.py
# ─────────────────────────────────────────
# Make BLAS/OMP single-threaded => fewer surprise restarts on tiny hosts
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, gc, tempfile, shutil, time
import numpy as np
import torch
torch.set_num_threads(1)

import streamlit as st
import nibabel as nib
import scipy.ndimage
import matplotlib.pyplot as plt

# ========== modules/ imports ==========
sys.path.append(os.path.abspath("modules"))
from unet3d import UNet3D
try:
    from explainability import ExplainableUNet3D
    HAS_EXPLAIN = True
except Exception:
    HAS_EXPLAIN = False
    ExplainableUNet3D = None  # type: ignore
try:
    from ground_truth import parse_tag_file as parse_tag_file_disk
except Exception:
    parse_tag_file_disk = None

# ========== constants ==========
MODEL_PATH      = "models/brain_shift_model_fulldataset.pt"  # FULL DATASET MODEL (only one)
TARGET_DEFAULT  = 128
COARSE_DEFAULT  = 192
CAM_DEFAULT     = 64  # downscaled GCAM working cube

# ========== UI ==========
st.set_page_config(page_title="Brain Shift (iUS) – Demo", layout="wide")
st.markdown(
    "<h1 style='display:flex;align-items:center;gap:.5rem'>"
    "🧠 Brain Shift Prediction (iUS) – Streamlit Demo</h1>",
    unsafe_allow_html=True,
)

# Session state: results + viewer controls
ss = st.session_state
if "result" not in ss: ss.result = None
if "gradcam" not in ss: ss.gradcam = None
if "playing" not in ss: ss.playing = False
if "last_idx" not in ss: ss.last_idx = 0
if "axis" not in ss: ss.axis = "axial"

# ── Sidebar: simple controls for end users
with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: {MODEL_PATH}")

    # Hide advanced processing knobs in an expander with human text
    with st.expander("Advanced (power users)", expanded=False):
        st.write(
            "- **Target cube size**: internal resize of volumes before inference. "
            "Bigger → sharper but heavier.\n"
            "- **Grad-CAM working size**: smaller → more likely to work on small servers.\n"
            "- **Coarse subsample**: caps the largest dimension *before* any heavy ops to cut memory."
        )
        target_dim = st.select_slider("Target cube size", [96,112,128,160], value=TARGET_DEFAULT)
        cam_dim    = st.select_slider("Grad-CAM working size", [48,64,80,96], value=CAM_DEFAULT)
        coarse_max_dim = st.select_slider("Coarse subsample (caps largest dim)", [128,160,192,224,256], value=COARSE_DEFAULT)

    use_cuda = st.toggle("Use CUDA if available", value=False) and torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"

    want_tag = st.toggle("Compute GT stats from .tag (optional)", value=False)
    want_gradcam = st.toggle("Enable Grad-CAM overlay", value=False) if HAS_EXPLAIN else False
    if not HAS_EXPLAIN:
        st.caption("Grad-CAM disabled (missing modules/explainability.py).")

# ========== caching ==========
@st.cache_resource(show_spinner=False)
def load_model_for_inference(weights: str, device: str = "cpu"):
    net = UNet3D().to(device)
    if os.path.exists(weights):
        sd = torch.load(weights, map_location=device)
        net.load_state_dict(sd)
    net.eval()
    return net

@st.cache_resource(show_spinner=False)
def load_model_for_gradcam(weights: str, device: str = "cpu"):
    assert HAS_EXPLAIN, "Explainable model not available"
    net = ExplainableUNet3D().to(device)
    if os.path.exists(weights):
        sd = torch.load(weights, map_location=device)
        net.load_state_dict(sd)
    net.eval()
    return net

@st.cache_data(show_spinner=False)
def read_minc_stream_to_array(uploaded_file, max_dim: int):
    """Stream an UploadedFile to temp .mnc, load via nibabel (mmap), and coarse-subsample."""
    with tempfile.NamedTemporaryFile(suffix=".mnc", delete=False) as tmp:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp)
        tmp.flush()
        path = tmp.name
    try:
        img = nib.load(path, mmap=True)
        shape = img.shape
        factor = max(1, int(np.ceil(max(shape) / max_dim)))
        slicer = tuple(slice(None, None, factor) for _ in shape)
        arr = np.asarray(img.dataobj[slicer], dtype=np.float32, order="C")
        return arr, img.affine, shape, factor
    finally:
        try: os.remove(path)
        except Exception: pass

# ========== utils ==========
def normalize01(v: np.ndarray) -> np.ndarray:
    vmin, vmax = float(v.min()), float(v.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(v, dtype=np.float32)
    return ((v - vmin) / (vmax - vmin)).astype(np.float32, copy=False)

def resize_cube(v: np.ndarray, dim: int) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype=np.float32)
    f = [dim / v.shape[i] for i in range(3)]
    out = scipy.ndimage.zoom(v, f, order=1, mode="nearest", prefilter=False)
    return out.astype(np.float32, copy=False)

@torch.no_grad()
def run_inference(pre_arr: np.ndarray, post_arr: np.ndarray, device: torch.device, model: torch.nn.Module, dim: int):
    """Plain inference with UNet3D (no hooks)."""
    pre  = resize_cube(normalize01(pre_arr), dim)
    post = resize_cube(normalize01(post_arr), dim)
    pre_t  = torch.from_numpy(pre)[None, None, ...].to(device)
    post_t = torch.from_numpy(post)[None, None, ...].to(device)
    flow_t = model(pre_t, post_t)                    # (1,3,D,H,W)
    flow   = flow_t.detach().cpu().numpy()[0]        # (3,D,H,W)
    mag    = np.linalg.norm(flow, axis=0)            # (D,H,W)  <-- flow magnitude
    mean_mm, max_mm = float(mag.mean()), float(mag.max())
    return mean_mm, max_mm, mag, flow, pre, post

def slice2d(vol: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "axial":   return vol[idx, :, :]
    if axis == "coronal": return vol[:, idx, :]
    return vol[:, :, idx]  # sagittal

def big_subplot(img, title, cmap=None, overlay=None, alpha=0.5):
    """Bigger, clearer matplotlib figure."""
    fig, ax = plt.subplots(figsize=(6.8, 6.8))  # BIG
    ax.imshow(img, cmap=cmap or "gray", origin="lower")
    if overlay is not None:
        im = ax.imshow(overlay, origin="lower", alpha=alpha)
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, label="heat")
    ax.set_title(title); ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

def draw_triptych(pre: np.ndarray, post: np.ndarray, heat: np.ndarray, axis: str, idx: int, alpha: float = 0.5):
    pre2d  = slice2d(pre,  axis, idx)
    post2d = slice2d(post, axis, idx)
    heat2d = slice2d(heat, axis, idx)
    cols = st.columns([1,1,1])
    with cols[0]: big_subplot(pre2d,  f"PRE • {axis} {idx}")
    with cols[1]: big_subplot(post2d, f"POST • {axis} {idx}")
    with cols[2]: big_subplot(pre2d,  f"Overlay • {axis} {idx}", overlay=heat2d, alpha=alpha)

def compute_gradcam(pre: np.ndarray, post: np.ndarray, device: torch.device, cam_dim: int, view_dim: int):
    """Low-RAM Grad-CAM: downscale to cam_dim^3, compute with grads, then upsample."""
    import torch.nn.functional as F
    try:
        modelGC = load_model_for_gradcam(MODEL_PATH, str(device))
        def _resize(v, d):
            f = [d / v.shape[i] for i in range(3)]
            return scipy.ndimage.zoom(np.ascontiguousarray(v, np.float32), f, order=1, mode="nearest", prefilter=False)
        pre_small  = _resize(pre,  cam_dim)
        post_small = _resize(post, cam_dim)

        with torch.enable_grad():
            pre_t  = torch.from_numpy(pre_small)[None, None, ...].to(device).requires_grad_(True)
            post_t = torch.from_numpy(post_small)[None, None, ...].to(device).requires_grad_(True)

            modelGC.zero_grad(set_to_none=True)
            out    = modelGC(pre_t, post_t)                 # (1,3,d,h,w)
            target = out.norm(p=2, dim=1).mean()            # scalar
            target.backward()

            grads   = modelGC.get_activations_gradient()    # (1,C,d,h,w)
            acts    = modelGC.get_activations().detach()    # (1,C,d,h,w)
            weights = grads.mean(dim=(2,3,4), keepdim=True)
            cam     = (weights * acts).sum(dim=1).squeeze(0)  # (d,h,w)
            cam     = F.relu(cam)
            cam_np  = cam.detach().cpu().numpy().astype(np.float32)
            cam_np  = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        # Upsample to viewer size
        if cam_np.shape != (view_dim, view_dim, view_dim):
            f = [view_dim / cam_np.shape[i] for i in range(3)]
            cam_np = scipy.ndimage.zoom(cam_np, f, order=1, mode="nearest", prefilter=False)

        # Cleanup
        del pre_t, post_t, out, cam, grads, acts, weights
        if device.type == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return cam_np

    except MemoryError:
        st.warning("Grad-CAM OOM. Lower the ‘Grad-CAM working size’.")
        return None
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        return None

# ========== uploads ==========
c1, c2 = st.columns(2)
with c1: pre_file  = st.file_uploader("Upload PRE .mnc",  type=["mnc"])
with c2: post_file = st.file_uploader("Upload POST .mnc", type=["mnc"])

tag_file = st.file_uploader("Upload .tag (optional)", type=["tag"]) if want_tag else None
run = st.button("Run Prediction", type="primary", disabled=not (pre_file and post_file))

# ========== run ONCE ==========
if run:
    try:
        device = torch.device(device_str)
        model_inf = load_model_for_inference(MODEL_PATH, device_str)

        with st.status("Loading volumes…", expanded=False) as s:
            pre_arr,  pre_aff,  pre_shape,  pre_stride  = read_minc_stream_to_array(pre_file,  max_dim=COARSE_DEFAULT if "coarse_max_dim" not in locals() else coarse_max_dim)
            post_arr, post_aff, post_shape, post_stride = read_minc_stream_to_array(post_file, max_dim=COARSE_DEFAULT if "coarse_max_dim" not in locals() else coarse_max_dim)
            s.update(label=f"Loaded. PRE {pre_shape}→/×{pre_stride}, POST {post_shape}→/×{post_stride}")

        with st.status("Running inference…", expanded=False) as s:
            dim = TARGET_DEFAULT if "target_dim" not in locals() else target_dim
            mean_mm, max_mm, mag, flow, pre128, post128 = run_inference(pre_arr, post_arr, device, model_inf, dim)
            s.update(label="Inference complete.")

        # Optional GT stats
        gt_mean = gt_max = None
        if tag_file and parse_tag_file_disk is not None:
            with tempfile.NamedTemporaryFile(suffix=".tag", delete=False) as tmp:
                tmp.write(tag_file.getvalue()); tmp.flush(); tag_path = tmp.name
            try:
                pre_lm, post_lm = parse_tag_file_disk(tag_path)
                if pre_lm.size and post_lm.size and pre_lm.shape == post_lm.shape:
                    gt = np.linalg.norm(pre_lm - post_lm, axis=1)
                    gt_mean, gt_max = float(gt.mean()), float(gt.max())
            finally:
                try: os.remove(tag_path)
                except Exception: pass

        # Store for live viewer
        ss.result = {
            "mean_mm": mean_mm, "max_mm": max_mm,
            "mag": mag,        # (D,H,W) flow magnitude
            "flow": flow,      # (3,D,H,W)
            "pre": pre128, "post": post128,
            "gt_mean": gt_mean, "gt_max": gt_max,
            "dim": dim
        }
        ss.gradcam = None
        ss.playing = False
        ss.last_idx = 0
        ss.axis = "axial"
        st.success("✅ Done!")
    except Exception as e:
        ss.result = None
        ss.gradcam = None
        ss.playing = False
        st.error(f"Error: {e}")
        st.exception(e)
    finally:
        gc.collect()

# ========== viewer (no recompute) ==========
res = ss.result
if res is not None:
    # Top metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean brain shift (pred)", f"{res['mean_mm']:.3f} mm")
    m2.metric("Max brain shift (pred)",  f"{res['max_mm']:.3f} mm")
    if res.get("gt_mean") is not None:
        m3.metric("GT mean / max (mm)", f"{res['gt_mean']:.3f} / {res['gt_max']:.3f}")
    else:
        m3.write(f"Volume (after resize): {res['mag'].shape}")

    st.subheader("Viewer")

    # Controls column + autoplay
    vcol1, vcol2 = st.columns([1,3])
    with vcol1:
        # View axis
        axis = st.radio("View", options=["axial", "coronal", "sagittal"], index=["axial","coronal","sagittal"].index(ss.axis), key="axis_radio")
        ss.axis = axis

        # Max index for chosen axis
        idx_max = {"axial": res["mag"].shape[0]-1, "coronal": res["mag"].shape[1]-1, "sagittal": res["mag"].shape[2]-1}[axis]
        # Slider; moving it pauses playback automatically
        idx = st.slider("Slice index", 0, idx_max, ss.last_idx if ss.last_idx <= idx_max else idx_max, key="idx_slider")
        if idx != ss.last_idx:
            ss.playing = False
        ss.last_idx = idx

        # Heatmap choice
        heat_choice = "Flow magnitude"
        if HAS_EXPLAIN and want_gradcam:
            heat_choice = st.radio("Heatmap", options=["Flow magnitude","Grad-CAM"], index=0 if ss.gradcam is None else 1)

        alpha  = st.slider("Overlay alpha", 0.0, 1.0, 0.5, 0.05)

        # Autoplay controls
        play_cols = st.columns(2)
        with play_cols[0]:
            if st.button("▶️ Play" if not ss.playing else "⏸ Pause"):
                ss.playing = not ss.playing
        with play_cols[1]:
            fps = st.slider("FPS", 1, 20, 6)

        # Compute Grad-CAM on demand (once)
        if heat_choice == "Grad-CAM" and HAS_EXPLAIN and want_gradcam and ss.gradcam is None:
            with st.spinner("Computing Grad-CAM…"):
                device = torch.device(device_str)
                ss.gradcam = compute_gradcam(res["pre"], res["post"], device, cam_dim=CAM_DEFAULT if "cam_dim" not in locals() else cam_dim, view_dim=res["dim"])

    # Right: big triptych
    with vcol2:
        heat = res["mag"] if (heat_choice != "Grad-CAM" or ss.gradcam is None) else ss.gradcam
        draw_triptych(res["pre"], res["post"], heat, ss.axis, ss.last_idx, alpha=alpha)

    # Auto-play loop: bump index, rerun
    if ss.playing:
        time.sleep(1.0 / fps)
        ss.last_idx = (ss.last_idx + 1) % (idx_max + 1)
        st.experimental_rerun()

    # Export flow if needed
    if st.sidebar.toggle("Export flow .npy", value=False, key="export_flow_toggle"):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, res["flow"]); path = tmp.name
        with open(path, "rb") as fh:
            st.download_button("Download flow (.npy)", data=fh, file_name="flow.npy", mime="application/octet-stream")
        try: os.remove(path)
        except Exception: pass
