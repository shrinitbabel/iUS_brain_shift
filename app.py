# app.py
import os, sys, gc, tempfile, shutil
import numpy as np
import torch
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

# ========== UI ==========
st.set_page_config(page_title="Brain Shift (iUS) – Demo", layout="wide")
st.title("🧠 Brain Shift Prediction (iUS) – Streamlit Demo")

with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: {MODEL_PATH}")
    target_dim = st.select_slider("Target cube size", [96,112,128,160], value=TARGET_DEFAULT)
    cam_dim = st.select_slider("Grad-CAM working size", [80, 96, 112, 128], value=96)
    coarse_max_dim = st.select_slider("Coarse subsample (caps largest dim)", [128,160,192,224,256], value=COARSE_DEFAULT)
    use_cuda   = st.toggle("Use CUDA if available", value=False) and torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    want_tag   = st.toggle("Compute GT stats from .tag", value=False)
    want_flow_export = st.toggle("Export flow .npy", value=False)
    want_gradcam = st.toggle("Compute Grad-CAM", value=False) if HAS_EXPLAIN else False
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

def draw_triptych(pre: np.ndarray, post: np.ndarray, heat: np.ndarray, axis: str, idx: int, alpha: float = 0.5):
    """Three side-by-side: PRE, POST, and PRE + heat overlay."""
    pre2d  = slice2d(pre,  axis, idx)
    post2d = slice2d(post, axis, idx)
    heat2d = slice2d(heat, axis, idx)

    cols = st.columns(3)

    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.imshow(pre2d, cmap="gray", origin="lower"); ax1.set_title(f"PRE • {axis} {idx}"); ax1.axis("off")
    cols[0].pyplot(fig1); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.imshow(post2d, cmap="gray", origin="lower"); ax2.set_title(f"POST • {axis} {idx}"); ax2.axis("off")
    cols[1].pyplot(fig2); plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(4,4))
    ax3.imshow(pre2d, cmap="gray", origin="lower")
    im = ax3.imshow(heat2d, origin="lower", alpha=alpha)   # default colormap
    ax3.set_title(f"Overlay • {axis} {idx}"); ax3.axis("off")
    fig3.colorbar(im, ax=ax3, shrink=0.8, label="heat")
    cols[2].pyplot(fig3); plt.close(fig3)

def compute_gradcam(pre: np.ndarray, post: np.ndarray, device: torch.device, cam_dim: int, view_dim: int):
    """Grad-CAM with downscaled inputs to cut memory; upsample CAM for display."""
    import torch.nn.functional as F
    try:
        modelGC = load_model_for_gradcam(MODEL_PATH, str(device))

        # downscale to cam_dim for Grad-CAM
        def _resize(v, d): 
            f = [d / v.shape[i] for i in range(3)]
            return scipy.ndimage.zoom(np.ascontiguousarray(v, np.float32), f, order=1, mode="nearest", prefilter=False)

        pre_small  = _resize(pre,  cam_dim)
        post_small = _resize(post, cam_dim)

        with torch.enable_grad():
            pre_t  = torch.from_numpy(pre_small)[None, None, ...].to(device).requires_grad_(True)
            post_t = torch.from_numpy(post_small)[None, None, ...].to(device).requires_grad_(True)

            modelGC.zero_grad()
            out = modelGC(pre_t, post_t)               # (1,3,d,h,w)
            target = out.norm(p=2, dim=1).mean()       # scalar objective
            target.backward()

            grads = modelGC.get_activations_gradient() # (1,C,d,h,w)
            acts  = modelGC.get_activations().detach() # (1,C,d,h,w)
            weights = grads.mean(dim=(2,3,4), keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze(0)  # (d,h,w)
            cam = F.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_np = cam.detach().cpu().numpy().astype(np.float32)

        # upsample CAM to viewer size (target_dim)
        if cam_np.shape != (view_dim, view_dim, view_dim):
            f = [view_dim / cam_np.shape[i] for i in range(3)]
            cam_np = scipy.ndimage.zoom(cam_np, f, order=1, mode="nearest", prefilter=False)

        # cleanup
        del pre_t, post_t, out, cam, grads, acts
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()
        return cam_np

    except MemoryError:
        # don’t crash the app; surface a graceful warning
        st.warning("Grad-CAM ran out of memory at this size. Try a smaller ‘Grad-CAM working size’.")
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

# ========== session state ==========
if "result" not in st.session_state:
    st.session_state.result = None
if "gradcam" not in st.session_state:
    st.session_state.gradcam = None

# ========== run ONCE ==========
if run:
    try:
        device = torch.device(device_str)
        model_inf = load_model_for_inference(MODEL_PATH, device_str)

        with st.status("Loading volumes…", expanded=False) as s:
            pre_arr,  pre_aff,  pre_shape,  pre_stride  = read_minc_stream_to_array(pre_file,  max_dim=coarse_max_dim)
            post_arr, post_aff, post_shape, post_stride = read_minc_stream_to_array(post_file, max_dim=coarse_max_dim)
            s.update(label=f"Loaded. PRE {pre_shape}→/×{pre_stride}, POST {post_shape}→/×{post_stride}")

        with st.status("Running inference…", expanded=False) as s:
            mean_mm, max_mm, mag, flow, pre128, post128 = run_inference(pre_arr, post_arr, device, model_inf, target_dim)
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
        st.session_state.result = {
            "mean_mm": mean_mm, "max_mm": max_mm,
            "mag": mag,          # (D,H,W) flow magnitude
            "flow": flow,        # (3,D,H,W)
            "pre": pre128, "post": post128,
            "gt_mean": gt_mean, "gt_max": gt_max
        }
        st.session_state.gradcam = None
        st.success("✅ Done!")
    except Exception as e:
        st.session_state.result = None
        st.session_state.gradcam = None
        st.error(f"Error: {e}")
        st.exception(e)
    finally:
        gc.collect()

# ========== viewer (no recompute) ==========
res = st.session_state.result
if res is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean brain shift (pred)", f"{res['mean_mm']:.3f} mm")
    m2.metric("Max brain shift (pred)",  f"{res['max_mm']:.3f} mm")
    if res.get("gt_mean") is not None:
        m3.metric("GT mean / max (mm)", f"{res['gt_mean']:.3f} / {res['gt_max']:.3f}")
    else:
        m3.write(f"Volume (after resize): {res['mag'].shape}")

    st.subheader("Viewer")
    vcol1, vcol2 = st.columns([1,3])

    with vcol1:
        axis   = st.radio("View", options=["axial", "coronal", "sagittal"], index=0)
        idx_max = {"axial": res["mag"].shape[0]-1, "coronal": res["mag"].shape[1]-1, "sagittal": res["mag"].shape[2]-1}[axis]
        idx    = st.slider("Slice index", 0, idx_max, idx_max//2)
        heat_source = st.radio("Heatmap", options=(["Flow magnitude"] if not HAS_EXPLAIN else ["Flow magnitude","Grad-CAM"]), index=0)
        alpha  = st.slider("Overlay alpha", 0.0, 1.0, 0.5, 0.05)

        # Compute Grad-CAM once on demand
        if heat_source == "Grad-CAM" and HAS_EXPLAIN and st.session_state.gradcam is None:
            with st.spinner("Computing Grad-CAM…"):
                device = torch.device(device_str)
                st.session_state.gradcam = compute_gradcam(res["pre"], res["post"], device, cam_dim=cam_dim, view_dim=target_dim)


    with vcol2:
        heat = res["mag"] if (heat_source != "Grad-CAM" or st.session_state.gradcam is None) else st.session_state.gradcam
        draw_triptych(res["pre"], res["post"], heat, axis, idx, alpha=alpha)

    # Export flow
    if want_flow_export:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, res["flow"]); path = tmp.name
        with open(path, "rb") as fh:
            st.download_button("Download flow (.npy)", data=fh, file_name="flow.npy", mime="application/octet-stream")
        try: os.remove(path)
        except Exception: pass
