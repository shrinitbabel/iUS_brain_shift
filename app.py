# app.py
# ─────────────────────────────────────────
# Keep BLAS/OMP single-threaded for stability on tiny hosts
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, gc, tempfile, shutil, time
from pathlib import Path
import gzip
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
MODEL_PATH      = "models/brain_shift_model_fulldataset.pt"  # FULL DATASET MODEL only
TARGET_DEFAULT  = 128
COARSE_DEFAULT  = 192
CAM_FIXED       = 48  # Grad-CAM working cube
TEST_DIR        = Path("Test/02")  # adjust if your sample lives elsewhere

# ========== UI ==========
st.set_page_config(
    page_title="Brain Shift (iUS) – Streamlit Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠",
)

# Inject minimal CSS: font + gradient-shine button + some polish
st.markdown("""
<style>
/* Match Tailwind-ish font stack for visual parity with babels.ai */
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI,
               Roboto, Helvetica Neue, Arial, "Noto Sans", "Apple Color Emoji",
               "Segoe UI Emoji","Segoe UI Symbol";
}

/* Big title spacing */
h1 { margin-bottom: 1rem; }

/* Pretty gradient-shine button (purple → fuchsia → red) */
.btn-gradient {
  --g1: #7c3aed;  /* purple-600 */
  --g2: #d946ef;  /* fuchsia-500 */
  --g3: #ef4444;  /* red-500 */
  position: relative;
  display: inline-flex; align-items: center; gap: .5rem;
  padding: 0.6rem 1rem;
  border-radius: 9999px;
  color: #fff !important;
  text-decoration: none !important;
  background: linear-gradient(90deg, var(--g1), var(--g2), var(--g3));
  background-size: 200% 200%;
  transition: transform .15s ease, box-shadow .25s ease, opacity .25s ease;
  box-shadow: 0 6px 18px rgba(124,58,237,.25);
}
.btn-gradient:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(124,58,237,.35);
}
.btn-gradient .shine {
  position: absolute; inset: 0; content: "";
  background: linear-gradient(120deg, rgba(255,255,255,0) 0%,
                                         rgba(255,255,255,.35) 50%,
                                         rgba(255,255,255,0) 100%);
  transform: translateX(-100%);
  transition: transform .8s ease;
}
.btn-gradient:hover .shine { transform: translateX(100%); }

/* Make triptych big and neat */
.block-divider { border-top: 1px solid #e5e7eb; margin: 1rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Header with return button
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.markdown(
        "<h1>🧠 <strong>Brain Shift Prediction (iUS)</strong> — Streamlit Demo</h1>",
        unsafe_allow_html=True
    )
with hdr_r:
    st.markdown(
        """
        <div style="display:flex;justify-content:flex-end">
          <a class="btn-gradient" href="https://www.babels.ai" target="_self" title="Back to Babels.ai">
            ← Return&nbsp;to&nbsp;Babels.ai
            <span class="shine"></span>
          </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Session state handle
ss = st.session_state
# Session state
if "result" not in ss: ss.result = None
if "gradcam" not in ss: ss.gradcam = None
if "playing" not in ss: ss.playing = False
if "last_idx" not in ss: ss.last_idx = 0
if "axis" not in ss: ss.axis = "axial"
if "gcam_ok" not in ss: ss.gcam_ok = False  # did Grad-CAM succeed last run?

# ── Sidebar
with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: {MODEL_PATH}")

    use_sample = st.toggle(
        "Use built-in sample (no upload)",
        value=False,
        help="Loads sample from ./Test/02 (pre.* & post.*: .mnc / .mnc.gz / .nii / .nii.gz)"
    )

    with st.expander("Advanced (power users)", expanded=False):
        st.write(
            "- **Target cube size**: internal resize before inference. "
            "Bigger → sharper but heavier.\n"
            "- **Coarse subsample**: caps the largest dimension *before* heavy ops to cut memory.\n"
            "- **Grad-CAM size**: fixed to 48 for reliability."
        )
        target_dim = st.select_slider("Target cube size", [96, 112, 128, 160], value=TARGET_DEFAULT)
        coarse_max_dim = st.select_slider("Coarse subsample (caps largest dim)", [128, 160, 192, 224, 256], value=COARSE_DEFAULT)
        st.caption("Grad-CAM working size: 48 (fixed)")

    use_cuda   = st.toggle("Use CUDA if available", value=False) and torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    want_tag   = st.toggle("Compute GT stats from .tag (optional)", value=False)

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
    """Stream UploadedFile to temp, robustly load with nibabel, coarse-subsample."""
    name = uploaded_file.name or "volume.mnc"
    suffix = "".join(Path(name).split('.')[-2:])  # keep extension for naming
    # Use actual suffix from name for temp file to preserve extension
    ext = "".join(Path(name).suffixes) or ".mnc"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp)
        tmp.flush()
        path = tmp.name
    try:
        load_path = path
        if ext.endswith(".mnc.gz"):
            with tempfile.NamedTemporaryFile(suffix=".mnc", delete=False) as unz:
                with gzip.open(path, "rb") as src:
                    shutil.copyfileobj(src, unz)
                unz.flush()
                load_path = unz.name
        img = nib.load(load_path, mmap=True)
        shape = img.shape
        factor = max(1, int(np.ceil(max(shape) / max_dim)))
        slicer = tuple(slice(None, None, factor) for _ in shape)
        arr = np.asarray(img.dataobj[slicer], dtype=np.float32, order="C")
        return arr, img.affine, shape, factor
    finally:
        try: os.remove(path)
        except Exception: pass
        if 'load_path' in locals() and load_path != path:
            try: os.remove(load_path)
            except Exception: pass

@st.cache_data(show_spinner=False)
def read_minc_path_to_array(path_in: Path, max_dim: int):
    """Load from a local path with robust .mnc.gz handling and coarse subsample."""
    path_in = Path(path_in)
    if not path_in.exists():
        raise FileNotFoundError(f"Sample not found: {path_in}")
    load_path = str(path_in); tmp_to_clean = None
    try:
        if str(path_in).endswith(".mnc.gz"):
            with tempfile.NamedTemporaryFile(suffix=".mnc", delete=False) as unz:
                with gzip.open(str(path_in), "rb") as src:
                    shutil.copyfileobj(src, unz)
                unz.flush()
                load_path = unz.name
                tmp_to_clean = unz.name
        img = nib.load(load_path, mmap=True)
        shape = img.shape
        factor = max(1, int(np.ceil(max(shape) / max_dim)))
        slicer = tuple(slice(None, None, factor) for _ in shape)
        arr = np.asarray(img.dataobj[slicer], dtype=np.float32, order="C")
        return arr, img.affine, shape, factor
    finally:
        if tmp_to_clean:
            try: os.remove(tmp_to_clean)
            except Exception: pass

# ========== sample detection ==========
def find_sample_pair() -> tuple[Path, Path] | None:
    """
    Try common sample names in Test/ or Test/02 for pre/post with multiple extensions.
    Order: .mnc → .mnc.gz → .nii.gz → .nii
    """
    bases = [Path("Test"), Path("Test/02"), TEST_DIR]
    exts  = [".mnc", ".mnc.gz", ".nii.gz", ".nii"]
    for base in bases:
        for ext in exts:
            p = base / f"pre{ext}"
            q = base / f"post{ext}"
            if p.exists() and q.exists():
                return (p, q)
    return None

# ========== core ----------
@torch.no_grad()
def run_inference(pre_arr: np.ndarray, post_arr: np.ndarray, device: torch.device, model: torch.nn.Module, dim: int):
    pre  = resize_cube(normalize01(pre_arr), dim)
    post = resize_cube(normalize01(post_arr), dim)
    pre_t  = torch.from_numpy(pre)[None, None, ...].to(device)   # (1,1,D,H,W)
    post_t = torch.from_numpy(post)[None, None, ...].to(device)  # (1,1,D,H,W)
    flow_t = model(pre_t, post_t)                                 # (1,3,D,H,W)
    flow   = flow_t.detach().cpu().numpy()[0]                     # (3,D,H,W)
    mag    = np.linalg.norm(flow, axis=0)                         # (D,H,W)
    mean_mm, max_mm = float(mag.mean()), float(mag.max())
    return mean_mm, max_mm, mag, flow, pre, post

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

def compute_gradcam(pre: np.ndarray, post: np.ndarray, device: torch.device, cam_dim: int, view_dim: int):
    """Low-RAM Grad-CAM: downscale to cam_dim^3, compute with grads, then upsample."""
    if not HAS_EXPLAIN:
        return None
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
            target = out.norm(p=2, dim=1).mean()
            target.backward()

            grads   = modelGC.get_activations_gradient()    # (1,C,d,h,w)
            acts    = modelGC.get_activations().detach()    # (1,C,d,h,w)
            weights = grads.mean(dim=(2,3,4), keepdim=True)
            cam     = (weights * acts).sum(dim=1).squeeze(0)  # (d,h,w)
            cam     = F.relu(cam)
            cam_np  = cam.detach().cpu().numpy().astype(np.float32)
            cam_np  = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        if cam_np.shape != (view_dim, view_dim, view_dim):
            f = [view_dim / cam_np.shape[i] for i in range(3)]
            cam_np = scipy.ndimage.zoom(cam_np, f, order=1, mode="nearest", prefilter=False)

        del pre_t, post_t, out, cam, grads, acts, weights
        if device.type == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return cam_np

    except MemoryError:
        st.warning("Grad-CAM OOM. Showing flow magnitude instead.")
        return None
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}. Showing flow magnitude instead.")
        return None

def slice2d(vol: np.ndarray, axis: str, idx: int) -> np.ndarray:
    """
    Return a 2D slice from a (D,H,W) volume along the chosen axis.
    axis: 'axial' (Z), 'coronal' (Y), 'sagittal' (X)
    """
    if axis == "axial":     # Z index
        return vol[idx, :, :]
    elif axis == "coronal": # Y index
        return vol[:, idx, :]
    else:                   # sagittal -> X index
        return vol[:, :, idx]

def draw_triptych_big(pre2d: np.ndarray, post2d: np.ndarray, heat2d: np.ndarray,
                      axis: str, idx: int, alpha: float = 0.5) -> None:
    """
    Render PRE, POST, and PRE+overlay (heat) as large side-by-side figures.
    """
    import matplotlib.pyplot as plt
    def _panel(img, title, overlay=None, alpha=0.5):
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        ax.imshow(img, cmap="gray", origin="lower")
        if overlay is not None:
            im = ax.imshow(overlay, origin="lower", alpha=alpha)
            fig.colorbar(im, ax=ax, shrink=0.72, label="heat")
        ax.set_title(title); ax.axis("off")
        st.pyplot(fig); plt.close(fig)

    c1, c2, c3 = st.columns([1,1,1])
    with c1: _panel(pre2d,  f"PRE • {axis} {idx}")
    with c2: _panel(post2d, f"POST • {axis} {idx}")
    with c3: _panel(pre2d,  f"Overlay • {axis} {idx}", overlay=heat2d, alpha=alpha)

# ========== uploads ==========
c1, c2 = st.columns(2)
with c1:
    pre_file  = st.file_uploader(
        "Upload PRE (.mnc / .mnc.gz / .nii / .nii.gz)",
        type=["mnc","gz","nii"],
        disabled=False  # can still upload even if sample is toggled
    )
with c2:
    post_file = st.file_uploader(
        "Upload POST (.mnc / .mnc.gz / .nii / .nii.gz)",
        type=["mnc","gz","nii"],
        disabled=False
    )

sample_pair = None
if use_sample:
    sample_pair = None
    # Prefer local sample if present
    sample_pair = None if not TEST_DIR.exists() else next(((p, q) for p,q in [
        (TEST_DIR / "pre.mnc",      TEST_DIR / "post.mnc"),
        (TEST_DIR / "pre.mnc.gz",   TEST_DIR / "post.mnc.gz"),
        (TEST_DIR / "pre.nii.gz",   TEST_DIR / "post.nii.gz"),
        (TEST_DIR / "pre.nii",      TEST_DIR / "post.nii"),
    ] if p.exists() and q.exists()), None)
    if sample_pair:
        st.info(f"Using sample: {sample_pair[0].name} + {sample_pair[1].name}")
    else:
        st.warning("No sample files found in ./Test/02 (expected pre/post with .mnc | .mnc.gz | .nii | .nii.gz).")

# tag upload only for user files (not shipping sample tag by default)
tag_file = st.file_uploader("Upload .tag (optional)", type=["tag"]) if (want_tag and not use_sample) else None

# Run button
can_run = (sample_pair is not None and use_sample) or (pre_file and post_file)
run = st.button("Run Prediction", type="primary", disabled=not can_run)

# ========== run ONCE ==========
if run:
    try:
        device = torch.device(device_str)
        model_inf = load_model_for_inference(MODEL_PATH, device_str)

        with st.status("Loading volumes…", expanded=False) as s:
            if use_sample and sample_pair:
                pre_arr,  pre_aff,  pre_shape,  pre_stride  = read_minc_path_to_array(sample_pair[0], max_dim=coarse_max_dim)
                post_arr, post_aff, post_shape, post_stride = read_minc_path_to_array(sample_pair[1], max_dim=coarse_max_dim)
            else:
                pre_arr,  pre_aff,  pre_shape,  pre_stride  = read_minc_stream_to_array(pre_file,  max_dim=coarse_max_dim)
                post_arr, post_aff, post_shape, post_stride = read_minc_stream_to_array(post_file, max_dim=coarse_max_dim)

            s.update(label=f"Loaded. PRE {pre_shape}→/×{pre_stride}, POST {post_shape}→/×{post_stride}")

        with st.status("Running inference…", expanded=False) as s:
            dim = target_dim
            mean_mm, max_mm, mag, flow, pre128, post128 = run_inference(pre_arr, post_arr, device, model_inf, dim)
            s.update(label="Inference complete.")

        # Always try Grad-CAM (fixed size = 48)
        gcam = compute_gradcam(pre128, post128, device, cam_dim=CAM_FIXED, view_dim=dim)
        gcam_ok = gcam is not None

        # Optional GT stats
        gt_mean = gt_max = None
        if tag_file and parse_tag_file_disk is not None and not use_sample:
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

        # Store results
        ss.result = {
            "mean_mm": mean_mm, "max_mm": max_mm,
            "mag": mag,            # (D,H/W) flow magnitude
            "flow": flow,          # (3,D,H,W)
            "pre": pre128, "post": post128,
            "gt_mean": gt_mean, "gt_max": gt_max,
            "dim": dim
        }
        ss.gradcam = gcam
        ss.gcam_ok = gcam_ok
        ss.playing = True
        ss.last_idx = 0
        ss.axis = "axial"
        if not gcam_ok:
            st.info("Showing flow magnitude (Grad-CAM unavailable).")
        st.success("✅ Done!")
    except Exception as e:
        ss.result = None
        ss.gradcam = None
        ss.gcam_ok = False
        ss.playing = False
        st.error(f"Error: {e}")
        st.exception(e)
    finally:
        gc.collect()

# ========== viewer (no recompute) ==========
res = ss.result
if res is not None:
    # Top metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean brain shift (pred)", f"{res['mean_mm']:.3f} mm")
    m2.metric("Max brain shift (pred)",  f"{res['max_mm']:.3f} mm")
    if res.get("gt_mean") is not None:
        m3.metric("GT mean / max (mm)", f"{res['gt_mean']:.3f} / {res['gt_max']:.3f}")
    else:
        m3.write(f"Volume (after resize): {res['mag'].shape}")

    st.subheader("Viewer")

    # Controls
    vcol1, _ = st.columns([1,3])
    with vcol1:
        axis = st.radio("View", options=["axial", "coronal", "sagittal"], index=["axial","coronal","sagittal"].index(ss.axis))
        ss.axis = axis
        idx_max = {"axial": res["mag"].shape[0]-1, "coronal": res["mag"].shape[1]-1, "sagittal": res["mag"].shape[2]-1}[axis]
        idx = st.slider("Slice index", 0, idx_max, ss.last_idx if ss.last_idx <= idx_max else idx_max)
        if idx != ss.last_idx:
            ss.playing = False
        ss.last_idx = idx

        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, 0.05)

        # Play/Pause (Play restarts from slice 0)
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            if st.button("▶️ Play" if not ss.playing else "⏸ Pause"):
                if not ss.playing:
                    ss.last_idx = 0
                ss.playing = not ss.playing
        with pcol2:
            fps = st.slider("FPS", 1, 20, 6)

    # Big viewer under controls — prefer Grad-CAM if available
    st.markdown("---")
    heat_vol = ss.gradcam if (ss.gcam_ok and ss.gradcam is not None) else res["mag"]
    pre2d  = slice2d(res["pre"],  ss.axis, ss.last_idx)
    post2d = slice2d(res["post"], ss.axis, ss.last_idx)
    heat2d = slice2d(heat_vol,    ss.axis, ss.last_idx)
    draw_triptych_big(pre2d, post2d, heat2d, ss.axis, ss.last_idx, alpha=alpha)

    # Autoplay
    if ss.playing:
        time.sleep(1.0 / fps)
        ss.last_idx = (ss.last_idx + 1) % (idx_max + 1)
        st.rerun()
