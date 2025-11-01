# app.py
import os, io, gc, tempfile, shutil, sys
import numpy as np
import torch
import streamlit as st
import nibabel as nib
import scipy.ndimage
import matplotlib.pyplot as plt

# --- Make your modules/ importable if app.py lives at repo root
sys.path.append(os.path.abspath("modules"))
from unet3d import UNet3D  # your network
# Optional: if you want .tag GT comparison later
try:
    from ground_truth import parse_tag_file as parse_tag_file_disk
except Exception:
    parse_tag_file_disk = None

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Brain Shift (iUS) – Demo", layout="wide")

st.title("🧠 Brain Shift Prediction (iUS) – Streamlit Demo")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input(
        "Model weights (.pt)",
        value="models/brain_shift_model_02.pt",
        help="Path relative to this app’s working directory.",
    )
    target_dim = st.select_slider("Target cube size", options=[96, 112, 128, 160], value=128)
    coarse_max_dim = st.select_slider("Coarse subsample before resize (caps largest dim)",
                                      options=[128, 160, 192, 224, 256], value=192)
    device_str = "cuda" if (torch.cuda.is_available() and st.toggle("Use CUDA if available", value=False)) else "cpu"
    show_tag = st.toggle("Provide landmarks .tag (optional GT stats)", value=False)
    save_flow = st.toggle("Export flow .npy after run", value=False)

# -----------------------------
# CACHING
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(weights_path: str, device: str = "cpu"):
    model = UNet3D().to(device)
    if os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location=device)
        model.load_state_dict(sd)
    model.eval()
    return model

@st.cache_data(show_spinner=False)
def read_minc_stream_to_array(uploaded_file, max_dim: int):
    """
    Streams an UploadedFile (pre/post .mnc) to disk, then loads with nibabel (mmap),
    and returns a coarse-subsampled array (so peak RAM is small).
    """
    import numpy as np, nibabel as nib, tempfile, os, shutil
    with tempfile.NamedTemporaryFile(suffix=".mnc", delete=False) as tmp:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp)
        tmp.flush()
        path = tmp.name
    try:
        img = nib.load(path, mmap=True)  # MINC1/2; requires h5py for MINC2
        shape = img.shape
        factor = max(1, int(np.ceil(max(shape) / max_dim)))
        slicer = tuple(slice(None, None, factor) for _ in shape)
        arr = np.asarray(img.dataobj[slicer], dtype=np.float32, order="C")
        affine = img.affine
        return arr, affine, shape, factor
    finally:
        try: os.remove(path)
        except Exception: pass

def normalize01(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = float(vol.min()), float(vol.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32, copy=False)

def resize_to_cube(vol: np.ndarray, tgt: int) -> np.ndarray:
    v = np.ascontiguousarray(vol, dtype=np.float32)
    factors = [tgt / v.shape[i] for i in range(3)]
    out = scipy.ndimage.zoom(v, factors, order=1, mode="nearest", prefilter=False)
    return out.astype(np.float32, copy=False)

@torch.no_grad()
def run_inference(pre_arr: np.ndarray, post_arr: np.ndarray, device: torch.device, model: torch.nn.Module):
    pre = resize_to_cube(normalize01(pre_arr), target_dim)
    post = resize_to_cube(normalize01(post_arr), target_dim)
    pre_t  = torch.from_numpy(pre)[None, None, ...].to(device)   # (1,1,D,H,W)
    post_t = torch.from_numpy(post)[None, None, ...].to(device)

    flow_t = model(pre_t, post_t)                # (1,3,D,H,W)
    flow = flow_t.detach().cpu().numpy()[0]      # (3,D,H,W)
    mag = np.linalg.norm(flow, axis=0)           # (D,H,W)
    mean_mm, max_mm = float(mag.mean()), float(mag.max())
    return mean_mm, max_mm, mag, flow

def visualize_slice(mag: np.ndarray, axis: str, index: int):
    """
    Show a single slice of the magnitude volume as a heatmap.
    axis in {'axial','sagittal','coronal'}
    """
    if axis == "axial":     img2d = mag[index, :, :]   # z, y, x
    elif axis == "coronal": img2d = mag[:, index, :]   # z, y, x -> y fixed
    else:                   img2d = mag[:, :, index]   # sagittal -> x fixed

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img2d, origin="lower")
    ax.set_title(f"{axis.capitalize()} slice {index}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="|flow|")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# UPLOAD WIDGETS
# -----------------------------
col_u1, col_u2 = st.columns(2)
with col_u1:
    pre_file = st.file_uploader("Upload PRE .mnc", type=["mnc"], accept_multiple_files=False)
with col_u2:
    post_file = st.file_uploader("Upload POST .mnc", type=["mnc"], accept_multiple_files=False)

tag_file = None
if show_tag:
    tag_file = st.file_uploader("Upload landmarks .tag (optional)", type=["tag"], accept_multiple_files=False)

run = st.button("Run Prediction", type="primary", disabled=not (pre_file and post_file))

# -----------------------------
# RUN
# -----------------------------
if run:
    try:
        device = torch.device(device_str)
        model = load_model_cached(model_path, device_str)

        with st.status("Loading volumes…", expanded=False) as s:
            pre_arr, pre_aff, pre_shape, pre_stride = read_minc_stream_to_array(pre_file, max_dim=coarse_max_dim)
            post_arr, post_aff, post_shape, post_stride = read_minc_stream_to_array(post_file, max_dim=coarse_max_dim)
            s.update(label=f"Loaded. PRE {pre_shape}→/{pre_stride} stride, POST {post_shape}→/{post_stride}")

        with st.status("Running inference…", expanded=False) as s:
            mean_mm, max_mm, mag, flow = run_inference(pre_arr, post_arr, device, model)
            s.update(label="Inference complete.")

        st.success("✅ Done!")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Mean brain shift (pred)", f"{mean_mm:.3f} mm")
        mcol2.metric("Max brain shift (pred)", f"{max_mm:.3f} mm")
        mcol3.write(f"Volume (after resize): {mag.shape}")

        # Optional GT stats from .tag
        if tag_file and parse_tag_file_disk is not None:
            # Save tag to temp and use your parser
            with tempfile.NamedTemporaryFile(suffix=".tag", delete=False) as tmp:
                tmp.write(tag_file.getvalue()); tmp.flush(); tag_path = tmp.name
            try:
                pre_lm, post_lm = parse_tag_file_disk(tag_path)
                if pre_lm.size and post_lm.size and pre_lm.shape == post_lm.shape:
                    gt_shifts = np.linalg.norm(pre_lm - post_lm, axis=1)
                    st.info(f"GT mean: {gt_shifts.mean():.3f} mm | GT max: {gt_shifts.max():.3f} mm  (N={gt_shifts.shape[0]})")
                else:
                    st.warning("Parsed .tag has no/unequal landmarks.")
            finally:
                try: os.remove(tag_path)
                except Exception: pass

        # Visualize slices
        st.subheader("Flow magnitude slice")
        axis = st.radio("View", options=["axial", "coronal", "sagittal"], horizontal=True)
        idx_max = dict(axial=mag.shape[0]-1, coronal=mag.shape[1]-1, sagittal=mag.shape[2]-1)[axis]
        idx = st.slider("Slice index", 0, idx_max, idx_max//2)
        visualize_slice(mag, axis, idx)

        # Download raw flow (optional)
        if save_flow:
            # Save to a temp .npy and expose download button
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                np.save(tmp.name, flow)
                path = tmp.name
            with open(path, "rb") as fh:
                st.download_button("Download flow (.npy)", data=fh, file_name="flow.npy", mime="application/octet-stream")
            try: os.remove(path)
            except Exception: pass

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
    finally:
        gc.collect()
