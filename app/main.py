## app/main.py

import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.schemas import ShiftSummary
from app.io_utils import load_mnc_bytes, parse_tag_bytes
from app.inference import run_model, summarize_tag
from app.settings import settings

import torch

# --- FastAPI init
app = FastAPI(title="Brain Shift API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model once at startup
try:
    from modules.unet3d import UNet3D
    device_str = settings.DEVICE
    device = torch.device(device_str if torch.cuda.is_available() and device_str=="cuda" else "cpu")
    model = UNet3D().to(device)
    if os.path.exists(settings.MODEL_PATH):
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    model.eval()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    model = None
    device = torch.device("cpu")
    print(f"Model failed to load: {e}")

# --- Helpers
MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024

def _check_size(f: UploadFile):
    if f.size and f.size > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {settings.MAX_UPLOAD_MB} MB): {f.filename}")

@app.post("/predict", response_model=ShiftSummary)
async def predict(
    pre: UploadFile = File(..., description="Pre-resection .mnc"),
    post: UploadFile = File(..., description="Post-resection .mnc"),
    tag: UploadFile | None = File(None, description="Optional landmarks .tag"),
    save_field: bool = Query(False, description="If true, save flow to /tmp and return path"),
):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    _check_size(pre); _check_size(post)
    pre_bytes = await pre.read()
    post_bytes = await post.read()

    try:
        pre_vol, _ = load_mnc_bytes(pre_bytes)
        post_vol, _ = load_mnc_bytes(post_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read .mnc: {e}")

    mean_pred, max_pred, flow = run_model(pre_vol, post_vol, device, model)

    mean_tag = max_tag = None
    if tag is not None:
        tag_bytes = await tag.read()
        try:
            pre_pts, post_pts = parse_tag_bytes(tag_bytes)
            mean_tag, max_tag = summarize_tag(pre_pts, post_pts)
        except Exception as e:
            # Do not fail request if tag parse fails; return only prediction
            print(f"Tag parse error: {e}")

    saved_artifact = None
    if save_field:
        with tempfile.NamedTemporaryFile(prefix="flow_", suffix=".npy", delete=False, dir="/tmp") as tmp:
            npy_path = tmp.name
        import numpy as np
        np.save(npy_path, flow)
        saved_artifact = npy_path

    return ShiftSummary(
        mean_shift_pred_mm=round(float(mean_pred), 3),
        max_shift_pred_mm=round(float(max_pred), 3),
        mean_shift_tag_mm=round(float(mean_tag), 3) if mean_tag is not None else None,
        max_shift_tag_mm=round(float(max_tag), 3) if max_tag is not None else None,
        saved_artifact=saved_artifact,
    )

@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_READY}
