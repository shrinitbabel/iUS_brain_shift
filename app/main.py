from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ShiftSummary
from app.io_utils import load_mnc_bytes
from app.inference import run_model
from app.settings import settings

import torch, os, tempfile, numpy as np

app = FastAPI(title="Brain Shift API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model once
try:
    from modules.unet3d import UNet3D
    device_str = settings.DEVICE
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
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

MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024
def _check_size(f: UploadFile):
    if getattr(f, "size", None) and f.size > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {settings.MAX_UPLOAD_MB} MB): {f.filename}")

@app.post("/predict", response_model=ShiftSummary)
async def predict(
    pre: UploadFile = File(..., description="Pre-resection .mnc"),
    post: UploadFile = File(..., description="Post-resection .mnc"),
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

    saved_artifact = None
    if save_field:
        with tempfile.NamedTemporaryFile(prefix="flow_", suffix=".npy", delete=False, dir="/tmp") as tmp:
            np.save(tmp.name, flow)
            saved_artifact = tmp.name

    return ShiftSummary(
        mean_shift_pred_mm=round(float(mean_pred), 3),
        max_shift_pred_mm=round(float(max_pred), 3),
        mean_shift_tag_mm=None,
        max_shift_tag_mm=None,
        saved_artifact=saved_artifact,
    )

@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_READY}
