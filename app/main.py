from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ShiftSummary
from app.io_utils import load_mnc_bytes
from app.inference import run_model
from app.settings import settings
import torch, os, tempfile, numpy as np
import time, traceback, sys
from fastapi.responses import JSONResponse

app = FastAPI(title="Brain Shift API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_errors(request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[MW-UNHANDLED] {e}\n{tb}", file=sys.stderr, flush=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unhandled error: {e.__class__.__name__}: {e}", "trace": tb[-2000:]},  # short tail
        )
    finally:
        dur = (time.time() - start) * 1000
        print(f"[REQ] {request.method} {request.url.path} took {dur:.1f} ms", flush=True)

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

# in app/main.py
from fastapi import UploadFile, File, HTTPException
from app.io_utils import load_mnc_bytes
import numpy as np, traceback

@app.post("/probe")
async def probe(pre: UploadFile = File(...), post: UploadFile = File(...)):
    try:
        pre_b = await pre.read()
        post_b = await post.read()
        pre_arr, _ = load_mnc_bytes(pre_b)
        post_arr, _ = load_mnc_bytes(post_b)
        return {
            "pre_bytes": len(pre_b),
            "post_bytes": len(post_b),
            "pre_shape": list(pre_arr.shape),
            "post_shape": list(post_arr.shape),
            "pre_minmax": [float(np.min(pre_arr)), float(np.max(pre_arr))],
            "post_minmax": [float(np.min(post_arr)), float(np.max(post_arr))],
        }
    except Exception as e:
        print("[PROBE-ERR]", e, traceback.format_exc(), flush=True)
        raise HTTPException(status_code=400, detail=f"Probe error: {e}")

# in app/main.py
from fastapi import Query
from app.inference import preprocess_pair, run_model
import time

@app.post("/diagnose")
async def diagnose(
    pre: UploadFile = File(...),
    post: UploadFile = File(...),
    run_inference: bool = Query(False, description="Also run the model"),
):
    report = {}
    t0 = time.time()

    # 1) Read bytes
    pre_b = await pre.read(); post_b = await post.read()
    report["bytes"] = {"pre": len(pre_b), "post": len(post_b)}

    # 2) MINC load
    t1 = time.time()
    pre_arr, _ = load_mnc_bytes(pre_b)
    post_arr, _ = load_mnc_bytes(post_b)
    report["load_s"] = round(time.time() - t1, 3)
    report["shapes"] = {"pre": list(pre_arr.shape), "post": list(post_arr.shape)}

    # 3) Preprocess (normalize+resize)
    t2 = time.time()
    pre_t, post_t = preprocess_pair(pre_arr, post_arr)
    report["preprocess_s"] = round(time.time() - t2, 3)
    report["tensor_shapes"] = {"pre": list(pre_t.shape), "post": list(post_t.shape)}

    # 4) Inference (optional)
    if run_inference:
        t3 = time.time()
        mean_pred, max_pred, _ = run_model(pre_arr, post_arr, device, model)
        report["inference_s"] = round(time.time() - t3, 3)
        report["pred_mm"] = {"mean": round(float(mean_pred), 3), "max": round(float(max_pred), 3)}

    report["total_s"] = round(time.time() - t0, 3)
    return report

@app.post("/predict", response_model=ShiftSummary)
async def predict(
    pre: UploadFile = File(..., description="Pre-resection .mnc"),
    post: UploadFile = File(..., description="Post-resection .mnc"),
    save_field: bool = Query(False, description="If true, save flow to /tmp and return path"),
):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        _check_size(pre); _check_size(post)
        pre_bytes = await pre.read()
        post_bytes = await post.read()

        # MINC load (now supports MINC2 via h5py)
        try:
            pre_vol, _ = load_mnc_bytes(pre_bytes)
            post_vol, _ = load_mnc_bytes(post_bytes)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[MINC-LOAD] {e}\n{tb}")
            raise HTTPException(status_code=400, detail=f"Failed to read .mnc: {e}")

        # Inference (includes normalize+resize to 128^3)
        try:
            mean_pred, max_pred, flow = run_model(pre_vol, post_vol, device, model)
        except MemoryError as me:
            tb = traceback.format_exc()
            print(f"[INFER-OMEM] {me}\n{tb}")
            raise HTTPException(status_code=413, detail="Out of memory during inference.")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[INFER-ERR] {e}\n{tb}")
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

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

    except HTTPException:
        # already handled with detail + status
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[UNHANDLED] {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")

@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_READY}
