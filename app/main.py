from __future__ import annotations
import shutil
import uuid
from pathlib import Path
from typing import Optional


from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


from .settings import WORK_DIR, MAX_UPLOAD_BYTES
from .inference import preprocess_pair, predict_flow, warp_image, shift_metrics, make_gradcam
from .utils import save_numpy, parse_tag_file
from .plotting import save_center_slices, save_mip, save_quiver



app = FastAPI(title="iUS Brain Shift")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # tighten to your Vercel domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/health")
def health():
    return {"ok": True}



@app.post("/api/predict")
async def api_predict(
    pre: UploadFile = File(...),
    post: UploadFile = File(...),
    tag: Optional[UploadFile] = File(None),
    make_cam: bool = False,
    make_quiver: bool = False,
):
    # Size guards
    if pre.size and pre.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail="Pre file too large")
    if post.size and post.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail="Post file too large")


    run_id = str(uuid.uuid4())[:8]
    run_dir = WORK_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)


    pre_path = run_dir / f"pre_{pre.filename}"
    post_path = run_dir / f"post_{post.filename}"
    with open(pre_path, "wb") as f: shutil.copyfileobj(pre.file, f)
    with open(post_path, "wb") as f: shutil.copyfileobj(post.file, f)


    tag_path = None
    if tag is not None:
        tag_path = run_dir / f"{tag.filename}"
        with open(tag_path, "wb") as f: shutil.copyfileobj(tag.file, f)


    # Inference
    tensor_pack = preprocess_pair(pre_path, post_path)
    flow = predict_flow(tensor_pack["pre_t"], tensor_pack["post_t"]) # (1,3,D,H,W)
    warped = warp_image(tensor_pack["pre_t"], flow)


    metrics = shift_metrics(flow)


    # Artifacts
    flow_path = save_numpy(flow.cpu().numpy(), run_dir / "flow.npy")
    warped_png = save_center_slices(warped, run_dir / "warped_pre_slices.png", title="Warped Pre (center slices)")
    import numpy as np
    mag = (flow.norm(dim=1)).squeeze(0).cpu().numpy()
    mip_png = save_mip(mag, run_dir / "shift_magnitude_mip.png", axis=0, title="Shift magnitude MIP")


    cam_pngs = []
    if make_cam:
        cam_ax_png, cam_cor_png, cam_sag_png = make_gradcam(tensor_pack["pre_t"], tensor_pack["post_t"], flow, run_dir)
        cam_pngs = [f"/files/{run_id}/{p.name}" for p in [cam_ax_png, cam_cor_png, cam_sag_png]]


    quiver_png = None
    if make_quiver and tag_path is not None:
        try:
            quiver_png = save_quiver(
                run_dir,
                tag_path,
                flow,
                pre_path,
        )
        except Exception as e:
            quiver_png = None


    resp = {
        "run_id": run_id,
        "metrics": metrics,
        "artifacts": {
            "warped_png": f"/files/{run_id}/warped_pre_slices.png",
            "mip_png": f"/files/{run_id}/shift_magnitude_mip.png",
            "flow_npy": f"/files/{run_id}/flow.npy",
            "gradcam_pngs": cam_pngs,
            "quiver_png": (f"/files/{run_id}/{Path(quiver_png).name}" if quiver_png else None),
        },
    }
    return JSONResponse(resp)



@app.get("/files/{run_id}/{filename}")
def get_file(run_id: str, filename: str):
    path = WORK_DIR / run_id / filename
    if not path.exists():
        raise HTTPException(404, detail="Not found")
    return FileResponse(path)


# Minimal HTML demo page (optional)
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

