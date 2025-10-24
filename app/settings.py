import os
from pathlib import Path


# Model path can be overridden via env var on Render
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/brain_shift_model_fulldataset.pt"))


# Where to stash uploaded files & outputs (Render ephemeral disk is fine)
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/ius_runs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)


# Target 3D shape expected by model
TARGET_SHAPE = (128, 128, 128)


# Max upload size guard (bytes) ~ 1.2 GB
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(1_200_000_000)))