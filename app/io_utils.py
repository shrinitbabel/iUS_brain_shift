import io
import numpy as np
import nibabel as nib
from typing import Tuple, Optional


# MINC (.mnc) reader using nibabel
# Returns float32 array and affine for spacing/origin if needed


def load_mnc_bytes(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    bio = io.BytesIO(file_bytes)
    img = nib.load(bio)
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    return arr, img.affine


# Optional: parse a .tag file with two landmark lists
# Assumes your existing modules.ground_truth.parse_tag_file(pre)


def parse_tag_bytes(file_bytes: bytes):
    from modules.ground_truth import parse_tag_file # your function expects a path
    # Write to temp then parse
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".tag", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        path = tmp.name
    try:
        pre_pts, post_pts = parse_tag_file(path)
        return pre_pts, post_pts
    finally:
        try:
            os.remove(path)
        except Exception:
            pass