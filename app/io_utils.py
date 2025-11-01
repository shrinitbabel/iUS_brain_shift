import io
import numpy as np
import nibabel as nib
from typing import Tuple, Optional
import os, tempfile, shutil
from fastapi import UploadFile

# MINC (.mnc) reader using nibabel
# Returns float32 array and affine for spacing/origin if needed


# app/io_utils.py
def load_mnc_bytes(file_bytes: bytes,
                   max_dim: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load .mnc with mmap and coarse subsample if needed so the largest
    dimension <= max_dim before any dense ops.
    """
    with tempfile.NamedTemporaryFile(suffix=".mnc", delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush(); path = tmp.name
    try:
        img = nib.load(path, mmap=True)  # MINC2 uses h5py underneath
        shape = img.shape  # (Z, Y, X) typically
        # integer stride so biggest dim is <= max_dim
        factor = max(1, int(np.ceil(max(shape) / max_dim)))
        slicer = tuple(slice(None, None, factor) for _ in shape)
        # This slices the memmapped data (low-memory)
        arr = np.asarray(img.dataobj[slicer], dtype=np.float32, order="C")
        return arr, img.affine
    finally:
        try: os.remove(path)
        except Exception: pass

def load_mnc_uploadfile(f: UploadFile, max_dim: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stream an UploadFile to a temp .mnc on disk (no full .read() into RAM),
    then load with nibabel using mmap and coarse subsampling so the largest
    dimension <= max_dim before any dense ops.
    """
    # Write the streamed file to disk
    with tempfile.NamedTemporaryFile(suffix=".mnc", delete=False) as tmp:
        # copy file-like to disk without buffering everything in memory
        f.file.seek(0)
        shutil.copyfileobj(f.file, tmp)
        tmp.flush()
        path = tmp.name

    try:
        # mmap avoids materializing entire volume
        img = nib.load(path, mmap=True)
        shape = img.shape  # e.g. (Z, Y, X)
        # integer stride so biggest dim <= max_dim
        factor = max(1, int(np.ceil(max(shape) / max_dim)))
        slicer = tuple(slice(None, None, factor) for _ in shape)
        arr = np.asarray(img.dataobj[slicer], dtype=np.float32, order="C")
        return arr, img.affine
    finally:
        try: os.remove(path)
        except Exception: pass


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