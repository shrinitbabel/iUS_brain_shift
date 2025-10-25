from pydantic import BaseModel
from typing import Optional


class ShiftSummary(BaseModel):
    mean_shift_pred_mm: float
    max_shift_pred_mm: float
    mean_shift_tag_mm: Optional[float] = None
    max_shift_tag_mm: Optional[float] = None
    saved_artifact: Optional[str] = None # path to npy/nii if returned

