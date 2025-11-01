from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    MODEL_PATH: str = Field(default="models/brain_shift_model_02.pt")
    DEVICE: str = Field(default="cuda") # falls back to cpu if not available
    ALLOWED_ORIGINS: List[str] = [
        "https://www.babels.ai",
        "https://babels.ai",
        "http://localhost:3000",
    ]
    MAX_UPLOAD_MB: int = 500 # reject very large uploads


settings = Settings()