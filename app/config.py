from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Auth
    SERVICE_TOKENS: List[str] = []
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # File validation
    MAX_FILE_SIZE_MB: int = 5
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]

    # Face matching thresholds
    FACE_MATCH_PASS_THRESHOLD: float = 0.75
    FACE_MATCH_REVIEW_THRESHOLD: float = 0.55

    # Document types
    ALLOWED_DOC_TYPES: List[str] = ["nin", "passport", "drivers_license"]

    # OCR confidence threshold
    OCR_CONFIDENCE_THRESHOLD: float = 0.60

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()