import os
os.environ["FLAGS_use_mkldnn"] = "0"  # ← ADD THIS
os.environ["FLAGS_onednn_cpu"] = "0" 
import re
import cv2
import numpy as np
from app.models.engine import engine
from app.config import settings
from app.utils.image import decode_image, assess_document_quality


def preprocess(img: np.ndarray) -> np.ndarray:
    """Deskew, denoise, enhance contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


FIELD_PATTERNS = {
    "nin": {
        "nin_number": r"\b\d{11}\b",
        "full_name": r"(?:Name|NAME)[:\s]+([A-Z\s]+)",
        "date_of_birth": r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b",
        "gender": r"\b(Male|Female|M|F)\b",
    },
    "passport": {
        "passport_number": r"\b[A-Z]{1}\d{8}\b",
        "full_name": r"(?:Surname|Given Names)[:\s]+([A-Z\s]+)",
        "date_of_birth": r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b",
        "expiry_date": r"(?:Expiry|Expiration)[:\s]+(\d{2}[\/\-]\d{2}[\/\-]\d{4})",
        "nationality": r"(?:Nationality)[:\s]+([A-Z\s]+)",
    },
    "drivers_license": {
        "license_number": r"\b[A-Z]{3}\d{6}[A-Z]{2}\d{2}\b",
        "full_name": r"(?:Name|NAME)[:\s]+([A-Z\s]+)",
        "date_of_birth": r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b",
        "expiry_date": r"(?:Expiry|Exp)[:\s]+(\d{2}[\/\-]\d{2}[\/\-]\d{4})",
        "state": r"(?:State)[:\s]+([A-Z\s]+)",
    },
}


def extract_fields(ocr_results: list, doc_type: str) -> dict:
    patterns = FIELD_PATTERNS.get(doc_type, {})
    full_text = " ".join([text for (_, text, _) in ocr_results])
    confidences = {text: conf for (_, text, conf) in ocr_results}

    extracted = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            value = match.group(1) if match.lastindex else match.group(0)
            value = value.strip()
            conf = confidences.get(value, settings.OCR_CONFIDENCE_THRESHOLD)
            extracted[field] = {
                "value": value,
                "confidence": round(float(conf), 4),
                "low_confidence": float(conf) < settings.OCR_CONFIDENCE_THRESHOLD,
            }
        else:
            extracted[field] = {
                "value": None,
                "confidence": 0.0,
                "low_confidence": True,
            }
    return extracted


def run_document_extraction(image_bytes: bytes, doc_type: str) -> dict:
    img = decode_image(image_bytes)
    quality = assess_document_quality(img)
    processed = preprocess(img)

    ocr_results = engine.ocr_reader.ocr(processed)
    ocr_results = [(line[0], line[1][0], line[1][1]) for batch in ocr_results for line in batch]
    if not ocr_results:
        raise ValueError("TEXT_UNREADABLE")

    fields = extract_fields(ocr_results, doc_type)

    del img, processed

    return {
        "doc_type": doc_type,
        "fields": fields,
        "quality_flags": quality,
    }