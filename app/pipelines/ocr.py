import os
os.environ["FLAGS_use_mkldnn"] = "0"  # ← ADD THIS
os.environ["FLAGS_onednn_cpu"] = "0" 
import re
import cv2
import numpy as np
from app.models.engine import engine
from app.config import settings
from app.utils.image import decode_image, assess_document_quality
from app.utils.logger import logger

def preprocess(img: np.ndarray) -> np.ndarray:
    """Deskew, denoise, enhance contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


FIELD_PATTERNS = {
    "nin": {
    "nin_number":   r"(?:NIN)[:\s]*([\d]{11})",
    "surname":      r"(?:Surname)[:\s]+([A-Z]+)",
    "first_name":   r"(?:First\s*Name)[:\s]+([A-Z]+)",
    "middle_name":  r"(?:Middle\s*Name)[:\s]+([A-Z]+)",
    "gender":       r"(?:Gender)[:\s]+(Male|Female|M|F)\b",
    "tracking_id":  r"(?:Tracking\s*ID)[:\s]*([A-Z0-9]+)",
    "address":      r"(?:Address)[:\s]+([A-Za-z0-9\s\'\,\.]+)",
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
    "nepa_bill": {
        "company_name":     r"(?:PORT HARCOURT|PHED|EEDC|IBEDC|EKEDC|BEDC|AEDC|KAEDCO)[A-Za-z\s]*(?:ELECTRICITY|DISTRIBUTION|COMPANY)?",
        "account_number":   r"(?:Account\s*No|Account\s*Numbe?r?|ACC\.?\s*NO)[:\s\.]*(\d+)",
        "customer_name":    r"(?:Name)[:\s]+([A-Z][A-Za-z\s\.]+)",
        "meter_number":     r"(?:Meter)[:\s#]*([A-Z0-9]+)",
        "amount_due":       r"(?:PAY TOTAL DUE|PAY TOTAL DUE NOW|Total Bill)[^\d]*([\d,]+\.?\d*)",
        "due_date":         r"(?:DUE DATE|Due Date)[:\s]+(\d{1,2}\s+[A-Z]{3}\s+\d{4})",
        "supply_address":   r"(?:Supply Address)[:\s]+([A-Za-z0-9\s\,\.]+)",
        "bill_address":     r"(?:Bill Delivery Address)[:\s]+([A-Za-z0-9\s\,\.]+)",
        "bill_id":          r"(?:Bill\s*ID)[:\s]*(\d+)",
        "billing_month":    r"(?:BILLING MONTH|energy used in)[:\s]*([A-Z]{3}[-\s]\d{4})",
        "tin":              r"(?:TIN)[:\s]*([\d\-]+)",
        "previous_balance": r"(?:Previous Bal)[:\s#.]*([^\s]+)",
        "current_charges":  r"(?:Current Charges)[:\s#.]*([\d,]+\.?\d*)",
        "vat":              r"(?:VAT)[^\d]*([\d,]+\.?\d*)",
        "arrears":          r"(?:Net Arrears|Arrears)[:\s#.]*([\d,]+\.?\d*)",
        "tariff_rate":      r"(?:TARIFF RATE|Rate)[:\s]*([\d\.]+)",
        "mobile_number":    r"(?:Mobile No)[:\s]*([\d]+)",
        "bill_period":      r"(?:MR DATE|Bill Prod\.\s*Date)[:\s]*(\d{1,2}\s+[A-Z]{3}\s+\d{4}|\d{2}\s+\w+\s+\d{4})",
        "units_billed":     r"(?:UNITS BILLED|Units\s*Billed)[:\s]*([\d\.]+)",
        "total_charges":    r"(?:TOTAL AMOUNT|TOTAL CHARGES)[^\d]*([\d,]+\.?\d*)",
         "cin":              r"(?:CIN)[:\s]*([A-Z0-9]+)",
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

    raw = engine.ocr_reader.ocr(processed)
    logger.info({"event": "ocr_keys", "data": str(list(raw[0].keys())) if raw else "empty"})
    logger.info({"event": "ocr_raw_output", "data": str(raw)[:1000]})
    ocr_results = []
    for item in raw:
        # New PaddleOCR format returns dict with 'rec_texts', 'rec_scores', 'det_polys'
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])
        boxes = item.get("rec_polys", [])
        for box, text, conf in zip(boxes, texts, scores):
            ocr_results.append((box, text, conf))

    if not ocr_results:
        raise ValueError("TEXT_UNREADABLE")

    fields = extract_fields(ocr_results, doc_type)

    del img, processed

    return {
        "doc_type": doc_type,
        "fields": fields,
        "quality_flags": quality,
    }