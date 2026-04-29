import numpy as np
import cv2


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into OpenCV image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def assess_quality(image: np.ndarray) -> dict:
    """Shared quality flags — blur and size."""
    quality_flags = {
        "is_blurry": False,
        "is_too_small": False,
    }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 100:
        quality_flags["is_blurry"] = True
    height, width = image.shape[:2]
    if height < 200 or width < 200:
        quality_flags["is_too_small"] = True
    return quality_flags


def assess_document_quality(image: np.ndarray) -> dict:
    """Extended quality flags for documents — adds glare detection."""
    flags = assess_quality(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags["possibly_glared"] = float(np.percentile(gray, 95)) > 240
    return flags