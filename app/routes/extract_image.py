import time
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import Response
from app.models.engine import engine
from app.config import settings
from app.utils.response import error_response, ms_since
from app.utils.image import decode_image

router = APIRouter()

@router.post("/extract-image")
async def extract_image(
    request: Request,
    document: UploadFile = File(...),
    request_id: str = None,
):
    start = time.perf_counter()
    request_id = request_id or str(uuid.uuid4())

    # Validate MIME type
    if document.content_type not in settings.ALLOWED_MIME_TYPES:
        return error_response(400, "INVALID_INPUT", f"Unsupported MIME type: {document.content_type}", ms_since(start))

    doc_bytes = await document.read()

    # Validate file size
    if len(doc_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        return error_response(400, "INVALID_INPUT", f"File too large", ms_since(start))

    img = decode_image(doc_bytes)

    # Detect face
    faces = engine.face_app.get(img)

    if len(faces) == 0:
        return error_response(400, "FACE_NOT_DETECTED", "No face found in document", ms_since(start))

    # Crop face from image
    face = faces[0]
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    # Add padding
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    cropped = img[y1:y2, x1:x2]

    # Convert to JPEG bytes
    _, buffer = cv2.imencode(".jpg", cropped)
    image_bytes = buffer.tobytes()

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={"X-Request-ID": request_id}
    )