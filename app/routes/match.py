import time
import uuid
from fastapi import APIRouter, UploadFile, File, Form, Request
from app.pipelines.face import run_face_match
from app.config import settings
from app.utils.response import success_response, error_response, ms_since
from app.utils.logger import logger

router = APIRouter()


@router.post("/match")
async def match_faces(
    request: Request,
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    request_id: str = Form(default=None),
):
    start = time.perf_counter()
    request_id = request_id or str(uuid.uuid4())
    token_hash = getattr(request.state, "token_hash", "unknown")

    # Validate MIME types
    for img, label in [(image_a, "image_a"), (image_b, "image_b")]:
        if img.content_type not in settings.ALLOWED_MIME_TYPES:
            return error_response(
                400, "INVALID_INPUT",
                f"{label} has unsupported MIME type: {img.content_type}",
                ms_since(start),
            )

    # Validate file sizes
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    for data, label in [(bytes_a, "image_a"), (bytes_b, "image_b")]:
        if len(data) > max_bytes:
            return error_response(
                400, "INVALID_INPUT",
                f"{label} exceeds maximum file size of {settings.MAX_FILE_SIZE_MB}MB",
                ms_since(start),
            )

    try:
        result = run_face_match(bytes_a, bytes_b, settings)
        elapsed = ms_since(start)

        logger.info({
            "event": "match_success",
            "request_id": request_id,
            "token_hash": token_hash,
            "decision": result["decision"],
            "latency_ms": elapsed,
        })

        return success_response({"request_id": request_id, **result}, elapsed)

    except ValueError as e:
        elapsed = ms_since(start)
        logger.warning({
            "event": "match_error",
            "request_id": request_id,
            "token_hash": token_hash,
            "code": str(e),
            "latency_ms": elapsed,
        })
        return error_response(200, str(e), f"Face matching failed: {str(e)}", elapsed)

    except Exception as e:
        elapsed = ms_since(start)
        logger.error({
            "event": "match_exception",
            "request_id": request_id,
            "token_hash": token_hash,
            "error": str(e),
            "latency_ms": elapsed,
        })
        return error_response(500, "INTERNAL_ERROR", "An unexpected error occurred", elapsed)