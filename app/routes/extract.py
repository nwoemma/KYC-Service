import time
import uuid
from fastapi import APIRouter, UploadFile, File, Form, Request
from app.pipelines.ocr import run_document_extraction
from app.config import settings
from app.utils.response import success_response, error_response, ms_since
from app.utils.logger import logger

router = APIRouter()


@router.post("/extract")
async def extract_document(
    request: Request,
    document: UploadFile = File(...),
    doc_type: str = Form(...),
    request_id: str = Form(default=None),
):
    start = time.perf_counter()
    request_id = request_id or str(uuid.uuid4())
    token_hash = getattr(request.state, "token_hash", "unknown")

    # Validate doc_type
    if doc_type not in settings.ALLOWED_DOC_TYPES:
        return error_response(
            400, "INVALID_INPUT",
            f"Unsupported doc_type '{doc_type}'. Allowed: {settings.ALLOWED_DOC_TYPES}",
            ms_since(start),
        )

    # Validate MIME type
    if document.content_type not in settings.ALLOWED_MIME_TYPES:
        return error_response(
            400, "INVALID_INPUT",
            f"Unsupported MIME type: {document.content_type}",
            ms_since(start),
        )

    # Validate file size
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    doc_bytes = await document.read()

    if len(doc_bytes) > max_bytes:
        return error_response(
            400, "INVALID_INPUT",
            f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB",
            ms_since(start),
        )

    try:
        result = run_document_extraction(doc_bytes, doc_type)
        elapsed = ms_since(start)

        logger.info({
            "event": "extract_success",
            "request_id": request_id,
            "token_hash": token_hash,
            "doc_type": doc_type,
            "latency_ms": elapsed,
        })

        return success_response({"request_id": request_id, **result}, elapsed)

    except ValueError as e:
        elapsed = ms_since(start)
        logger.warning({
            "event": "extract_error",
            "request_id": request_id,
            "token_hash": token_hash,
            "code": str(e),
            "latency_ms": elapsed,
        })
        return error_response(200, str(e), f"Extraction failed: {str(e)}", elapsed)

    except Exception as e:
        elapsed = ms_since(start)
        logger.error({
            "event": "extract_exception",
            "request_id": request_id,
            "token_hash": token_hash,
            "error": str(e),
            "latency_ms": elapsed,
        })
        return error_response(500, "INTERNAL_ERROR", "An unexpected error occurred", elapsed)