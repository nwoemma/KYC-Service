from fastapi.responses import JSONResponse
from typing import Any, Optional
import time


def success_response(data: dict, processing_time_ms: float) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "processing_time_ms": round(processing_time_ms, 2),
            **data,
        },
    )


def error_response(
    status_code: int,
    code: str,
    message: str,
    processing_time_ms: float = 0.0,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "code": code,
            "message": message,
            "processing_time_ms": round(processing_time_ms, 2),
        },
    )


def ms_since(start: float) -> float:
    """Return elapsed milliseconds since start time."""
    return (time.perf_counter() - start) * 1000