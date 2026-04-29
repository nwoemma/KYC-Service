from fastapi import Request
from fastapi.responses import JSONResponse
from app.config import settings
from app.utils.logger import logger, hash_token
from app.utils.response import error_response


async def auth_middleware(request: Request, call_next):
    # Skip auth for health check
    if request.url.path == "/health":
        return await call_next(request)

    token = request.headers.get("X-Service-Token")

    if not token:
        logger.warning(json_extra("missing_token", request))
        return error_response(401, "UNAUTHORIZED", "Missing X-Service-Token header")

    if token not in settings.SERVICE_TOKENS:
        logger.warning(json_extra("invalid_token", request, token))
        return error_response(401, "UNAUTHORIZED", "Invalid X-Service-Token")

    # Attach hashed token to request state for logging downstream
    request.state.token_hash = hash_token(token)
    return await call_next(request)


def json_extra(event: str, request: Request, token: str = "") -> str:
    return {
        "event": event,
        "path": request.url.path,
        "token_hash": hash_token(token) if token else "none",
    }