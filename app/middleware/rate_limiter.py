from fastapi import Request
from collections import defaultdict, deque
from app.config import settings
from app.utils.response import error_response
from app.utils.logger import logger
import time


# In-memory store: token_hash -> deque of request timestamps
_request_log: dict[str, deque] = defaultdict(deque)


async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)

    token_hash = getattr(request.state, "token_hash", "anonymous")
    now = time.time()
    window = 60  # 1 minute sliding window

    queue = _request_log[token_hash]

    # Remove timestamps outside the window
    while queue and queue[0] < now - window:
        queue.popleft()

    if len(queue) >= settings.RATE_LIMIT_PER_MINUTE:
        logger.warning({
            "event": "rate_limit_exceeded",
            "token_hash": token_hash,
            "path": request.url.path,
        })
        return error_response(429, "RATE_LIMIT_EXCEEDED", "Too many requests. Please slow down.")

    queue.append(now)
    return await call_next(request)