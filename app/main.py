import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.models.engine import engine
from app.middleware.auth import auth_middleware
from app.middleware.rate_limiter import rate_limit_middleware
from app.routes import match, extract
from app.utils.logger import logger

startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    engine.load()
    yield
    # Nothing to clean up — stateless


app = FastAPI(
    title="KYC Image Matching & Document Extraction Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,   # Disable Swagger in production
    redoc_url=None,
)

# Middleware — order matters: auth runs first, then rate limit
app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware)
app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)


# Routes
app.include_router(match.router)
app.include_router(extract.router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": engine.is_ready,
        "uptime_s": round(time.time() - startup_time, 2),
    }