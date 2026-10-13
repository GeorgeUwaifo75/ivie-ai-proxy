"""Ivie AI — FastAPI entrypoint.

Serves the talking-avatar frontend from /static and exposes:
  GET  /health  → health check for Render
  POST /chat    → chat with Ivie
  GET  /        → the avatar UI
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .llm import clear_history, generate_reply
from .schemas import ChatRequest, ChatResponse, HealthResponse

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ivie")

settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Talking AI avatar — FastAPI port for Render.com",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files -----------------------------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning("Static directory not found: %s", STATIC_DIR)


# --- Routes -----------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the avatar UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Render / uptime health check."""
    provider = settings.llm_provider
    status = "ok"
    if provider == "openai" and not settings.openai_api_key:
        status = "degraded"
    return HealthResponse(
        status=status,          # type: ignore[arg-type]
        service=settings.app_name,
        provider=provider,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Main chat endpoint used by the avatar frontend."""
    # Build a stable session id: client header wins, then payload, then IP.
    session_id = (
        payload.session_id
        or request.headers.get("X-Session-Id")
        or (request.client.host if request.client else "anon")
    )

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    logger.info("chat | session=%s | chars=%d", session_id, len(message))
    reply, provider = await generate_reply(session_id, message)

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        model=provider,
    )


@app.post("/reset", tags=["chat"])
async def reset(request: Request) -> dict:
    """Clear conversation history for a session."""
    session_id = request.headers.get("X-Session-Id") or (
        request.client.host if request.client else "anon"
    )
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# --- Global error handler ---------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": 500},
    )


# --- Dev entrypoint ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )