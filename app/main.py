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
from . import llm as llm_module
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
    # no-cache: browsers must pick up UI updates (a stale page = no memory)
    return FileResponse(index_file, headers={"Cache-Control": "no-cache"})


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
    # Build a stable session id: payload wins, then header, then IP.
    # The frontend sends its own session_id. The IP fallback is only for simple
    # clients; behind Render's proxy many visitors can share one IP, so any
    # real frontend should send a session_id (conversation memory is keyed on it).
    session_id = (
        payload.session_id
        or request.headers.get("X-Session-Id")
        or (request.client.host if request.client else "anon")
    )

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    # The page sends its conversation as "history" (role/content messages).
    # Read it from the raw body so this works whatever fields ChatRequest declares.
    try:
        raw = await request.json()
    except Exception:  # noqa: BLE001
        raw = {}
    client_history = raw.get("history") if isinstance(raw, dict) else None

    logger.info("chat | session=%s | chars=%d | client_history=%s",
                session_id, len(message),
                len(client_history) if isinstance(client_history, list) else "none")
    reply, provider = await generate_reply(session_id, message, client_history)

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


@app.get("/diagnose", include_in_schema=False)
async def diagnose(key: str = "") -> dict:
    """
    Memory self-test. Enable by setting DIAGNOSE_KEY on Render, then open
    /diagnose?key=<that value>. It runs a short conversation through the real
    Space and reports where (if anywhere) conversation memory breaks.
    """
    expected = getattr(settings, "diagnose_key", "")
    if not expected or key != expected:
        raise HTTPException(status_code=404, detail="Not found")

    from .hf_client import HuggingFaceClient
    client = HuggingFaceClient(settings.hf_space_url, settings.hf_token, settings.hf_timeout)

    q1, q2, q3 = "What is the capital of Nigeria?", "What about Ghana?", "What was my last question?"
    r1 = await client.chat_raw(q1, [], debug=True)
    h = [{"role": "user", "content": q1}, {"role": "assistant", "content": r1.get("response", "")}]
    r2 = await client.chat_raw(q2, h, debug=True)
    h += [{"role": "user", "content": q2}, {"role": "assistant", "content": r2.get("response", "")}]
    r3 = await client.chat_raw(q3, h, debug=True)

    if "mode" not in r1:
        verdict = ("Space is running OLD code (no 'mode' in its reply). Deploy the updated "
                   "app.py to the Space and let it rebuild.")
    elif r2.get("mode") != "followup":
        verdict = (f"Space received the history but treated 'What about Ghana?' as "
                   f"'{r2.get('mode')}'. See debug for what it saw.")
    elif r3.get("mode") != "recall":
        verdict = f"Recall question was handled as '{r3.get('mode')}' instead of 'recall'."
    else:
        verdict = ("Space memory works. If real chats still have no memory, the problem is in "
                   "this proxy: see 'proxy' below (several workers? old llm.py? no session_id?).")

    return {
        "verdict": verdict,
        "space": {"turn2_mode": r2.get("mode"), "turn2_context_turns":
                  (r2.get("debug") or {}).get("context_turns_used"),
                  "turn2_prompt": (r2.get("debug") or {}).get("prompt"),
                  "turn2_reply": r2.get("response"), "turn3_reply": r3.get("response")},
        "proxy": {
            "pid": os.getpid(),                      # changes between calls => several workers
            "llm_py_is_new_version": hasattr(llm_module, "_MAX_MESSAGES"),
            "config_is_new_version": hasattr(settings, "hf_history_turns"),
            "stored_sessions_in_this_process": len(llm_module._HISTORY),
            "provider": settings.llm_provider,
        },
    }


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