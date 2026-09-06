# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import sys
import os
import httpx
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── Daily chat request limits ──────────────────────────────────────
REGULAR_DAILY_LIMIT = 5
PREMIUM_DAILY_LIMIT = 15

# Optional persistent usage store (JSONBin) — matches the JSONBin-based
# storage already used elsewhere in this project. If these env vars are
# not set on Render, usage falls back to an in-memory counter that resets
# whenever this service restarts or spins down after inactivity — fine
# for testing, but set these for the daily limit to actually hold up.
JSONBIN_API_KEY = os.getenv("JSONBIN_API_KEY")
JSONBIN_USAGE_BIN_ID = os.getenv("JSONBIN_USAGE_BIN_ID")
JSONBIN_BASE_URL = "https://api.jsonbin.io/v3/b"

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    is_premium: Optional[bool] = False

class ChatResponse(BaseModel):
    success: bool
    response: str
    sentences: Optional[List[str]] = None
    session_id: Optional[str] = None
    interaction_count: Optional[int] = None
    model_loading: Optional[bool] = False
    limit_reached: Optional[bool] = False
    remaining_today: Optional[int] = None
    daily_limit: Optional[int] = None

class SessionResponse(BaseModel):
    session_id: str
    session: dict

# Chat session class
class ChatSession:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.history: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def add_interaction(self, user_message: str, ai_response: str):
        self.history.append({"trigger": user_message, "reply": ai_response})
        self.last_updated = datetime.now()
    
    def to_json(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "total_interactions": len(self.history),
            "history": self.history
        }

# Session manager
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        session = ChatSession()
        self.sessions[session.session_id] = session
        self.current_session_id = session.session_id
        return session
    
    def create_session(self) -> ChatSession:
        session = ChatSession()
        self.sessions[session.session_id] = session
        self.current_session_id = session.session_id
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.sessions.get(session_id)
    
    def get_current_session(self) -> Optional[ChatSession]:
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return None
    
    def get_all_sessions(self) -> List[Dict]:
        sessions = []
        for s in self.sessions.values():
            sessions.append({
                'session_id': s.session_id,
                'created_at': s.created_at.isoformat(),
                'last_updated': s.last_updated.isoformat(),
                'total_interactions': len(s.history)
            })
        sessions.sort(key=lambda x: x['last_updated'], reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            if self.current_session_id == session_id:
                self.current_session_id = None
            del self.sessions[session_id]
            return True
        return False


# ── Daily usage tracking ────────────────────────────────────────────
# Keyed by user_id -> {"date": "YYYY-MM-DD", "count": N}. Persisted to
# JSONBin when configured; otherwise kept in memory for this process only.
_usage_cache: Dict[str, Dict] = {}
_usage_lock = asyncio.Lock()


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


async def _load_usage_data() -> Dict:
    global _usage_cache
    if JSONBIN_API_KEY and JSONBIN_USAGE_BIN_ID:
        try:
            resp = await client.get(
                f"{JSONBIN_BASE_URL}/{JSONBIN_USAGE_BIN_ID}/latest",
                headers={"X-Master-Key": JSONBIN_API_KEY},
                timeout=8.0
            )
            if resp.status_code == 200:
                _usage_cache = resp.json().get("record", {}) or {}
        except Exception as e:
            logger.warning(f"JSONBin usage load failed, using in-memory cache: {e}")
    return _usage_cache


async def _save_usage_data(data: Dict):
    global _usage_cache
    _usage_cache = data
    if JSONBIN_API_KEY and JSONBIN_USAGE_BIN_ID:
        try:
            await client.put(
                f"{JSONBIN_BASE_URL}/{JSONBIN_USAGE_BIN_ID}",
                headers={"X-Master-Key": JSONBIN_API_KEY, "Content-Type": "application/json"},
                json=data,
                timeout=8.0
            )
        except Exception as e:
            logger.warning(f"JSONBin usage save failed: {e}")


async def check_and_increment_usage(user_id: str, is_premium: bool) -> Dict:
    """
    Checks today's usage for user_id against their daily limit. If they're
    still under the limit, increments the count and returns allowed=True.
    If they've already hit the limit, returns allowed=False WITHOUT
    incrementing further.
    """
    async with _usage_lock:
        data = await _load_usage_data()
        today = _today_str()
        limit = PREMIUM_DAILY_LIMIT if is_premium else REGULAR_DAILY_LIMIT

        record = data.get(user_id, {})
        if record.get("date") != today:
            record = {"date": today, "count": 0}

        used = record["count"]
        if used >= limit:
            data[user_id] = record
            await _save_usage_data(data)
            return {"allowed": False, "remaining": 0, "limit": limit, "used": used}

        record["count"] = used + 1
        data[user_id] = record
        await _save_usage_data(data)
        return {"allowed": True, "remaining": limit - record["count"], "limit": limit, "used": record["count"]}


async def get_usage_status(user_id: str, is_premium: bool) -> Dict:
    """Read-only lookup of today's usage, without incrementing anything."""
    async with _usage_lock:
        data = await _load_usage_data()
        today = _today_str()
        limit = PREMIUM_DAILY_LIMIT if is_premium else REGULAR_DAILY_LIMIT
        record = data.get(user_id, {})
        used = record.get("count", 0) if record.get("date") == today else 0
        return {"used": used, "limit": limit, "remaining": max(0, limit - used)}

# Create FastAPI app
app = FastAPI(
    title="IvieAI Chat API (Spaces Proxy)",
    description="Proxy for IvieAI on HuggingFace Spaces",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize session manager
session_manager = SessionManager()

# HuggingFace Space configuration
SPACE_URL = "https://GeorgeUwaifo-IvieAI.hf.space"
CHAT_ENDPOINT = f"{SPACE_URL}/chat"
HEALTH_ENDPOINT = f"{SPACE_URL}/health"

# HTTP client for making requests to Spaces
client = httpx.AsyncClient(timeout=60.0)  # Longer timeout for model loading

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up HTTP client on shutdown."""
    await client.aclose()

# Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat interface."""
    with open("app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    """Health check - checks both proxy and Spaces."""
    try:
        # Check if Spaces is reachable
        response = await client.get(HEALTH_ENDPOINT, timeout=5.0)
        if response.status_code == 200:
            spaces_status = response.json()
            return JSONResponse(content={
                "status": "healthy",
                "spaces_reachable": True,
                "spaces_status": spaces_status
            })
        else:
            return JSONResponse(content={
                "status": "degraded",
                "spaces_reachable": False,
                "error": f"Spaces returned status {response.status_code}"
            }, status_code=503)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(content={
            "status": "unhealthy",
            "spaces_reachable": False,
            "error": str(e)
        }, status_code=503)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Proxy chat messages to HuggingFace Spaces."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")

        # ── Enforce the daily request limit before doing anything else,
        # so a user who's already used up their quota never costs us a
        # call to the Spaces backend.
        user_id = request.user_id or "anonymous"
        is_premium = bool(request.is_premium)
        usage = await check_and_increment_usage(user_id, is_premium)

        if not usage["allowed"]:
            return ChatResponse(
                success=True,
                response=(
                    f"You've reached your daily limit of {usage['limit']} chat "
                    f"requests. Please come back tomorrow"
                    + ("." if is_premium else ", or upgrade to Premium for more chats per day.")
                ),
                session_id=request.session_id,
                limit_reached=True,
                remaining_today=0,
                daily_limit=usage["limit"]
            )
        
        # Check if Spaces is healthy
        try:
            health_check = await client.get(HEALTH_ENDPOINT, timeout=5.0)
            if health_check.status_code != 200:
                return ChatResponse(
                    success=True,
                    response="The AI model is currently unavailable. Please try again later.",
                    model_loading=True,
                    remaining_today=usage["remaining"],
                    daily_limit=usage["limit"]
                )
        except Exception:
            return ChatResponse(
                success=True,
                response="Cannot connect to the AI service. Please check your connection and try again.",
                model_loading=True,
                remaining_today=usage["remaining"],
                daily_limit=usage["limit"]
            )
        
        # Get or create session
        session = session_manager.get_or_create_session(request.session_id)
        
        # Forward request to Spaces
        payload = {
            "message": request.message,
            "session_id": session.session_id
        }
        
        logger.info(f"Forwarding request to Spaces: {payload}")
        response = await client.post(CHAT_ENDPOINT, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Spaces returned error: {response.status_code} - {response.text}")
            return ChatResponse(
                success=True,
                response="The AI service encountered an issue. Please try again.",
                session_id=session.session_id,
                model_loading=True,
                remaining_today=usage["remaining"],
                daily_limit=usage["limit"]
            )
        
        result = response.json()
        logger.info(f"Spaces response: {result}")
        
        # Process the response
        if result.get("success"):
            ai_response = result.get("response", "I couldn't generate a response. Please try again.")
            sentences = result.get("sentences", [])
            
            # Add interaction to session
            session.add_interaction(request.message, ai_response)
            
            return ChatResponse(
                success=True,
                response=ai_response,
                sentences=sentences,
                session_id=session.session_id,
                interaction_count=len(session.history),
                model_loading=result.get("model_loading", False),
                remaining_today=usage["remaining"],
                daily_limit=usage["limit"]
            )
        else:
            error_msg = result.get("error", "Unknown error from AI service")
            return ChatResponse(
                success=True,
                response=f"Error from AI service: {error_msg}",
                session_id=session.session_id,
                model_loading=True,
                remaining_today=usage["remaining"],
                daily_limit=usage["limit"]
            )
    
    except httpx.TimeoutException:
        logger.error("Request to Spaces timed out")
        return ChatResponse(
            success=True,
            response="The AI is taking too long to respond. Please try again.",
            model_loading=True
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage/{user_id}")
async def usage_status(user_id: str, is_premium: bool = False):
    """Look up today's remaining chat quota for a user without using one up."""
    status = await get_usage_status(user_id, is_premium)
    return JSONResponse(content=status)

@app.get("/sessions")
async def list_sessions():
    """List all chat sessions."""
    return JSONResponse(content={"sessions": session_manager.get_all_sessions()})

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get a specific session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(content=session.to_json())

@app.post("/session/new")
async def new_session():
    """Create a new session."""
    session = session_manager.create_session()
    return SessionResponse(
        session_id=session.session_id,
        session=session.to_json()
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(content={"message": "Session deleted"})

@app.get("/session/current")
async def get_current_session():
    """Get the current session."""
    session = session_manager.get_current_session()
    if not session:
        raise HTTPException(status_code=404, detail="No active session")
    return JSONResponse(content=session.to_json())

@app.get("/export/{session_id}/{fmt}")
async def export_session(session_id: str, fmt: str):
    """Export a session in JSON or text format."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if fmt == "json":
        return JSONResponse(content=session.to_json())
    elif fmt == "text":
        return JSONResponse(content={"text": session.to_text()})
    else:
        raise HTTPException(status_code=400, detail="Invalid format")

@app.get("/test")
async def test():
    """Test endpoint."""
    try:
        response = await client.get(HEALTH_ENDPOINT, timeout=5.0)
        return JSONResponse(content={
            "proxy_status": "healthy",
            "spaces_reachable": True,
            "spaces_status_code": response.status_code,
            "spaces_response": response.json() if response.status_code == 200 else None
        })
    except Exception as e:
        return JSONResponse(content={
            "proxy_status": "degraded",
            "spaces_reachable": False,
            "error": str(e)
        })
