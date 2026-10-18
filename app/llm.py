"""LLM abstraction.

Supports:
  - "huggingface": the IvieAI Space (default)
  - "openai"     : any OpenAI-compatible endpoint
  - "echo"       : offline stub

Conversation history is kept in-process per session_id (bounded, least-recently-
used conversations are evicted). For production you'd swap this for Redis / a DB —
but it's fine for a single Render instance.

Only SUCCESSFUL exchanges are recorded. Error notices ("I'm having trouble
reaching my backend…") and "still waking up" messages never enter the history,
so they can't be mistaken for real answers in later follow-ups.
"""
from __future__ import annotations

import logging
from collections import OrderedDict, deque
from typing import Deque

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# --- In-memory conversation store ------------------------------------------
# Keep enough for whichever provider needs the most turns; each provider then
# slices what it sends.
_HF_TURNS = getattr(settings, "hf_history_turns", 20)
_MAX_SESSIONS = getattr(settings, "max_sessions", 2000)
_MAX_MESSAGES = max(settings.max_history_turns, _HF_TURNS) * 2
_HISTORY: "OrderedDict[str, Deque[dict]]" = OrderedDict()


def _record(session_id: str, role: str, content: str) -> None:
    turns = _HISTORY.get(session_id)
    if turns is None:
        turns = _HISTORY[session_id] = deque(maxlen=_MAX_MESSAGES)
    turns.append({"role": role, "content": content})
    _HISTORY.move_to_end(session_id)
    while len(_HISTORY) > _MAX_SESSIONS:      # evict least recently used
        _HISTORY.popitem(last=False)


def get_history(session_id: str) -> list[dict]:
    return list(_HISTORY.get(session_id, ()))


def clear_history(session_id: str) -> None:
    _HISTORY.pop(session_id, None)


def clean_client_history(raw) -> list[dict]:
    """
    Validate history sent by the browser: a list of {"role": "user"|"assistant",
    "content": str}. Anything else is dropped; size is bounded.

    The page keeps the conversation and sends it with every request, so memory
    survives server restarts, several workers, or a changed session id. The
    server-side store below is only the fallback for clients that send none.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content.strip()[:2000]})
    return out[-_MAX_MESSAGES:]


# --- Echo provider ----------------------------------------------------------
def _echo_reply(message: str) -> str:
    """Deterministic offline reply so the app is fully usable without an API key."""
    msg = message.strip()
    low = msg.lower()

    if any(g in low for g in ("hello", "hi ", "hey", "good morning", "good evening")) or low in {"hi", "hey", "hello"}:
        return "Hello! I'm Ivie. It's lovely to meet you. How can I help you today?"
    if "your name" in low or "who are you" in low:
        return "I'm Ivie, your friendly AI assistant. I'm here to chat, answer questions, and help however I can."
    if "how are you" in low:
        return "I'm doing wonderfully, thank you for asking! And you — how are you doing today?"
    if low.endswith("?"):
        return (
            f"That's a great question about '{msg.rstrip('?')}'. "
            "In the offline demo mode I can't reach a language model, but once you set "
            "LLM_PROVIDER=openai and an OPENAI_API_KEY, I'll give you a full answer."
        )
    return (
        f"I hear you say: \"{msg}\". "
        "I'm running in offline demo mode right now. Set LLM_PROVIDER=openai "
        "and OPENAI_API_KEY on Render to unlock my full brain!"
    )


# --- OpenAI provider --------------------------------------------------------
async def _openai_reply(history: list[dict], message: str) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    # Import lazily so the module loads even when openai isn't installed.
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    messages = [{"role": "system", "content": settings.system_prompt}]
    messages.extend(history[-settings.max_history_turns * 2:])
    messages.append({"role": "user", "content": message})

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.8,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


# --- HuggingFace provider ---------------------------------------------------
async def _hf_reply(session_id: str, history: list[dict], message: str) -> str:
    """Forward to the IvieAI Space, sending prior turns so it can follow up."""
    from .hf_client import HuggingFaceClient  # local import → optional dependency

    client = HuggingFaceClient(
        base_url=settings.hf_space_url,
        token=settings.hf_token,
        timeout=settings.hf_timeout,
    )
    # History is read BEFORE the new message is recorded, so it contains only
    # earlier turns; the Space receives the new message separately.
    logger.info("memory | session=%s | prior_msgs=%d | known_sessions=%d",
                session_id, len(history), len(_HISTORY))
    return await client.chat(message, history)


# --- Public API -------------------------------------------------------------
async def generate_reply(session_id: str, message: str,
                         client_history: list[dict] | None = None) -> tuple[str, str]:
    """Return (reply_text, provider_name_used).

    client_history (sent by the browser) wins over the server-side store.
    """
    provider = settings.llm_provider
    history = clean_client_history(client_history) or get_history(session_id)

    try:
        if provider == "huggingface":
            reply = await _hf_reply(session_id, history, message)
        elif provider == "openai":
            reply = await _openai_reply(history, message)
        else:
            reply = _echo_reply(message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM provider '%s' failed: %s", provider, exc)
        # Shown to the user but deliberately NOT recorded in the history.
        reply = getattr(exc, "user_message", None) or (
            f"I'm having trouble reaching my backend right now. "
            f"({type(exc).__name__}: {exc}). Please try again in a moment."
        )
        return reply, provider

    _record(session_id, "user", message)
    _record(session_id, "assistant", reply)
    return reply, provider
