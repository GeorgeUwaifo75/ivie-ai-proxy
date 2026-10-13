"""LLM abstraction.

Supports:
  - "echo"   : offline stub — returns a canned but useful reply (default, no API key needed)
  - "openai" : any OpenAI-compatible endpoint (OpenAI, Azure, Together, Groq, Ollama, etc.)

Conversation history is kept in-process per session_id. For production you'd
swap this for Redis / a DB — but it's fine for a single Render instance.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# --- In-memory conversation store ------------------------------------------
_HISTORY: dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=settings.max_history_turns * 2))


def _record(session_id: str, role: str, content: str) -> None:
    _HISTORY[session_id].append({"role": role, "content": content})


def get_history(session_id: str) -> list[dict]:
    return list(_HISTORY[session_id])


def clear_history(session_id: str) -> None:
    _HISTORY.pop(session_id, None)


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
async def _openai_reply(session_id: str, message: str) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    # Import lazily so the module loads even when openai isn't installed.
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    messages = [{"role": "system", "content": settings.system_prompt}]
    messages.extend(get_history(session_id))
    messages.append({"role": "user", "content": message})

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.8,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


# --- Public API -------------------------------------------------------------
# --- HuggingFace provider ---------------------------------------------------
async def _hf_reply(session_id: str, message: str) -> str:
    """Forward to the IvieAI Space and return its reply."""
    from .hf_client import HuggingFaceClient  # local import → optional dependency

    client = HuggingFaceClient(
        base_url=settings.hf_space_url,
        token=settings.hf_token,
        timeout=settings.hf_timeout,
    )
    # Send prior turns so IvieAI keeps context
    history = get_history(session_id)
    return await client.chat(message, history)


# --- Public API -------------------------------------------------------------
async def generate_reply(session_id: str, message: str) -> tuple[str, str]:
    """Return (reply_text, provider_name_used)."""
    provider = settings.llm_provider

    try:
        if provider == "huggingface":
            reply = await _hf_reply(session_id, message)
        elif provider == "openai":
            reply = await _openai_reply(session_id, message)
        else:
            reply = _echo_reply(message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM provider '%s' failed: %s", provider, exc)
        # Optional: fall back to echo so the UI always shows *something*
        reply = (
            f"I'm having trouble reaching my backend right now. "
            f"({type(exc).__name__}: {exc}). Please try again in a moment."
        )

    _record(session_id, "user", message)
    _record(session_id, "assistant", reply)
    return reply, provider