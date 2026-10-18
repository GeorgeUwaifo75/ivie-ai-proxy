"""HuggingFace Space client for IvieAI.

The IvieAI Space (https://georgeuwaifo-ivieai.hf.space) is a Flask app whose
chat route is POST /chat. It accepts:

    {"message": "<new user message>",
     "history": [{"role": "user", "content": "..."},
                 {"role": "assistant", "content": "..."}, ...]}

and answers {"success": true, "response": "...", "mode": "followup|recall|standalone", ...}.

`history` is what gives the model its conversation memory: the Space decides
whether the new message is a follow-up (and needs earlier turns), a question
about earlier turns, or a standalone question.

NOTE: the previous version of this file probed ~80 path/payload combinations,
and cached the first one that worked, {"message": ...}. That payload carries
no history, so every request reached the Space as a brand-new conversation.
The route and payload are known now, so the probing has been removed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import get_settings

logger = logging.getLogger("ivie.hf")
settings = get_settings()

# Status codes a sleeping / restarting Space returns while it boots.
_WAKING_STATUSES = {502, 503, 504}


class SpaceWarmingUp(RuntimeError):
    """The Space is up but the model has not finished loading yet."""

    user_message = "I'm still waking up. Please try again in a few seconds."


class HuggingFaceClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 120.0) -> None:
        cleaned = (base_url or "").strip().rstrip("/")
        if not cleaned:
            raise ValueError("HF_SPACE_URL is empty — set it in Render env vars.")
        parsed = urlparse(cleaned)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"HF_SPACE_URL is not a valid URL: {cleaned!r}")

        self.base_url = f"{parsed.scheme}://{parsed.netloc.lower()}"
        self.chat_url = f"{self.base_url}{settings.hf_chat_path}"
        self.timeout = timeout

        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    async def chat(self, message: str, history: list[dict] | None = None) -> str:
        """Send the new message plus prior turns; return the Space's reply."""
        payload: dict[str, Any] = {
            "message": message,
            "history": (history or [])[-settings.hf_history_turns * 2:],
        }

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            try:
                r = await client.post(self.chat_url, headers=self._headers, json=payload)
                if r.status_code in _WAKING_STATUSES:
                    raise httpx.TransportError(f"Space returned HTTP {r.status_code}")
            except httpx.TransportError as exc:
                # Sleeping Spaces wake on the first request but often drop it.
                # Ping the root once, give it a moment, then retry one time.
                logger.warning("Space not ready (%s) — waking it and retrying once.", exc)
                try:
                    await client.get(self.base_url, timeout=30.0)
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(3)
                r = await client.post(self.chat_url, headers=self._headers, json=payload)

        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from Space: {r.text[:200]}")

        try:
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Invalid JSON from Space: {exc} | body={r.text[:200]}")

        if isinstance(data, dict) and data.get("model_loading"):
            raise SpaceWarmingUp()

        reply = self._extract_reply(data)
        if not reply:
            raise RuntimeError(f"No reply found in Space response: {data!r}")

        if isinstance(data, dict):
            logger.info("HF ✓ mode=%s | history_sent=%d msgs | reply_chars=%d",
                        data.get("mode", "n/a"), len(payload["history"]), len(reply))
        return reply

    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(f"{self.base_url}/health")
                body = r.json() if "json" in r.headers.get("content-type", "") else {}
                return {"ok": r.status_code == 200 and body.get("ready", True),
                        "status": r.status_code, **body}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _extract_reply(data: Any) -> str | None:
        """Find the assistant reply in the Space's JSON."""
        if isinstance(data, str):
            return data.strip() or None
        if isinstance(data, dict):
            for key in ("response", "reply", "answer", "output", "message", "text", "content"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None
