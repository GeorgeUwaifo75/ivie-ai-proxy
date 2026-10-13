"""HuggingFace Space client for IvieAI.

The IvieAI Space (https://GeorgeUwaifo-IvieAI.hf.space) exposes a
Gradio interface. Gradio apps can be called programmatically through
the /gradio_api/call/<endpoint> route using Server-Sent Events (SSE).

We use this to forward user messages to IvieAI and stream back the reply.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger("ivie.hf")
settings = get_settings()


class HuggingFaceClient:
    """Thin async wrapper around the IvieAI Gradio Space."""

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # Discovery — Gradio exposes an /info route describing its endpoints
    # ------------------------------------------------------------------
    async def _list_endpoints(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        """Return the Space's API map (which functions are callable)."""
        try:
            r = await client.get(f"{self.base_url}/gradio_api/info", headers=self._headers)
            if r.status_code == 200:
                return r.json().get("named_endpoints", {}) and list(r.json().get("named_endpoints", {}).keys()) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not fetch gradio info: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Public API — called by llm.py
    # ------------------------------------------------------------------
    async def chat(self, message: str, history: list[dict] | None = None) -> str:
        """Send one user message to the Space and return the assistant reply."""
        history = history or []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Step 1: find the chat endpoint name
            endpoint = await self._find_chat_endpoint(client)
            if not endpoint:
                raise RuntimeError(
                    "Could not discover the Gradio chat endpoint on the Space. "
                    "The Space may be asleep or its API changed."
                )

            # Step 2: POST to /gradio_api/call/<endpoint> with the message + history
            payload = {"data": [message, history]}
            call_url = f"{self.base_url}/gradio_api/call/{endpoint}"
            logger.info("HF call → %s | msg=%r", endpoint, message[:80])

            r = await client.post(call_url, headers=self._headers, json=payload)
            if r.status_code != 200:
                raise RuntimeError(
                    f"HuggingFace Space returned HTTP {r.status_code}: {r.text[:200]}"
                )

            event_id = r.json().get("event_id")
            if not event_id:
                raise RuntimeError(f"No event_id in Space response: {r.text[:200]}")

            # Step 3: GET /gradio_api/call/<endpoint>/<event_id> → stream SSE
            stream_url = f"{call_url}/{event_id}"
            async with client.stream("GET", stream_url, headers=self._headers) as s:
                async for line in s.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # Gradio emits: {"msg": "process_completed", "output": {"data": [...]}}
                    if obj.get("msg") == "process_completed":
                        out = obj.get("output", {}).get("data", [])
                        # The first string in data is usually the reply
                        for item in out:
                            if isinstance(item, str) and item.strip():
                                return item.strip()
                            # some spaces return [[user, bot], history]
                            if isinstance(item, list) and item and isinstance(item[-1], list) and item[-1]:
                                last = item[-1]
                                if len(last) >= 2 and isinstance(last[1], str):
                                    return last[1].strip()
                        # If nothing found, return whatever we got as a string
                        return json.dumps(out)

        raise RuntimeError("Space closed the stream without a completion event.")

    async def health(self) -> dict[str, Any]:
        """Cheap reachability check."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.base_url}/", headers=self._headers)
                return {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _find_chat_endpoint(self, client: httpx.AsyncClient) -> str | None:
        """Guess the chat endpoint name.

        Common names in Gradio Spaces: 'chat', 'respond', 'predict',
        'chat_1', 'generate', 'infer'. We try /info first; if that
        fails, we fall back to trying a list of common names directly.
        """
        try:
            r = await client.get(f"{self.base_url}/gradio_api/info", headers=self._headers)
            if r.status_code == 200:
                info = r.json()
                named = list(info.get("named_endpoints", {}).keys())
                logger.debug("Gradio endpoints: %s", named)
                # Prefer names that clearly indicate a chat
                for candidate in named:
                    low = candidate.lower()
                    if any(k in low for k in ("chat", "respond", "predict", "message", "ask")):
                        return candidate.lstrip("/")
                # Otherwise take the first
                if named:
                    return named[0].lstrip("/")
        except Exception as exc:  # noqa: BLE001
            logger.debug("/info lookup failed: %s", exc)

        # Fallback: try common names
        for name in ("chat", "respond", "predict", "chat_1", "message", "generate"):
            return name
        return None