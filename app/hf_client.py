"""HuggingFace Space client for IvieAI.

The IvieAI Space (https://georgeuwaifo-ivieai.hf.space) exposes a
Gradio interface. Gradio apps can be called programmatically through
the /gradio_api/call/<endpoint> route using Server-Sent Events (SSE).

We use this to forward user messages to IvieAI and stream back the reply.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import get_settings

logger = logging.getLogger("ivie.hf")
settings = get_settings()


class HuggingFaceClient:
    """Thin async wrapper around the IvieAI Gradio Space."""

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 120.0) -> None:
        # ── Normalize & validate the URL ────────────────────────────────
        cleaned = (base_url or "").strip().rstrip("/")
        if not cleaned:
            raise ValueError("HF_SPACE_URL is empty — set it in Render env vars.")

        parsed = urlparse(cleaned)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"HF_SPACE_URL is not a valid URL: {cleaned!r}")

        # Force lowercase host (DNS is case-sensitive on some resolvers)
        host = parsed.netloc.lower()
        self.base_url = f"{parsed.scheme}://{host}"
        self.token = token
        self.timeout = timeout

        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

        logger.info("HF client initialized | base_url=%s | token=%s",
                    self.base_url, "yes" if token else "no")

    # ------------------------------------------------------------------
    # Discovery — Gradio exposes an /info route describing its endpoints
    # ------------------------------------------------------------------
    async def _list_endpoints(self, client: httpx.AsyncClient) -> list[str]:
        try:
            r = await client.get(f"{self.base_url}/gradio_api/info", headers=self._headers)
            if r.status_code == 200:
                return list(r.json().get("named_endpoints", {}).keys())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not fetch gradio info: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Public API — called by llm.py
    # ------------------------------------------------------------------
    async def chat(self, message: str, history: list[dict] | None = None) -> str:
        history = history or []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            endpoint = await self._find_chat_endpoint(client)
            if not endpoint:
                raise RuntimeError(
                    "Could not discover the Gradio chat endpoint on the Space."
                )

            payload = {"data": [message, history]}
            call_url = f"{self.base_url}/gradio_api/call/{endpoint}"
            logger.info("HF POST %s | payload=%s", call_url, str(payload)[:200])

            try:
                r = await client.post(call_url, headers=self._headers, json=payload)
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot reach HuggingFace Space at {self.base_url}. "
                    f"Check that HF_SPACE_URL is correct and the Space is running. "
                    f"({exc})"
                ) from exc

            if r.status_code != 200:
                raise RuntimeError(
                    f"HuggingFace Space returned HTTP {r.status_code}: {r.text[:300]}"
                )

            event_id = r.json().get("event_id")
            if not event_id:
                raise RuntimeError(f"No event_id in Space response: {r.text[:300]}")

            stream_url = f"{call_url}/{event_id}"
            logger.info("HF SSE %s", stream_url)

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

                    if obj.get("msg") == "process_completed":
                        out = obj.get("output", {}).get("data", [])
                        for item in out:
                            if isinstance(item, str) and item.strip():
                                return item.strip()
                            if isinstance(item, list) and item:
                                last = item[-1]
                                if isinstance(last, list) and len(last) >= 2 and isinstance(last[1], str):
                                    return last[1].strip()

                        return json.dumps(out)

        raise RuntimeError("Space closed the stream without a completion event.")

    async def health(self) -> dict[str, Any]:
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
        # Try the live /info route first
        try:
            r = await client.get(f"{self.base_url}/gradio_api/info", headers=self._headers)
            if r.status_code == 200:
                info = r.json()
                named = list(info.get("named_endpoints", {}).keys())
                logger.info("Gradio named_endpoints: %s", named)
                for candidate in named:
                    low = candidate.lower()
                    if any(k in low for k in ("chat", "respond", "predict", "message", "ask")):
                        return candidate.lstrip("/")
                if named:
                    return named[0].lstrip("/")
            else:
                logger.warning("GET /gradio_api/info → HTTP %s", r.status_code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not query /gradio_api/info: %s", exc)

        # Fallback names
        for name in ("chat", "respond", "predict", "chat_1", "message", "generate"):
            return name
        return None