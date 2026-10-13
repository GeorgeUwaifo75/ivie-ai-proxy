"""HuggingFace Space client for IvieAI.

The IvieAI Space (https://georgeuwaifo-ivieai.hf.space) is a FastAPI
application (NOT Gradio). It exposes:
    POST /chat    { "message": "..." }  →  { "reply": "..." }  (or similar)
    GET  /health                        →  { "status": "ok", ... }

We call it as a plain REST endpoint — no SSE, no Gradio protocol.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import get_settings

logger = logging.getLogger("ivie.hf")
settings = get_settings()


class HuggingFaceClient:
    """Async REST wrapper around the IvieAI FastAPI Space."""

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 120.0) -> None:
        cleaned = (base_url or "").strip().rstrip("/")
        if not cleaned:
            raise ValueError("HF_SPACE_URL is empty — set it in Render env vars.")

        parsed = urlparse(cleaned)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"HF_SPACE_URL is not a valid URL: {cleaned!r}")

        self.base_url = f"{parsed.scheme}://{parsed.netloc.lower()}"
        self.token = token
        self.timeout = timeout

        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

        logger.info(
            "HF client initialized | base_url=%s | token=%s | timeout=%.0fs",
            self.base_url, "yes" if token else "no", timeout,
        )

    # ------------------------------------------------------------------
    # Main entry point — called by llm.py
    # ------------------------------------------------------------------
    async def chat(self, message: str, history: list[dict] | None = None) -> str:
        """POST the message to the Space's /chat endpoint and return the reply."""
        history = history or []

        # Try a few common payload shapes — different FastAPI Spaces expect
        # different keys. We try the most likely one first, then fall back.
        payloads_to_try: list[dict[str, Any]] = [
            {"message": message, "history": history},   # most common
            {"message": message},                        # simple
            {"text": message},                           # some apps
            {"prompt": message},                         # some apps
        ]

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            last_error: Exception | None = None

            # ── First, wake up the Space if it's asleep ──────────────────
            try:
                await client.get(self.base_url, headers=self._headers, timeout=30.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not ping Space root: %s", exc)

            # ── Try each payload shape against /chat ─────────────────────
            for idx, payload in enumerate(payloads_to_try):
                url = f"{self.base_url}/chat"
                logger.info("HF POST %s | try=%d | payload=%s",
                            url, idx + 1, str(payload)[:160])

                try:
                    r = await client.post(url, headers=self._headers, json=payload)
                except httpx.ConnectError as exc:
                    raise RuntimeError(
                        f"Cannot reach HuggingFace Space at {self.base_url}. "
                        f"Check HF_SPACE_URL. ({exc})"
                    ) from exc

                if r.status_code == 404:
                    # /chat doesn't exist — try to discover via /openapi.json
                    logger.warning("POST /chat → 404. Querying /openapi.json ...")
                    discovered = await self._discover_routes(client)
                    if discovered:
                        logger.info("Discovered routes: %s", discovered)
                    last_error = RuntimeError(
                        f"Space has no /chat endpoint. Available routes: {discovered}"
                    )
                    break

                if r.status_code >= 400:
                    logger.warning("POST /chat → HTTP %d: %s",
                                   r.status_code, r.text[:300])
                    last_error = RuntimeError(
                        f"Space responded HTTP {r.status_code}: {r.text[:300]}"
                    )
                    continue  # try next payload shape

                # ── Extract the reply from whatever JSON shape came back ─
                try:
                    data = r.json()
                except Exception:
                    text = r.text.strip()
                    if text:
                        return text
                    last_error = RuntimeError("Empty response from Space")
                    continue

                reply = self._extract_reply(data)
                if reply:
                    return reply

                last_error = RuntimeError(f"Could not find reply in: {data!r}")

        raise last_error or RuntimeError("All payload shapes failed.")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(f"{self.base_url}/health", headers=self._headers)
                return {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_reply(data: Any) -> str | None:
        """Find the assistant's reply in a variety of response shapes."""
        if isinstance(data, str):
            return data.strip() or None

        if isinstance(data, dict):
            # Try common keys first
            for key in ("reply", "response", "answer", "output",
                        "message", "text", "content", "result"):
                if key in data:
                    val = data[key]
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                    # Recurse if it's a nested object
                    nested = HuggingFaceClient._extract_reply(val)
                    if nested:
                        return nested
            # Otherwise, take the first non-empty string value
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()

        if isinstance(data, list) and data:
            return HuggingFaceClient._extract_reply(data[0])

        return None

    async def _discover_routes(self, client: httpx.AsyncClient) -> list[str]:
        """Fetch /openapi.json (FastAPI) to find out which routes exist."""
        try:
            r = await client.get(f"{self.base_url}/openapi.json",
                                 headers=self._headers, timeout=20.0)
            if r.status_code == 200:
                spec = r.json()
                return sorted(spec.get("paths", {}).keys())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not fetch /openapi.json: %s", exc)
        return []