"""HuggingFace Space client for IvieAI.

The IvieAI Space (https://georgeuwaifo-ivieai.hf.space) runs a
Flask application (it returns the classic Flask/Werkzeug 404 page
for unknown routes). It is NOT Gradio and NOT FastAPI.

We therefore probe the most common Flask chat-endpoint conventions
until one answers, then reuse the winning combination for all
subsequent calls.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import get_settings

logger = logging.getLogger("ivie.hf")
settings = get_settings()


# ---------------------------------------------------------------------------
# Endpoint + payload candidates, in order of likelihood
# ---------------------------------------------------------------------------
CANDIDATE_PATHS: list[str] = [
    "/chat",
    "/api/chat",
    "/api/v1/chat",
    "/respond",
    "/message",
    "/api/message",
    "/generate",
    "/api/generate",
    "/ask",
    "/api/ask",
]


def _payload_variants(message: str, history: list[dict]) -> list[dict[str, Any]]:
    """Different Flask apps expect different JSON keys. Try them all."""
    return [
        {"message": message},                                    # 1. most common
        {"message": message, "history": history},                # 2. common w/ history
        {"prompt": message},                                     # 3. some apps
        {"text": message},                                       # 4. some apps
        {"user_message": message},                               # 5. explicit naming
        {"input": message},                                      # 6. generic
        {"query": message},                                      # 7. LLM-style
        {"messages": [{"role": "user", "content": message}]},    # 8. OpenAI-style
    ]


class HuggingFaceClient:
    """Async REST wrapper tuned for a Flask Space."""

    # Class-level memory of the winning (path, payload_key) so we only probe once
    _winner: tuple[str, str] | None = None

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
            "HF client | base=%s | token=%s | timeout=%.0fs",
            self.base_url, "yes" if token else "no", timeout,
        )

    # ------------------------------------------------------------------
    # Public API — called by llm.py
    # ------------------------------------------------------------------
    async def chat(self, message: str, history: list[dict] | None = None) -> str:
        history = history or []

        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True
        ) as client:

            # ── Wake the Space (HuggingFace Spaces sleep after inactivity) ──
            try:
                await client.get(self.base_url, timeout=30.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Space root ping failed: %s", exc)

            # ── If we already know the winning combo, use it directly ──────
            if HuggingFaceClient._winner is not None:
                path, key = HuggingFaceClient._winner
                try:
                    return await self._try_combo(
                        client, path, {key: message} if key != "messages"
                        else {"messages": [{"role": "user", "content": message}]}
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Cached combo %s failed (%s) — re-probing.",
                                   path, exc)
                    HuggingFaceClient._winner = None

            # ── Probe every (path, payload) combination ────────────────────
            errors: list[str] = []
            for path in CANDIDATE_PATHS:
                for payload in _payload_variants(message, history):
                    try:
                        reply = await self._try_combo(client, path, payload)
                        # Success — remember for next time
                        key = next(iter(payload))
                        HuggingFaceClient._winner = (path, key)
                        logger.info("HF ✓ winner | path=%s | payload_key=%s",
                                    path, key)
                        return reply
                    except _NotFoundError:
                        # This path doesn't exist — stop trying payloads for it
                        errors.append(f"{path}: 404")
                        break
                    except _BadRequestError as exc:
                        # Path exists but payload shape wrong — try next variant
                        errors.append(f"{path} {list(payload)}: {exc}")
                        continue
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{path}: {type(exc).__name__}: {exc}")
                        continue

        # Nothing worked
        detail = "; ".join(errors[-12:])  # last 12 attempts
        raise RuntimeError(
            f"No working /chat endpoint found on {self.base_url}. "
            f"Tried: {detail}. "
            f"Check the Space's app.py to see what route it actually exposes."
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                r = await client.get(self.base_url)
                return {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _try_combo(
        self, client: httpx.AsyncClient, path: str, payload: dict[str, Any]
    ) -> str:
        url = f"{self.base_url}{path}"
        logger.info("HF → POST %s | keys=%s", url, list(payload))

        r = await client.post(url, headers=self._headers, json=payload)

        if r.status_code == 404:
            raise _NotFoundError()
        if r.status_code == 400:
            raise _BadRequestError(f"400: {r.text[:120]}")
        if r.status_code == 422:
            raise _BadRequestError(f"422: {r.text[:120]}")
        if r.status_code >= 500:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

        # ── Parse the reply ─────────────────────────────────────────────
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            try:
                data = r.json()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Invalid JSON: {exc} | body={r.text[:200]}")
            reply = self._extract_reply(data)
            if not reply:
                raise RuntimeError(f"No reply found in JSON: {data!r}")
            return reply

        # Plain text / HTML response — return the text (with a length guard)
        text = r.text.strip()
        if text and len(text) < 5000 and "<html" not in text.lower()[:200]:
            return text
        raise RuntimeError(f"Unexpected response: {text[:200]}")

    @staticmethod
    def _extract_reply(data: Any) -> str | None:
        """Find the assistant reply in whatever JSON shape the Space uses."""
        if isinstance(data, str):
            return data.strip() or None

        if isinstance(data, dict):
            for key in (
                "reply", "response", "answer", "output", "message",
                "text", "content", "result", "generated_text",
                "bot", "assistant", "ai_response",
            ):
                if key in data:
                    val = data[key]
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                    nested = HuggingFaceClient._extract_reply(val)
                    if nested:
                        return nested

            # Fallback: first non-empty string value
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()

        if isinstance(data, list) and data:
            # Might be [{role: user}, {role: bot, content: "..."}]
            for item in reversed(data):
                if isinstance(item, dict):
                    for key in ("content", "message", "text", "reply"):
                        v = item.get(key)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
            return HuggingFaceClient._extract_reply(data[-1])

        return None


class _NotFoundError(Exception):
    pass


class _BadRequestError(Exception):
    pass