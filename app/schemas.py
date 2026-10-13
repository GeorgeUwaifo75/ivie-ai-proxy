"""Pydantic request/response models."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    session_id: str | None = Field(default=None, description="Optional conversation id")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant reply text")
    session_id: str | None = None
    model: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    provider: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    code: int | None = None
    extra: dict[str, Any] | None = None