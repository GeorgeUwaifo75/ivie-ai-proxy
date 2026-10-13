"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Runtime settings. Values come from env vars (Render dashboard or .env)."""

    def __init__(self) -> None:
        # --- Server ---
        self.app_name: str = os.getenv("APP_NAME", "Ivie AI")
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"

        # --- CORS ---
        # Comma-separated list. Use "*" during development.
        self.allowed_origins: list[str] = [
            o.strip()
            for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
            if o.strip()
        ]

        # --- LLM router ---
        # "huggingface" (default) | "openai" | "echo"
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "huggingface").lower()

        # --- HuggingFace Space (IvieAI backend) ---
        self.hf_space_url: str = os.getenv(
            "HF_SPACE_URL", "https://GeorgeUwaifo-IvieAI.hf.space"
        )
        self.hf_token: str | None = os.getenv("HF_TOKEN") or None
        self.hf_timeout: float = float(os.getenv("HF_TIMEOUT", "120"))

        # --- OpenAI (only used when LLM_PROVIDER=openai) ---
        self.openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None

        # --- Persona (only used by the OpenAI provider) ---
        self.system_prompt: str = os.getenv(
            "SYSTEM_PROMPT",
            (
                "You are Ivie, a warm, confident, and helpful AI assistant "
                "with a West African (Nigerian) personality. You speak in a "
                "friendly, natural, conversational tone. Keep answers clear "
                "and concise — two to four sentences unless the user asks for "
                "more detail."
            ),
        )
        self.max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "8"))


@lru_cache
def get_settings() -> Settings:
    return Settings()