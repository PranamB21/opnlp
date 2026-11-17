"""Environment-backed settings for the NLP service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    # Upstream WhisperLiveKit ASR WebSocket (audio + transcript backbone).
    asr_ws_url: str = field(default_factory=lambda: os.environ.get("WLK_WS_URL", "ws://localhost:8000/asr"))

    # LLM provider: auto | anthropic | openai | mock
    llm_provider: str = field(default_factory=lambda: os.environ.get("NLP_LLM_PROVIDER", "auto").strip().lower())
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    llm_timeout_seconds: float = field(default_factory=lambda: _env_float("NLP_LLM_TIMEOUT", 60.0))
    llm_max_output_tokens: int = field(default_factory=lambda: _env_int("NLP_LLM_MAX_TOKENS", 2000))
    extra_headers: dict = field(default_factory=dict)

    # Rolling-window segmentation.
    window_seconds: float = field(default_factory=lambda: _env_float("NLP_WINDOW_SECONDS", 40.0))
    min_window_seconds: float = field(default_factory=lambda: _env_float("NLP_MIN_WINDOW_SECONDS", 20.0))
    max_window_seconds: float = field(default_factory=lambda: _env_float("NLP_MAX_WINDOW_SECONDS", 90.0))
    pause_gap_seconds: float = field(default_factory=lambda: _env_float("NLP_PAUSE_GAP_SECONDS", 2.5))
    history_topics: int = field(default_factory=lambda: _env_int("NLP_HISTORY_TOPICS", 6))

    # Slack incoming webhook for the "end meeting" action.
    slack_webhook_url: str = field(default_factory=lambda: os.environ.get("SLACK_WEBHOOK_URL", ""))

    # Where the extended WhisperLiveKit dashboard assets live.
    web_dir: str = field(default_factory=lambda: os.environ.get("WLK_WEB_DIR", ""))

    def resolved_provider(self) -> str:
        """Pick a provider from explicit config, falling back on available keys."""
        if self.llm_provider in {"anthropic", "openai", "mock"}:
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "mock"