from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8989
    prefill_batch_size: int = 1
    prefill_seq_len: int = 128
    decode_batch_size: int = 1
    decode_seq_len: int = 1
    decode_past_len: int = 127

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            host=os.getenv("LLM_VIEWER_HOST", "127.0.0.1"),
            port=int(os.getenv("LLM_VIEWER_PORT", "8989")),
            prefill_batch_size=int(os.getenv("LLM_VIEWER_PREFILL_BATCH", "1")),
            prefill_seq_len=int(os.getenv("LLM_VIEWER_PREFILL_SEQ", "128")),
            decode_batch_size=int(os.getenv("LLM_VIEWER_DECODE_BATCH", "1")),
            decode_seq_len=int(os.getenv("LLM_VIEWER_DECODE_SEQ", "1")),
            decode_past_len=int(os.getenv("LLM_VIEWER_DECODE_PAST", "127")),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def configure_settings(settings: Settings) -> None:
    global _settings
    _settings = settings
