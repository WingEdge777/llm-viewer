from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llm_viewer.settings import get_settings


class ProfileName(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class RuntimeProfile:
    name: ProfileName
    batch_size: int
    seq_len: int
    past_len: int
    use_cache: bool = True


def get_profile(name: ProfileName) -> RuntimeProfile:
    settings = get_settings()
    if name is ProfileName.PREFILL:
        return RuntimeProfile(
            name=name,
            batch_size=settings.prefill_batch_size,
            seq_len=settings.prefill_seq_len,
            past_len=0,
        )
    if name is ProfileName.DECODE:
        return RuntimeProfile(
            name=name,
            batch_size=settings.decode_batch_size,
            seq_len=settings.decode_seq_len,
            past_len=settings.decode_past_len,
        )
    raise ValueError(f"Unsupported profile: {name}")
