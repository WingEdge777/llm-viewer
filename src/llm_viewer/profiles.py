from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    if name is ProfileName.PREFILL:
        return RuntimeProfile(name=name, batch_size=1, seq_len=128, past_len=0)
    if name is ProfileName.DECODE:
        return RuntimeProfile(name=name, batch_size=1, seq_len=1, past_len=127)
    raise ValueError(f"Unsupported profile: {name}")
