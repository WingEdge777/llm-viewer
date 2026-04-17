from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig


@dataclass(frozen=True)
class DecoderSpec:
    model_type: str
    config_class: str
    architecture: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int
    tie_word_embeddings: bool


def spec_from_transformers(config: dict[str, Any]) -> DecoderSpec:
    model_type = str(config["model_type"])
    config_kwargs = dict(config)
    config_kwargs.pop("model_type", None)
    hf_config = AutoConfig.for_model(model_type, **config_kwargs)

    hidden_size = int(hf_config.hidden_size)
    num_heads = int(hf_config.num_attention_heads)
    num_kv_heads = int(getattr(hf_config, "num_key_value_heads", num_heads))
    intermediate_size = int(getattr(hf_config, "intermediate_size", hidden_size * 4))
    vocab_size = int(getattr(hf_config, "vocab_size", 0))
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_heads))
    architectures = getattr(hf_config, "architectures", None) or []
    architecture = (
        architectures[0]
        if architectures
        else f"{type(hf_config).__name__.removesuffix('Config')}ForCausalLM"
    )

    return DecoderSpec(
        model_type=model_type,
        config_class=type(hf_config).__name__,
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=int(hf_config.num_hidden_layers),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=head_dim,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
    )
