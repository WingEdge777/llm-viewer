"""Adapter registry for building graph bundles from model configs."""

from __future__ import annotations

from typing import Any

from llm_viewer.adapters import DecoderOnlyAdapter, GraphAdapter
from llm_viewer.profiles import RuntimeProfile
from llm_viewer.schema import GraphBundle

_ADAPTERS: list[GraphAdapter] = [DecoderOnlyAdapter()]


def build_graph_bundle(config: dict[str, Any], profile: RuntimeProfile) -> GraphBundle:
    """Build a graph bundle from a HuggingFace model config.

    Args:
        config: HuggingFace model config dictionary with 'model_type' field.
        profile: Runtime profile (prefill/decode) for shape inference.

    Returns:
        GraphBundle containing model and block graphs.

    Raises:
        ValueError: If model_type is missing or unsupported.
    """
    model_type = str(config.get("model_type", ""))
    if not model_type:
        raise ValueError("Missing required config field: model_type")

    for adapter in _ADAPTERS:
        if adapter.supports(model_type):
            return adapter.build(config=config, profile=profile)
    supported = sorted({model for adapter in _ADAPTERS for model in adapter.model_types})
    raise ValueError(f"Unsupported model_type: {model_type}. Supported: {', '.join(supported)}")
