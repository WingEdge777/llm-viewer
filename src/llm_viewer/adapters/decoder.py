from __future__ import annotations

from typing import Any

from llm_viewer.adapters.base import GraphAdapter
from llm_viewer.adapters.block_graph import build_block_graph
from llm_viewer.adapters.model_graph import build_model_graph
from llm_viewer.adapters.spec import spec_from_transformers
from llm_viewer.profiles import RuntimeProfile
from llm_viewer.schema import GraphBundle


class DecoderOnlyAdapter(GraphAdapter):
    model_types = {
        "code_llama",
        "diffllama",
        "doge",
        "gemma",
        "gemma2",
        "gemma3_text",
        "gemma4_text",
        "granite",
        "helium",
        "llama",
        "ministral",
        "mistral",
        "olmo",
        "qwen2",
        "qwen3",
        "seed_oss",
        "smollm3",
        "stablelm",
    }

    def build(self, config: dict[str, Any], profile: RuntimeProfile) -> GraphBundle:
        spec = spec_from_transformers(config)
        metadata = {
            "model_type": spec.model_type,
            "profile": profile.name.value,
            "hidden_size": spec.hidden_size,
            "num_hidden_layers": spec.num_layers,
            "num_attention_heads": spec.num_heads,
            "num_key_value_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
            "architecture": spec.architecture,
            "transformers_config_class": spec.config_class,
            "shape_inference": "symbolic-from-transformers-config",
        }
        graphs = [
            build_model_graph(spec=spec, profile=profile),
            build_block_graph(spec=spec, profile=profile),
        ]
        return GraphBundle(metadata=metadata, graphs=graphs)
