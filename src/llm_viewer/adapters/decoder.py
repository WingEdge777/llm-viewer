from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig

from llm_viewer.adapters.base import GraphAdapter
from llm_viewer.profiles import ProfileName, RuntimeProfile
from llm_viewer.schema import Edge, Graph, GraphBundle, Node


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
        spec = self._spec_from_transformers(config)
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
            self._build_model_graph(
                spec=spec,
                profile=profile,
            ),
            self._build_block_graph(
                spec=spec,
                profile=profile,
            ),
        ]
        return GraphBundle(metadata=metadata, graphs=graphs)

    def _build_model_graph(
        self,
        *,
        spec: DecoderSpec,
        profile: RuntimeProfile,
    ) -> Graph:
        batch = profile.batch_size
        seq = profile.seq_len
        token_shape = [batch, seq]
        hidden_shape = [batch, seq, spec.hidden_size]
        logits_shape = [batch, seq, spec.vocab_size or "vocab"]

        nodes = [
            Node(
                id="tokens",
                name="Input Tokens",
                kind="input",
                op_family="Input",
                output_shapes=[token_shape],
                attrs={"profile": profile.name.value},
                module_path="input_ids",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="embedding",
                name="Embedding",
                kind="embedding",
                op_family="Embedding",
                input_shapes=[token_shape],
                output_shapes=[hidden_shape],
                param_shapes=[[spec.vocab_size or "vocab", spec.hidden_size]],
                module_path="model.embed_tokens",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="repeated_block",
                name=f"Repeated Block x{spec.num_layers}",
                kind="block_stack",
                op_family="TransformerBlock",
                input_shapes=[hidden_shape],
                output_shapes=[hidden_shape],
                attrs={
                    "repeat": spec.num_layers,
                    "block_graph_id": "block",
                    "profile": profile.name.value,
                    "model_type": spec.model_type,
                    "architecture": spec.architecture,
                },
                module_path="model.layers[*]",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="final_norm",
                name="Final Norm",
                kind="norm",
                op_family="Norm",
                input_shapes=[hidden_shape],
                output_shapes=[hidden_shape],
                param_shapes=[[spec.hidden_size]],
                module_path="model.norm",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="lm_head",
                name="LM Head",
                kind="projection",
                op_family="LMHead",
                input_shapes=[hidden_shape],
                output_shapes=[logits_shape],
                param_shapes=[[spec.hidden_size, spec.vocab_size or "vocab"]],
                attrs={"tie_word_embeddings": spec.tie_word_embeddings},
                module_path="lm_head",
                source_file=self._source_file(spec.model_type),
            ),
        ]
        edges = [
            Edge(source="tokens", target="embedding", shape=token_shape, tensor_name="input_ids"),
            Edge(source="embedding", target="repeated_block", shape=hidden_shape, tensor_name="hidden_states"),
            Edge(source="repeated_block", target="final_norm", shape=hidden_shape, tensor_name="hidden_states"),
            Edge(source="final_norm", target="lm_head", shape=hidden_shape, tensor_name="hidden_states"),
        ]
        return Graph(
            id="model",
            name="Model Graph",
            level="model",
            nodes=nodes,
            edges=edges,
            attrs={"model_type": spec.model_type, "profile": profile.name.value},
        )

    def _build_block_graph(
        self,
        *,
        spec: DecoderSpec,
        profile: RuntimeProfile,
    ) -> Graph:
        batch = profile.batch_size
        seq = profile.seq_len
        past = profile.past_len

        hidden_shape = [batch, seq, spec.hidden_size]
        q_proj_shape = [batch, seq, spec.hidden_size]
        kv_proj_shape = [batch, seq, spec.num_kv_heads * spec.head_dim]
        q_heads_shape = [batch, spec.num_heads, seq, spec.head_dim]
        kv_heads_shape = [batch, spec.num_kv_heads, seq, spec.head_dim]
        total_kv_shape = [batch, spec.num_kv_heads, seq + past, spec.head_dim]
        attn_out_shape = [batch, seq, spec.hidden_size]
        mlp_mid_shape = [batch, seq, spec.intermediate_size]

        nodes = [
            Node(
                id="block_input",
                name="Block Input",
                kind="input",
                op_family="Input",
                output_shapes=[hidden_shape],
                module_path="model.layers.*",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="input_norm",
                name="Input Norm",
                kind="norm",
                op_family="Norm",
                input_shapes=[hidden_shape],
                output_shapes=[hidden_shape],
                param_shapes=[[spec.hidden_size]],
                module_path="model.layers.*.input_layernorm",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="q_proj",
                name="Q Projection",
                kind="linear",
                op_family="Attention",
                input_shapes=[hidden_shape],
                output_shapes=[q_proj_shape],
                param_shapes=[[spec.hidden_size, spec.hidden_size]],
                module_path="model.layers.*.self_attn.q_proj",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="k_proj",
                name="K Projection",
                kind="linear",
                op_family="Attention",
                input_shapes=[hidden_shape],
                output_shapes=[kv_proj_shape],
                param_shapes=[[spec.hidden_size, spec.num_kv_heads * spec.head_dim]],
                module_path="model.layers.*.self_attn.k_proj",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="v_proj",
                name="V Projection",
                kind="linear",
                op_family="Attention",
                input_shapes=[hidden_shape],
                output_shapes=[kv_proj_shape],
                param_shapes=[[spec.hidden_size, spec.num_kv_heads * spec.head_dim]],
                module_path="model.layers.*.self_attn.v_proj",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="attn_mix",
                name=self._attention_name(spec.model_type),
                kind="attention_core",
                op_family="Attention",
                input_shapes=[q_heads_shape, total_kv_shape, total_kv_shape],
                output_shapes=[attn_out_shape],
                attrs={
                    "num_heads": spec.num_heads,
                    "num_key_value_heads": spec.num_kv_heads,
                    "head_dim": spec.head_dim,
                    "config_class": spec.config_class,
                },
                module_path="model.layers.*.self_attn",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="attn_output",
                name="Attention Output Projection",
                kind="linear",
                op_family="Attention",
                input_shapes=[attn_out_shape],
                output_shapes=[attn_out_shape],
                param_shapes=[[spec.hidden_size, spec.hidden_size]],
                module_path="model.layers.*.self_attn.o_proj",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="attn_residual",
                name="Attention Residual Add",
                kind="residual",
                op_family="Residual",
                input_shapes=[hidden_shape, attn_out_shape],
                output_shapes=[hidden_shape],
                module_path="model.layers.* (attention residual)",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="post_attn_norm",
                name="Post-Attention Norm",
                kind="norm",
                op_family="Norm",
                input_shapes=[hidden_shape],
                output_shapes=[hidden_shape],
                param_shapes=[[spec.hidden_size]],
                module_path="model.layers.*.post_attention_layernorm",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="gate_proj",
                name="Gate Projection",
                kind="linear",
                op_family="MLP",
                input_shapes=[hidden_shape],
                output_shapes=[mlp_mid_shape],
                param_shapes=[[spec.hidden_size, spec.intermediate_size]],
                module_path="model.layers.*.mlp.gate_proj",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="up_proj",
                name="Up Projection",
                kind="linear",
                op_family="MLP",
                input_shapes=[hidden_shape],
                output_shapes=[mlp_mid_shape],
                param_shapes=[[spec.hidden_size, spec.intermediate_size]],
                module_path="model.layers.*.mlp.up_proj",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="down_proj",
                name="Down Projection",
                kind="linear",
                op_family="MLP",
                input_shapes=[mlp_mid_shape],
                output_shapes=[hidden_shape],
                param_shapes=[[spec.intermediate_size, spec.hidden_size]],
                module_path="model.layers.*.mlp.down_proj",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="mlp_residual",
                name="MLP Residual Add",
                kind="residual",
                op_family="Residual",
                input_shapes=[hidden_shape, hidden_shape],
                output_shapes=[hidden_shape],
                module_path="model.layers.* (mlp residual)",
                source_file=self._source_file(spec.model_type),
            ),
            Node(
                id="block_output",
                name="Block Output",
                kind="output",
                op_family="Output",
                input_shapes=[hidden_shape],
                output_shapes=[hidden_shape],
                module_path="model.layers.*",
                source_file=self._source_file(spec.model_type),
            ),
        ]
        edges = [
            Edge(source="block_input", target="input_norm", shape=hidden_shape, tensor_name="hidden_states"),
            Edge(source="input_norm", target="q_proj", shape=hidden_shape, tensor_name="normed_hidden_states"),
            Edge(source="input_norm", target="k_proj", shape=hidden_shape, tensor_name="normed_hidden_states"),
            Edge(source="input_norm", target="v_proj", shape=hidden_shape, tensor_name="normed_hidden_states"),
            Edge(source="q_proj", target="attn_mix", shape=q_heads_shape, tensor_name="q"),
            Edge(source="attn_mix", target="attn_output", shape=attn_out_shape, tensor_name="context"),
            Edge(source="block_input", target="attn_residual", shape=hidden_shape, tensor_name="residual", edge_kind="residual"),
            Edge(source="attn_output", target="attn_residual", shape=attn_out_shape, tensor_name="attn_out"),
            Edge(source="attn_residual", target="post_attn_norm", shape=hidden_shape, tensor_name="hidden_states"),
            Edge(source="post_attn_norm", target="gate_proj", shape=hidden_shape, tensor_name="normed_hidden_states"),
            Edge(source="post_attn_norm", target="up_proj", shape=hidden_shape, tensor_name="normed_hidden_states"),
            Edge(source="gate_proj", target="down_proj", shape=mlp_mid_shape, tensor_name="gated_mlp"),
            Edge(source="up_proj", target="down_proj", shape=mlp_mid_shape, tensor_name="up_mlp"),
            Edge(source="attn_residual", target="mlp_residual", shape=hidden_shape, tensor_name="residual", edge_kind="residual"),
            Edge(source="down_proj", target="mlp_residual", shape=hidden_shape, tensor_name="mlp_out"),
            Edge(source="mlp_residual", target="block_output", shape=hidden_shape, tensor_name="hidden_states"),
        ]

        if profile.name is ProfileName.DECODE:
            cache_nodes = [
                Node(
                    id="past_kv",
                    name="Past KV Cache",
                    kind="cache",
                    op_family="Cache",
                    output_shapes=[[batch, spec.num_kv_heads, past, spec.head_dim], [batch, spec.num_kv_heads, past, spec.head_dim]],
                    attrs={"past_len": past},
                    module_path="past_key_values",
                    source_file=self._source_file(spec.model_type),
                ),
                Node(
                    id="cache_update",
                    name="Cache Update",
                    kind="cache_update",
                    op_family="Cache",
                    input_shapes=[kv_heads_shape, kv_heads_shape, [batch, spec.num_kv_heads, past, spec.head_dim]],
                    output_shapes=[total_kv_shape, total_kv_shape],
                    attrs={"new_total_len": seq + past},
                    module_path="model.layers.*.self_attn.past_key_values",
                    source_file=self._source_file(spec.model_type),
                ),
            ]
            nodes[5:5] = cache_nodes
            edges.extend(
                [
                    Edge(source="past_kv", target="cache_update", shape=[batch, spec.num_kv_heads, past, spec.head_dim], tensor_name="past_k"),
                    Edge(source="k_proj", target="cache_update", shape=kv_heads_shape, tensor_name="new_k"),
                    Edge(source="v_proj", target="cache_update", shape=kv_heads_shape, tensor_name="new_v"),
                    Edge(source="cache_update", target="attn_mix", shape=total_kv_shape, tensor_name="cache_kv"),
                ]
            )
        else:
            edges.extend(
                [
                    Edge(source="k_proj", target="attn_mix", shape=total_kv_shape, tensor_name="k"),
                    Edge(source="v_proj", target="attn_mix", shape=total_kv_shape, tensor_name="v"),
                ]
            )

        return Graph(
            id="block",
            name="Transformer Block",
            level="block",
            nodes=nodes,
            edges=edges,
            attrs={
                "model_type": spec.model_type,
                "profile": profile.name.value,
                "supports_kv_cache": True,
                "shape_inference": "symbolic-from-transformers-config",
            },
        )

    @staticmethod
    def _spec_from_transformers(config: dict[str, Any]) -> DecoderSpec:
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
        architecture = architectures[0] if architectures else f"{type(hf_config).__name__.removesuffix('Config')}ForCausalLM"

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

    @staticmethod
    def _attention_name(model_type: str) -> str:
        return {
            "llama": "Grouped Query Attention",
            "mistral": "Sliding Window Attention",
            "qwen2": "Qwen2 Attention",
            "qwen3": "Qwen3 Attention",
            "gemma": "Gemma Attention",
            "gemma2": "Gemma2 Attention",
        }.get(model_type, "Attention")

    @staticmethod
    def _source_file(model_type: str) -> str:
        return {
            "code_llama": "transformers/models/llama/modeling_llama.py",
            "diffllama": "transformers/models/diffllama/modeling_diffllama.py",
            "doge": "transformers/models/doge/modeling_doge.py",
            "gemma": "transformers/models/gemma/modeling_gemma.py",
            "llama": "transformers/models/llama/modeling_llama.py",
            "granite": "transformers/models/granite/modeling_granite.py",
            "gemma3_text": "transformers/models/gemma3/modeling_gemma3.py",
            "gemma4_text": "transformers/models/gemma4/modeling_gemma4.py",
            "helium": "transformers/models/helium/modeling_helium.py",
            "ministral": "transformers/models/ministral/modeling_ministral.py",
            "mistral": "transformers/models/mistral/modeling_mistral.py",
            "olmo": "transformers/models/olmo/modeling_olmo.py",
            "qwen2": "transformers/models/qwen2/modeling_qwen2.py",
            "qwen3": "transformers/models/qwen3/modeling_qwen3.py",
            "gemma2": "transformers/models/gemma2/modeling_gemma2.py",
            "seed_oss": "transformers/models/seed_oss/modeling_seed_oss.py",
            "smollm3": "transformers/models/smollm3/modeling_smollm3.py",
            "stablelm": "transformers/models/stablelm/modeling_stablelm.py",
        }.get(model_type, f"transformers/models/{model_type}/modeling_{model_type}.py")
