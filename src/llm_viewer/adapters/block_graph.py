from __future__ import annotations

from llm_viewer.adapters.spec import DecoderSpec
from llm_viewer.adapters.utils import get_attention_name, get_source_file
from llm_viewer.profiles import ProfileName, RuntimeProfile
from llm_viewer.schema import Edge, Graph, Node, Shape


def build_block_graph(spec: DecoderSpec, profile: RuntimeProfile) -> Graph:
    batch = profile.batch_size
    seq = profile.seq_len
    past = profile.past_len
    source_file = get_source_file(spec.model_type)

    hidden_shape: Shape = [batch, seq, spec.hidden_size]
    q_proj_shape: Shape = [batch, seq, spec.hidden_size]
    kv_proj_shape: Shape = [batch, seq, spec.num_kv_heads * spec.head_dim]
    q_heads_shape: Shape = [batch, spec.num_heads, seq, spec.head_dim]
    kv_heads_shape: Shape = [batch, spec.num_kv_heads, seq, spec.head_dim]
    total_kv_shape: Shape = [batch, spec.num_kv_heads, seq + past, spec.head_dim]
    attn_out_shape: Shape = [batch, seq, spec.hidden_size]
    mlp_mid_shape: Shape = [batch, seq, spec.intermediate_size]

    nodes = [
        Node(
            id="block_input",
            name="Block Input",
            kind="input",
            op_family="Input",
            output_shapes=[hidden_shape],
            module_path="model.layers.*",
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
        ),
        Node(
            id="attn_mix",
            name=get_attention_name(spec.model_type),
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
            source_file=source_file,
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
            source_file=source_file,
        ),
        Node(
            id="attn_residual",
            name="Attention Residual Add",
            kind="residual",
            op_family="Residual",
            input_shapes=[hidden_shape, attn_out_shape],
            output_shapes=[hidden_shape],
            module_path="model.layers.* (attention residual)",
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
        ),
        Node(
            id="mlp_residual",
            name="MLP Residual Add",
            kind="residual",
            op_family="Residual",
            input_shapes=[hidden_shape, hidden_shape],
            output_shapes=[hidden_shape],
            module_path="model.layers.* (mlp residual)",
            source_file=source_file,
        ),
        Node(
            id="block_output",
            name="Block Output",
            kind="output",
            op_family="Output",
            input_shapes=[hidden_shape],
            output_shapes=[hidden_shape],
            module_path="model.layers.*",
            source_file=source_file,
        ),
    ]

    edges = [
        Edge(
            source="block_input",
            target="input_norm",
            shape=hidden_shape,
            tensor_name="hidden_states",
        ),
        Edge(
            source="input_norm",
            target="q_proj",
            shape=hidden_shape,
            tensor_name="normed_hidden_states",
        ),
        Edge(
            source="input_norm",
            target="k_proj",
            shape=hidden_shape,
            tensor_name="normed_hidden_states",
        ),
        Edge(
            source="input_norm",
            target="v_proj",
            shape=hidden_shape,
            tensor_name="normed_hidden_states",
        ),
        Edge(source="q_proj", target="attn_mix", shape=q_heads_shape, tensor_name="q"),
        Edge(source="attn_mix", target="attn_output", shape=attn_out_shape, tensor_name="context"),
        Edge(
            source="block_input",
            target="attn_residual",
            shape=hidden_shape,
            tensor_name="residual",
            edge_kind="residual",
        ),
        Edge(
            source="attn_output",
            target="attn_residual",
            shape=attn_out_shape,
            tensor_name="attn_out",
        ),
        Edge(
            source="attn_residual",
            target="post_attn_norm",
            shape=hidden_shape,
            tensor_name="hidden_states",
        ),
        Edge(
            source="post_attn_norm",
            target="gate_proj",
            shape=hidden_shape,
            tensor_name="normed_hidden_states",
        ),
        Edge(
            source="post_attn_norm",
            target="up_proj",
            shape=hidden_shape,
            tensor_name="normed_hidden_states",
        ),
        Edge(source="gate_proj", target="down_proj", shape=mlp_mid_shape, tensor_name="gated_mlp"),
        Edge(source="up_proj", target="down_proj", shape=mlp_mid_shape, tensor_name="up_mlp"),
        Edge(
            source="attn_residual",
            target="mlp_residual",
            shape=hidden_shape,
            tensor_name="residual",
            edge_kind="residual",
        ),
        Edge(source="down_proj", target="mlp_residual", shape=hidden_shape, tensor_name="mlp_out"),
        Edge(
            source="mlp_residual",
            target="block_output",
            shape=hidden_shape,
            tensor_name="hidden_states",
        ),
    ]

    if profile.name is ProfileName.DECODE:
        _add_decode_cache_nodes(
            nodes, edges, spec, batch, seq, past, kv_heads_shape, total_kv_shape
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


def _add_decode_cache_nodes(
    nodes: list[Node],
    edges: list[Edge],
    spec: DecoderSpec,
    batch: int,
    seq: int,
    past: int,
    kv_heads_shape: Shape,
    total_kv_shape: Shape,
) -> None:
    source_file = get_source_file(spec.model_type)

    cache_nodes = [
        Node(
            id="past_kv",
            name="Past KV Cache",
            kind="cache",
            op_family="Cache",
            output_shapes=[
                [batch, spec.num_kv_heads, past, spec.head_dim],
                [batch, spec.num_kv_heads, past, spec.head_dim],
            ],
            attrs={"past_len": past},
            module_path="past_key_values",
            source_file=source_file,
        ),
        Node(
            id="cache_update",
            name="Cache Update",
            kind="cache_update",
            op_family="Cache",
            input_shapes=[
                kv_heads_shape,
                kv_heads_shape,
                [batch, spec.num_kv_heads, past, spec.head_dim],
            ],
            output_shapes=[total_kv_shape, total_kv_shape],
            attrs={"new_total_len": seq + past},
            module_path="model.layers.*.self_attn.past_key_values",
            source_file=source_file,
        ),
    ]
    nodes[5:5] = cache_nodes
    edges.extend(
        [
            Edge(
                source="past_kv",
                target="cache_update",
                shape=[batch, spec.num_kv_heads, past, spec.head_dim],
                tensor_name="past_k",
            ),
            Edge(source="k_proj", target="cache_update", shape=kv_heads_shape, tensor_name="new_k"),
            Edge(source="v_proj", target="cache_update", shape=kv_heads_shape, tensor_name="new_v"),
            Edge(
                source="cache_update",
                target="attn_mix",
                shape=total_kv_shape,
                tensor_name="cache_kv",
            ),
        ]
    )
