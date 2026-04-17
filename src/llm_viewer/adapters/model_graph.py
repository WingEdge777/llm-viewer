from __future__ import annotations

from llm_viewer.adapters.spec import DecoderSpec
from llm_viewer.adapters.utils import get_source_file
from llm_viewer.profiles import RuntimeProfile
from llm_viewer.schema import Edge, Graph, Node, Shape


def build_model_graph(spec: DecoderSpec, profile: RuntimeProfile) -> Graph:
    batch = profile.batch_size
    seq = profile.seq_len
    token_shape: Shape = [batch, seq]
    hidden_shape: Shape = [batch, seq, spec.hidden_size]
    logits_shape: Shape = [batch, seq, spec.vocab_size or "vocab"]
    source_file = get_source_file(spec.model_type)

    nodes = [
        Node(
            id="tokens",
            name="Input Tokens",
            kind="input",
            op_family="Input",
            output_shapes=[token_shape],
            attrs={"profile": profile.name.value},
            module_path="input_ids",
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
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
            source_file=source_file,
        ),
    ]

    edges = [
        Edge(source="tokens", target="embedding", shape=token_shape, tensor_name="input_ids"),
        Edge(
            source="embedding",
            target="repeated_block",
            shape=hidden_shape,
            tensor_name="hidden_states",
        ),
        Edge(
            source="repeated_block",
            target="final_norm",
            shape=hidden_shape,
            tensor_name="hidden_states",
        ),
        Edge(
            source="final_norm",
            target="lm_head",
            shape=hidden_shape,
            tensor_name="hidden_states",
        ),
    ]

    return Graph(
        id="model",
        name="Model Graph",
        level="model",
        nodes=nodes,
        edges=edges,
        attrs={"model_type": spec.model_type, "profile": profile.name.value},
    )
