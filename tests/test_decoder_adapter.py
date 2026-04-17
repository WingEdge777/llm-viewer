from llm_viewer.profiles import ProfileName, get_profile
from llm_viewer.registry import build_graph_bundle

BASE_CONFIG = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
    "vocab_size": 32000,
}


def test_prefill_builds_model_and_block_graphs():
    bundle = build_graph_bundle(BASE_CONFIG, get_profile(ProfileName.PREFILL))

    assert bundle.metadata["model_type"] == "llama"
    assert bundle.metadata["transformers_config_class"] == "LlamaConfig"
    assert bundle.metadata["shape_inference"] == "symbolic-from-transformers-config"
    assert [graph.id for graph in bundle.graphs] == ["model", "block"]

    model_graph = bundle.graphs[0]
    block_graph = bundle.graphs[1]

    repeated_block = next(node for node in model_graph.nodes if node.id == "repeated_block")
    attn_mix = next(node for node in block_graph.nodes if node.id == "attn_mix")

    assert repeated_block.attrs["repeat"] == 32
    assert attn_mix.output_shapes == [[1, 128, 4096]]


def test_decode_adds_cache_nodes():
    bundle = build_graph_bundle(BASE_CONFIG, get_profile(ProfileName.DECODE))
    block_graph = bundle.graphs[1]

    node_ids = {node.id for node in block_graph.nodes}
    edge_names = {edge.tensor_name for edge in block_graph.edges}
    direct_targets = {(edge.source, edge.target) for edge in block_graph.edges}

    assert "past_kv" in node_ids
    assert "cache_update" in node_ids
    assert "cache_kv" in edge_names
    assert ("k_proj", "attn_mix") not in direct_targets
    assert ("v_proj", "attn_mix") not in direct_targets


def test_qwen3_is_supported():
    qwen3_config = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
        "head_dim": 128,
        "vocab_size": 151936,
    }

    bundle = build_graph_bundle(qwen3_config, get_profile(ProfileName.PREFILL))
    model_graph = bundle.graphs[0]
    block_graph = bundle.graphs[1]

    assert bundle.metadata["model_type"] == "qwen3"
    assert bundle.metadata["transformers_config_class"] == "Qwen3Config"
    assert (
        next(node for node in model_graph.nodes if node.id == "repeated_block").attrs["repeat"]
        == 36
    )
    assert (
        next(node for node in block_graph.nodes if node.id == "attn_mix").name == "Qwen3 Attention"
    )


def test_stablelm_is_supported():
    config = {
        "model_type": "stablelm",
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 6912,
        "head_dim": 80,
        "vocab_size": 100352,
    }

    bundle = build_graph_bundle(config, get_profile(ProfileName.PREFILL))

    assert bundle.metadata["model_type"] == "stablelm"
    assert bundle.metadata["transformers_config_class"] == "StableLmConfig"
    assert bundle.graphs[1].attrs["model_type"] == "stablelm"
