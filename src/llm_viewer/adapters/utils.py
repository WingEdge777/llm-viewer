from __future__ import annotations


def get_attention_name(model_type: str) -> str:
    return {
        "llama": "Grouped Query Attention",
        "mistral": "Sliding Window Attention",
        "qwen2": "Qwen2 Attention",
        "qwen3": "Qwen3 Attention",
        "gemma": "Gemma Attention",
        "gemma2": "Gemma2 Attention",
    }.get(model_type, "Attention")


def get_source_file(model_type: str) -> str:
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
