import pytest

from llm_viewer.config import load_model_config
from llm_viewer.profiles import ProfileName, get_profile
from llm_viewer.registry import build_graph_bundle


def test_missing_model_type_raises_error():
    config = {"hidden_size": 4096}

    with pytest.raises(ValueError, match="Missing required config field: model_type"):
        build_graph_bundle(config, get_profile(ProfileName.PREFILL))


def test_unsupported_model_type_raises_error():
    config = {"model_type": "unknown_model", "hidden_size": 4096}

    with pytest.raises(ValueError, match="Unsupported model_type: unknown_model"):
        build_graph_bundle(config, get_profile(ProfileName.PREFILL))


def test_missing_required_config_fields():
    config = {
        "model_type": "llama",
    }

    bundle = build_graph_bundle(config, get_profile(ProfileName.PREFILL))

    assert bundle.metadata["model_type"] == "llama"


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_model_config("/nonexistent/path/config.json")


def test_invalid_json_config(tmp_path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{ invalid json }")

    with pytest.raises(Exception):
        load_model_config(config_path)


def test_config_directory_fallback(tmp_path):
    config_dir = tmp_path / "model"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text('{"model_type": "llama", "hidden_size": 4096}')

    config = load_model_config(config_dir)

    assert config["model_type"] == "llama"


def test_invalid_profile_name():
    with pytest.raises(ValueError, match="Unsupported profile"):
        get_profile("invalid_profile")  # type: ignore
