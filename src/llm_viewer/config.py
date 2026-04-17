"""Config loading utilities for HuggingFace model configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load a HuggingFace model config from a JSON file.

    Args:
        path: Path to config.json or directory containing config.json.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if config_path.is_dir():
        config_path = config_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
