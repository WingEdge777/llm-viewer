from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_viewer.profiles import RuntimeProfile
from llm_viewer.schema import GraphBundle


class GraphAdapter(ABC):
    model_types: set[str]

    def supports(self, model_type: str) -> bool:
        return model_type in self.model_types

    @abstractmethod
    def build(self, config: dict[str, Any], profile: RuntimeProfile) -> GraphBundle:
        raise NotImplementedError
