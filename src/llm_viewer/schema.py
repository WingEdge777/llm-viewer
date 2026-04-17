from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Shape = list[int | str]


@dataclass
class Node:
    id: str
    name: str
    kind: str
    op_family: str
    input_shapes: list[Shape] = field(default_factory=list)
    output_shapes: list[Shape] = field(default_factory=list)
    param_shapes: list[Shape] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    module_path: str | None = None
    source_file: str | None = None
    source_line: int | None = None


@dataclass
class Edge:
    source: str
    target: str
    shape: Shape | None = None
    tensor_name: str | None = None
    edge_kind: str = "data"


@dataclass
class Graph:
    id: str
    name: str
    level: str
    nodes: list[Node]
    edges: list[Edge]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphBundle:
    metadata: dict[str, Any]
    graphs: list[Graph]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
