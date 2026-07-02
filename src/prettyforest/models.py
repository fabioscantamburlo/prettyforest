from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum


class ComparisonOp(Enum):
    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"
    EQ = "=="
    NE = "!="


class EnsembleType(Enum):
    SINGLE = "single"
    VOTE_BASED = "vote_based"
    ADDITIVE = "additive"


@dataclass
class UnifiedNode:
    node_id: str
    depth: int
    feature_name: str | None = None
    threshold: float | None = None
    comparison_op: ComparisonOp | None = None
    left_child: UnifiedNode | None = None
    right_child: UnifiedNode | None = None
    prediction_value: float | None = None
    class_distribution: dict[str, float] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


@dataclass
class UnifiedTree:
    root: UnifiedNode
    node_count: int
    max_depth: int
    feature_names: list[str]
    is_classifier: bool
    class_names: list[str] | None = None

    def iter_nodes(self) -> Iterator[UnifiedNode]:
        queue: deque[UnifiedNode] = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            if node.left_child is not None:
                queue.append(node.left_child)
            if node.right_child is not None:
                queue.append(node.right_child)

    def get_node(self, node_id: str) -> UnifiedNode | None:
        for node in self.iter_nodes():
            if node.node_id == node_id:
                return node
        return None


@dataclass
class LeafDistribution:
    class_proportions: dict[str, float] | None = None
    histogram_bins: list[float] | None = None
    histogram_counts: list[int] | None = None


@dataclass
class FlowResult:
    sample_counts: dict[str, int]
    edge_fractions: dict[tuple[str, str], float]
    leaf_distributions: dict[str, LeafDistribution]
    total_samples: int


@dataclass
class PathStep:
    node_id: str
    decision_outcome: str | None = None
    prediction: float | dict[str, float] | None = None


@dataclass
class NodePosition:
    x: float
    y: float
    width: float
    height: float


@dataclass
class EnsembleMeta:
    ensemble_type: EnsembleType
    tree_count: int
    cumulative_contributions: list[float] | None = None
    vote_proportions: dict[str, float] | None = None
