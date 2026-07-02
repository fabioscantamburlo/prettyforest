"""Route dataset samples through a tree and compute per-node statistics."""

from __future__ import annotations

import numpy as np
import polars as pl

from prettyforest.models import (
    ComparisonOp,
    FlowResult,
    LeafDistribution,
    UnifiedNode,
    UnifiedTree,
)

_OP_FUNCS = {
    ComparisonOp.LE: lambda col, t: col <= t,
    ComparisonOp.LT: lambda col, t: col < t,
    ComparisonOp.GE: lambda col, t: col >= t,
    ComparisonOp.GT: lambda col, t: col > t,
    ComparisonOp.EQ: lambda col, t: col == t,
    ComparisonOp.NE: lambda col, t: col != t,
}


class FlowComputer:
    def compute(
        self,
        tree: UnifiedTree,
        data: pl.DataFrame,
        target: pl.Series | None = None,
    ) -> FlowResult:
        if data.height == 0:
            msg = "Flow Mode requires at least one sample. Provided dataset has 0 rows."
            raise ValueError(msg)

        required = set(tree.feature_names)
        available = set(data.columns)
        # Only check features actually used in splits
        used_features = {
            n.feature_name
            for n in tree.iter_nodes()
            if not n.is_leaf and n.feature_name is not None
        }
        missing = used_features - available
        if missing:
            msg = f"Dataset is missing features required by the tree: {sorted(missing)}"
            raise KeyError(msg)

        total = data.height
        sample_counts: dict[str, int] = {}
        edge_fractions: dict[tuple[str, str], float] = {}
        leaf_distributions: dict[str, LeafDistribution] = {}

        # Use row indices for routing
        all_indices = np.arange(total)
        self._route(
            tree.root, data, all_indices, total,
            sample_counts, edge_fractions, leaf_distributions,
            tree.is_classifier, target,
        )

        return FlowResult(
            sample_counts=sample_counts,
            edge_fractions=edge_fractions,
            leaf_distributions=leaf_distributions,
            total_samples=total,
        )

    def _route(
        self,
        node: UnifiedNode,
        data: pl.DataFrame,
        indices: np.ndarray,
        total: int,
        sample_counts: dict[str, int],
        edge_fractions: dict[tuple[str, str], float],
        leaf_distributions: dict[str, LeafDistribution],
        is_classifier: bool,
        target: pl.Series | None,
    ) -> None:
        count = len(indices)
        sample_counts[node.node_id] = count

        if node.is_leaf:
            leaf_distributions[node.node_id] = self._compute_leaf_distribution(
                indices, is_classifier, target
            )
            return

        # Evaluate split condition
        col_values = data[node.feature_name].to_numpy()[indices]
        op_func = _OP_FUNCS[node.comparison_op]
        mask = op_func(col_values, node.threshold)

        left_indices = indices[mask]
        right_indices = indices[~mask]

        # Edge fractions
        if total > 0:
            edge_fractions[(node.node_id, node.left_child.node_id)] = len(left_indices) / total
            edge_fractions[(node.node_id, node.right_child.node_id)] = len(right_indices) / total

        self._route(
            node.left_child, data, left_indices, total,
            sample_counts, edge_fractions, leaf_distributions,
            is_classifier, target,
        )
        self._route(
            node.right_child, data, right_indices, total,
            sample_counts, edge_fractions, leaf_distributions,
            is_classifier, target,
        )

    def _compute_leaf_distribution(
        self,
        indices: np.ndarray,
        is_classifier: bool,
        target: pl.Series | None,
    ) -> LeafDistribution:
        if target is None or len(indices) == 0:
            return LeafDistribution()

        values = target.to_numpy()[indices]

        if is_classifier:
            unique, counts = np.unique(values, return_counts=True)
            total = counts.sum()
            proportions = {str(u): float(c / total) for u, c in zip(unique, counts)}
            return LeafDistribution(class_proportions=proportions)
        else:
            counts, bin_edges = np.histogram(values, bins=10)
            return LeafDistribution(
                histogram_bins=bin_edges.tolist(),
                histogram_counts=counts.tolist(),
            )
