"""Trace a single sample's path from root to leaf."""

from __future__ import annotations

from prettyforest.models import ComparisonOp, PathStep, UnifiedNode, UnifiedTree

_OP_EVAL = {
    ComparisonOp.LE: lambda v, t: v <= t,
    ComparisonOp.LT: lambda v, t: v < t,
    ComparisonOp.GE: lambda v, t: v >= t,
    ComparisonOp.GT: lambda v, t: v > t,
    ComparisonOp.EQ: lambda v, t: v == t,
    ComparisonOp.NE: lambda v, t: v != t,
}


class PathTracer:
    def trace(self, tree: UnifiedTree, sample: dict[str, float]) -> list[PathStep]:
        """Trace a sample from root to leaf, returning the path with decision outcomes."""
        path: list[PathStep] = []
        node = tree.root
        while not node.is_leaf:
            value = sample[node.feature_name]
            satisfied = _OP_EVAL[node.comparison_op](value, node.threshold)
            outcome = "satisfied" if satisfied else "not_satisfied"
            path.append(PathStep(node_id=node.node_id, decision_outcome=outcome))
            node = node.left_child if satisfied else node.right_child

        # Leaf step
        prediction = node.class_distribution if node.class_distribution else node.prediction_value
        path.append(PathStep(node_id=node.node_id, prediction=prediction))
        return path

    def trace_by_index(
        self, tree: UnifiedTree, data, row_index: int
    ) -> list[PathStep]:
        """Trace a sample by row index from a DataFrame or array."""
        import polars as pl

        if isinstance(data, pl.DataFrame):
            n_rows = data.height
        else:
            n_rows = len(data)

        if row_index < 0 or row_index >= n_rows:
            msg = f"Row index {row_index} out of range. Valid range: [0, {n_rows})."
            raise IndexError(msg)

        if isinstance(data, pl.DataFrame):
            row = data.row(row_index, named=True)
            sample = {k: float(v) for k, v in row.items()}
        else:
            import numpy as np
            sample = {tree.feature_names[i]: float(data[row_index, i]) for i in range(len(tree.feature_names))}

        return self.trace(tree, sample)
