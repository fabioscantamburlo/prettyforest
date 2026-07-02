"""Unit tests for PathTracer."""

import pytest
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from prettyforest.analysis import PathTracer
from prettyforest.extraction import TreeExtractorRegistry


@pytest.fixture
def tree_and_data():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    tree = TreeExtractorRegistry().extract(model)
    df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    return tree, df


class TestPathTrace:
    def test_path_starts_at_root(self, tree_and_data):
        tree, df = tree_and_data
        tracer = PathTracer()
        path = tracer.trace_by_index(tree, df, 0)
        assert path[0].node_id == tree.root.node_id

    def test_path_ends_at_leaf(self, tree_and_data):
        tree, df = tree_and_data
        tracer = PathTracer()
        path = tracer.trace_by_index(tree, df, 0)
        last = path[-1]
        leaf_node = tree.get_node(last.node_id)
        assert leaf_node.is_leaf

    def test_path_has_prediction_at_leaf(self, tree_and_data):
        tree, df = tree_and_data
        tracer = PathTracer()
        path = tracer.trace_by_index(tree, df, 0)
        assert path[-1].prediction is not None

    def test_decision_outcomes_are_consistent(self, tree_and_data):
        tree, df = tree_and_data
        tracer = PathTracer()
        row = df.row(0, named=True)
        path = tracer.trace_by_index(tree, df, 0)

        node = tree.root
        for step in path[:-1]:  # all except leaf
            assert step.decision_outcome in ("satisfied", "not_satisfied")
            value = row[node.feature_name]
            from prettyforest.models import ComparisonOp
            ops = {
                ComparisonOp.LE: lambda v, t: v <= t,
                ComparisonOp.LT: lambda v, t: v < t,
                ComparisonOp.GE: lambda v, t: v >= t,
                ComparisonOp.GT: lambda v, t: v > t,
            }
            expected_satisfied = ops[node.comparison_op](value, node.threshold)
            expected_outcome = "satisfied" if expected_satisfied else "not_satisfied"
            assert step.decision_outcome == expected_outcome
            node = node.left_child if expected_satisfied else node.right_child


class TestErrorCases:
    def test_negative_index_raises(self, tree_and_data):
        tree, df = tree_and_data
        tracer = PathTracer()

        with pytest.raises(IndexError, match="out of range"):
            tracer.trace_by_index(tree, df, -1)

    def test_out_of_bounds_index_raises(self, tree_and_data):
        tree, df = tree_and_data
        tracer = PathTracer()

        with pytest.raises(IndexError, match="out of range"):
            tracer.trace_by_index(tree, df, 999)
