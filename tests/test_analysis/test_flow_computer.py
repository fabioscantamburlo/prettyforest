"""Unit tests for FlowComputer."""

import pytest
import polars as pl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression

from prettyforest.analysis import FlowComputer
from prettyforest.extraction import TreeExtractorRegistry


@pytest.fixture
def cls_tree_and_data():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    tree = TreeExtractorRegistry().extract(model)
    df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    target = pl.Series("target", y)
    return tree, df, target


@pytest.fixture
def reg_tree_and_data():
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    tree = TreeExtractorRegistry().extract(model)
    df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    target = pl.Series("target", y)
    return tree, df, target


class TestSampleCountAdditivity:
    def test_root_equals_total_rows(self, cls_tree_and_data):
        tree, df, target = cls_tree_and_data
        result = FlowComputer().compute(tree, df, target)
        assert result.sample_counts[tree.root.node_id] == result.total_samples == 50

    def test_children_sum_to_parent(self, cls_tree_and_data):
        tree, df, target = cls_tree_and_data
        result = FlowComputer().compute(tree, df, target)

        for node in tree.iter_nodes():
            if not node.is_leaf:
                parent_count = result.sample_counts[node.node_id]
                left_count = result.sample_counts[node.left_child.node_id]
                right_count = result.sample_counts[node.right_child.node_id]
                assert left_count + right_count == parent_count


class TestLeafDistributions:
    def test_classification_proportions_sum_to_one(self, cls_tree_and_data):
        tree, df, target = cls_tree_and_data
        result = FlowComputer().compute(tree, df, target)

        for node in tree.iter_nodes():
            if node.is_leaf and node.node_id in result.leaf_distributions:
                dist = result.leaf_distributions[node.node_id]
                if dist.class_proportions and result.sample_counts[node.node_id] > 0:
                    assert abs(sum(dist.class_proportions.values()) - 1.0) < 1e-9

    def test_regression_histograms_have_10_bins(self, reg_tree_and_data):
        tree, df, target = reg_tree_and_data
        result = FlowComputer().compute(tree, df, target)

        for node in tree.iter_nodes():
            if node.is_leaf and node.node_id in result.leaf_distributions:
                dist = result.leaf_distributions[node.node_id]
                if dist.histogram_bins is not None:
                    assert len(dist.histogram_bins) == 11  # 11 edges = 10 bins
                    assert len(dist.histogram_counts) == 10


class TestEdgeFractions:
    def test_fractions_within_bounds(self, cls_tree_and_data):
        tree, df, target = cls_tree_and_data
        result = FlowComputer().compute(tree, df, target)

        for fraction in result.edge_fractions.values():
            assert 0.0 <= fraction <= 1.0


class TestErrorCases:
    def test_zero_row_dataset_raises_value_error(self, cls_tree_and_data):
        tree, df, _ = cls_tree_and_data
        empty_df = df.head(0)

        with pytest.raises(ValueError, match="at least one sample"):
            FlowComputer().compute(tree, empty_df)

    def test_missing_features_raises_key_error(self, cls_tree_and_data):
        tree, _, _ = cls_tree_and_data
        wrong_df = pl.DataFrame({"wrong_col": [1.0, 2.0, 3.0]})

        with pytest.raises(KeyError, match="missing features"):
            FlowComputer().compute(tree, wrong_df)
