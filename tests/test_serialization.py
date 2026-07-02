"""Unit tests for tree serialization."""

import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression

from prettyforest import serialize, deserialize
from prettyforest.extraction import TreeExtractorRegistry


@pytest.fixture
def cls_tree():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return TreeExtractorRegistry().extract(model)


@pytest.fixture
def reg_tree():
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    return TreeExtractorRegistry().extract(model)


class TestRoundTrip:
    def test_classifier_round_trip(self, cls_tree):
        json_str = serialize(cls_tree)
        restored = deserialize(json_str)

        assert restored.node_count == cls_tree.node_count
        assert restored.max_depth == cls_tree.max_depth
        assert restored.is_classifier == cls_tree.is_classifier
        assert restored.feature_names == cls_tree.feature_names

    def test_regressor_round_trip(self, reg_tree):
        json_str = serialize(reg_tree)
        restored = deserialize(json_str)

        assert restored.node_count == reg_tree.node_count
        assert restored.max_depth == reg_tree.max_depth
        assert restored.is_classifier is False

    def test_threshold_precision(self, cls_tree):
        json_str = serialize(cls_tree)
        restored = deserialize(json_str)

        # Compare root threshold with high precision
        assert abs(restored.root.threshold - cls_tree.root.threshold) < 1e-10


class TestDeserializationErrors:
    def test_malformed_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            deserialize("{not valid json")

    def test_missing_tree_field(self):
        with pytest.raises(ValueError, match="missing required field 'tree'"):
            deserialize('{"version": "1.0"}')

    def test_missing_root_field(self):
        with pytest.raises(ValueError, match="missing required field 'root'"):
            deserialize('{"tree": {"node_count": 1, "max_depth": 0, "is_classifier": true, "feature_names": []}}')

    def test_unknown_node_type(self):
        json_str = '{"tree": {"node_count": 1, "max_depth": 0, "is_classifier": true, "feature_names": [], "root": {"node_id": "0", "depth": 0, "type": "unknown"}}}'
        with pytest.raises(ValueError, match="unknown node type"):
            deserialize(json_str)

    def test_unknown_comparison_op(self):
        json_str = '{"tree": {"node_count": 1, "max_depth": 0, "is_classifier": true, "feature_names": [], "root": {"node_id": "0", "depth": 0, "type": "internal", "feature_name": "x", "threshold": 1.0, "comparison_op": "??", "left": {"node_id": "1", "depth": 1, "type": "leaf"}, "right": {"node_id": "2", "depth": 1, "type": "leaf"}}}}'
        with pytest.raises(ValueError, match="unknown comparison operator"):
            deserialize(json_str)
