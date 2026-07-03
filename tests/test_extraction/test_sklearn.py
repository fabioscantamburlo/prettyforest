"""Unit tests for SklearnExtractor."""

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.extraction.sklearn_extractor import SklearnExtractor
from prettyforest.models import ComparisonOp


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    return X, y


class TestDecisionTreeClassifier:
    def test_extraction_matches_sklearn_internals(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        extractor = SklearnExtractor()
        trees = extractor.extract(model)

        assert len(trees) == 1
        tree = trees[0]
        assert tree.node_count == model.tree_.node_count
        assert tree.max_depth == model.tree_.max_depth
        assert tree.is_classifier is True

    def test_feature_names_and_thresholds(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        extractor = SklearnExtractor()
        tree = extractor.extract(model)[0]

        assert tree.root.feature_name is not None
        assert tree.root.threshold is not None
        assert tree.root.comparison_op == ComparisonOp.LE
        assert not tree.root.is_leaf

    def test_leaf_nodes_have_class_distribution(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        tree = SklearnExtractor().extract(model)[0]
        leaves = [n for n in tree.iter_nodes() if n.is_leaf]

        for leaf in leaves:
            assert leaf.class_distribution is not None
            assert abs(sum(leaf.class_distribution.values()) - 1.0) < 1e-9


class TestDecisionTreeRegressor:
    def test_extraction_preserves_structure(self, reg_data):
        X, y = reg_data
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)

        tree = SklearnExtractor().extract(model)[0]
        assert tree.node_count == model.tree_.node_count
        assert tree.is_classifier is False

    def test_leaf_nodes_have_prediction_value(self, reg_data):
        X, y = reg_data
        model = DecisionTreeRegressor(max_depth=2, random_state=42)
        model.fit(X, y)

        tree = SklearnExtractor().extract(model)[0]
        leaves = [n for n in tree.iter_nodes() if n.is_leaf]

        for leaf in leaves:
            assert leaf.prediction_value is not None


class TestRandomForest:
    def test_all_trees_extracted(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        trees = registry.extract(model)
        assert len(trees) == 5

    def test_single_tree_by_index(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        tree = registry.extract(model, tree_index=2)
        assert tree.node_count == model.estimators_[2].tree_.node_count


class TestGradientBoosting:
    def test_all_trees_extracted(self, cls_data):
        X, y = cls_data
        model = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        trees = registry.extract(model)
        assert len(trees) == 3


class TestErrorCases:
    def test_unfitted_model_raises_value_error(self):
        model = DecisionTreeClassifier()
        registry = TreeExtractorRegistry()

        with pytest.raises(ValueError, match="must be fitted"):
            registry.extract(model)

    def test_invalid_tree_index_raises_index_error(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        registry = TreeExtractorRegistry()

        with pytest.raises(IndexError, match="out of range"):
            registry.extract(model, tree_index=10)

        with pytest.raises(IndexError, match="out of range"):
            registry.extract(model, tree_index=-1)
