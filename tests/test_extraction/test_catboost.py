"""Unit tests for CatBoostExtractor."""

import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import make_classification, make_regression

from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.extraction.catboost_extractor import CatBoostExtractor


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    return X, y


class TestCatBoostClassifier:
    def test_extraction_produces_correct_tree_count(self, cls_data):
        X, y = cls_data
        model = CatBoostClassifier(iterations=3, depth=3, verbose=0)
        model.fit(X, y)

        extractor = CatBoostExtractor()
        trees = extractor.extract(model)
        assert len(trees) == 3

    def test_tree_structure_has_valid_nodes(self, cls_data):
        X, y = cls_data
        model = CatBoostClassifier(iterations=2, depth=2, verbose=0)
        model.fit(X, y)

        tree = CatBoostExtractor().extract(model)[0]
        assert tree.root.feature_name is not None
        assert tree.root.threshold is not None
        assert not tree.root.is_leaf
        assert tree.is_classifier is True

    def test_leaf_nodes_have_class_distribution(self, cls_data):
        X, y = cls_data
        model = CatBoostClassifier(iterations=2, depth=2, verbose=0)
        model.fit(X, y)

        tree = CatBoostExtractor().extract(model)[0]
        leaves = [n for n in tree.iter_nodes() if n.is_leaf]
        assert len(leaves) > 0
        for leaf in leaves:
            assert leaf.class_distribution is not None
            assert abs(sum(leaf.class_distribution.values()) - 1.0) < 1e-9


class TestCatBoostRegressor:
    def test_extraction_produces_trees(self, reg_data):
        X, y = reg_data
        model = CatBoostRegressor(iterations=3, depth=3, verbose=0)
        model.fit(X, y)

        trees = CatBoostExtractor().extract(model)
        assert len(trees) == 3
        assert trees[0].is_classifier is False

    def test_leaf_nodes_have_prediction_values(self, reg_data):
        X, y = reg_data
        model = CatBoostRegressor(iterations=2, depth=2, verbose=0)
        model.fit(X, y)

        tree = CatBoostExtractor().extract(model)[0]
        leaves = [n for n in tree.iter_nodes() if n.is_leaf]
        for leaf in leaves:
            assert leaf.prediction_value is not None


class TestErrorCases:
    def test_unfitted_model_raises_value_error(self):
        model = CatBoostClassifier(verbose=0)
        registry = TreeExtractorRegistry()

        with pytest.raises(ValueError, match="must be fitted"):
            registry.extract(model)

    def test_invalid_tree_index_raises_index_error(self, cls_data):
        X, y = cls_data
        model = CatBoostClassifier(iterations=3, depth=3, verbose=0)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        with pytest.raises(IndexError, match="out of range"):
            registry.extract(model, tree_index=10)
