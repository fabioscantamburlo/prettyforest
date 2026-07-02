"""Unit tests for LightGBMExtractor."""

import pytest
import lightgbm as lgb
from sklearn.datasets import make_classification, make_regression

from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.extraction.lightgbm_extractor import LightGBMExtractor


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    return X, y


class TestLGBMClassifier:
    def test_extraction_produces_correct_tree_count(self, cls_data):
        X, y = cls_data
        model = lgb.LGBMClassifier(n_estimators=3, max_depth=3, verbose=-1)
        model.fit(X, y)

        extractor = LightGBMExtractor()
        trees = extractor.extract(model)
        assert len(trees) == 3

    def test_tree_structure_has_valid_nodes(self, cls_data):
        X, y = cls_data
        model = lgb.LGBMClassifier(n_estimators=2, max_depth=2, verbose=-1)
        model.fit(X, y)

        tree = LightGBMExtractor().extract(model)[0]
        assert tree.root.feature_name is not None
        assert tree.root.threshold is not None
        assert not tree.root.is_leaf
        assert tree.is_classifier is True

    def test_leaf_nodes_have_prediction_values(self, cls_data):
        X, y = cls_data
        model = lgb.LGBMClassifier(n_estimators=2, max_depth=2, verbose=-1)
        model.fit(X, y)

        tree = LightGBMExtractor().extract(model)[0]
        leaves = [n for n in tree.iter_nodes() if n.is_leaf]
        assert len(leaves) > 0
        for leaf in leaves:
            assert leaf.prediction_value is not None


class TestLGBMRegressor:
    def test_extraction_produces_trees(self, reg_data):
        X, y = reg_data
        model = lgb.LGBMRegressor(n_estimators=3, max_depth=3, verbose=-1)
        model.fit(X, y)

        trees = LightGBMExtractor().extract(model)
        assert len(trees) == 3
        assert trees[0].is_classifier is False

    def test_leaf_nodes_have_prediction_values(self, reg_data):
        X, y = reg_data
        model = lgb.LGBMRegressor(n_estimators=2, max_depth=2, verbose=-1)
        model.fit(X, y)

        tree = LightGBMExtractor().extract(model)[0]
        leaves = [n for n in tree.iter_nodes() if n.is_leaf]
        for leaf in leaves:
            assert leaf.prediction_value is not None


class TestBoosterDirect:
    def test_booster_extraction(self, cls_data):
        X, y = cls_data
        train_data = lgb.Dataset(X, label=y)
        params = {"objective": "binary", "max_depth": 3, "verbose": -1}
        booster = lgb.train(params, train_data, num_boost_round=3)

        extractor = LightGBMExtractor()
        assert extractor.is_fitted(booster)
        trees = extractor.extract(booster)
        assert len(trees) == 3


class TestErrorCases:
    def test_unfitted_model_raises_value_error(self):
        model = lgb.LGBMClassifier()
        registry = TreeExtractorRegistry()

        with pytest.raises(ValueError, match="must be fitted"):
            registry.extract(model)

    def test_invalid_tree_index_raises_index_error(self, cls_data):
        X, y = cls_data
        model = lgb.LGBMClassifier(n_estimators=3, max_depth=3, verbose=-1)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        with pytest.raises(IndexError, match="out of range"):
            registry.extract(model, tree_index=10)
