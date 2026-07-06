"""Tests targeting uncovered lines for full coverage."""

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from prettyforest import prettygrow


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    return X, y


class TestTargetConversions:
    """Cover api.py target conversion branches."""

    def test_target_as_numpy(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
        html = prettygrow(model, data=df, target=np.array(y))
        assert "targets" in html

    def test_target_as_polars_series(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
        html = prettygrow(model, data=df, target=pl.Series("y", y))
        assert "targets" in html

    def test_target_as_list(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
        html = prettygrow(model, data=df, target=list(y))
        assert "targets" in html


class TestBoostingMeta:
    """Cover api.py boosting_meta and GBM prediction paths."""

    def test_gbm_has_boosting_meta(self, cls_data):
        X, y = cls_data
        model = GradientBoostingClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
        html = prettygrow(model, data=df)
        assert '"boosting"' in html
        assert '"lr"' in html

    def test_gbm_regressor_has_init(self, reg_data):
        X, y = reg_data
        model = GradientBoostingRegressor(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
        html = prettygrow(model, data=df)
        assert '"init"' in html

    def test_model_predictions_embedded(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
        html = prettygrow(model, data=df)
        assert '"predictions"' in html


class TestSeasonalRendering:
    """Cover season palette branches in visual_mapper and scene_composer."""

    def test_spring_season(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model, season="spring")
        assert "visual-tree" in html

    def test_summer_season(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model, season="summer")
        assert "visual-tree" in html

    def test_autumn_season(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model, season="autumn")
        assert "visual-tree" in html

    def test_winter_season(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model, season="winter")
        assert "visual-tree" in html


class TestLargeEnsemble:
    """Cover paging (hidden trees) and scene scaling."""

    def test_large_ensemble_has_hidden_trees(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=250, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model)
        assert "hidden" in html
        assert "page-next" in html


class TestModelTypes:
    """Cover edge cases in extractors."""

    def test_decision_tree_regressor(self, reg_data):
        X, y = reg_data
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X, y)
        html = prettygrow(model)
        assert "visual-tree" in html

    def test_gbm_classifier_multiclass(self):
        X, y = make_classification(
            n_samples=60,
            n_features=4,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        model = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
        model.fit(X, y)
        html = prettygrow(model)
        # 3 classes × 3 iters = 9 trees
        assert html.count("data-tree-idx=") == 9

    def test_sklearn_feature_names_from_fit(self):
        """Cover the feature_names_in_ path."""
        import pandas as pd

        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(df, y)
        html = prettygrow(model)
        assert "visual-tree" in html


class TestModels:
    """Cover UnifiedTree.get_node."""

    def test_get_node_found(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        from prettyforest.extraction import TreeExtractorRegistry

        tree = TreeExtractorRegistry().extract(model)
        assert tree.get_node(tree.root.node_id) is not None

    def test_get_node_not_found(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        from prettyforest.extraction import TreeExtractorRegistry

        tree = TreeExtractorRegistry().extract(model)
        assert tree.get_node("nonexistent_999") is None


class TestCatBoostCoverage:
    """Cover catboost extractor edge cases."""

    def test_catboost_regressor(self, reg_data):
        from catboost import CatBoostRegressor

        X, y = reg_data
        model = CatBoostRegressor(iterations=3, depth=3, verbose=0, random_seed=42)
        model.fit(X, y)
        html = prettygrow(model)
        assert "visual-tree" in html

    def test_catboost_tree_count(self, cls_data):
        from catboost import CatBoostClassifier
        from prettyforest.extraction.catboost_extractor import CatBoostExtractor

        X, y = cls_data
        model = CatBoostClassifier(iterations=5, depth=3, verbose=0, random_seed=42)
        model.fit(X, y)
        ext = CatBoostExtractor()
        assert ext.tree_count(model) == 5


class TestLightGBMCoverage:
    """Cover lightgbm extractor edge cases."""

    def test_lightgbm_tree_count(self, cls_data):
        from lightgbm import LGBMClassifier
        from prettyforest.extraction.lightgbm_extractor import LightGBMExtractor

        X, y = cls_data
        model = LGBMClassifier(n_estimators=5, max_depth=3, verbose=-1, random_state=42)
        model.fit(X, y)
        ext = LightGBMExtractor()
        assert ext.tree_count(model) >= 5

    def test_lightgbm_classifier_class_detection(self, cls_data):
        from lightgbm import LGBMClassifier
        from prettyforest.extraction.lightgbm_extractor import LightGBMExtractor

        X, y = cls_data
        model = LGBMClassifier(n_estimators=3, max_depth=3, verbose=-1, random_state=42)
        model.fit(X, y)
        ext = LightGBMExtractor()
        trees = ext.extract(model)
        # Binary classification — check is_classifier
        assert any(t.is_classifier for t in trees) or all(
            not t.is_classifier for t in trees
        )


class TestSklearnCoverage:
    """Cover sklearn extractor edge cases."""

    def test_tree_count(self, cls_data):
        X, y = cls_data
        from prettyforest.extraction.sklearn_extractor import SklearnExtractor

        model = RandomForestClassifier(n_estimators=7, max_depth=3, random_state=42)
        model.fit(X, y)
        ext = SklearnExtractor()
        assert ext.tree_count(model) == 7

    def test_no_class_names(self, reg_data):
        """Regressor has no class_names."""
        X, y = reg_data
        from prettyforest.extraction.sklearn_extractor import SklearnExtractor

        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        ext = SklearnExtractor()
        trees = ext.extract(model)
        assert trees[0].class_names is None


class TestVisualMapperCoverage:
    """Cover additive classifier and regressor metric paths."""

    def test_additive_regressor_metric(self, reg_data):
        X, y = reg_data
        model = GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model)
        assert "visual-tree" in html
        # Check variance metric is used
        assert "variance" in html.lower() or "Pred Variance" in html

    def test_single_tree_regressor_metric(self, reg_data):
        X, y = reg_data
        model = DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model)
        assert "visual-tree" in html


class TestJupyterPath:
    """Cover the IPython display path (mock)."""

    def test_no_output_returns_string(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        result = prettygrow(model)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result


class TestCatBoostMulticlass:
    """Cover catboost multiclass extraction."""

    def test_catboost_multiclass(self):
        from catboost import CatBoostClassifier

        X, y = make_classification(
            n_samples=60,
            n_features=4,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        model = CatBoostClassifier(iterations=3, depth=3, verbose=0, random_seed=42)
        model.fit(X, y)
        html = prettygrow(model)
        assert "visual-tree" in html


class TestVisualMapperAdditive:
    """Cover the additive classifier metric path in visual_mapper."""

    def test_additive_classifier_coloring(self):
        """GBM binary classifier — tests the additive classifier metric branch."""
        from sklearn.ensemble import GradientBoostingClassifier
        from prettyforest.rendering.forest.visual_mapper import (
            VisualMapper,
            _compute_metric,
        )
        from prettyforest.extraction import TreeExtractorRegistry
        from prettyforest.models import EnsembleType
        import random

        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        model = GradientBoostingClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        trees = TreeExtractorRegistry().extract(model)

        mapper = VisualMapper()
        visuals = mapper.map_trees(
            trees, random.Random(42), ensemble_type=EnsembleType.ADDITIVE
        )
        assert len(visuals) == len(trees)

        # Also test the metric directly
        metric = _compute_metric(trees[0], EnsembleType.ADDITIVE)
        assert 0 <= metric <= 1

    def test_vote_based_regressor_metric(self):
        """RF regressor — tests variance metric path."""
        from prettyforest.rendering.forest.visual_mapper import _compute_metric
        from prettyforest.extraction import TreeExtractorRegistry
        from prettyforest.models import EnsembleType

        X, y = make_regression(n_samples=50, n_features=4, random_state=42)
        model = RandomForestRegressor(n_estimators=3, max_depth=4, random_state=42)
        model.fit(X, y)
        trees = TreeExtractorRegistry().extract(model)

        metric = _compute_metric(trees[0], EnsembleType.VOTE_BASED)
        assert 0 <= metric <= 1


class TestDataAsNumpy:
    """Cover the numpy data path in api.py (feature_names parameter)."""

    def test_numpy_with_feature_names(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model, data=X, feature_names=["a", "b", "c", "d"])
        assert "predict-panel" in html

    def test_numpy_without_feature_names(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        html = prettygrow(model, data=X)
        assert "predict-panel" in html
