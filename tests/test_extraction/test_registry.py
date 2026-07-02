"""Unit tests for TreeExtractorRegistry."""

import pytest

from prettyforest.extraction import TreeExtractorRegistry


class TestUnsupportedModels:
    def test_plain_object_raises_type_error(self):
        registry = TreeExtractorRegistry()

        with pytest.raises(TypeError, match="Unsupported model type"):
            registry.extract(object())

    def test_string_raises_type_error(self):
        registry = TreeExtractorRegistry()

        with pytest.raises(TypeError, match="Unsupported model type"):
            registry.extract("not a model")

    def test_error_message_lists_supported_types(self):
        registry = TreeExtractorRegistry()

        with pytest.raises(TypeError, match="DecisionTreeClassifier"):
            registry.extract(42)


class TestDispatch:
    def test_routes_sklearn(self):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=4, random_state=0)
        model = DecisionTreeClassifier(max_depth=2, random_state=0)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        tree = registry.extract(model)
        assert tree.node_count > 0

    def test_routes_lightgbm(self):
        import lightgbm as lgb
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=4, random_state=0)
        model = lgb.LGBMClassifier(n_estimators=2, max_depth=2, verbose=-1)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        trees = registry.extract(model)
        assert len(trees) == 2

    def test_routes_catboost(self):
        from catboost import CatBoostClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=50, n_features=4, random_state=0)
        model = CatBoostClassifier(iterations=2, depth=2, verbose=0)
        model.fit(X, y)

        registry = TreeExtractorRegistry()
        trees = registry.extract(model)
        assert len(trees) == 2
