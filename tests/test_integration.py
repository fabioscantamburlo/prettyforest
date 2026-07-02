"""End-to-end integration tests."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from prettyforest import visualize


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    return X, y


class TestSklearnEndToEnd:
    def test_blueprint_single_tree(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        assert isinstance(html, str)
        assert "<svg" in html
        assert "<!DOCTYPE html>" in html

    def test_flow_single_tree(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
        target = pl.Series("target", y)

        html = visualize(model, data=df, target=target, mode="flow")
        assert "n=50" in html
        assert 'class="pie-chart"' in html

    def test_ensemble_random_forest(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        assert "Tree 0" in html
        assert "Tree 4" in html
        assert "tree-selector" in html

    def test_ensemble_flow_mode(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
        target = pl.Series("target", y)

        html = visualize(model, data=df, target=target, mode="flow")
        assert "Ensemble Vote" in html


class TestLightGBMEndToEnd:
    def test_blueprint(self, cls_data):
        X, y = cls_data
        model = LGBMClassifier(n_estimators=3, max_depth=3, verbose=-1)
        model.fit(X, y)

        html = visualize(model)
        assert "<svg" in html
        assert "Tree 0" in html


class TestCatBoostEndToEnd:
    def test_blueprint(self, cls_data):
        X, y = cls_data
        model = CatBoostClassifier(iterations=3, depth=3, verbose=0)
        model.fit(X, y)

        html = visualize(model)
        assert "<svg" in html
        assert "Tree 0" in html


class TestOutputHandling:
    def test_file_output(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        result = visualize(model, output_path=path)
        assert result is None
        content = Path(path).read_text(encoding="utf-8")
        assert "<svg" in content
        Path(path).unlink()

    def test_unwritable_path_raises(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        with pytest.raises(OSError, match="Cannot write"):
            visualize(model, output_path="/nonexistent/dir/out.html")

    def test_string_return_when_no_path(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_numpy_data_input(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(
            model,
            data=X,
            target=np.array(y),
            mode="flow",
            feature_names=[f"feature_{i}" for i in range(4)],
        )
        assert "n=50" in html


class TestSingleTreeFromEnsemble:
    def test_tree_index_renders_single(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model, tree_index=2)
        # Single tree view — no ensemble navigation
        assert "tree-selector" not in html
        assert "<svg" in html
