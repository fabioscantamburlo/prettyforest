"""End-to-end integration tests for forest visualization."""

import tempfile
from pathlib import Path

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


class TestForestVisualization:
    def test_single_tree(self, cls_data):
        X, y = cls_data
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        assert isinstance(html, str)
        assert "<svg" in html
        assert "forest-svg" in html

    def test_random_forest(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        assert "visual-tree" in html
        assert "forest-svg" in html

    def test_gradient_boosting(self, cls_data):
        X, y = cls_data
        model = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        assert "visual-tree" in html

    def test_with_data_adds_predict_panel(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})

        html = visualize(model, data=df)
        assert "predict-panel" in html
        assert "predict-data" in html

    def test_predict_panel_has_close_button(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)
        df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})

        html = visualize(model, data=df)
        assert "predict-close" in html

    def test_without_data_no_predict_panel(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        # The panel div should not be present (JS still references it but handles null)
        assert 'id="predict-panel"' not in html

    def test_numpy_data_input(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model, data=X, feature_names=[f"f{i}" for i in range(4)])
        assert "predict-panel" in html

    def test_tree_structures_always_embedded(self, cls_data):
        X, y = cls_data
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
        model.fit(X, y)

        html = visualize(model)
        assert "trees-data" in html


class TestLightGBM:
    def test_renders(self, cls_data):
        X, y = cls_data
        model = LGBMClassifier(n_estimators=3, max_depth=3, verbose=-1)
        model.fit(X, y)

        html = visualize(model)
        assert "visual-tree" in html


class TestCatBoost:
    def test_renders(self, cls_data):
        X, y = cls_data
        model = CatBoostClassifier(iterations=3, depth=3, verbose=0)
        model.fit(X, y)

        html = visualize(model)
        assert "visual-tree" in html


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
