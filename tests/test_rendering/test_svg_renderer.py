"""Unit tests for SVGRenderer."""

import pytest
import polars as pl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression

from prettyforest.analysis import FlowComputer, PathTracer
from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.rendering.layout_engine import LayoutEngine, compute_initial_collapse_state
from prettyforest.rendering.svg_renderer import SVGRenderer


@pytest.fixture
def cls_tree_and_flow():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    tree = TreeExtractorRegistry().extract(model)
    df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    target = pl.Series("target", y)
    flow = FlowComputer().compute(tree, df, target)
    return tree, flow, df


@pytest.fixture
def reg_tree_and_flow():
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    tree = TreeExtractorRegistry().extract(model)
    df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    target = pl.Series("target", y)
    flow = FlowComputer().compute(tree, df, target)
    return tree, flow, df


class TestBlueprintMode:
    def test_svg_contains_split_conditions(self, cls_tree_and_flow):
        tree, _, _ = cls_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout)

        # Check that visible internal nodes have their feature/threshold
        for node in tree.iter_nodes():
            if not node.is_leaf and node.node_id in layout:
                assert node.feature_name in svg
                assert node.comparison_op.value in svg

    def test_svg_contains_leaf_predictions(self, cls_tree_and_flow):
        tree, _, _ = cls_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout)

        assert "Class:" in svg

    def test_edges_have_labels(self, cls_tree_and_flow):
        tree, _, _ = cls_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout)

        assert "✓" in svg  # left = satisfied
        assert "✗" in svg  # right = not satisfied


class TestFlowMode:
    def test_sample_counts_displayed(self, cls_tree_and_flow):
        tree, flow, _ = cls_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout, flow=flow)

        assert "n=50" in svg  # root should show total

    def test_edge_thickness_in_bounds(self, cls_tree_and_flow):
        tree, flow, _ = cls_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout, flow=flow)

        # All stroke-width values should be present
        assert 'stroke-width="' in svg

    def test_classification_has_pie_chart(self, cls_tree_and_flow):
        tree, flow, _ = cls_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout, flow=flow)

        assert 'class="pie-chart"' in svg

    def test_regression_has_histogram(self, reg_tree_and_flow):
        tree, flow, _ = reg_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout, flow=flow)

        assert 'class="histogram"' in svg

    def test_purity_legend_present(self, cls_tree_and_flow):
        tree, flow, _ = cls_tree_and_flow
        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout, flow=flow)

        assert 'class="purity-legend"' in svg
        assert "High purity" in svg
        assert "Mid purity" in svg
        assert "Low purity" in svg


class TestPathHighlighting:
    def test_highlighted_path_has_distinct_color(self, cls_tree_and_flow):
        tree, flow, df = cls_tree_and_flow
        tracer = PathTracer()
        path = tracer.trace_by_index(tree, df, 0)

        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout, flow=flow, highlighted_path=path)

        # Highlighted color
        assert "#1565c0" in svg

    def test_non_highlighted_have_reduced_opacity(self, cls_tree_and_flow):
        tree, flow, df = cls_tree_and_flow
        tracer = PathTracer()
        path = tracer.trace_by_index(tree, df, 0)

        engine = LayoutEngine()
        layout = engine.compute_layout(tree)
        svg = SVGRenderer().render(tree, layout, flow=flow, highlighted_path=path)

        # Non-highlighted elements should have opacity <= 0.4
        assert 'opacity="0.3"' in svg
