"""Unit tests for LayoutEngine."""

import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.rendering.layout_engine import (
    LayoutEngine,
    compute_initial_collapse_state,
    count_descendants,
)


@pytest.fixture
def simple_tree():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return TreeExtractorRegistry().extract(model)


@pytest.fixture
def deep_tree():
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model.fit(X, y)
    return TreeExtractorRegistry().extract(model)


class TestLayoutNoOverlap:
    def test_no_overlapping_nodes(self, simple_tree):
        engine = LayoutEngine()
        layout = engine.compute_layout(simple_tree)

        positions = list(layout.values())
        for i, a in enumerate(positions):
            for b in positions[i + 1:]:
                # Check bounding boxes don't overlap
                x_overlap = a.x < b.x + b.width and b.x < a.x + a.width
                y_overlap = a.y < b.y + b.height and b.y < a.y + a.height
                assert not (x_overlap and y_overlap), (
                    f"Nodes overlap: ({a.x},{a.y}) and ({b.x},{b.y})"
                )

    def test_collapsed_subtree_not_positioned(self, simple_tree):
        engine = LayoutEngine()
        root_id = simple_tree.root.node_id
        collapse_state = {root_id: True}
        layout = engine.compute_layout(simple_tree, collapse_state)

        # Only root should be positioned
        assert root_id in layout
        assert len(layout) == 1


class TestInitialCollapseState:
    def test_shallow_tree_all_expanded(self, simple_tree):
        assert simple_tree.max_depth <= 5
        state = compute_initial_collapse_state(simple_tree)
        assert state == {}

    def test_deep_tree_collapses_beyond_depth_3(self, deep_tree):
        assert deep_tree.max_depth > 5
        state = compute_initial_collapse_state(deep_tree)

        for node in deep_tree.iter_nodes():
            if not node.is_leaf and node.depth >= 3:
                assert state.get(node.node_id, False) is True
            elif not node.is_leaf and node.depth < 3:
                assert node.node_id not in state or state[node.node_id] is False


class TestDescendantCount:
    def test_leaf_has_zero_descendants(self, simple_tree):
        leaves = [n for n in simple_tree.iter_nodes() if n.is_leaf]
        for leaf in leaves:
            assert count_descendants(leaf) == 0

    def test_root_descendants_equals_node_count_minus_one(self, simple_tree):
        assert count_descendants(simple_tree.root) == simple_tree.node_count - 1
