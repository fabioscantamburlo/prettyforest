"""Unit tests for EnsembleRenderer."""

import pytest
import polars as pl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification

from prettyforest.analysis import FlowComputer
from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.models import EnsembleType
from prettyforest.rendering import EnsembleRenderer


@pytest.fixture
def rf_trees_and_flow():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    trees = TreeExtractorRegistry().extract(model)
    df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    target = pl.Series("target", y)
    flow_results = [FlowComputer().compute(t, df, target) for t in trees]
    return trees, flow_results


@pytest.fixture
def gbm_trees_and_flow():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)
    trees = TreeExtractorRegistry().extract(model)
    df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    target = pl.Series("target", y)
    flow_results = [FlowComputer().compute(t, df, target) for t in trees]
    return trees, flow_results


class TestVoteBasedEnsemble:
    def test_all_trees_present(self, rf_trees_and_flow):
        trees, flow_results = rf_trees_and_flow
        renderer = EnsembleRenderer()
        html = renderer.render(trees, EnsembleType.VOTE_BASED, flow_results=flow_results)

        for i in range(5):
            assert f"Tree {i}" in html

    def test_vote_proportions_displayed(self, rf_trees_and_flow):
        trees, flow_results = rf_trees_and_flow
        vote_props = {"0": 0.4, "1": 0.6}
        renderer = EnsembleRenderer()
        html = renderer.render(
            trees, EnsembleType.VOTE_BASED,
            flow_results=flow_results,
            vote_proportions=vote_props,
        )

        assert "Ensemble Vote" in html
        assert "40.0%" in html
        assert "60.0%" in html

    def test_navigation_controls_present(self, rf_trees_and_flow):
        trees, flow_results = rf_trees_and_flow
        renderer = EnsembleRenderer()
        html = renderer.render(trees, EnsembleType.VOTE_BASED)

        assert "tree-selector" in html
        assert "5 trees" in html


class TestAdditiveEnsemble:
    def test_all_trees_present(self, gbm_trees_and_flow):
        trees, flow_results = gbm_trees_and_flow
        renderer = EnsembleRenderer()
        html = renderer.render(trees, EnsembleType.ADDITIVE, flow_results=flow_results)

        for i in range(3):
            assert f"Tree {i}" in html

    def test_cumulative_contribution_shown(self, gbm_trees_and_flow):
        trees, flow_results = gbm_trees_and_flow
        cumulative = [0.3, 0.45, 0.52]
        renderer = EnsembleRenderer()
        html = renderer.render(
            trees, EnsembleType.ADDITIVE,
            flow_results=flow_results,
            cumulative_contributions=cumulative,
        )

        assert "cumulative: 0.3000" in html
        assert "cumulative: 0.4500" in html

    def test_navigation_controls_present(self, gbm_trees_and_flow):
        trees, _ = gbm_trees_and_flow
        renderer = EnsembleRenderer()
        html = renderer.render(trees, EnsembleType.ADDITIVE)

        assert "tree-selector" in html
        assert "3 trees" in html
