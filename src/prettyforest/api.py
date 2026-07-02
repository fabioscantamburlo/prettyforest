"""Public API for PrettyForest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from prettyforest.analysis import FlowComputer, PathTracer, numpy_to_polars
from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.models import EnsembleType, FlowResult, UnifiedTree
from prettyforest.rendering import HTMLAssembler, EnsembleRenderer, LayoutEngine, SVGRenderer
from prettyforest.rendering.layout_engine import compute_initial_collapse_state
from prettyforest.serialization import serialize

MAX_EMBEDDED_SAMPLES = 200


def visualize(
    model: Any,
    *,
    data: pl.DataFrame | np.ndarray | None = None,
    target: pl.Series | np.ndarray | None = None,
    mode: Literal["blueprint", "flow", "forest"] = "blueprint",
    tree_index: int | None = None,
    output_path: str | Path | None = None,
    feature_names: list[str] | None = None,
    highlight_samples: list[int] | None = None,
    seed: int = 42,
    season: Literal["summer", "autumn", "winter"] | None = None,
) -> str | None:
    """Visualize a trained tree-based model.

    Args:
        model: Trained tree model (sklearn, LightGBM, or CatBoost)
        data: Dataset for Flow Mode (Polars DataFrame or NumPy array)
        target: Target values for leaf distribution visualization
        mode: "blueprint" for structure only, "flow" for data overlay, "forest" for aesthetic view
        tree_index: If set, visualize only this tree from an ensemble
        output_path: File path to write HTML; if None, returns string or renders in Jupyter
        feature_names: Column names when data is a NumPy array
        highlight_samples: Row indices to embed for path highlighting. If None,
            embeds the first 200 rows. Pass specific indices to include exactly
            those rows (e.g. [5, 42, 150, 999]).
        seed: Random seed for forest mode (default 42)
        season: Color theme for forest mode ("summer", "autumn", "winter", or None)

    Returns:
        HTML string if no output_path and not in Jupyter; None otherwise
    """
    registry = TreeExtractorRegistry()
    result = registry.extract(model, tree_index=tree_index)

    if isinstance(result, list):
        trees = result
        ensemble_type = _detect_ensemble_type(model)
    else:
        trees = [result]
        ensemble_type = EnsembleType.SINGLE

    # Forest mode — purely aesthetic
    if mode == "forest":
        from prettyforest.rendering.forest import ForestConfig, ForestRenderer
        config = ForestConfig(seed=seed, season=season)

        forest_data = None
        if data is not None:
            if isinstance(data, np.ndarray):
                names = feature_names or trees[0].feature_names
                forest_data = numpy_to_polars(data, names)
            else:
                forest_data = data

        html = ForestRenderer().render(
            trees, config, ensemble_type=ensemble_type, data=forest_data
        )
        return _handle_output(html, output_path)

    df: pl.DataFrame | None = None
    target_series: pl.Series | None = None

    if mode == "flow" and data is not None:
        if isinstance(data, np.ndarray):
            names = feature_names or trees[0].feature_names
            df = numpy_to_polars(data, names)
        else:
            df = data

        if target is not None:
            if isinstance(target, np.ndarray):
                target_series = pl.Series("target", target)
            else:
                target_series = target

    if ensemble_type != EnsembleType.SINGLE and len(trees) > 1:
        html = _render_ensemble(trees, ensemble_type, df, target_series, mode)
    else:
        tree = trees[0]
        html = _render_single(tree, df, target_series, mode, highlight_samples)

    return _handle_output(html, output_path)


def _build_flow_data_json(
    df: pl.DataFrame,
    tree: UnifiedTree,
    highlight_samples: list[int] | None,
) -> str:
    """Build the JSON payload for client-side path tracing.

    Embeds a mapping of row_index -> {feature: value} for the selected rows.
    """
    n_rows = df.height
    feature_cols = [c for c in df.columns if c in set(tree.feature_names)]

    if highlight_samples is not None:
        indices = [i for i in highlight_samples if 0 <= i < n_rows]
    else:
        indices = list(range(min(n_rows, MAX_EMBEDDED_SAMPLES)))

    subset = df.select(feature_cols)
    embedded: dict[str, dict[str, float]] = {}
    for idx in indices:
        row = subset.row(idx, named=True)
        embedded[str(idx)] = {k: float(v) for k, v in row.items()}

    return json.dumps({"samples": embedded, "total_rows": n_rows})


def _detect_ensemble_type(model: Any) -> EnsembleType:
    module = type(model).__module__ or ""
    name = type(model).__name__

    if "RandomForest" in name:
        return EnsembleType.VOTE_BASED
    if "GradientBoosting" in name or "LGBM" in name or "CatBoost" in name:
        return EnsembleType.ADDITIVE
    if module.startswith("lightgbm") or module.startswith("catboost"):
        return EnsembleType.ADDITIVE
    return EnsembleType.SINGLE


def _render_single(
    tree: UnifiedTree,
    df: pl.DataFrame | None,
    target: pl.Series | None,
    mode: str,
    highlight_samples: list[int] | None = None,
) -> str:
    layout_engine = LayoutEngine()
    svg_renderer = SVGRenderer()
    html_assembler = HTMLAssembler()

    collapse_state = compute_initial_collapse_state(tree)
    layout = layout_engine.compute_layout(tree, collapse_state)

    flow: FlowResult | None = None
    flow_data_json = '{"samples": {}, "total_rows": 0}'
    if mode == "flow" and df is not None:
        flow = FlowComputer().compute(tree, df, target)
        flow_data_json = _build_flow_data_json(df, tree, highlight_samples)

    svg = svg_renderer.render(tree, layout, flow=flow, collapse_state=collapse_state)
    tree_json = serialize(tree)
    return html_assembler.assemble(
        svg, tree_json=tree_json, flow_data_json=flow_data_json, mode=mode
    )


def _render_ensemble(
    trees: list[UnifiedTree],
    ensemble_type: EnsembleType,
    df: pl.DataFrame | None,
    target: pl.Series | None,
    mode: str,
) -> str:
    flow_results: list[FlowResult] | None = None
    cumulative_contributions: list[float] | None = None
    vote_proportions: dict[str, float] | None = None

    if mode == "flow" and df is not None:
        fc = FlowComputer()
        flow_results = [fc.compute(tree, df, target) for tree in trees]

        if ensemble_type == EnsembleType.ADDITIVE:
            cumulative_contributions = _compute_cumulative(trees, flow_results)
        elif ensemble_type == EnsembleType.VOTE_BASED:
            vote_proportions = _compute_vote_proportions(trees, flow_results)

    renderer = EnsembleRenderer()
    return renderer.render(
        trees,
        ensemble_type,
        flow_results=flow_results,
        cumulative_contributions=cumulative_contributions,
        vote_proportions=vote_proportions,
    )


def _compute_cumulative(
    trees: list[UnifiedTree], flow_results: list[FlowResult]
) -> list[float]:
    cumulative: list[float] = []
    running = 0.0
    for tree, flow in zip(trees, flow_results):
        total = flow.total_samples
        if total == 0:
            cumulative.append(running)
            continue
        weighted_sum = 0.0
        for node in tree.iter_nodes():
            if node.is_leaf and node.prediction_value is not None:
                count = flow.sample_counts.get(node.node_id, 0)
                weighted_sum += node.prediction_value * count
        contribution = weighted_sum / total
        running += contribution
        cumulative.append(running)
    return cumulative


def _compute_vote_proportions(
    trees: list[UnifiedTree], flow_results: list[FlowResult]
) -> dict[str, float]:
    vote_counts: dict[str, float] = {}

    for tree, flow in zip(trees, flow_results):
        if not tree.is_classifier:
            continue
        max_count = -1
        majority_class = None
        for node in tree.iter_nodes():
            if node.is_leaf and node.class_distribution:
                count = flow.sample_counts.get(node.node_id, 0)
                if count > max_count:
                    max_count = count
                    majority = max(node.class_distribution, key=node.class_distribution.get)
                    majority_class = majority

        if majority_class:
            vote_counts[majority_class] = vote_counts.get(majority_class, 0) + 1

    total_votes = sum(vote_counts.values())
    if total_votes > 0:
        return {cls: count / total_votes for cls, count in vote_counts.items()}
    return vote_counts


def _handle_output(html: str, output_path: str | Path | None) -> str | None:
    if output_path is not None:
        path = Path(output_path)
        try:
            path.write_text(html, encoding="utf-8")
        except (OSError, PermissionError) as e:
            msg = f"Cannot write to '{path}': {e}"
            raise OSError(msg) from e
        return None

    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None and "IPKernelApp" in shell.config:
            from IPython.display import HTML, display
            display(HTML(html))
            return None
    except (ImportError, AttributeError):
        pass

    return html
