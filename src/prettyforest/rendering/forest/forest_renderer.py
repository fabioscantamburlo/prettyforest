"""ForestRenderer — orchestrates the aesthetic forest rendering pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path

from prettyforest.models import EnsembleType, UnifiedTree
from prettyforest.rendering.forest.models import ForestConfig
from prettyforest.rendering.forest.scene_composer import (
    MAX_VISIBLE,
    SceneComposer,
    TreeMeta,
)
from prettyforest.rendering.forest.tree_shape_generator import TreeShapeGenerator
from prettyforest.rendering.forest.visual_mapper import VisualMapper

_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def _get_static_css() -> str:
    return (_STATIC_DIR / "forest.css").read_text(encoding="utf-8")


def _get_static_js() -> str:
    return (_STATIC_DIR / "forest.js").read_text(encoding="utf-8")


class ForestRenderer:
    def __init__(self):
        self._mapper = VisualMapper()
        self._shape_gen = TreeShapeGenerator()
        self._composer = SceneComposer()

    def render(
        self,
        trees: list[UnifiedTree],
        config: ForestConfig = ForestConfig(),
        ensemble_type: EnsembleType = EnsembleType.SINGLE,
        data=None,
        boosting_meta: dict | None = None,
        target: list | None = None,
        model_name: str = "Unknown Model",
        model_predictions: list | None = None,
    ) -> str:
        rng = random.Random(config.seed)
        total_count = len(trees)

        visuals = self._mapper.map_trees(
            trees, rng, season=config.season, ensemble_type=ensemble_type
        )
        paths = [self._shape_gen.generate(v, rng) for v in visuals]
        metas = [
            self._compute_meta(tree, i, ensemble_type) for i, tree in enumerate(trees)
        ]

        svg = self._composer.compose(paths, visuals, metas, config, total_count, rng)
        # Determine if this is a classification task from the model name
        is_classifier = "Classification" in model_name or "Classifier" in model_name

        # Build prediction data if data was provided
        predict_json = ""
        if data is not None:
            predict_json = self._build_predict_json(
                trees,
                data,
                ensemble_type,
                boosting_meta,
                target,
                is_classifier,
                model_predictions,
            )

        # Always embed tree structures for detail view
        trees_json = self._build_trees_json(trees)

        return self._wrap_html(
            svg,
            ensemble_type,
            is_classifier,
            total_count,
            predict_json,
            trees_json,
            model_name,
        )

    def _compute_meta(
        self, tree: UnifiedTree, index: int, ensemble_type: EnsembleType
    ) -> TreeMeta:
        leaves = [n for n in tree.iter_nodes() if n.is_leaf]
        n_leaves = len(leaves)

        avg_purity = None
        pred_variance = None
        leaf_magnitude = None

        if tree.is_classifier:
            if ensemble_type == EnsembleType.VOTE_BASED:
                purities = []
                for leaf in leaves:
                    if leaf.class_distribution:
                        purities.append(max(leaf.class_distribution.values()))
                avg_purity = sum(purities) / len(purities) if purities else None
            else:
                magnitudes = []
                for leaf in leaves:
                    if leaf.prediction_value is not None:
                        magnitudes.append(abs(leaf.prediction_value))
                    elif leaf.class_distribution:
                        magnitudes.append(
                            max(abs(v) for v in leaf.class_distribution.values())
                        )
                leaf_magnitude = (
                    sum(magnitudes) / len(magnitudes) if magnitudes else None
                )
        else:
            values = [
                leaf.prediction_value
                for leaf in leaves
                if leaf.prediction_value is not None
            ]
            if len(values) >= 2:
                mean = sum(values) / len(values)
                pred_variance = sum((v - mean) ** 2 for v in values) / len(values)

        return TreeMeta(
            index=index,
            depth=tree.max_depth,
            node_count=tree.node_count,
            n_leaves=n_leaves,
            avg_purity=avg_purity,
            pred_variance=pred_variance,
            leaf_magnitude=leaf_magnitude,
        )

    def _build_predict_json(
        self,
        trees: list[UnifiedTree],
        data,
        ensemble_type: EnsembleType = EnsembleType.SINGLE,
        boosting_meta: dict | None = None,
        target: list | None = None,
        is_classifier: bool = True,
        model_predictions: list | None = None,
    ) -> str:
        """Build compact JSON with tree structures + sample rows for client-side prediction."""
        import json
        import polars as pl

        # Embed up to 100 sample rows
        max_samples = 100
        if isinstance(data, pl.DataFrame):
            n_rows = data.height
            # Use the tree's feature names (what splits reference) not the user's column names
            tree_feature_names = trees[0].feature_names if trees else data.columns
            data_columns = data.columns
            rows = []
            for i in range(min(n_rows, max_samples)):
                row = data.row(i, named=True)
                # Map: tree_feature_name -> value from the positional column
                mapped = {}
                for col_idx, tree_fname in enumerate(tree_feature_names):
                    if col_idx < len(data_columns):
                        mapped[tree_fname] = float(row[data_columns[col_idx]])
                rows.append(mapped)
            feature_names = tree_feature_names
        else:
            return ""

        # Compact tree structures: only splits + leaf predictions
        def serialize_node(node):
            if node.is_leaf:
                n = {"t": "l"}
                if node.prediction_value is not None:
                    n["v"] = round(node.prediction_value, 6)
                if node.class_distribution is not None:
                    n["c"] = {
                        k: round(v, 4) for k, v in node.class_distribution.items()
                    }
                return n
            return {
                "t": "s",
                "f": node.feature_name,
                "th": round(node.threshold, 6),
                "op": node.comparison_op.value,
                "l": serialize_node(node.left_child),
                "r": serialize_node(node.right_child),
            }

        compact_trees = [serialize_node(tree.root) for tree in trees]

        # Aggregation method: "avg" for vote-based, "sum" for additive
        agg = (
            "avg"
            if ensemble_type == EnsembleType.VOTE_BASED
            or ensemble_type == EnsembleType.SINGLE
            else "sum"
        )

        payload = {
            "features": feature_names,
            "samples": rows,
            "n_rows": n_rows,
            "trees": compact_trees,
            "is_classifier": is_classifier,
            "aggregation": agg,
        }
        if boosting_meta:
            payload["boosting"] = boosting_meta
        if target is not None:
            # Embed targets for the same rows we embedded
            max_idx = min(n_rows, max_samples)
            payload["targets"] = [
                target[i] if i < len(target) else None for i in range(max_idx)
            ]
        if model_predictions is not None:
            payload["predictions"] = model_predictions[:max_samples]
        return json.dumps(payload)

    def _build_trees_json(self, trees: list[UnifiedTree]) -> str:
        """Build compact JSON of all tree structures for the detail modal."""
        import json

        def serialize_node(node):
            if node.is_leaf:
                n = {"t": "l"}
                if node.prediction_value is not None:
                    n["v"] = round(node.prediction_value, 6)
                if node.class_distribution is not None:
                    n["c"] = {
                        k: round(v, 4) for k, v in node.class_distribution.items()
                    }
                return n
            return {
                "t": "s",
                "f": node.feature_name,
                "th": round(node.threshold, 6),
                "op": node.comparison_op.value,
                "l": serialize_node(node.left_child),
                "r": serialize_node(node.right_child),
            }

        return json.dumps([serialize_node(tree.root) for tree in trees])

    def _wrap_html(
        self,
        svg: str,
        ensemble_type: EnsembleType,
        is_classifier: bool,
        total: int,
        predict_json: str = "",
        trees_json: str = "",
        model_name: str = "Unknown Model",
    ) -> str:
        if not is_classifier:
            metric_name = "variance"
            metric_label = "Pred Variance"
        elif ensemble_type == EnsembleType.VOTE_BASED:
            metric_name = "purity"
            metric_label = "Leaf Purity"
        else:
            metric_name = "magnitude"
            metric_label = "Leaf Magnitude"

        page_size = min(total, MAX_VISIBLE)

        parts = [
            "<!DOCTYPE html>\n",
            '<html lang="en">\n',
            "<head>\n",
            "<title>PrettyForest — Aesthetic Forest View</title>\n",
            '<meta charset="utf-8"/>\n',
            f"<style>{_get_static_css()}</style>\n",
            "</head>\n",
            "<body>\n",
            '<div class="header">\n',
            f"<h2>PrettyForest — {model_name}</h2>\n",
            '<div class="zoom-controls">\n',
            '<button id="dark-toggle" title="Toggle dark mode">🌙</button>\n',
            '<button id="info-btn" title="How it works / Model Info">❓</button>\n',
            '<select id="season-toggle" class="tool-select" title="Season theme">\n',
            '<option value="">🌳 Natural</option>\n',
            '<option value="spring">🌸 Spring</option>\n',
            '<option value="summer">🌿 Summer</option>\n',
            '<option value="autumn">🍂 Autumn</option>\n',
            '<option value="winter">❄️ Winter</option>\n',
            "</select>\n",
            '<button id="zoom-in" title="Zoom In">+</button>\n',
            '<button id="zoom-reset" title="Reset Zoom">⟳</button>\n',
            '<button id="zoom-out" title="Zoom Out">−</button>\n',
            '<span id="zoom-level">Zoom: 100%</span>\n',
            "</div>\n",
            "</div>\n",
            '<div class="toolbar">\n',
            '<div class="tool-group">\n',
            "<label>Sort by</label>\n",
            '<select id="sort-by">\n',
            '<option value="natural">Natural</option>\n',
            '<option value="depth">Depth</option>\n',
            '<option value="nodes">Nodes</option>\n',
            '<option value="leaves">Leaves</option>\n',
            f'<option value="metric">{metric_label}</option>\n',
            "</select>\n",
            "</div>\n",
            '<div class="tool-group">\n',
            "<label>Showing</label>\n",
            '<button id="page-prev" class="tool-btn" title="Previous page">◀</button>\n',
            f'<span id="page-info">1–{page_size} of {total}</span>\n',
            '<button id="page-next" class="tool-btn" title="Next page">▶</button>\n',
            "</div>\n",
            '<div class="tool-group">\n',
            '<button id="reset-all" class="tool-btn">⟳ Reset</button>\n',
            "</div>\n",
            "</div>\n",
            # Model info panel (toggled by ? button)
            '<div class="model-info-panel" id="model-info-panel">\n',
            '<button id="info-panel-close">✕</button>\n',
            f"<strong>{model_name}</strong>\n",
            '<div id="model-description" class="model-desc"></div>\n',
            "</div>\n",
        ]

        if predict_json:
            parts.extend(
                [
                    '<div class="predict-panel" id="predict-panel">\n',
                    '<div class="predict-header">\n',
                    "<strong>🔍 Predict</strong>\n",
                    '<button id="predict-close" title="Close">✕</button>\n',
                    "</div>\n",
                    '<div class="predict-body">\n',
                    "<label>Sample row:</label>\n",
                    '<input type="number" id="predict-row" min="0" value="0" style="width:60px"/>\n',
                    '<button id="predict-go" class="tool-btn">Trace</button>\n',
                    '<button id="predict-clear" class="tool-btn">Clear</button>\n',
                    '<span id="predict-result"></span>\n',
                    "</div>\n",
                    "</div>\n",
                ]
            )

        # Sample data display (inline bar, shown when a sample is traced)
        parts.append('<div class="sample-display" id="sample-display"></div>\n')

        parts.extend(
            [
                '<div class="forest-container" id="forest-container">\n',
                f"{svg}\n",
                "</div>\n",
                '<div class="tooltip" id="tooltip"></div>\n',
            ]
        )

        if predict_json:
            parts.append(
                f'<script id="predict-data" type="application/json">{predict_json}</script>\n'
            )

        # Always embed tree structures for detail modal
        parts.append(
            f'<script id="trees-data" type="application/json">{trees_json}</script>\n'
        )

        # Detail modal (rendered on the fly via JS)
        parts.append('<div class="detail-modal" id="detail-modal">\n')
        parts.append(
            '<div class="detail-header"><span id="detail-title">Tree #0</span><button id="detail-close">✕</button></div>\n'
        )
        parts.append('<div class="detail-note" id="detail-note"></div>\n')
        parts.append('<div class="detail-sample" id="detail-sample"></div>\n')
        parts.append('<div class="detail-body" id="detail-body"></div>\n')
        parts.append("</div>\n")

        config_data = {
            "METRIC_KEY": metric_name,
            "METRIC_LABEL": metric_label,
            "TOTAL": total,
            "PAGE_SIZE": page_size,
            "HAS_PREDICT": bool(predict_json),
            "IS_BOOSTED": ensemble_type == EnsembleType.ADDITIVE,
            "MODEL_NAME": model_name,
        }
        parts.extend(
            [
                f'<script id="forest-config" type="application/json">{json.dumps(config_data)}</script>\n',
                f"<script>{_get_static_js()}\ninitForest(document, {json.dumps(config_data)});</script>\n",
                "</body>\n",
                "</html>",
            ]
        )

        return "".join(parts)
