"""ForestRenderer — orchestrates the aesthetic forest rendering pipeline."""

from __future__ import annotations

import random

from prettyforest.models import EnsembleType, UnifiedTree
from prettyforest.rendering.forest.models import ForestConfig
from prettyforest.rendering.forest.scene_composer import (
    MAX_VISIBLE,
    SceneComposer,
    TreeMeta,
)
from prettyforest.rendering.forest.tree_shape_generator import TreeShapeGenerator
from prettyforest.rendering.forest.visual_mapper import VisualMapper


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
            f"<style>{_CSS}</style>\n",
            "</head>\n",
            "<body>\n",
            '<div class="header">\n',
            f"<h2>PrettyForest — {model_name}</h2>\n",
            '<div class="zoom-controls">\n',
            '<button id="dark-toggle" title="Toggle dark mode">🌙</button>\n',
            '<button id="info-btn" title="How it works / Model Info">❓</button>\n',
            '<select id="season-toggle" title="Season theme" style="padding:3px 6px;border:1px solid #ccc;border-radius:4px;font-size:11px;cursor:pointer">\n',
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
            '<div class="spotlight-panel" id="spotlight-panel">\n',
            '<button id="spotlight-close">✕</button>\n',
            '<div id="spotlight-content"></div>\n',
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

        has_predict = "true" if predict_json else "false"
        is_boosted = "true" if ensemble_type == EnsembleType.ADDITIVE else "false"
        parts.extend(
            [
                f'<script>var METRIC_KEY="{metric_name}",METRIC_LABEL="{metric_label}",TOTAL={total},PAGE_SIZE={page_size},HAS_PREDICT={has_predict},IS_BOOSTED={is_boosted},MODEL_NAME="{model_name}";</script>\n',
                f"<script>{_JS}</script>\n",
                "</body>\n",
                "</html>",
            ]
        )

        return "".join(parts)


_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f2f6f2; height: 100vh; overflow: hidden; display: flex; flex-direction: column; color: #2e3d30; }

/* Glassmorphism base classes */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    z-index: 100;
}
.header h2 { font-size: 18px; font-weight: 600; color: #1c281e; letter-spacing: -0.02em; }

.zoom-controls { display: flex; align-items: center; gap: 8px; }
.zoom-controls button {
    width: 32px;
    height: 32px;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.8);
    font-size: 15px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.zoom-controls button:hover {
    background: #e8f5e9;
    border-color: #4caf50;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.15);
}
.zoom-controls button:active {
    transform: translateY(0);
}
#zoom-level { font-size: 12px; color: #667568; min-width: 65px; font-weight: 500; text-align: center; }

.toolbar {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 8px 24px;
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    flex-wrap: wrap;
    z-index: 90;
}
.tool-group { display: flex; align-items: center; gap: 8px; }
.tool-group label { font-size: 12px; color: #556256; font-weight: 600; white-space: nowrap; text-transform: uppercase; letter-spacing: 0.05em; }
.tool-group select {
    padding: 4px 10px;
    border: 1px solid rgba(0, 0, 0, 0.15);
    border-radius: 6px;
    font-size: 12px;
    background: rgba(255, 255, 255, 0.9);
    font-family: inherit;
    font-weight: 500;
    outline: none;
    cursor: pointer;
    transition: border-color 0.2s;
}
.tool-group select:focus {
    border-color: #4caf50;
}
#page-info { font-size: 12px; font-weight: 600; color: #2e3d30; min-width: 95px; text-align: center; }

.tool-btn {
    padding: 4px 12px;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 6px;
    font-size: 12px;
    font-family: inherit;
    font-weight: 500;
    background: rgba(255, 255, 255, 0.85);
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.tool-btn:hover {
    background: #e8f5e9;
    border-color: #4caf50;
    transform: translateY(-1px);
}
.tool-btn:active {
    transform: translateY(0);
}
.tool-btn.active { background: #c8e6c9; border-color: #4caf50; color: #1b5e20; }
.tool-btn:disabled { opacity: 0.4; cursor: default; transform: none !important; box-shadow: none !important; }

.spotlight-panel {
    position: fixed;
    top: 120px;
    right: 20px;
    width: 280px;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
    padding: 20px;
    display: none;
    z-index: 500;
    border: 1px solid rgba(255, 255, 255, 0.5);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.spotlight-panel.visible { display: block; }
#spotlight-close {
    position: absolute;
    top: 12px;
    right: 14px;
    border: none;
    background: none;
    font-size: 18px;
    cursor: pointer;
    color: #888;
    transition: color 0.2s;
}
#spotlight-close:hover { color: #1c281e; }
#spotlight-content { font-size: 13px; line-height: 1.8; }
#spotlight-content strong { font-size: 15px; font-weight: 700; display: block; margin-bottom: 8px; color: #1c281e; }
.stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.04); }
.stat-label { color: #556256; font-weight: 400; }
.stat-value { font-weight: 600; color: #1c281e; }

.forest-container { width: 100%; flex: 1; min-height: 0; overflow: hidden; position: relative; }
.forest-container svg { display: block; width: 100%; height: 100%; cursor: grab; transition: transform 0.15s ease; }
.forest-container svg:active { cursor: grabbing; }

.tooltip {
    position: fixed;
    display: none;
    background: rgba(20, 28, 21, 0.88);
    backdrop-filter: blur(8px);
    color: #f5fcf6;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 13px;
    line-height: 1.7;
    pointer-events: none;
    z-index: 1000;
    max-width: 260px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.visual-tree { cursor: pointer; transition: opacity 0.3s, filter 0.3s, transform 0.6s ease; }
.visual-tree:hover { filter: brightness(1.08) drop-shadow(0 4px 12px rgba(0,0,0,0.15)); }
.visual-tree.hidden { display: none; }
.visual-tree.highlighted { filter: drop-shadow(0 0 8px #ffeb3b) drop-shadow(0 0 16px rgba(255,235,59,0.6)); }
.visual-tree.spotlit { filter: drop-shadow(0 0 10px #42a5f5) drop-shadow(0 0 20px rgba(66,165,245,0.5)); }
.visual-tree { opacity: 0; }
.visual-tree.grown { opacity: 1; }
.visual-tree .trunk { transform: scaleY(0); transform-origin: bottom center; }
.visual-tree.grow-trunk .trunk { transform: scaleY(1); transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1); }
.visual-tree .branch { opacity: 0; stroke-dasharray: 100; stroke-dashoffset: 100; }
.visual-tree.grow-branches .branch { opacity: 1; stroke-dashoffset: 0; transition: stroke-dashoffset 0.5s ease, opacity 0.2s ease; }
.visual-tree .canopy { opacity: 0; transform: scale(0); transform-origin: center 40%; }
.visual-tree.grow-canopy .canopy { opacity: 0.9; transform: scale(1); transition: transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.3s ease; }

.predict-panel {
    padding: 10px 24px;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    z-index: 80;
}
.predict-header { display: flex; align-items: center; gap: 8px; }
.predict-header strong { font-size: 13px; color: #1c281e; }
#predict-close { border: none; background: none; font-size: 16px; cursor: pointer; color: #888; transition: color 0.2s; }
#predict-close:hover { color: #1c281e; }
.predict-body { display: flex; align-items: center; gap: 10px; flex-wrap: nowrap; }
.predict-body label { font-size: 12px; color: #556256; font-weight: 500; }
.predict-body input {
    padding: 4px 8px;
    border: 1px solid rgba(0, 0, 0, 0.15);
    border-radius: 6px;
    font-size: 12px;
    font-family: inherit;
    background: rgba(255, 255, 255, 0.9);
    outline: none;
    transition: border-color 0.2s;
}
.predict-body input:focus { border-color: #4caf50; }
#predict-result { font-size: 13px; color: #1c281e; margin-left: 12px; white-space: nowrap; }
#predict-result strong { color: #1565c0; font-weight: 700; }
.predict-badge { font-size: 9px; font-weight: bold; fill: white; }
.predict-badge-bg { rx: 3; }

.detail-modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(240, 244, 240, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    z-index: 2000;
    flex-direction: column;
}
.detail-modal.open { display: flex; }
.detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(200, 230, 201, 0.5);
    background: rgba(232, 245, 233, 0.5);
}
.detail-header span { font-size: 16px; font-weight: 600; color: #2e7d32; }
#detail-close { border: none; background: none; font-size: 26px; cursor: pointer; color: #888; padding: 4px 8px; transition: color 0.2s; }
#detail-close:hover { color: #1c281e; }

.detail-body {
    flex: 1;
    overflow: hidden;
    padding: 24px;
    display: flex;
    align-items: flex-start;
    cursor: grab;
    background: radial-gradient(ellipse at center, rgba(249, 253, 249, 0.5) 0%, rgba(238, 245, 238, 0.5) 100%);
}
.detail-body svg { height: auto; cursor: grab; flex-shrink: 0; }
.detail-body svg:active { cursor: grabbing; }
.detail-body .node-rect { fill: #fffde7; stroke: #5D4037; stroke-width: 2; rx: 8; }
.detail-body .node-rect.leaf { fill: #e8f5e9; stroke: #2e7d32; stroke-width: 2; }
.detail-body .node-rect.on-path { fill: #e3f2fd; stroke: #1565c0; stroke-width: 3; }
.detail-body .node-rect.leaf.on-path { fill: #bbdefb; stroke: #1565c0; stroke-width: 3; }
.detail-body .node-rect.dimmed { opacity: 0.2; }
.detail-body .edge-line { stroke: #8d6e63; stroke-width: 2.5; stroke-linecap: round; }
.detail-body .edge-line.on-path { stroke: #1565c0; stroke-width: 3.5; }
.detail-body .edge-line.dimmed { opacity: 0.15; }
.detail-body .node-text { font-size: 11px; fill: #333; text-anchor: middle; font-family: inherit; pointer-events: none; }
.detail-body .node-text.dimmed { opacity: 0.2; }
.detail-body .node-text.sample-val { font-size: 9px; fill: #1565c0; font-style: italic; font-weight: 500; }
.detail-body .edge-label { font-size: 11px; fill: #5D4037; text-anchor: middle; font-weight: 600; }
.detail-body .edge-label.dimmed { opacity: 0.15; }

.sample-display {
    display: none;
    padding: 8px 24px;
    background: rgba(245,248,245,0.7);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    overflow-x: auto;
    white-space: nowrap;
}
.sample-display.visible { display: block; }
.sample-display .sample-title { font-size: 12px; font-weight: 600; color: #1c281e; margin-bottom: 6px; display: block; }
.sample-display .sample-chips { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; }
.sample-display .chip { background: rgba(255, 255, 255, 0.8); border: 1px solid rgba(0, 0, 0, 0.08); border-radius: 6px; padding: 4px 10px; font-size: 11px; white-space: nowrap; flex-shrink: 0; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
.sample-display .chip .chip-name { color: #556256; }
.sample-display .chip .chip-val { color: #1565c0; font-weight: 600; margin-left: 4px; }

.detail-sample {
    padding: 8px 24px;
    background: rgba(232, 245, 233, 0.4);
    border-bottom: 1px solid rgba(200, 230, 201, 0.4);
    overflow-x: auto;
    white-space: nowrap;
    display: none;
}
.detail-sample.visible { display: block; }
.detail-sample .sample-chips { display: flex; gap: 8px; }
.detail-sample .chip { background: #fff; border: 1px solid rgba(200, 230, 201, 0.6); border-radius: 6px; padding: 4px 10px; font-size: 11px; white-space: nowrap; flex-shrink: 0; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
.detail-sample .chip .chip-name { color: #2e7d32; }
.detail-sample .chip .chip-val { color: #1565c0; font-weight: 600; margin-left: 4px; }

.detail-note { padding: 6px 24px; font-size: 12px; color: #e65100; background: rgba(255, 248, 225, 0.6); backdrop-filter: blur(8px); border-bottom: 1px solid rgba(255, 224, 130, 0.5); display: none; }
.detail-note.visible { display: block; }

.model-info-panel {
    position: fixed;
    top: 80px;
    left: 50%;
    transform: translateX(-50%);
    width: 520px;
    max-width: 90vw;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 16px;
    box-shadow: 0 12px 50px rgba(0, 0, 0, 0.1);
    padding: 24px;
    display: none;
    z-index: 600;
    border: 1px solid rgba(255, 255, 255, 0.5);
    transition: all 0.3s ease;
}
.model-info-panel.visible { display: block; }
#info-panel-close { position: absolute; top: 12px; right: 16px; border: none; background: none; font-size: 18px; cursor: pointer; color: #888; transition: color 0.2s; }
#info-panel-close:hover { color: #1c281e; }
.model-info-panel strong { font-size: 16px; color: #2e7d32; display: block; margin-bottom: 12px; font-weight: 600; }
.model-desc { font-size: 13px; line-height: 1.8; color: #2e3d30; }
.model-desc p { margin: 8px 0; }
.model-desc .key { font-weight: 600; color: #1c281e; }

/* Dark mode */
body.dark { background: #0b0b14; color: #cfd8d0; }
body.dark .header {
    background: rgba(15, 15, 28, 0.75);
    border-color: rgba(255, 255, 255, 0.06);
}
body.dark .header h2 { color: #ecf3ed; }

body.dark .zoom-controls button {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
    border-color: rgba(255, 255, 255, 0.08);
}
body.dark .zoom-controls button:hover {
    background: rgba(76, 175, 80, 0.2);
    border-color: #66bb6a;
}
body.dark #zoom-level { color: #9bb09e; }

body.dark .toolbar {
    background: rgba(15, 15, 28, 0.65);
    border-color: rgba(255, 255, 255, 0.05);
}
body.dark .toolbar label { color: #9bb09e; }
body.dark .toolbar select {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
    border-color: rgba(255, 255, 255, 0.08);
}
body.dark .toolbar select:focus {
    border-color: #66bb6a;
}
body.dark #page-info { color: #ecf3ed; }

body.dark .tool-btn {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
    border-color: rgba(255, 255, 255, 0.08);
}
body.dark .tool-btn:hover {
    background: rgba(76, 175, 80, 0.2);
    border-color: #66bb6a;
}
body.dark .tool-btn.active {
    background: rgba(76, 175, 80, 0.3);
    border-color: #66bb6a;
    color: #a5d6a7;
}

body.dark .predict-panel {
    background: rgba(15, 15, 28, 0.7);
    border-color: rgba(255, 255, 255, 0.05);
}
body.dark .predict-panel label { color: #9bb09e; }
body.dark .predict-panel input {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
    border-color: rgba(255, 255, 255, 0.08);
}
body.dark .predict-panel input:focus { border-color: #66bb6a; }
body.dark #predict-result { color: #ecf3ed; }
body.dark #predict-result strong { color: #64b5f6; }

body.dark .sample-display {
    background: rgba(15, 15, 28, 0.75);
    border-color: rgba(255, 255, 255, 0.05);
}
body.dark .sample-display .sample-title { color: #ecf3ed; }
body.dark .sample-display .chip {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.08);
}
body.dark .sample-display .chip .chip-name { color: #9bb09e; }
body.dark .sample-display .chip .chip-val { color: #64b5f6; }

body.dark .tooltip {
    background: rgba(15, 15, 28, 0.9);
    border-color: rgba(255, 255, 255, 0.12);
    color: #ecf3ed;
}

body.dark .spotlight-panel {
    background: rgba(15, 15, 28, 0.8);
    border-color: rgba(255, 255, 255, 0.08);
    color: #cfd8d0;
    box-shadow: 0 12px 50px rgba(0, 0, 0, 0.25);
}
body.dark #spotlight-close:hover { color: #ecf3ed; }
body.dark #spotlight-content strong { color: #ecf3ed; }
body.dark .stat-row { border-color: rgba(255, 255, 255, 0.05); }
body.dark .stat-label { color: #9bb09e; }
body.dark .stat-value { color: #ecf3ed; }

body.dark .detail-modal {
    background: rgba(10, 10, 18, 0.88);
}
body.dark .detail-header {
    background: rgba(15, 30, 20, 0.5);
    border-color: rgba(255, 255, 255, 0.05);
}
body.dark .detail-header span { color: #81c784; }
body.dark #detail-close:hover { color: #ecf3ed; }

body.dark .detail-body {
    background: radial-gradient(ellipse at center, rgba(15, 25, 15, 0.5) 0%, rgba(10, 10, 15, 0.5) 100%);
}
body.dark .detail-sample {
    background: rgba(15, 30, 20, 0.4);
    border-color: rgba(255, 255, 255, 0.05);
}
body.dark .detail-sample .chip {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.08);
}
body.dark .detail-sample .chip .chip-name { color: #81c784; }
body.dark .detail-sample .chip .chip-val { color: #64b5f6; }

body.dark .detail-note {
    color: #ffb74d;
    background: rgba(255, 183, 77, 0.08);
    border-color: rgba(255, 183, 77, 0.15);
}

body.dark .model-info-panel {
    background: rgba(15, 15, 28, 0.85);
    border-color: rgba(255, 255, 255, 0.08);
    color: #cfd8d0;
    box-shadow: 0 15px 60px rgba(0, 0, 0, 0.3);
}
body.dark #info-panel-close:hover { color: #ecf3ed; }
body.dark .model-info-panel strong { color: #81c784; }
body.dark .model-desc { color: #cfd8d0; }
body.dark .model-desc .key { color: #ecf3ed; }
"""

_JS = r"""
(function(ROOT) {
  var $ = function(id) { return ROOT.querySelector('#' + id); };
  var svg = $('forest-svg');
  var container = $('forest-container');
  var tooltip = $('tooltip');
  var sortBy = $('sort-by');
  var pagePrev = $('page-prev');
  var pageNext = $('page-next');
  var pageInfo = $('page-info');
  var resetAll = $('reset-all');
  var spotlightPanel = $('spotlight-panel');
  var spotlightClose = $('spotlight-close');
  var spotlightContent = $('spotlight-content');
  var zoomIn = $('zoom-in');
  var zoomOut = $('zoom-out');
  var zoomReset = $('zoom-reset');
  var zoomLabel = $('zoom-level');
  if (!svg || !container) return;

  // --- Collect all tree elements and their data ---
  var traceActive = false; // set to true when user clicks Trace, false on Clear
  var allTrees = Array.prototype.slice.call(svg.querySelectorAll('.visual-tree'));
  var treeData = allTrees.map(function(el) {
    return {
      el: el,
      idx: parseInt(el.getAttribute('data-tree-idx')) || 0,
      depth: parseInt(el.getAttribute('data-depth')) || 0,
      nodes: parseInt(el.getAttribute('data-nodes')) || 0,
      leaves: parseInt(el.getAttribute('data-leaves')) || 0,
      purity: parseFloat(el.getAttribute('data-purity')) || null,
      magnitude: parseFloat(el.getAttribute('data-magnitude')) || null,
      variance: parseFloat(el.getAttribute('data-variance')) || null,
      origTransform: el.getAttribute('transform') || ''
    };
  });

  function metric(d) {
    if (METRIC_KEY === 'purity') return d.purity;
    if (METRIC_KEY === 'magnitude') return d.magnitude;
    if (METRIC_KEY === 'variance') return d.variance;
    return null;
  }

  // --- Paging state ---
  var currentPage = 0;
  var sortedData = treeData.slice(); // full sorted list
  var totalPages = Math.ceil(TOTAL / PAGE_SIZE);

  // Pre-compute position slots from initially visible trees (sorted by DOM order = back to front)
  var positions = [];
  (function() {
    var visible = allTrees.filter(function(t) { return !t.classList.contains('hidden'); });
    for (var i = 0; i < visible.length; i++) {
      positions.push(visible[i].getAttribute('transform') || '');
    }
  })();

  function showPage() {
    var start = currentPage * PAGE_SIZE;
    var end = Math.min(start + PAGE_SIZE, sortedData.length);
    var count = end - start;

    // Hide all trees
    treeData.forEach(function(d) {
      d.el.classList.add('hidden');
      d.el.classList.remove('grown','grow-trunk','grow-branches','grow-canopy','highlighted','spotlit');
      d.el.style.opacity = '';
    });

    // Show current page — assign position slots (cycling if needed)
    for (var i = 0; i < count; i++) {
      var d = sortedData[start + i];
      var posIdx = i % positions.length;
      d.el.setAttribute('transform', positions[posIdx]);
      d.el.classList.remove('hidden');
      d.el.style.opacity = '1';
      d.el.classList.add('grown', 'grow-trunk', 'grow-branches', 'grow-canopy');
    }

    pageInfo.textContent = (start + 1) + '–' + end + ' of ' + sortedData.length;
    pagePrev.disabled = (currentPage === 0);
    pageNext.disabled = (end >= sortedData.length);
  }

  // Initial page
  showPage();

  pagePrev.addEventListener('click', function() {
    if (currentPage > 0) { currentPage--; showPage(); }
  });
  pageNext.addEventListener('click', function() {
    if ((currentPage + 1) * PAGE_SIZE < sortedData.length) { currentPage++; showPage(); }
  });

  // --- Zoom & Pan ---
  var scale = 1, tx = 0, ty = 0, dragging = false, sx = 0, sy = 0;
  function applyZoom() {
    svg.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    svg.style.transformOrigin = 'center center';
    zoomLabel.textContent = 'Zoom: ' + Math.round(scale * 100) + '%';
  }
  zoomIn.addEventListener('click', function() { scale = Math.min(scale * 1.25, 5); applyZoom(); });
  zoomOut.addEventListener('click', function() { scale = Math.max(scale / 1.25, 0.2); applyZoom(); });
  zoomReset.addEventListener('click', function() { scale = 1; tx = 0; ty = 0; applyZoom(); });

  // Dark mode toggle
  var darkBtn = $('dark-toggle');
  var themeRoot = ROOT === document ? document.body : ROOT;
  if (darkBtn) {
    darkBtn.addEventListener('click', function() {
      themeRoot.classList.toggle('dark');
      darkBtn.textContent = themeRoot.classList.contains('dark') ? '☀️' : '🌙';
    });
  }

  // Season toggle — recolors canopies and ground live
  var seasonSelect = $('season-toggle');
  if (seasonSelect) {
    var seasonPalettes = {
      spring: { canopy: ['#90EE90','#98FB98','#FFB7C5','#FF69B4','#DDA0DD','#87CEAB'], ground: '#a8d8a0', sky: '#f1fff1' },
      summer: { canopy: ['#2E8B57','#3CB371','#6B8E23','#228B22','#32CD32'], ground: '#8cc97a', sky: '#dceef5' },
      autumn: { canopy: ['#D2691E','#B22222','#DAA520','#CD853F','#FF8C00'], ground: '#8B6914', sky: '#fff3e0' },
      winter: { canopy: [], ground: '#B0C4DE', sky: '#e3f2fd', bare: true }
    };

    // Store all originals immediately
    var groundStops = svg.querySelectorAll('#ground-gradient stop');
    var skyStops = svg.querySelectorAll('#sky-gradient stop');
    var patches = svg.querySelectorAll('ellipse[data-patch]');
    var origGround = []; groundStops.forEach(function(s) { origGround.push(s.getAttribute('stop-color')); });
    var origSky = []; skyStops.forEach(function(s) { origSky.push(s.getAttribute('stop-color')); });
    var origPatches = []; patches.forEach(function(e) { origPatches.push(e.getAttribute('fill')); });

    seasonSelect.addEventListener('change', function() {
      var season = this.value;
      var trees = svg.querySelectorAll('.visual-tree');

      if (!season) {
        // Restore everything
        trees.forEach(function(t) {
          var canopy = t.querySelector('.canopy');
          if (canopy && canopy.dataset.origFill) { canopy.setAttribute('fill', canopy.dataset.origFill); canopy.setAttribute('stroke', canopy.dataset.origStroke || canopy.dataset.origFill); }
          if (canopy) canopy.style.display = '';
        });
        groundStops.forEach(function(s, i) { if (origGround[i]) s.setAttribute('stop-color', origGround[i]); });
        skyStops.forEach(function(s, i) { if (origSky[i]) s.setAttribute('stop-color', origSky[i]); });
        patches.forEach(function(e, i) { if (origPatches[i]) e.setAttribute('fill', origPatches[i]); });
        return;
      }

      var pal = seasonPalettes[season];
      if (!pal) return;

      // Canopies
      trees.forEach(function(t) {
        var canopy = t.querySelector('.canopy');
        if (!canopy) return;
        if (!canopy.dataset.origFill) { canopy.dataset.origFill = canopy.getAttribute('fill'); canopy.dataset.origStroke = canopy.getAttribute('stroke'); }
        if (pal.bare) { canopy.style.display = 'none'; }
        else { canopy.style.display = ''; var c = pal.canopy[Math.floor(Math.random()*pal.canopy.length)]; canopy.setAttribute('fill', c); canopy.setAttribute('stroke', c); }
      });

      // Ground
      if (groundStops.length >= 2) { groundStops[0].setAttribute('stop-color', pal.ground); groundStops[1].setAttribute('stop-color', pal.ground); }

      // Sky
      if (skyStops.length >= 1) { skyStops[0].setAttribute('stop-color', pal.sky); }

      // Patches
      patches.forEach(function(e) { e.setAttribute('fill', pal.ground); });
    });
  }

  container.addEventListener('wheel', function(e) {
    e.preventDefault();
    scale = Math.max(0.2, Math.min(5, scale * (e.deltaY > 0 ? 0.9 : 1.1)));
    applyZoom();
  }, {passive: false});
  container.addEventListener('mousedown', function(e) {
    if (e.target !== svg && !svg.contains(e.target)) return;
    if (e.button !== 0) return; dragging = true; sx = e.clientX - tx; sy = e.clientY - ty; svg.style.cursor = 'grabbing';
  });
  document.addEventListener('mousemove', function(e) {
    if (!dragging) return; tx = e.clientX - sx; ty = e.clientY - sy; svg.style.transition = 'none'; applyZoom();
  });
  document.addEventListener('mouseup', function() { dragging = false; svg.style.cursor = 'grab'; svg.style.transition = 'transform 0.15s ease'; });
  document.addEventListener('keydown', function(e) {
    switch(e.key) {
      case 'ArrowLeft': tx += 40; break; case 'ArrowRight': tx -= 40; break;
      case 'ArrowUp': ty += 40; break; case 'ArrowDown': ty -= 40; break;
      case '+': case '=': scale = Math.min(scale * 1.15, 5); break;
      case '-': scale = Math.max(scale / 1.15, 0.2); break;
      case 'Escape': closeSpotlight(); return;
      default: return;
    }
    e.preventDefault(); applyZoom();
  });

  // --- Tooltip ---
  function findTree(el) {
    while (el && el !== svg) {
      if (el.getAttribute && (el.getAttribute('class') || '').indexOf('visual-tree') !== -1) return el;
      el = el.parentNode;
    }
    return null;
  }
  svg.addEventListener('mouseover', function(e) {
    var tree = findTree(e.target);
    if (!tree) { tooltip.style.display = 'none'; return; }
    var d = treeData.find(function(t) { return t.el === tree; });
    if (!d) return;
    var h = '<strong>Tree #' + d.idx + '</strong><br>';
    h += 'Depth: ' + d.depth + '<br>Nodes: ' + d.nodes + '<br>Leaves: ' + d.leaves + '<br>';
    if (d.purity !== null) h += 'Purity: ' + (d.purity*100).toFixed(1) + '%<br>';
    if (d.magnitude !== null) h += 'Magnitude: ' + d.magnitude.toFixed(4) + '<br>';
    if (d.variance !== null) h += 'Variance: ' + d.variance.toFixed(2) + '<br>';
    tooltip.innerHTML = h; tooltip.style.display = 'block';
  });
  svg.addEventListener('mousemove', function(e) {
    if (tooltip.style.display === 'block') { tooltip.style.left=(e.clientX+14)+'px'; tooltip.style.top=(e.clientY+14)+'px'; }
  });
  svg.addEventListener('mouseout', function(e) { if (!findTree(e.target)) tooltip.style.display='none'; });
  svg.addEventListener('mouseleave', function() { tooltip.style.display='none'; });

  // --- Sort (applies globally, resets to page 0) ---
  sortBy.addEventListener('change', function() {
    var mode = this.value;
    if (mode === 'natural') {
      sortedData = treeData.slice();
    } else {
      sortedData = treeData.slice().sort(function(a, b) {
        if (mode === 'depth') return b.depth - a.depth;
        if (mode === 'nodes') return b.nodes - a.nodes;
        if (mode === 'leaves') return b.leaves - a.leaves;
        if (mode === 'metric') return (metric(b)||0) - (metric(a)||0);
        return 0;
      });
    }
    currentPage = 0;
    totalPages = Math.ceil(sortedData.length / PAGE_SIZE);
    showPage();
  });

  // --- Reset ---
  resetAll.addEventListener('click', function() {
    closeSpotlight();
    sortBy.value = 'natural';
    sortedData = treeData.slice();
    currentPage = 0;
    totalPages = Math.ceil(sortedData.length / PAGE_SIZE);
    treeData.forEach(function(d) {
      d.el.classList.remove('hidden', 'highlighted', 'spotlit');
      d.el.setAttribute('transform', d.origTransform);
    });
    showPage();
  });

  // --- Click to spotlight ---
  var spotlitEl = null;
  svg.addEventListener('click', function(e) {
    var tree = findTree(e.target);
    if (!tree) { closeSpotlight(); return; }
    if (spotlitEl === tree) { closeSpotlight(); return; }
    if (spotlitEl) spotlitEl.classList.remove('spotlit');
    spotlitEl = tree;
    tree.classList.add('spotlit');

    var d = treeData.find(function(t) { return t.el === tree; });
    if (!d) return;
    var h = '<strong>Tree #' + d.idx + '</strong>';
    h += '<div class="stat-row"><span class="stat-label">Depth</span><span class="stat-value">' + d.depth + '</span></div>';
    h += '<div class="stat-row"><span class="stat-label">Nodes</span><span class="stat-value">' + d.nodes + '</span></div>';
    h += '<div class="stat-row"><span class="stat-label">Leaves</span><span class="stat-value">' + d.leaves + '</span></div>';
    if (d.purity !== null) h += '<div class="stat-row"><span class="stat-label">Purity</span><span class="stat-value">' + (d.purity*100).toFixed(1) + '%</span></div>';
    if (d.magnitude !== null) h += '<div class="stat-row"><span class="stat-label">Magnitude</span><span class="stat-value">' + d.magnitude.toFixed(4) + '</span></div>';
    if (d.variance !== null) h += '<div class="stat-row"><span class="stat-label">Variance</span><span class="stat-value">' + d.variance.toFixed(2) + '</span></div>';
    var ranked = treeData.slice().filter(function(t) { return metric(t) !== null; });
    ranked.sort(function(a, b) { return (metric(b)||0) - (metric(a)||0); });
    var rank = ranked.findIndex(function(t) { return t.el === tree; }) + 1;
    if (rank > 0) h += '<div class="stat-row" style="margin-top:4px"><span class="stat-label">' + METRIC_LABEL + ' rank</span><span class="stat-value">#' + rank + '/' + ranked.length + '</span></div>';
    spotlightContent.innerHTML = h;
    spotlightPanel.classList.add('visible');
  });
  spotlightClose.addEventListener('click', closeSpotlight);
  function closeSpotlight() {
    if (spotlitEl) spotlitEl.classList.remove('spotlit');
    spotlitEl = null; spotlightPanel.classList.remove('visible');
  }

  // --- Info button + Model description ---
  (function() {
    var infoBtn = $('info-btn');
    var svgInfoBtn = $('svg-info-btn');
    var infoPanel = $('model-info-panel');
    var infoClose = $('info-panel-close');
    var descEl = $('model-description');
    if (!infoPanel) return;

    var descriptions = {
      'Random Forest (Classification)': '<p><span class="key">How it works:</span> Trains multiple independent trees on random subsets of data (bagging). Each tree votes for a class. Final prediction = majority vote.</p><p><span class="key">Each tree:</span> Splits on original features with real class proportions in leaves. Fully interpretable individually.</p><p><span class="key">Reading tips:</span> Purity shows how cleanly each tree separates classes. High purity = the tree is confident in its leaves.</p>',
      'Random Forest (Regression)': '<p><span class="key">How it works:</span> Trains multiple independent trees on random subsets of data. Each tree predicts a value. Final prediction = average of all trees.</p><p><span class="key">Each tree:</span> Splits on original features with target means in leaves. Each leaf is a direct prediction.</p><p><span class="key">Reading tips:</span> Variance shows how spread out the leaf predictions are within each tree.</p>',
      'Gradient Boosting (Classification)': '<p><span class="key">How it works:</span> Trains trees sequentially. Each tree corrects the errors of the previous ensemble by fitting gradients (residuals).</p><p><span class="key">Each tree:</span> Splits on original features, but leaf values are small <em>gradient corrections</em>, not class predictions. A leaf value of +0.12 means "push the score slightly toward this class."</p><p><span class="key">Reading tips:</span> Final prediction = initial value + learning_rate × sum of all tree corrections. Individual tree leaf values are not standalone predictions.</p>',
      'Gradient Boosting (Regression)': '<p><span class="key">How it works:</span> Trains trees sequentially. Each tree predicts the residual (error) of the current ensemble.</p><p><span class="key">Each tree:</span> Splits on original features, leaf values are residual corrections. Final prediction = initial mean + lr × sum(leaf values).</p><p><span class="key">Reading tips:</span> Early trees make large corrections, later trees fine-tune. Magnitude decreases over iterations.</p>',
      'LightGBM (Classification)': '<p><span class="key">How it works:</span> Gradient boosting with histogram-based splits for speed. Trains trees on gradients sequentially.</p><p><span class="key">Each tree:</span> Splits on original features using histogram bins. Leaf values are log-odds corrections (not class probabilities).</p><p><span class="key">Reading tips:</span> Final prediction = sum of all leaf values, then softmax for probabilities. Individual leaves show gradient steps.</p>',
      'LightGBM (Regression)': '<p><span class="key">How it works:</span> Fast gradient boosting with histogram splits. Each tree predicts the residual error.</p><p><span class="key">Each tree:</span> Leaf values are additive corrections. Final prediction = sum of all tree leaf values.</p>',
      'CatBoost (Classification)': '<p><span class="key">How it works:</span> Ordered boosting with symmetric trees. Handles categorical features natively.</p><p><span class="key">Each tree:</span> Uses oblivious (symmetric) decision trees — same split at each depth level. Leaf values are gradient corrections.</p><p><span class="key">Reading tips:</span> Leaf values are small corrections. Final prediction comes from summing all trees. Exact reconstruction may differ slightly due to internal scaling.</p>',
      'CatBoost (Regression)': '<p><span class="key">How it works:</span> Ordered boosting with symmetric trees on residuals.</p><p><span class="key">Each tree:</span> Symmetric structure, leaf values are residual corrections summed for final prediction.</p>',
      'Decision Tree (Classification)': '<p><span class="key">How it works:</span> A single tree that recursively splits the data to separate classes.</p><p><span class="key">The tree:</span> Each split uses the feature that best separates classes (by Gini or entropy). Leaves show class proportions from training data.</p><p><span class="key">Reading tips:</span> The forest shows one tree. This IS the full model — no ensemble aggregation.</p>',
      'Decision Tree (Regression)': '<p><span class="key">How it works:</span> A single tree that recursively splits to minimize prediction error.</p><p><span class="key">The tree:</span> Leaves contain the mean target value of training samples that landed there. This is a direct prediction.</p>',
    };

    var desc = descriptions[MODEL_NAME] || '<p>Tree-based ensemble model.</p>';
    desc += '<hr style="margin:10px 0;border:none;border-top:1px solid rgba(0, 0, 0, 0.08)"><p style="font-size:11px;color:#666"><strong>Visual encoding:</strong> Tree height = depth, trunk width = node count, canopy color = green (pure/low variance) to amber (impure/high variance).</p>';
    if (descEl) descEl.innerHTML = desc;

    var toggleInfo = function(e) {
      e.stopPropagation();
      infoPanel.classList.toggle('visible');
    };

    if (infoBtn) infoBtn.addEventListener('click', toggleInfo);
    if (svgInfoBtn) svgInfoBtn.addEventListener('click', toggleInfo);
    if (infoClose) infoClose.addEventListener('click', function() { infoPanel.classList.remove('visible'); });
  })();

  // --- Double-click to open tree detail modal ---
  (function() {
    var modal = $('detail-modal');
    var modalBody = $('detail-body');
    var modalTitle = $('detail-title');
    var closeBtn = $('detail-close');
    var treesEl = $('trees-data');
    if (!modal || !treesEl) return;

    var allTreeStructures = JSON.parse(treesEl.textContent);
    var detailScale = 1, detailTx = 0, detailTy = 0;
    var detailSvg = null;

    function openDetail(treeIdx, depth, nodes) {
      var treeStruct = allTreeStructures[treeIdx];
      if (!treeStruct) return;

      modalTitle.textContent = 'Tree #' + treeIdx + ' — Depth: ' + depth + ', Nodes: ' + nodes;
      modalBody.innerHTML = '';
      detailScale = 1; detailTx = 0; detailTy = 0;

      // Show boosted model note
      var noteEl = $('detail-note');
      if (noteEl) {
        if (IS_BOOSTED) {
          noteEl.textContent = '\u26a0\ufe0f This is a boosted tree — leaf values are gradient corrections (residuals), not final predictions. The sample path and splits are on original features.';
          noteEl.classList.add('visible');
        } else {
          noteEl.classList.remove('visible');
        }
      }

      // Get traced sample if available
      var tracedSample = null;
      if (traceActive) {
        var predDataEl = $('predict-data');
        var rowInput = $('predict-row');
        if (predDataEl && rowInput) {
          try {
            var pd = JSON.parse(predDataEl.textContent);
            var idx = parseInt(rowInput.value);
            if (pd.samples && idx >= 0 && idx < pd.samples.length) {
              tracedSample = pd.samples[idx];
            }
          } catch(e) {}
        }
      }

      // Show sample data in detail header
      var detailSampleEl = $('detail-sample');
      if (detailSampleEl) {
        if (tracedSample) {
          var html = '<div class="sample-chips">';
          for (var key in tracedSample) {
            var val = tracedSample[key];
            var display = (typeof val === 'number') ? val.toFixed(3) : String(val);
            html += '<span class="chip"><span class="chip-name">' + key + ':</span><span class="chip-val">' + display + '</span></span>';
          }
          html += '</div>';
          detailSampleEl.innerHTML = html;
          detailSampleEl.classList.add('visible');
        } else {
          detailSampleEl.classList.remove('visible');
          detailSampleEl.innerHTML = '';
        }
      }

      detailSvg = renderTreeSVG(treeStruct, tracedSample);
      detailSvg.style.transition = 'none';
      detailSvg.style.transformOrigin = '0 0';
      modalBody.appendChild(detailSvg);
      modal.classList.add('open');
      // Center the SVG within the modal body
      requestAnimationFrame(function() {
        var bodyW = modalBody.clientWidth;
        var svgW = detailSvg.getBoundingClientRect().width / detailScale;
        detailTx = Math.max(0, (bodyW - svgW) / 2);
        detailTy = 0;
        detailSvg.style.transform = 'translate(' + detailTx + 'px,' + detailTy + 'px) scale(' + detailScale + ')';
        requestAnimationFrame(function() { detailSvg.style.transition = 'transform 0.15s ease'; });
      });
    }

    svg.addEventListener('dblclick', function(e) {
      var tree = findTree(e.target);
      if (!tree) return;
      var d = treeData.find(function(t) { return t.el === tree; });
      if (!d) return;
      openDetail(d.idx, d.depth, d.nodes);
    });

    // Also open from spotlight panel on double-click the panel title
    var spotlightEl = $('spotlight-content');
    if (spotlightEl) {
      spotlightEl.addEventListener('dblclick', function() {
        if (!spotlitEl) return;
        var d = treeData.find(function(t) { return t.el === spotlitEl; });
        if (d) openDetail(d.idx, d.depth, d.nodes);
      });
    }

    closeBtn.addEventListener('click', closeDetail);
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && modal.classList.contains('open')) { closeDetail(); e.stopPropagation(); }
    });
    modal.addEventListener('click', function(e) { if (e.target === modal) closeDetail(); });

    function closeDetail() { modal.classList.remove('open'); detailSvg = null; }

    // Zoom/pan in detail modal
    modalBody.addEventListener('wheel', function(e) {
      if (!detailSvg) return;
      e.preventDefault();
      detailScale = Math.max(0.3, Math.min(5, detailScale * (e.deltaY > 0 ? 0.9 : 1.1)));
      applyDetailTransform();
    }, {passive: false});

    var detailDrag = false, detailSx = 0, detailSy = 0;
    modalBody.addEventListener('mousedown', function(e) {
      if (e.button !== 0 || !detailSvg) return;
      detailDrag = true; detailSx = e.clientX - detailTx; detailSy = e.clientY - detailTy;
      modalBody.style.cursor = 'grabbing';
    });
    document.addEventListener('mousemove', function(e) {
      if (!detailDrag) return;
      detailTx = e.clientX - detailSx; detailTy = e.clientY - detailSy;
      if (detailSvg) detailSvg.style.transition = 'none';
      applyDetailTransform();
    });
    document.addEventListener('mouseup', function() {
      if (detailDrag) { detailDrag = false; modalBody.style.cursor = 'grab'; if (detailSvg) detailSvg.style.transition = 'transform 0.15s ease'; }
    });

    function applyDetailTransform() {
      if (!detailSvg) return;
      detailSvg.style.transform = 'translate(' + detailTx + 'px,' + detailTy + 'px) scale(' + detailScale + ')';
      detailSvg.style.transformOrigin = '0 0';
    }

    // --- Lightweight tree layout + SVG renderer (per-node expand) ---
    var NODE_W = 160, NODE_H = 50, H_GAP = 16, V_GAP = 60;
    var INITIAL_DEPTH = 3;
    var currentTreeStruct = null, currentSample = null;
    var expandedNodes = {};

    function countNodes(node) {
      if (!node || node.t === 'l') return 1;
      return 1 + countNodes(node.l) + countNodes(node.r);
    }

    function computeLayout(node, depth, baseMax, path) {
      if (node.t === 'l') return { node: node, depth: depth, width: NODE_W, children: null, truncated: 0, path: path };
      var extra = expandedNodes[path] || 0;
      if (depth >= baseMax + extra) {
        return { node: node, depth: depth, width: NODE_W, children: null, truncated: countNodes(node) - 1, path: path };
      }
      var left = computeLayout(node.l, depth + 1, baseMax, path + '.l');
      var right = computeLayout(node.r, depth + 1, baseMax, path + '.r');
      var w = left.width + H_GAP + right.width;
      return { node: node, depth: depth, width: Math.max(w, NODE_W), children: [left, right], truncated: 0, path: path };
    }

    function assignPos(layout, cx, y, positions, nodeH) {
      positions.push({ layout: layout, x: cx, y: y });
      if (!layout.children) return;
      var totalW = layout.children[0].width + H_GAP + layout.children[1].width;
      assignPos(layout.children[0], cx - totalW/2 + layout.children[0].width/2, y + nodeH + V_GAP, positions, nodeH);
      assignPos(layout.children[1], cx + totalW/2 - layout.children[1].width/2, y + nodeH + V_GAP, positions, nodeH);
    }

    function traceP(node, sample) {
      if (!sample || node.t === 'l') return [];
      var fv = sample[node.f]; if (fv === undefined) return [];
      var gl = (node.op==='<='?fv<=node.th:node.op==='<'?fv<node.th:node.op==='>='?fv>=node.th:node.op==='>'?fv>node.th:fv<=node.th);
      return [gl?'l':'r'].concat(traceP(gl?node.l:node.r, sample));
    }

    function getPS(positions, ts, sample) {
      if (!sample) return new Set();
      var path = traceP(ts, sample), onP = new Set([0]), cur = positions[0].layout;
      for (var i = 0; i < path.length; i++) {
        if (!cur.children) break;
        var ch = cur.children[path[i]==='l'?0:1];
        for (var j = 0; j < positions.length; j++) { if (positions[j].layout === ch) { onP.add(j); break; } }
        cur = ch;
      }
      return onP;
    }

    function renderTreeSVG(ts, sample) { currentTreeStruct=ts; currentSample=sample; expandedNodes={}; return buildSvg(); }
    function rerender() { if(!currentTreeStruct) return; modalBody.innerHTML=''; detailSvg=buildSvg(); detailSvg.style.transition='none'; detailSvg.style.transformOrigin='0 0'; detailSvg.style.transform='translate('+detailTx+'px,'+detailTy+'px) scale('+detailScale+')'; modalBody.appendChild(detailSvg); requestAnimationFrame(function(){if(detailSvg)detailSvg.style.transition='transform 0.15s ease';}); }

    function buildSvg() {
      var sample=currentSample, ts=currentTreeStruct, NH=sample?70:NODE_H;
      var layout=computeLayout(ts,0,INITIAL_DEPTH,'R'), positions=[];
      assignPos(layout, layout.width/2, 20, positions, NH);
      var pathSet=getPS(positions,ts,sample), hasP=pathSet.size>0;
      var ns='http://www.w3.org/2000/svg', mxX=0,mxY=0;
      positions.forEach(function(p){mxX=Math.max(mxX,p.x+NODE_W/2);mxY=Math.max(mxY,p.y+NH);});
      var svgW=mxX+40,svgH=mxY+40;
      var el=document.createElementNS(ns,'svg');
      el.setAttribute('width',svgW);el.setAttribute('height',svgH);
      el.setAttribute('viewBox','0 0 '+svgW+' '+svgH);el.style.cursor='grab';

      for(var i=0;i<positions.length;i++){var p=positions[i];if(!p.layout.children)continue;var pOn=pathSet.has(i);
        p.layout.children.forEach(function(cl,ci){
          var cI=positions.findIndex(function(pp){return pp.layout===cl;});if(cI<0)return;
          var c=positions[cI],onE=pOn&&pathSet.has(cI),dim=hasP&&!onE;
          var ln=document.createElementNS(ns,'line');
          ln.setAttribute('x1',p.x);ln.setAttribute('y1',p.y+NH);ln.setAttribute('x2',c.x);ln.setAttribute('y2',c.y);
          ln.setAttribute('class','edge-line'+(onE?' on-path':'')+(dim?' dimmed':''));
          ln.style.stroke=onE?'#1565c0':'#8d6e63';ln.style.strokeWidth=onE?'3.5':'2.5';ln.style.strokeLinecap='round';
          if(dim)ln.style.opacity='0.15';
          el.appendChild(ln);
          var lb=document.createElementNS(ns,'text');lb.setAttribute('x',(p.x+c.x)/2+(ci===0?-10:10));
          lb.setAttribute('y',(p.y+NH+c.y)/2);lb.setAttribute('class','edge-label'+(dim?' dimmed':''));
          lb.style.fill=onE?'#1565c0':'#5D4037';lb.style.fontSize='11px';lb.style.textAnchor='middle';lb.style.fontWeight='600';
          if(dim)lb.style.opacity='0.15';
          lb.textContent=ci===0?'\u2713':'\u2717';el.appendChild(lb);
        });
      }

      for(var j=0;j<positions.length;j++){(function(j){
        var pos=positions[j],nd=pos.layout.node,trunc=pos.layout.truncated||0,nPath=pos.layout.path;
        var rx=pos.x-NODE_W/2,ry=pos.y,onP=pathSet.has(j),dim=hasP&&!onP;
        var rect=document.createElementNS(ns,'rect');
        rect.setAttribute('x',rx);rect.setAttribute('y',ry);rect.setAttribute('width',NODE_W);rect.setAttribute('height',NH);
        rect.setAttribute('class',(nd.t==='l'?'node-rect leaf':'node-rect')+(onP?' on-path':'')+(dim?' dimmed':''));
        // Inline fill/stroke to survive notebook CSS overrides
        if(onP){rect.style.fill=nd.t==='l'?'#bbdefb':'#e3f2fd';rect.style.stroke='#1565c0';rect.style.strokeWidth='3';}
        else if(nd.t==='l'){rect.style.fill='#e8f5e9';rect.style.stroke='#2e7d32';rect.style.strokeWidth='2';}
        else{rect.style.fill='#fffde7';rect.style.stroke='#5D4037';rect.style.strokeWidth='2';}
        rect.setAttribute('rx','8');
        if(dim){rect.style.opacity='0.2';}
        el.appendChild(rect);
        var t1=document.createElementNS(ns,'text');t1.setAttribute('x',pos.x);t1.setAttribute('y',ry+(sample?20:18));t1.setAttribute('class','node-text'+(dim?' dimmed':''));
        t1.style.fill='#333';t1.style.fontSize='11px';t1.style.textAnchor='middle';t1.style.pointerEvents='none';
        if(dim)t1.style.opacity='0.2';
        var t2=document.createElementNS(ns,'text');t2.setAttribute('x',pos.x);t2.setAttribute('y',ry+(sample?36:33));t2.setAttribute('class','node-text'+(dim?' dimmed':''));
        t2.style.fill='#333';t2.style.fontSize='11px';t2.style.textAnchor='middle';t2.style.pointerEvents='none';
        if(dim)t2.style.opacity='0.2';

        if(trunc>0){
          t1.textContent=nd.f+' '+nd.op+' '+nd.th.toFixed(4);
          t2.textContent='\u25bc expand (+'+trunc+')';t2.style.fill='#1565c0';t2.style.fontSize='9px';t2.style.cursor='pointer';
          rect.style.cursor='pointer';rect.style.strokeDasharray='4,2';
          var xp=function(e){e.stopPropagation();expandedNodes[nPath]=(expandedNodes[nPath]||0)+3;rerender();};
          rect.addEventListener('click',xp);t2.addEventListener('click',xp);
        }else if(nd.t==='s'){
          t1.textContent=nd.f+' '+nd.op+' '+nd.th.toFixed(4);t2.textContent='';
          if(sample&&onP&&sample[nd.f]!==undefined){var vt=document.createElementNS(ns,'text');vt.setAttribute('x',pos.x);vt.setAttribute('y',ry+55);vt.setAttribute('class','node-text sample-val');vt.style.fill='#1565c0';vt.style.fontSize='10px';vt.style.fontStyle='italic';vt.style.fontWeight='500';vt.style.textAnchor='middle';vt.style.pointerEvents='none';vt.textContent=nd.f+' = '+sample[nd.f].toFixed(3);el.appendChild(vt);}
        }else{
          if(nd.c){var best='',bv=-1;for(var k in nd.c){if(nd.c[k]>bv){bv=nd.c[k];best=k;}}
            if(IS_BOOSTED){t1.textContent='🌿 Leaf correction';var vals=[];for(var k2 in nd.c){vals.push(k2+':'+nd.c[k2].toFixed(3));}t2.textContent=vals.join(' ');}
            else{t1.textContent='🌿 Class: '+best;t2.textContent=(bv*100).toFixed(0)+'%';}}
          else if(nd.v!==undefined){t1.textContent='🌿 '+nd.v.toFixed(4);t2.textContent='';}
        }
        el.appendChild(t1);if(t2.textContent)el.appendChild(t2);
      })(j);}

      return el;
    }
  })();

  // --- Prediction panel ---
  (function() {
    if (!HAS_PREDICT) return;
    var dataEl = $('predict-data');
    if (!dataEl) return;
    var predData = JSON.parse(dataEl.textContent);
    var goBtn = $('predict-go');
    var clearBtn = $('predict-clear');
    var rowInput = $('predict-row');
    var resultEl = $('predict-result');
    var closeBtn = $('predict-close');
    var panel = $('predict-panel');

    if (closeBtn && panel) {
      closeBtn.addEventListener('click', function() { panel.style.display = 'none'; });
    }

    function traceTree(node, sample) {
      if (node.t === 'l') {
        if (node.c) {
          var best = null, bestV = -1;
          for (var k in node.c) { if (node.c[k] > bestV) { bestV = node.c[k]; best = k; } }
          return { cls: best, dist: node.c };
        }
        return { val: node.v };
      }
      var fv = sample[node.f];
      if (fv === undefined) return { err: 'missing ' + node.f };
      var goLeft = false;
      switch(node.op) {
        case '<=': goLeft = fv <= node.th; break;
        case '<': goLeft = fv < node.th; break;
        case '>=': goLeft = fv >= node.th; break;
        case '>': goLeft = fv > node.th; break;
        case '==': goLeft = fv == node.th; break;
        case '!=': goLeft = fv != node.th; break;
        default: goLeft = fv <= node.th;
      }
      return traceTree(goLeft ? node.l : node.r, sample);
    }

    function clearBadges() {
      svg.querySelectorAll('.pred-label').forEach(function(el) { el.remove(); });
      resultEl.textContent = '';
      var sd = $('sample-display');
      if (sd) sd.classList.remove('visible');
      traceActive = false;
    }

    function showSampleDisplay(sample, idx) {
      var sd = $('sample-display');
      if (!sd) return;
      var html = '<span class="sample-title">Sample #' + idx + '</span>';
      html += '<div class="sample-chips">';
      for (var key in sample) {
        var val = sample[key];
        var display = (typeof val === 'number') ? val.toFixed(3) : String(val);
        html += '<span class="chip"><span class="chip-name">' + key + ':</span><span class="chip-val">' + display + '</span></span>';
      }
      html += '</div>';
      sd.innerHTML = html;
      sd.classList.add('visible');
    }

    if (clearBtn) clearBtn.addEventListener('click', clearBadges);

    if (goBtn) goBtn.addEventListener('click', function() {
      clearBadges();
      var idx = parseInt(rowInput.value);
      if (isNaN(idx) || idx < 0 || idx >= predData.samples.length) {
        resultEl.textContent = 'Row out of range (0–' + (predData.samples.length - 1) + ')';
        return;
      }
      var sample = predData.samples[idx];
      showSampleDisplay(sample, idx);
      traceActive = true;
      var predictions = [];

      // Trace every tree
      for (var t = 0; t < predData.trees.length; t++) {
        predictions.push({ idx: t, result: traceTree(predData.trees[t], sample) });
      }

      // Show badges on visible trees
      var agg = predData.aggregation || 'avg';
      var visibleTrees = treeData.filter(function(d) { return !d.el.classList.contains('hidden'); });
      visibleTrees.forEach(function(d) {
        var pred = predictions[d.idx];
        if (!pred) return;
        var label = '';
        var color = '#1565c0';
        if (agg === 'sum') {
          // Boosted: show raw correction value
          if (pred.result.val !== undefined) {
            label = pred.result.val.toFixed(2);
            color = pred.result.val >= 0 ? '#2e7d32' : '#c62828';
          } else if (pred.result.cls !== undefined && pred.result.dist) {
            // class_distribution with single value — show the raw value
            var vals = Object.values(pred.result.dist);
            label = vals[0].toFixed(2);
            color = vals[0] >= 0 ? '#2e7d32' : '#c62828';
          }
        } else {
          // Vote-based: show class
          if (pred.result.cls !== undefined) {
            label = pred.result.cls;
            var colors = ['#1565c0','#c62828','#2e7d32','#f57c00','#6a1b9a','#00838f'];
            color = colors[parseInt(label) % colors.length];
          } else if (pred.result.val !== undefined) {
            label = pred.result.val.toFixed(2);
            color = '#333';
          }
        }
        if (!label) return;

        var badge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        badge.setAttribute('class', 'pred-label');
        var bgR = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bgR.setAttribute('x', '-18'); bgR.setAttribute('y', '-165');
        bgR.setAttribute('width', '36'); bgR.setAttribute('height', '16');
        bgR.setAttribute('rx', '3'); bgR.setAttribute('fill', color); bgR.setAttribute('opacity', '0.9');
        var txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        txt.setAttribute('x', '0'); txt.setAttribute('y', '-153');
        txt.setAttribute('text-anchor', 'middle'); txt.setAttribute('font-size', '10');
        txt.setAttribute('fill', 'white'); txt.setAttribute('font-weight', 'bold');
        txt.textContent = label;
        badge.appendChild(bgR); badge.appendChild(txt);
        d.el.appendChild(badge);
      });

      // Aggregate result — use pre-computed model prediction if available
      var agg = predData.aggregation || 'avg';
      var modelPred = (predData.predictions && idx < predData.predictions.length) ? predData.predictions[idx] : null;

      if (modelPred !== null) {
        resultEl.innerHTML = 'Ensemble prediction: <strong>' + modelPred + '</strong>';
      } else if (predData.is_classifier) {
        if (agg === 'sum') {
          var classSums = {};
          predictions.forEach(function(p) {
            if (p.result.val !== undefined) { classSums['_'] = (classSums['_']||0) + p.result.val; }
            else if (p.result.dist) { for (var k in p.result.dist) { classSums[k] = (classSums[k]||0) + p.result.dist[k]; } }
          });
          var bestCls = '', bestSum = -Infinity;
          for (var k in classSums) { if (classSums[k] > bestSum) { bestSum = classSums[k]; bestCls = k; } }
          resultEl.innerHTML = 'Ensemble score: <strong>' + bestSum.toFixed(3) + '</strong> (' + predictions.length + ' trees)';
        } else {
          var votes = {};
          predictions.forEach(function(p) { if (p.result.cls) votes[p.result.cls] = (votes[p.result.cls]||0) + 1; });
          var best = '', bestCount = 0;
          for (var k in votes) { if (votes[k] > bestCount) { bestCount = votes[k]; best = k; } }
          resultEl.innerHTML = 'Ensemble prediction: <strong>' + best + '</strong> (' + bestCount + '/' + predictions.length + ' votes)';
        }
      } else {
        // Regression
        var sum = 0, cnt = 0;
        predictions.forEach(function(p) { if (p.result.val !== undefined) { sum += p.result.val; cnt++; } });
        if (agg === 'sum') {
          var final = sum;
          var detail = 'sum of ' + cnt + ' trees';
          // Apply boosting constants if available (sklearn GBM)
          if (predData.boosting) {
            var lr = predData.boosting.lr || 1;
            var init = predData.boosting.init || 0;
            final = init + lr * sum;
            detail = 'init(' + init.toFixed(2) + ') + ' + lr + ' × sum(' + sum.toFixed(2) + ')';
          }
          resultEl.innerHTML = 'Ensemble prediction: <strong>' + final.toFixed(4) + '</strong> (' + detail + ')';
        } else {
          var avg = cnt > 0 ? (sum / cnt).toFixed(4) : '?';
          resultEl.innerHTML = 'Ensemble prediction: <strong>' + avg + '</strong> (avg of ' + cnt + ' trees)';
        }
      }
      // Show true label if available
      if (predData.targets && idx < predData.targets.length && predData.targets[idx] !== null) {
        var trueVal = predData.targets[idx];
        resultEl.innerHTML += ' | True: <strong style="color:#2e7d32">' + trueVal + '</strong>';
      }
    });
  })();

  // --- Growth animation (first page only, skip if >200) ---
  (function() {
    var visible = allTrees.filter(function(t) { return !t.classList.contains('hidden'); });
    if (visible.length > 200) {
      visible.forEach(function(t) { t.style.opacity='1'; t.classList.add('grown'); });
      return;
    }
    visible.sort(function(a, b) {
      var ay = parseFloat((a.getAttribute('transform')||'').replace(/.*translate\([^,]+,([^)]+)\).*/, '$1'))||0;
      var by = parseFloat((b.getAttribute('transform')||'').replace(/.*translate\([^,]+,([^)]+)\).*/, '$1'))||0;
      return ay - by;
    });
    var delay = Math.max(15, Math.min(50, 1200 / visible.length));
    visible.forEach(function(tree, i) {
      var d = i * delay;
      setTimeout(function() { tree.style.opacity='1'; tree.classList.add('grow-trunk'); }, d);
      setTimeout(function() { tree.classList.add('grow-branches'); }, d + 200);
      setTimeout(function() { tree.classList.add('grow-canopy'); }, d + 380);
      setTimeout(function() { tree.classList.add('grown'); }, d + 650);
    });
  })();
})(document);
"""
