"""ForestRenderer — orchestrates the aesthetic forest rendering pipeline."""

from __future__ import annotations

import random

from prettyforest.models import EnsembleType, UnifiedTree
from prettyforest.rendering.forest.models import ForestConfig
from prettyforest.rendering.forest.scene_composer import MAX_VISIBLE, SceneComposer, TreeMeta
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
    ) -> str:
        rng = random.Random(config.seed)
        total_count = len(trees)

        visuals = self._mapper.map_trees(trees, rng, season=config.season, ensemble_type=ensemble_type)
        paths = [self._shape_gen.generate(v, rng) for v in visuals]
        metas = [self._compute_meta(tree, i, ensemble_type) for i, tree in enumerate(trees)]

        svg = self._composer.compose(paths, visuals, metas, config, total_count, rng)
        is_classifier = trees[0].is_classifier if trees else True

        # Build prediction data if data was provided
        predict_json = ""
        if data is not None:
            predict_json = self._build_predict_json(trees, data)

        # Always embed tree structures for detail view
        trees_json = self._build_trees_json(trees)

        return self._wrap_html(svg, ensemble_type, is_classifier, total_count, predict_json, trees_json)

    def _compute_meta(self, tree: UnifiedTree, index: int, ensemble_type: EnsembleType) -> TreeMeta:
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
                        magnitudes.append(max(abs(v) for v in leaf.class_distribution.values()))
                leaf_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else None
        else:
            values = [l.prediction_value for l in leaves if l.prediction_value is not None]
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

    def _build_predict_json(self, trees: list[UnifiedTree], data) -> str:
        """Build compact JSON with tree structures + sample rows for client-side prediction."""
        import json
        import polars as pl

        # Embed up to 100 sample rows
        max_samples = 100
        if isinstance(data, pl.DataFrame):
            n_rows = data.height
            feature_names = data.columns
            rows = []
            for i in range(min(n_rows, max_samples)):
                row = data.row(i, named=True)
                rows.append({k: float(v) for k, v in row.items()})
        else:
            return ""

        # Compact tree structures: only splits + leaf predictions
        def serialize_node(node):
            if node.is_leaf:
                n = {"t": "l"}
                if node.prediction_value is not None:
                    n["v"] = round(node.prediction_value, 6)
                if node.class_distribution is not None:
                    n["c"] = {k: round(v, 4) for k, v in node.class_distribution.items()}
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

        payload = {
            "features": feature_names,
            "samples": rows,
            "n_rows": n_rows,
            "trees": compact_trees,
            "is_classifier": trees[0].is_classifier if trees else True,
        }
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
                    n["c"] = {k: round(v, 4) for k, v in node.class_distribution.items()}
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

    def _wrap_html(self, svg: str, ensemble_type: EnsembleType, is_classifier: bool, total: int, predict_json: str = "", trees_json: str = "") -> str:
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
            "<h2>PrettyForest</h2>\n",
            '<div class="zoom-controls">\n',
            '<button id="zoom-in" title="Zoom In">+</button>\n',
            '<button id="zoom-reset" title="Reset Zoom">⟳</button>\n',
            '<button id="zoom-out" title="Zoom Out">−</button>\n',
            '<span id="zoom-level">Zoom: 100%</span>\n',
            "</div>\n",
            "</div>\n",
            '<div class="toolbar">\n',
            '<div class="tool-group">\n',
            '<label>Sort by</label>\n',
            '<select id="sort-by">\n',
            '<option value="natural">Natural</option>\n',
            '<option value="depth">Depth</option>\n',
            '<option value="nodes">Nodes</option>\n',
            '<option value="leaves">Leaves</option>\n',
            f'<option value="metric">{metric_label}</option>\n',
            '</select>\n',
            "</div>\n",
            '<div class="tool-group">\n',
            '<label>Showing</label>\n',
            '<button id="page-prev" class="tool-btn" title="Previous page">◀</button>\n',
            f'<span id="page-info">1–{page_size} of {total}</span>\n',
            '<button id="page-next" class="tool-btn" title="Next page">▶</button>\n',
            "</div>\n",
            '<div class="tool-group">\n',
            '<label>Highlight</label>\n',
            '<button id="highlight-top" class="tool-btn">Top 3</button>\n',
            '<button id="highlight-bottom" class="tool-btn">Bottom 3</button>\n',
            "</div>\n",
            '<div class="tool-group">\n',
            '<button id="reset-all" class="tool-btn">⟳ Reset</button>\n',
            "</div>\n",
            "</div>\n",
            '<div class="spotlight-panel" id="spotlight-panel">\n',
            '<button id="spotlight-close">✕</button>\n',
            '<div id="spotlight-content"></div>\n',
            "</div>\n",
        ]

        if predict_json:
            parts.extend([
                '<div class="predict-panel" id="predict-panel">\n',
                '<div class="predict-header">\n',
                '<strong>🔍 Predict</strong>\n',
                '<button id="predict-close" title="Close">✕</button>\n',
                '</div>\n',
                '<div class="predict-body">\n',
                '<label>Sample row:</label>\n',
                '<input type="number" id="predict-row" min="0" value="0" style="width:60px"/>\n',
                '<button id="predict-go" class="tool-btn">Trace</button>\n',
                '<button id="predict-clear" class="tool-btn">Clear</button>\n',
                '<div id="predict-result"></div>\n',
                '</div>\n',
                "</div>\n",
            ])

        # Sample data display (inline bar, shown when a sample is traced)
        parts.append('<div class="sample-display" id="sample-display"></div>\n')

        parts.extend([
            '<div class="forest-container" id="forest-container">\n',
            f"{svg}\n",
            "</div>\n",
            '<div class="tooltip" id="tooltip"></div>\n',
        ])

        if predict_json:
            parts.append(f'<script id="predict-data" type="application/json">{predict_json}</script>\n')

        # Always embed tree structures for detail modal
        parts.append(f'<script id="trees-data" type="application/json">{trees_json}</script>\n')

        # Detail modal (rendered on the fly via JS)
        parts.append('<div class="detail-modal" id="detail-modal">\n')
        parts.append('<div class="detail-header"><span id="detail-title">Tree #0</span><button id="detail-close">✕</button></div>\n')
        parts.append('<div class="detail-body" id="detail-body"></div>\n')
        parts.append('</div>\n')

        has_predict = "true" if predict_json else "false"
        parts.extend([
            f'<script>var METRIC_KEY="{metric_name}",METRIC_LABEL="{metric_label}",TOTAL={total},PAGE_SIZE={page_size},HAS_PREDICT={has_predict};</script>\n',
            f"<script>{_JS}</script>\n",
            "</body>\n",
            "</html>",
        ])

        return "".join(parts)


_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f4f0; }
.header { display: flex; justify-content: space-between; align-items: center; padding: 10px 24px; background: rgba(255,255,255,0.97); border-bottom: 1px solid #e0e0e0; }
.header h2 { font-size: 16px; color: #333; }
.zoom-controls { display: flex; align-items: center; gap: 6px; }
.zoom-controls button { width: 28px; height: 28px; border: 1px solid #ccc; border-radius: 5px; background: white; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.zoom-controls button:hover { background: #e8f5e9; border-color: #4caf50; }
#zoom-level { font-size: 11px; color: #888; min-width: 60px; text-align: center; }
.toolbar { display: flex; align-items: center; gap: 18px; padding: 7px 24px; background: rgba(255,255,255,0.94); border-bottom: 1px solid #eee; flex-wrap: wrap; }
.tool-group { display: flex; align-items: center; gap: 6px; }
.tool-group label { font-size: 11px; color: #555; font-weight: 500; white-space: nowrap; }
.tool-group select { padding: 3px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 11px; }
#page-info { font-size: 11px; color: #333; min-width: 90px; text-align: center; }
.tool-btn { padding: 3px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 11px; background: white; cursor: pointer; }
.tool-btn:hover { background: #e8f5e9; border-color: #4caf50; }
.tool-btn.active { background: #c8e6c9; border-color: #4caf50; }
.tool-btn:disabled { opacity: 0.4; cursor: default; }
.spotlight-panel { position: fixed; top: 100px; right: 20px; width: 250px; background: rgba(255,255,255,0.97); border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); padding: 16px; display: none; z-index: 500; border: 1px solid #e0e0e0; }
.spotlight-panel.visible { display: block; }
#spotlight-close { position: absolute; top: 8px; right: 10px; border: none; background: none; font-size: 16px; cursor: pointer; color: #888; }
#spotlight-close:hover { color: #333; }
#spotlight-content { font-size: 12px; line-height: 1.8; }
#spotlight-content strong { font-size: 13px; display: block; margin-bottom: 4px; }
.stat-row { display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px solid #f0f0f0; }
.stat-label { color: #666; } .stat-value { font-weight: 500; }
.forest-container { width: 100%; height: calc(100vh - 94px); overflow: hidden; position: relative; }
.forest-container svg { display: block; width: 100%; height: 100%; cursor: grab; transition: transform 0.15s ease; }
.forest-container svg:active { cursor: grabbing; }
.tooltip { position: fixed; display: none; background: rgba(20,20,20,0.92); color: #f0f0f0; padding: 10px 14px; border-radius: 8px; font-size: 12px; line-height: 1.7; pointer-events: none; z-index: 1000; max-width: 240px; box-shadow: 0 6px 20px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); }
.visual-tree { cursor: pointer; transition: opacity 0.3s, filter 0.3s; }
.visual-tree:hover { filter: brightness(1.1); }
.visual-tree.hidden { display: none; }
.visual-tree.highlighted { filter: drop-shadow(0 0 6px #ffeb3b) drop-shadow(0 0 12px rgba(255,235,59,0.5)); }
.visual-tree.spotlit { filter: drop-shadow(0 0 8px #42a5f5) drop-shadow(0 0 16px rgba(66,165,245,0.4)); }
.visual-tree { opacity: 0; }
.visual-tree.grown { opacity: 1; }
.visual-tree .trunk { transform: scaleY(0); transform-origin: bottom center; }
.visual-tree.grow-trunk .trunk { transform: scaleY(1); transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1); }
.visual-tree .branch { opacity: 0; stroke-dasharray: 100; stroke-dashoffset: 100; }
.visual-tree.grow-branches .branch { opacity: 1; stroke-dashoffset: 0; transition: stroke-dashoffset 0.5s ease, opacity 0.2s ease; }
.visual-tree .canopy { opacity: 0; transform: scale(0); transform-origin: center 40%; }
.visual-tree.grow-canopy .canopy { opacity: 0.9; transform: scale(1); transition: transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.3s ease; }
.predict-panel { padding: 8px 24px; background: rgba(255,255,255,0.94); border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.predict-header { display: flex; align-items: center; gap: 8px; }
.predict-header strong { font-size: 12px; }
#predict-close { border: none; background: none; font-size: 14px; cursor: pointer; color: #888; }
.predict-body { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.predict-body label { font-size: 11px; color: #555; }
.predict-body input { padding: 3px 6px; border: 1px solid #ccc; border-radius: 4px; font-size: 11px; }
#predict-result { font-size: 11px; color: #333; margin-left: 8px; }
#predict-result strong { color: #1565c0; }
.predict-badge { font-size: 9px; font-weight: bold; fill: white; }
.predict-badge-bg { rx: 3; }
.detail-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(240,244,240,0.98); z-index: 2000; flex-direction: column; }
.detail-modal.open { display: flex; }
.detail-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 24px; border-bottom: 1px solid #c8e6c9; background: linear-gradient(to right, #e8f5e9, #f1f8e9); }
.detail-header span { font-size: 15px; font-weight: 600; color: #2e7d32; }
#detail-close { border: none; background: none; font-size: 22px; cursor: pointer; color: #888; padding: 4px 8px; }
#detail-close:hover { color: #333; }
.detail-body { flex: 1; overflow: auto; padding: 20px; display: flex; justify-content: center; cursor: grab; background: radial-gradient(ellipse at center, #f9fdf9 0%, #eef5ee 100%); }
.detail-body svg { height: auto; cursor: grab; }
.detail-body svg:active { cursor: grabbing; }
.detail-body .node-rect { fill: #fffde7; stroke: #5D4037; stroke-width: 2; rx: 8; }
.detail-body .node-rect.leaf { fill: #e8f5e9; stroke: #2e7d32; stroke-width: 2; }
.detail-body .node-rect.on-path { fill: #e3f2fd; stroke: #1565c0; stroke-width: 3; }
.detail-body .node-rect.leaf.on-path { fill: #bbdefb; stroke: #1565c0; stroke-width: 3; }
.detail-body .node-rect.dimmed { opacity: 0.2; }
.detail-body .edge-line { stroke: #8d6e63; stroke-width: 2.5; stroke-linecap: round; }
.detail-body .edge-line.on-path { stroke: #1565c0; stroke-width: 3.5; }
.detail-body .edge-line.dimmed { opacity: 0.15; }
.detail-body .node-text { font-size: 11px; fill: #333; text-anchor: middle; pointer-events: none; }
.detail-body .node-text.dimmed { opacity: 0.2; }
.detail-body .node-text.sample-val { font-size: 9px; fill: #1565c0; font-style: italic; }
.detail-body .edge-label { font-size: 11px; fill: #5D4037; text-anchor: middle; font-weight: 600; }
.detail-body .edge-label.dimmed { opacity: 0.15; }
.sample-display { display: none; padding: 6px 24px; background: rgba(245,248,245,0.98); border-bottom: 1px solid #eee; overflow-x: auto; white-space: nowrap; }
.sample-display.visible { display: block; }
.sample-display .sample-title { font-size: 11px; font-weight: 600; color: #333; margin-bottom: 6px; display: block; }
.sample-display .sample-chips { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
.sample-display .chip { background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 4px; padding: 3px 8px; font-size: 10px; white-space: nowrap; flex-shrink: 0; }
.sample-display .chip .chip-name { color: #666; }
.sample-display .chip .chip-val { color: #1565c0; font-weight: 600; margin-left: 3px; }
"""

_JS = r"""
(function() {
  var svg = document.getElementById('forest-svg');
  var container = document.getElementById('forest-container');
  var tooltip = document.getElementById('tooltip');
  var sortBy = document.getElementById('sort-by');
  var pagePrev = document.getElementById('page-prev');
  var pageNext = document.getElementById('page-next');
  var pageInfo = document.getElementById('page-info');
  var highlightTop = document.getElementById('highlight-top');
  var highlightBottom = document.getElementById('highlight-bottom');
  var resetAll = document.getElementById('reset-all');
  var spotlightPanel = document.getElementById('spotlight-panel');
  var spotlightClose = document.getElementById('spotlight-close');
  var spotlightContent = document.getElementById('spotlight-content');
  var zoomIn = document.getElementById('zoom-in');
  var zoomOut = document.getElementById('zoom-out');
  var zoomReset = document.getElementById('zoom-reset');
  var zoomLabel = document.getElementById('zoom-level');
  if (!svg || !container) return;

  // --- Collect all tree elements and their data ---
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

  // --- Highlight top/bottom 3 (shows only those 3) ---
  highlightTop.addEventListener('click', function() {
    var ranked = treeData.slice().filter(function(d) { return metric(d) !== null; });
    ranked.sort(function(a, b) { return (metric(b)||0) - (metric(a)||0); });
    showSubset(ranked.slice(0, 3));
    highlightTop.classList.add('active');
    highlightBottom.classList.remove('active');
  });
  highlightBottom.addEventListener('click', function() {
    var ranked = treeData.slice().filter(function(d) { return metric(d) !== null; });
    ranked.sort(function(a, b) { return (metric(a)||0) - (metric(b)||0); });
    showSubset(ranked.slice(0, 3));
    highlightBottom.classList.add('active');
    highlightTop.classList.remove('active');
  });

  function showSubset(subset) {
    var subSet = new Set(subset.map(function(d) { return d.el; }));
    treeData.forEach(function(d) {
      if (subSet.has(d.el)) {
        d.el.classList.remove('hidden');
        d.el.classList.add('highlighted', 'grown', 'grow-trunk', 'grow-branches', 'grow-canopy');
        d.el.style.opacity = '1';
      } else {
        d.el.classList.add('hidden');
        d.el.classList.remove('highlighted', 'grown', 'grow-trunk', 'grow-branches', 'grow-canopy', 'spotlit');
        d.el.style.opacity = '';
      }
    });
    pageInfo.textContent = subset.length + ' highlighted';
    pagePrev.disabled = true;
    pageNext.disabled = true;
  }

  // --- Reset ---
  resetAll.addEventListener('click', function() {
    closeSpotlight();
    highlightTop.classList.remove('active');
    highlightBottom.classList.remove('active');
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

  // --- Info button ---
  (function() {
    var infoBtn = document.getElementById('info-btn');
    var legendPanel = document.getElementById('legend-panel');
    if (!infoBtn || !legendPanel) return;
    infoBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      legendPanel.style.display = legendPanel.style.display === 'none' ? '' : 'none';
    });
  })();

  // --- Double-click to open tree detail modal ---
  (function() {
    var modal = document.getElementById('detail-modal');
    var modalBody = document.getElementById('detail-body');
    var modalTitle = document.getElementById('detail-title');
    var closeBtn = document.getElementById('detail-close');
    var treesEl = document.getElementById('trees-data');
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

      // Get traced sample if available
      var tracedSample = null;
      var predDataEl = document.getElementById('predict-data');
      var rowInput = document.getElementById('predict-row');
      if (predDataEl && rowInput) {
        try {
          var pd = JSON.parse(predDataEl.textContent);
          var idx = parseInt(rowInput.value);
          if (pd.samples && idx >= 0 && idx < pd.samples.length) {
            tracedSample = pd.samples[idx];
          }
        } catch(e) {}
      }

      detailSvg = renderTreeSVG(treeStruct, tracedSample);
      detailSvg.style.transition = 'transform 0.15s ease';
      modalBody.appendChild(detailSvg);
      modal.classList.add('open');
    }

    svg.addEventListener('dblclick', function(e) {
      var tree = findTree(e.target);
      if (!tree) return;
      var d = treeData.find(function(t) { return t.el === tree; });
      if (!d) return;
      openDetail(d.idx, d.depth, d.nodes);
    });

    // Also open from spotlight panel on double-click the panel title
    var spotlightEl = document.getElementById('spotlight-content');
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
      detailSvg.style.transformOrigin = 'center top';
    }

    // --- Lightweight tree layout + SVG renderer ---
    var NODE_W = 140, NODE_H = 50, H_GAP = 16, V_GAP = 60;

    function computeLayout(node, depth) {
      if (node.t === 'l') return { node: node, depth: depth, width: NODE_W, children: null };
      var left = computeLayout(node.l, depth + 1);
      var right = computeLayout(node.r, depth + 1);
      var w = left.width + H_GAP + right.width;
      return { node: node, depth: depth, width: Math.max(w, NODE_W), children: [left, right] };
    }

    function assignPositions(layout, cx, y, positions, nodeH) {
      positions.push({ layout: layout, x: cx, y: y });
      if (!layout.children) return;
      var left = layout.children[0], right = layout.children[1];
      var totalW = left.width + H_GAP + right.width;
      var leftCx = cx - totalW / 2 + left.width / 2;
      var rightCx = cx + totalW / 2 - right.width / 2;
      var childY = y + nodeH + V_GAP;
      assignPositions(left, leftCx, childY, positions, nodeH);
      assignPositions(right, rightCx, childY, positions, nodeH);
    }

    function tracePath(node, sample) {
      // Returns array of 'l' or 'r' decisions
      if (!sample || node.t === 'l') return [];
      var fv = sample[node.f];
      if (fv === undefined) return [];
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
      if (goLeft) return ['l'].concat(tracePath(node.l, sample));
      else return ['r'].concat(tracePath(node.r, sample));
    }

    function getPathPositions(positions, treeStruct, sample) {
      // Return set of position indices on the traced path
      if (!sample) return new Set();
      var path = tracePath(treeStruct, sample);
      var onPath = new Set();
      onPath.add(0); // root
      var posIdx = 0;
      var currentLayout = positions[0].layout;
      for (var i = 0; i < path.length; i++) {
        var childIdx = path[i] === 'l' ? 0 : 1;
        if (!currentLayout.children) break;
        var childLayout = currentLayout.children[childIdx];
        // Find position of this child
        for (var j = 0; j < positions.length; j++) {
          if (positions[j].layout === childLayout) { posIdx = j; break; }
        }
        onPath.add(posIdx);
        currentLayout = childLayout;
      }
      return onPath;
    }

    function renderTreeSVG(treeStruct, sample) {
      var NODE_H_EXT = sample ? 62 : NODE_H; // taller nodes when showing sample values
      var layout = computeLayout(treeStruct, 0);
      var positions = [];
      assignPositions(layout, layout.width / 2, 20, positions, NODE_H_EXT);

      var pathSet = getPathPositions(positions, treeStruct, sample);
      var hasPath = pathSet.size > 0;

      var maxX = 0, maxY = 0;
      positions.forEach(function(p) {
        if (p.x + NODE_W / 2 > maxX) maxX = p.x + NODE_W / 2;
        if (p.y + NODE_H_EXT > maxY) maxY = p.y + NODE_H_EXT;
      });
      var svgW = maxX + 40, svgH = maxY + 40;

      var ns = 'http://www.w3.org/2000/svg';
      var svgEl = document.createElementNS(ns, 'svg');
      svgEl.setAttribute('width', svgW);
      svgEl.setAttribute('height', svgH);
      svgEl.setAttribute('viewBox', '0 0 ' + svgW + ' ' + svgH);
      svgEl.style.cursor = 'grab';

      // Draw edges
      for (var i = 0; i < positions.length; i++) {
        var p = positions[i];
        if (!p.layout.children) continue;
        var leftChild = positions.find(function(pp) { return pp.layout === p.layout.children[0]; });
        var rightChild = positions.find(function(pp) { return pp.layout === p.layout.children[1]; });
        var pOnPath = pathSet.has(i);

        [['l', leftChild, '✓ yes'], ['r', rightChild, '✗ no']].forEach(function(pair) {
          var child = pair[1], label = pair[2];
          if (!child) return;
          var cIdx = positions.indexOf(child);
          var onEdge = pOnPath && pathSet.has(cIdx);
          var dimmed = hasPath && !onEdge;

          var line = document.createElementNS(ns, 'line');
          line.setAttribute('x1', p.x); line.setAttribute('y1', p.y + NODE_H_EXT);
          line.setAttribute('x2', child.x); line.setAttribute('y2', child.y);
          line.setAttribute('class', 'edge-line' + (onEdge ? ' on-path' : '') + (dimmed ? ' dimmed' : ''));
          svgEl.appendChild(line);

          var lbl = document.createElementNS(ns, 'text');
          var lx = (p.x + child.x) / 2 + (pair[0] === 'l' ? -12 : 12);
          var ly = (p.y + NODE_H_EXT + child.y) / 2;
          lbl.setAttribute('x', lx); lbl.setAttribute('y', ly);
          lbl.setAttribute('class', 'edge-label' + (dimmed ? ' dimmed' : ''));
          lbl.textContent = label;
          svgEl.appendChild(lbl);
        });
      }

      // Draw nodes
      for (var j = 0; j < positions.length; j++) {
        var pos = positions[j];
        var nd = pos.layout.node;
        var rx = pos.x - NODE_W / 2, ry = pos.y;
        var onPath = pathSet.has(j);
        var dimmed = hasPath && !onPath;

        var rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', rx); rect.setAttribute('y', ry);
        rect.setAttribute('width', NODE_W); rect.setAttribute('height', NODE_H_EXT);
        var cls = (nd.t === 'l' ? 'node-rect leaf' : 'node-rect') + (onPath ? ' on-path' : '') + (dimmed ? ' dimmed' : '');
        rect.setAttribute('class', cls);
        svgEl.appendChild(rect);

        var txt1 = document.createElementNS(ns, 'text');
        txt1.setAttribute('x', pos.x); txt1.setAttribute('y', ry + 18);
        txt1.setAttribute('class', 'node-text' + (dimmed ? ' dimmed' : ''));

        var txt2 = document.createElementNS(ns, 'text');
        txt2.setAttribute('x', pos.x); txt2.setAttribute('y', ry + 33);
        txt2.setAttribute('class', 'node-text' + (dimmed ? ' dimmed' : ''));

        if (nd.t === 's') {
          txt1.textContent = nd.f + ' ' + nd.op + ' ' + nd.th.toFixed(4);
          txt2.textContent = '';

          // Show sample's feature value at this split
          if (sample && onPath && sample[nd.f] !== undefined) {
            var valTxt = document.createElementNS(ns, 'text');
            valTxt.setAttribute('x', pos.x); valTxt.setAttribute('y', ry + (sample ? 50 : 44));
            valTxt.setAttribute('class', 'node-text sample-val');
            valTxt.textContent = nd.f + ' = ' + sample[nd.f].toFixed(3);
            svgEl.appendChild(valTxt);
          }
        } else {
          if (nd.c) {
            var best = '', bestV = -1;
            for (var k in nd.c) { if (nd.c[k] > bestV) { bestV = nd.c[k]; best = k; } }
            txt1.textContent = '🌿 Class: ' + best;
            txt2.textContent = (bestV * 100).toFixed(0) + '% confidence';
          } else if (nd.v !== undefined) {
            txt1.textContent = '🌿 Value: ' + nd.v.toFixed(4);
            txt2.textContent = '';
          }
        }
        svgEl.appendChild(txt1);
        if (txt2.textContent) svgEl.appendChild(txt2);
      }

      return svgEl;
    }
  })();

  // --- Prediction panel ---
  (function() {
    if (!HAS_PREDICT) return;
    var dataEl = document.getElementById('predict-data');
    if (!dataEl) return;
    var predData = JSON.parse(dataEl.textContent);
    var goBtn = document.getElementById('predict-go');
    var clearBtn = document.getElementById('predict-clear');
    var rowInput = document.getElementById('predict-row');
    var resultEl = document.getElementById('predict-result');
    var closeBtn = document.getElementById('predict-close');
    var panel = document.getElementById('predict-panel');

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
      var sd = document.getElementById('sample-display');
      if (sd) sd.classList.remove('visible');
    }

    function showSampleDisplay(sample, idx) {
      var sd = document.getElementById('sample-display');
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
      var predictions = [];

      // Trace every tree
      for (var t = 0; t < predData.trees.length; t++) {
        predictions.push({ idx: t, result: traceTree(predData.trees[t], sample) });
      }

      // Show badges on visible trees
      var visibleTrees = treeData.filter(function(d) { return !d.el.classList.contains('hidden'); });
      visibleTrees.forEach(function(d) {
        var pred = predictions[d.idx];
        if (!pred) return;
        var label = '';
        var color = '#1565c0';
        if (pred.result.cls !== undefined) {
          label = pred.result.cls;
          // Color by class
          var colors = ['#1565c0','#c62828','#2e7d32','#f57c00','#6a1b9a','#00838f'];
          var classIdx = parseInt(label) || 0;
          color = colors[classIdx % colors.length];
        } else if (pred.result.val !== undefined) {
          label = pred.result.val.toFixed(2);
          color = '#333';
        }
        if (!label) return;

        // Add a badge above the tree
        var badge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        badge.setAttribute('class', 'pred-label');
        var bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bg.setAttribute('x', '-18'); bg.setAttribute('y', '-165');
        bg.setAttribute('width', '36'); bg.setAttribute('height', '16');
        bg.setAttribute('rx', '3'); bg.setAttribute('fill', color); bg.setAttribute('opacity', '0.9');
        var txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        txt.setAttribute('x', '0'); txt.setAttribute('y', '-153');
        txt.setAttribute('text-anchor', 'middle'); txt.setAttribute('font-size', '10');
        txt.setAttribute('fill', 'white'); txt.setAttribute('font-weight', 'bold');
        txt.textContent = label;
        badge.appendChild(bg); badge.appendChild(txt);
        d.el.appendChild(badge);
      });

      // Aggregate result
      if (predData.is_classifier) {
        var votes = {};
        predictions.forEach(function(p) {
          if (p.result.cls) votes[p.result.cls] = (votes[p.result.cls]||0) + 1;
        });
        var best = '', bestCount = 0;
        for (var k in votes) { if (votes[k] > bestCount) { bestCount = votes[k]; best = k; } }
        resultEl.innerHTML = 'Ensemble prediction: <strong>' + best + '</strong> (' + bestCount + '/' + predictions.length + ' votes)';
      } else {
        var sum = 0, cnt = 0;
        predictions.forEach(function(p) { if (p.result.val !== undefined) { sum += p.result.val; cnt++; } });
        var avg = cnt > 0 ? (sum / cnt).toFixed(4) : '?';
        resultEl.innerHTML = 'Ensemble prediction: <strong>' + avg + '</strong> (avg of ' + cnt + ' trees)';
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
})();
"""
