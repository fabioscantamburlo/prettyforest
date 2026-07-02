"""Render ensemble visualizations with all trees and navigation."""

from __future__ import annotations

from prettyforest.models import (
    EnsembleMeta,
    EnsembleType,
    FlowResult,
    UnifiedTree,
)
from prettyforest.rendering.html_assembler import HTMLAssembler
from prettyforest.rendering.layout_engine import LayoutEngine, compute_initial_collapse_state
from prettyforest.rendering.svg_renderer import SVGRenderer


class EnsembleRenderer:
    def __init__(self):
        self._layout = LayoutEngine()
        self._svg = SVGRenderer()
        self._html = HTMLAssembler()

    def render(
        self,
        trees: list[UnifiedTree],
        ensemble_type: EnsembleType,
        flow_results: list[FlowResult] | None = None,
        cumulative_contributions: list[float] | None = None,
        vote_proportions: dict[str, float] | None = None,
    ) -> str:
        n = len(trees)
        tree_svgs: list[str] = []

        for i, tree in enumerate(trees):
            collapse_state = compute_initial_collapse_state(tree)
            layout = self._layout.compute_layout(tree, collapse_state)
            flow = flow_results[i] if flow_results and i < len(flow_results) else None
            svg = self._svg.render(tree, layout, flow=flow, collapse_state=collapse_state)

            # Add tree header
            header = f"<h3>Tree {i}"
            if ensemble_type == EnsembleType.ADDITIVE and cumulative_contributions:
                if i < len(cumulative_contributions):
                    header += f" (cumulative: {cumulative_contributions[i]:.4f})"
            header += "</h3>"
            tree_svgs.append(f'{header}\n<div class="tree-panel" data-tree-index="{i}">{svg}</div>')

        # Vote proportions summary for vote-based ensembles
        vote_summary = ""
        if ensemble_type == EnsembleType.VOTE_BASED and vote_proportions:
            items = [f"{cls}: {prop*100:.1f}%" for cls, prop in vote_proportions.items()]
            vote_summary = (
                '<div class="vote-summary">'
                f'<strong>Ensemble Vote:</strong> {", ".join(items)}'
                "</div>"
            )

        # Navigation controls
        nav = self._build_navigation(n)

        # Build full HTML
        ensemble_meta = EnsembleMeta(
            ensemble_type=ensemble_type,
            tree_count=n,
            cumulative_contributions=cumulative_contributions,
            vote_proportions=vote_proportions,
        )

        body = f"""
{vote_summary}
{nav}
<div class="ensemble-container">
{"".join(tree_svgs)}
</div>
"""
        return self._wrap_html(body, ensemble_meta)

    def _build_navigation(self, n_trees: int) -> str:
        options = "".join(
            f'<option value="{i}">Tree {i}</option>' for i in range(n_trees)
        )
        return (
            '<div class="ensemble-nav">'
            f'<select id="tree-selector">{options}</select>'
            f'<span class="tree-count">{n_trees} trees</span>'
            "</div>"
        )

    def _wrap_html(self, body: str, ensemble_meta: EnsembleMeta) -> str:
        css = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #fafafa; }
.ensemble-container { display: flex; flex-wrap: wrap; gap: 20px; }
.tree-panel { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; overflow: auto; }
.ensemble-nav { margin-bottom: 16px; }
.ensemble-nav select { padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; }
.tree-count { margin-left: 12px; color: #666; }
.vote-summary { margin-bottom: 16px; padding: 12px; background: #e3f2fd; border-radius: 6px; }
h3 { margin: 8px 0; color: #333; }
"""
        js = """
(function() {
  const selector = document.getElementById('tree-selector');
  if (!selector) return;
  const panels = document.querySelectorAll('.tree-panel');

  selector.addEventListener('change', function() {
    const idx = parseInt(this.value, 10);
    panels.forEach((p, i) => {
      p.style.display = (i === idx) ? '' : 'none';
    });
  });

  // Initially show all trees (scrollable)
})();
"""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            "<title>PrettyForest - Ensemble Visualization</title>\n"
            '<meta charset="utf-8"/>\n'
            f"<style>{css}</style>\n"
            "</head>\n"
            "<body>\n"
            "<h2>PrettyForest - Ensemble Visualization</h2>\n"
            f"{body}\n"
            f"<script>{js}</script>\n"
            "</body>\n"
            "</html>"
        )
