"""Generate SVG markup for tree visualizations."""

from __future__ import annotations

import math

from prettyforest.models import (
    FlowResult,
    NodePosition,
    PathStep,
    UnifiedNode,
    UnifiedTree,
)
from prettyforest.rendering.layout_engine import count_descendants


def _purity_tier(class_distribution: dict[str, float]) -> str:
    max_prop = max(class_distribution.values())
    if max_prop > 0.9:
        return "high"
    if max_prop <= 0.6:
        return "low"
    return "mid"


PURITY_STYLES = {
    "high": {"fill": "#c8e6c9", "stroke": "#2e7d32", "stroke-dasharray": "none"},
    "mid": {"fill": "#fff9c4", "stroke": "#f9a825", "stroke-dasharray": "none"},
    "low": {"fill": "#ffcdd2", "stroke": "#c62828", "stroke-dasharray": "5,3"},
}


class SVGRenderer:
    def render(
        self,
        tree: UnifiedTree,
        layout: dict[str, NodePosition],
        flow: FlowResult | None = None,
        highlighted_path: list[PathStep] | None = None,
        collapse_state: dict[str, bool] | None = None,
    ) -> str:
        if collapse_state is None:
            collapse_state = {}

        highlighted_ids = set()
        highlighted_edges = set()
        if highlighted_path:
            for step in highlighted_path:
                highlighted_ids.add(step.node_id)
            for i in range(len(highlighted_path) - 1):
                highlighted_edges.add(
                    (highlighted_path[i].node_id, highlighted_path[i + 1].node_id)
                )

        # Compute SVG bounds
        if not layout:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"></svg>'

        min_x = min(p.x for p in layout.values())
        min_y = min(p.y for p in layout.values())
        max_x = max(p.x + p.width for p in layout.values())
        max_y = max(p.y + p.height for p in layout.values())
        padding = 40
        svg_w = max_x - min_x + padding * 2
        svg_h = max_y - min_y + padding * 2

        parts: list[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_w:.0f}" height="{svg_h:.0f}" '
            f'viewBox="{min_x - padding} {min_y - padding} {svg_w} {svg_h}">'
        )

        # Render edges first (below nodes)
        parts.append(self._render_edges(
            tree.root, layout, flow, highlighted_edges, highlighted_ids, collapse_state
        ))

        # Render nodes
        parts.append(self._render_nodes(
            tree, layout, flow, highlighted_ids, collapse_state
        ))

        # Purity legend (only in flow mode with classifier)
        if flow and tree.is_classifier:
            parts.append(self._render_purity_legend(min_x - padding + 10, min_y - padding + 10))

        parts.append("</svg>")
        return "\n".join(parts)

    def _render_edges(
        self,
        node: UnifiedNode,
        layout: dict[str, NodePosition],
        flow: FlowResult | None,
        highlighted_edges: set[tuple[str, str]],
        highlighted_ids: set[str],
        collapse_state: dict[str, bool],
    ) -> str:
        parts: list[str] = []
        self._collect_edges(
            node, layout, flow, highlighted_edges, highlighted_ids, collapse_state, parts
        )
        return "\n".join(parts)

    def _collect_edges(
        self,
        node: UnifiedNode,
        layout: dict[str, NodePosition],
        flow: FlowResult | None,
        highlighted_edges: set[tuple[str, str]],
        highlighted_ids: set[str],
        collapse_state: dict[str, bool],
        parts: list[str],
    ) -> None:
        if node.is_leaf or node.node_id not in layout:
            return
        if collapse_state.get(node.node_id, False):
            return

        parent_pos = layout[node.node_id]
        px = parent_pos.x + parent_pos.width / 2
        py = parent_pos.y + parent_pos.height

        for child, label in [
            (node.left_child, "✓"),
            (node.right_child, "✗"),
        ]:
            if child.node_id not in layout:
                continue
            child_pos = layout[child.node_id]
            cx = child_pos.x + child_pos.width / 2
            cy = child_pos.y

            thickness = 2
            if flow:
                edge_key = (node.node_id, child.node_id)
                fraction = flow.edge_fractions.get(edge_key, 0)
                thickness = 1 + 19 * fraction
                thickness = max(1, min(20, thickness))

            is_highlighted = (node.node_id, child.node_id) in highlighted_edges
            opacity = 1.0
            if highlighted_ids and not is_highlighted:
                opacity = 0.3
            color = "#1565c0" if is_highlighted else "#616161"

            parts.append(
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                f'stroke="{color}" stroke-width="{thickness:.1f}" opacity="{opacity}" '
                f'data-edge="{node.node_id}-{child.node_id}"/>'
            )
            # Edge label
            mx = (px + cx) / 2
            my = (py + cy) / 2
            parts.append(
                f'<text x="{mx:.1f}" y="{my:.1f}" font-size="12" '
                f'text-anchor="middle" fill="{color}" opacity="{opacity}">{label}</text>'
            )

        # Recurse
        if node.left_child:
            self._collect_edges(
                node.left_child, layout, flow, highlighted_edges, highlighted_ids,
                collapse_state, parts,
            )
        if node.right_child:
            self._collect_edges(
                node.right_child, layout, flow, highlighted_edges, highlighted_ids,
                collapse_state, parts,
            )

    def _render_nodes(
        self,
        tree: UnifiedTree,
        layout: dict[str, NodePosition],
        flow: FlowResult | None,
        highlighted_ids: set[str],
        collapse_state: dict[str, bool],
    ) -> str:
        parts: list[str] = []
        for node in tree.iter_nodes():
            if node.node_id not in layout:
                continue
            pos = layout[node.node_id]

            opacity = 1.0
            if highlighted_ids and node.node_id not in highlighted_ids:
                opacity = 0.3

            # Determine styles
            fill = "#ffffff"
            stroke = "#424242"
            stroke_dash = "none"

            if node.is_leaf and flow and tree.is_classifier:
                # Compute purity from flow data
                dist = flow.leaf_distributions.get(node.node_id)
                if dist and dist.class_proportions:
                    tier = _purity_tier(dist.class_proportions)
                    style = PURITY_STYLES[tier]
                    fill = style["fill"]
                    stroke = style["stroke"]
                    stroke_dash = style["stroke-dasharray"]

            is_highlighted = node.node_id in highlighted_ids
            if is_highlighted:
                stroke = "#1565c0"

            parts.append(
                f'<rect x="{pos.x:.1f}" y="{pos.y:.1f}" '
                f'width="{pos.width:.1f}" height="{pos.height:.1f}" '
                f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2" '
                f'stroke-dasharray="{stroke_dash}" opacity="{opacity}" '
                f'data-node-id="{node.node_id}"/>'
            )

            # Node content
            cx = pos.x + pos.width / 2
            cy = pos.y + pos.height / 2
            lines = self._node_text(node, flow, collapse_state)
            line_height = 14
            start_y = cy - (len(lines) - 1) * line_height / 2

            for i, line in enumerate(lines):
                parts.append(
                    f'<text x="{cx:.1f}" y="{start_y + i * line_height:.1f}" '
                    f'font-size="11" text-anchor="middle" fill="#212121" '
                    f'opacity="{opacity}">{line}</text>'
                )

            # Mini chart for flow mode leaves
            if node.is_leaf and flow:
                chart = self._render_leaf_chart(node, flow, tree.is_classifier, pos)
                if chart:
                    parts.append(chart)

        return "\n".join(parts)

    def _node_text(
        self,
        node: UnifiedNode,
        flow: FlowResult | None,
        collapse_state: dict[str, bool],
    ) -> list[str]:
        lines: list[str] = []

        if node.is_leaf:
            if node.class_distribution:
                majority = max(node.class_distribution, key=node.class_distribution.get)
                lines.append(f"Class: {majority}")
            elif node.prediction_value is not None:
                lines.append(f"Value: {node.prediction_value:.4f}")
        else:
            lines.append(f"{node.feature_name} {node.comparison_op.value} {node.threshold:.4f}")
            if collapse_state.get(node.node_id, False):
                desc = count_descendants(node)
                lines.append(f"[+{desc} hidden]")

        if flow and node.node_id in flow.sample_counts:
            lines.append(f"n={flow.sample_counts[node.node_id]}")

        return lines

    def _render_leaf_chart(
        self,
        node: UnifiedNode,
        flow: FlowResult,
        is_classifier: bool,
        pos: NodePosition,
    ) -> str:
        dist = flow.leaf_distributions.get(node.node_id)
        if not dist:
            return ""

        cx = pos.x + pos.width / 2
        chart_y = pos.y + pos.height + 5
        chart_size = 30

        if is_classifier and dist.class_proportions:
            return self._render_pie_chart(cx, chart_y, chart_size, dist.class_proportions)
        elif not is_classifier and dist.histogram_counts:
            return self._render_histogram(cx, chart_y, chart_size, dist.histogram_counts)
        return ""

    def _render_pie_chart(
        self, cx: float, cy: float, radius: float, proportions: dict[str, float]
    ) -> str:
        colors = ["#42a5f5", "#ef5350", "#66bb6a", "#ffa726", "#ab47bc", "#26c6da"]
        parts = [f'<g class="pie-chart">']
        start_angle = 0.0

        for i, (_, prop) in enumerate(proportions.items()):
            if prop <= 0:
                continue
            end_angle = start_angle + prop * 360
            large_arc = 1 if (end_angle - start_angle) > 180 else 0

            x1 = cx + radius * math.cos(math.radians(start_angle - 90))
            y1 = cy + radius * math.sin(math.radians(start_angle - 90))
            x2 = cx + radius * math.cos(math.radians(end_angle - 90))
            y2 = cy + radius * math.sin(math.radians(end_angle - 90))

            color = colors[i % len(colors)]
            if prop >= 1.0:
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy + radius:.1f}" r="{radius}" fill="{color}"/>'
                )
            else:
                parts.append(
                    f'<path d="M {cx:.1f} {cy:.1f} L {x1:.1f} {y1:.1f} '
                    f'A {radius} {radius} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z" '
                    f'fill="{color}"/>'
                )
            start_angle = end_angle

        parts.append("</g>")
        return "\n".join(parts)

    def _render_histogram(
        self, cx: float, cy: float, size: float, counts: list[int]
    ) -> str:
        if not counts:
            return ""
        max_count = max(counts) if max(counts) > 0 else 1
        n_bins = len(counts)
        bar_width = size * 2 / n_bins
        start_x = cx - size

        parts = [f'<g class="histogram">']
        for i, count in enumerate(counts):
            bar_height = (count / max_count) * size
            x = start_x + i * bar_width
            y = cy + size - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="#42a5f5" stroke="#1565c0" '
                f'stroke-width="0.5"/>'
            )
        parts.append("</g>")
        return "\n".join(parts)

    def _render_purity_legend(self, x: float, y: float) -> str:
        parts = [f'<g class="purity-legend" transform="translate({x:.0f},{y:.0f})">']
        items = [
            ("High purity (>90%)", PURITY_STYLES["high"]),
            ("Mid purity (60-90%)", PURITY_STYLES["mid"]),
            ("Low purity (≤60%)", PURITY_STYLES["low"]),
        ]
        for i, (label, style) in enumerate(items):
            iy = i * 22
            parts.append(
                f'<rect x="0" y="{iy}" width="16" height="16" rx="2" '
                f'fill="{style["fill"]}" stroke="{style["stroke"]}" '
                f'stroke-dasharray="{style["stroke-dasharray"]}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="22" y="{iy + 12}" font-size="11" fill="#424242">{label}</text>'
            )
        parts.append("</g>")
        return "\n".join(parts)
