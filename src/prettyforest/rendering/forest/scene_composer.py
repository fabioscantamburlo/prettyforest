"""Arrange trees in a 2.5D isometric forest scene (Pokémon-style overhead perspective)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from prettyforest.rendering.forest.constants import SEASON_PALETTES
from prettyforest.rendering.forest.models import ForestConfig, TreePaths, TreeVisuals

# How many trees to display at once (scene is sized for this)
MAX_VISIBLE = 200

PADDING = 40
GROUND_MARGIN_TOP = 0.15
GROUND_MARGIN_BOTTOM = 0.05
MIN_SCALE = 0.45
MAX_SCALE = 1.0


@dataclass
class TreeMeta:
    index: int
    depth: int
    node_count: int
    n_leaves: int
    avg_purity: float | None = None
    pred_variance: float | None = None
    leaf_magnitude: float | None = None


class SceneComposer:
    def compose(
        self,
        tree_paths: list[TreePaths],
        tree_visuals: list[TreeVisuals],
        tree_metas: list[TreeMeta],
        config: ForestConfig,
        total_tree_count: int,
        rng: random.Random,
    ) -> str:
        n = len(tree_paths)
        visible_n = min(n, MAX_VISIBLE)

        # Scene sized for the visible window only
        width, height = self._compute_scene_size(visible_n)
        ground_top = height * GROUND_MARGIN_TOP
        ground_bottom = height * (1 - GROUND_MARGIN_BOTTOM)
        ground_h = ground_bottom - ground_top

        ground_color = "#8cc97a"
        ground_dark = _darken(ground_color, 0.88)
        sky_color = "#dceef5"
        if config.season and config.season in SEASON_PALETTES:
            palette = SEASON_PALETTES[config.season]
            ground_color = palette.get("ground", ground_color)
            ground_dark = _darken(ground_color, 0.88)

        parts: list[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="100%" height="100%" '
            f'viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'id="forest-svg">'
        )
        parts.append("<title>PrettyForest — Aesthetic Forest View</title>")
        parts.append(
            self._build_defs(ground_color, ground_dark, sky_color, config.season)
        )

        # Sky + ground
        parts.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#sky-gradient)"/>'
        )
        parts.append(
            self._render_ground(
                width,
                height,
                ground_top,
                ground_bottom,
                ground_color,
                rng,
                config.season,
            )
        )

        # Compute positions for ALL trees (using visible_n worth of space, cycling positions)
        positions = self._compute_placements(
            visible_n, width, ground_top, ground_bottom, rng
        )

        # Render all trees — first MAX_VISIBLE are visible, rest are hidden
        # Sort visible set back to front for proper overlap
        vis_order = sorted(range(visible_n), key=lambda i: positions[i][1])

        for i in vis_order:
            x, y = positions[i]
            depth_t = (y - ground_top) / ground_h if ground_h > 0 else 0.5
            scale = MIN_SCALE + depth_t * (MAX_SCALE - MIN_SCALE)
            meta = tree_metas[i] if i < len(tree_metas) else None
            parts.append(
                self._render_tree(
                    tree_paths[i],
                    tree_visuals[i],
                    x,
                    y,
                    scale,
                    config.season,
                    meta,
                    hidden=False,
                )
            )

        # Render remaining trees as hidden (no position needed, they'll be placed by JS when paged in)
        for i in range(visible_n, n):
            meta = tree_metas[i] if i < len(tree_metas) else None
            # Place at 0,0 hidden — JS will assign position when paged in
            parts.append(
                self._render_tree(
                    tree_paths[i],
                    tree_visuals[i],
                    0,
                    0,
                    0.6,
                    config.season,
                    meta,
                    hidden=True,
                )
            )

        # Legend (info button)
        parts.append(self._render_legend(width))
        parts.append("</svg>")
        return "\n".join(parts)

    def _compute_scene_size(self, n: int) -> tuple[int, int]:
        if n <= 1:
            return 450, 450
        base_w, base_h = 800, 620
        factor = max(1.0, math.sqrt(n / 30))
        scale_f = min(factor, 1.5)
        return int(base_w * scale_f), int(base_h * scale_f)

    def _build_defs(
        self, ground_color: str, ground_dark: str, sky_color: str, season: str | None
    ) -> str:
        return (
            "<defs>"
            '<linearGradient id="sky-gradient" x1="0%" y1="0%" x2="0%" y2="100%">'
            f'<stop offset="0%" stop-color="{sky_color}"/>'
            f'<stop offset="35%" stop-color="{_lighten(sky_color, 1.05)}"/>'
            f'<stop offset="100%" stop-color="{ground_color}" stop-opacity="0.3"/>'
            "</linearGradient>"
            '<linearGradient id="ground-gradient" x1="0%" y1="0%" x2="0%" y2="100%">'
            f'<stop offset="0%" stop-color="{ground_dark}"/>'
            f'<stop offset="100%" stop-color="{ground_color}"/>'
            "</linearGradient>"
            '<filter id="tree-shadow" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur in="SourceAlpha" stdDeviation="3"/>'
            '<feOffset dx="2" dy="4" result="shadow"/>'
            '<feFlood flood-color="rgba(0,0,0,0.2)"/>'
            '<feComposite in2="shadow" operator="in"/>'
            '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>'
            "</filter>"
            "</defs>"
        )

    def _render_ground(
        self,
        width: int,
        height: int,
        ground_top: float,
        ground_bottom: float,
        ground_color: str,
        rng: random.Random,
        season: str | None,
    ) -> str:
        parts: list[str] = []
        parts.append(
            f'<rect x="0" y="{ground_top - 10:.0f}" width="{width}" '
            f'height="{height - ground_top + 10:.0f}" fill="url(#ground-gradient)"/>'
        )
        patch_color = _lighten(ground_color, 1.08)
        dark_patch = _darken(ground_color, 0.92)
        n_patches = min(50, max(20, width * height // 20000))

        for _ in range(n_patches):
            px = rng.uniform(0, width)
            py = rng.uniform(ground_top, ground_bottom)
            depth_t = (py - ground_top) / (ground_bottom - ground_top)
            patch_w = rng.uniform(15, 45) * (0.4 + depth_t * 0.6)
            patch_h = patch_w * 0.3
            color = rng.choice([patch_color, dark_patch, ground_color])
            opacity = rng.uniform(0.12, 0.35)
            parts.append(
                f'<ellipse data-patch="1" cx="{px:.0f}" cy="{py:.0f}" rx="{patch_w:.0f}" ry="{patch_h:.0f}" '
                f'fill="{color}" opacity="{opacity:.2f}"/>'
            )
        return "\n".join(parts)

    def _compute_placements(
        self,
        n: int,
        width: int,
        ground_top: float,
        ground_bottom: float,
        rng: random.Random,
    ) -> list[tuple[float, float]]:
        if n == 0:
            return []

        margin_x = width * 0.07
        margin_y_top = 25
        margin_y_bot = 25
        usable_w = width - 2 * margin_x
        usable_h = (ground_bottom - ground_top) - margin_y_top - margin_y_bot

        area = usable_w * usable_h
        min_dist = max(18, min(45, math.sqrt(area / max(n, 1)) * 0.5))

        placements: list[tuple[float, float]] = []
        attempts = 0
        max_attempts = n * 100

        while len(placements) < n and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(0, usable_w)
            y = rng.uniform(0, usable_h)
            too_close = False
            for px, py in placements:
                if math.hypot(x - px, (y - py) * 1.5) < min_dist:
                    too_close = True
                    break
            if not too_close:
                placements.append((x, y))

        while len(placements) < n:
            placements.append((rng.uniform(0, usable_w), rng.uniform(0, usable_h)))

        # Center
        if placements:
            xs = [p[0] for p in placements]
            ys = [p[1] for p in placements]
            off_x = usable_w / 2 - (min(xs) + max(xs)) / 2
            off_y = usable_h / 2 - (min(ys) + max(ys)) / 2
            placements = [
                (margin_x + x + off_x, ground_top + margin_y_top + y + off_y)
                for x, y in placements
            ]

        return placements

    def _render_tree(
        self,
        paths: TreePaths,
        visuals: TreeVisuals,
        x: float,
        y: float,
        scale: float,
        season: str | None,
        meta: TreeMeta | None,
        hidden: bool = False,
    ) -> str:
        data_attrs = ""
        if meta:
            data_attrs = (
                f'data-tree-idx="{meta.index}" '
                f'data-depth="{meta.depth}" '
                f'data-nodes="{meta.node_count}" '
                f'data-leaves="{meta.n_leaves}" '
            )
            if meta.avg_purity is not None:
                data_attrs += f'data-purity="{meta.avg_purity:.3f}" '
            if meta.pred_variance is not None:
                data_attrs += f'data-variance="{meta.pred_variance:.4f}" '
            if meta.leaf_magnitude is not None:
                data_attrs += f'data-magnitude="{meta.leaf_magnitude:.4f}" '

        hidden_cls = " hidden" if hidden else ""
        parts: list[str] = []
        parts.append(
            f'<g class="visual-tree{hidden_cls}" '
            f'transform="translate({x:.1f},{y:.1f}) scale({scale:.3f})" '
            f'filter="url(#tree-shadow)" {data_attrs}>'
        )

        shadow_rx = visuals.trunk_width * 1.8
        shadow_ry = shadow_rx * 0.35
        parts.append(
            f'<ellipse cx="0" cy="4" rx="{shadow_rx:.1f}" ry="{shadow_ry:.1f}" fill="rgba(0,0,0,0.15)"/>'
        )

        trunk_color = "#5D4037" if season != "winter" else "#696969"
        parts.append(
            f'<path class="trunk" d="{paths.trunk}" fill="{trunk_color}" stroke="#3E2723" stroke-width="0.5"/>'
        )

        branch_color = "#795548" if season != "winter" else "#808080"
        branch_width = max(1.5, visuals.trunk_width * 0.2)
        for branch in paths.branches:
            parts.append(
                f'<path class="branch" d="{branch}" fill="none" stroke="{branch_color}" '
                f'stroke-width="{branch_width:.1f}" stroke-linecap="round"/>'
            )

        if paths.canopy:
            parts.append(
                f'<path class="canopy" d="{paths.canopy}" fill="{visuals.canopy_color}" '
                f'stroke="{_darken(visuals.canopy_color, 0.7)}" stroke-width="1.5" opacity="0.9"/>'
            )

        parts.append("</g>")
        return "\n".join(parts)

    def _render_legend(self, width: int) -> str:
        lx = width - 280
        ly = 12
        return (
            f'<g class="legend" id="legend-panel" transform="translate({lx},{ly})" style="display:none">'
            '<rect x="0" y="0" width="270" height="92" fill="rgba(255,255,255,0.95)" rx="6" stroke="#ccc" stroke-width="1"/>'
            '<text x="10" y="16" font-size="10" fill="#333" font-weight="bold">How to read this forest</text>'
            '<text x="10" y="32" font-size="9" fill="#555">↕ Taller tree = deeper (more decision splits)</text>'
            '<text x="10" y="46" font-size="9" fill="#555">↔ Thicker trunk = more nodes (complex tree)</text>'
            '<text x="10" y="60" font-size="9" fill="#555">🟢 Green canopy = pure / low variance</text>'
            '<text x="10" y="74" font-size="9" fill="#555">🟡 Amber canopy = impure / high variance</text>'
            '<text x="10" y="88" font-size="9" fill="#888">Hover any tree for detailed stats</text>'
            "</g>"
        )


def _darken(hex_color: str, factor: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"#{max(0,int(r*factor)):02x}{max(0,int(g*factor)):02x}{max(0,int(b*factor)):02x}"


def _lighten(hex_color: str, factor: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"#{min(255,int(r*factor)):02x}{min(255,int(g*factor)):02x}{min(255,int(b*factor)):02x}"
