"""Map model properties to visual dimensions."""

from __future__ import annotations

import random

from prettyforest.models import EnsembleType, UnifiedTree
from prettyforest.rendering.forest.constants import (
    MAX_HEIGHT,
    MAX_TRUNK_WIDTH,
    METRIC_GRADIENT,
    MIN_HEIGHT,
    MIN_TRUNK_WIDTH,
    SEASON_PALETTES,
)
from prettyforest.rendering.forest.models import CrownShape, TreeVisuals


class VisualMapper:
    def map_trees(
        self,
        trees: list[UnifiedTree],
        rng: random.Random,
        season: str | None = None,
        ensemble_type: EnsembleType = EnsembleType.SINGLE,
    ) -> list[TreeVisuals]:
        depths = [t.max_depth for t in trees]
        counts = [t.node_count for t in trees]

        min_d, max_d = (min(depths), max(depths)) if depths else (1, 1)
        min_c, max_c = (min(counts), max(counts)) if counts else (1, 1)

        results: list[TreeVisuals] = []
        for tree in trees:
            height = _lerp(tree.max_depth, min_d, max_d, MIN_HEIGHT, MAX_HEIGHT)
            trunk_width = _lerp(
                tree.node_count, min_c, max_c, MIN_TRUNK_WIDTH, MAX_TRUNK_WIDTH
            )
            canopy_color = _compute_color(tree, season, rng, ensemble_type)
            crown_shape = rng.choice(list(CrownShape))
            results.append(
                TreeVisuals(
                    height=height,
                    trunk_width=trunk_width,
                    canopy_color=canopy_color,
                    crown_shape=crown_shape,
                )
            )
        return results


def _lerp(
    value: float, src_min: float, src_max: float, dst_min: float, dst_max: float
) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    t = (value - src_min) / (src_max - src_min)
    return dst_min + t * (dst_max - dst_min)


def _compute_color(
    tree: UnifiedTree,
    season: str | None,
    rng: random.Random,
    ensemble_type: EnsembleType,
) -> str:
    if season and season in SEASON_PALETTES:
        palette = SEASON_PALETTES[season]
        canopy_colors = palette.get("canopy", [])
        if canopy_colors:
            return rng.choice(canopy_colors)
        return "#808080"

    metric = _compute_metric(tree, ensemble_type)
    return _interpolate_color(METRIC_GRADIENT[0], METRIC_GRADIENT[1], metric)


def _compute_metric(tree: UnifiedTree, ensemble_type: EnsembleType) -> float:
    """Return 0.0 (green/good) to 1.0 (amber/high variance or impure).

    Classification (vote-based): uses leaf purity (max class proportion).
    Classification (additive): uses normalized leaf magnitude (gradient strength).
    Regression: uses normalized prediction variance across leaves.
    """
    leaves = [n for n in tree.iter_nodes() if n.is_leaf]
    if not leaves:
        return 0.5

    if tree.is_classifier:
        if ensemble_type == EnsembleType.VOTE_BASED:
            # Real class proportions → purity makes sense
            purities = []
            for leaf in leaves:
                if leaf.class_distribution:
                    max_prop = max(leaf.class_distribution.values())
                    if 0 <= max_prop <= 1:
                        purities.append(max_prop)
            if not purities:
                return 0.5
            avg_purity = sum(purities) / len(purities)
            return 1.0 - avg_purity  # high purity → green (0), low → amber (1)
        else:
            # Additive: leaves store gradients/log-odds, use magnitude spread
            magnitudes = []
            for leaf in leaves:
                if leaf.prediction_value is not None:
                    magnitudes.append(abs(leaf.prediction_value))
                elif leaf.class_distribution:
                    magnitudes.append(
                        max(abs(v) for v in leaf.class_distribution.values())
                    )
            if not magnitudes:
                return 0.5
            # Normalize: low magnitude = early tree (green), high = later corrections (amber)
            max_mag = max(magnitudes) if magnitudes else 1.0
            avg_mag = sum(magnitudes) / len(magnitudes)
            return min(1.0, avg_mag / max_mag) if max_mag > 0 else 0.5
    else:
        # Regression: variance of leaf predictions
        values = [
            leaf.prediction_value
            for leaf in leaves
            if leaf.prediction_value is not None
        ]
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        max_var = max(abs(v) for v in values) ** 2 if values else 1.0
        return min(1.0, variance / max_var) if max_var > 0 else 0.0


def _interpolate_color(c1: str, c2: str, t: float) -> str:
    """Interpolate between two hex colors. t=0 → c1, t=1 → c2."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"
