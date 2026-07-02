"""Generate organic SVG paths for trunk, branches, and canopy."""

from __future__ import annotations

import math
import random

from prettyforest.rendering.forest.constants import TRUNK_HEIGHT_FRACTION
from prettyforest.rendering.forest.models import CrownShape, TreePaths, TreeVisuals


class TreeShapeGenerator:
    def generate(self, visuals: TreeVisuals, rng: random.Random) -> TreePaths:
        trunk_h = visuals.height * TRUNK_HEIGHT_FRACTION
        canopy_h = visuals.height - trunk_h
        base_w = visuals.trunk_width
        top_w = base_w * 0.4

        trunk = _generate_trunk(base_w, top_w, trunk_h)
        branches = _generate_branches(top_w, canopy_h, visuals.crown_shape, rng)
        canopy = _generate_canopy(canopy_h, visuals.crown_shape, rng)

        return TreePaths(trunk=trunk, branches=branches, canopy=canopy)


def _generate_trunk(base_w: float, top_w: float, height: float) -> str:
    """Closed cubic Bezier trunk with taper."""
    bx = base_w / 2
    tx = top_w / 2
    # Origin at bottom-center of trunk
    # Points: bottom-left, top-left, top-right, bottom-right
    bl_x, bl_y = -bx, 0
    tl_x, tl_y = -tx, -height
    tr_x, tr_y = tx, -height
    br_x, br_y = bx, 0

    # Control points for organic curves
    cp1_x = bl_x + (tl_x - bl_x) * 0.3
    cp1_y = bl_y + (tl_y - bl_y) * 0.4
    cp2_x = tl_x - (tl_x - bl_x) * 0.1
    cp2_y = tl_y + height * 0.15

    cp3_x = tr_x + (br_x - tr_x) * 0.1
    cp3_y = tr_y + height * 0.15
    cp4_x = br_x - (br_x - tr_x) * 0.3
    cp4_y = br_y + (tr_y - br_y) * 0.4

    return (
        f"M {bl_x:.1f},{bl_y:.1f} "
        f"C {cp1_x:.1f},{cp1_y:.1f} {cp2_x:.1f},{cp2_y:.1f} {tl_x:.1f},{tl_y:.1f} "
        f"L {tr_x:.1f},{tr_y:.1f} "
        f"C {cp3_x:.1f},{cp3_y:.1f} {cp4_x:.1f},{cp4_y:.1f} {br_x:.1f},{br_y:.1f} "
        "Z"
    )


def _generate_branches(
    top_w: float, canopy_h: float, crown_shape: CrownShape, rng: random.Random
) -> list[str]:
    """Generate 2-5 branch paths fanning from trunk top into canopy."""
    num = rng.randint(2, 5)
    branches: list[str] = []
    canopy_radius = canopy_h * 0.4

    for i in range(num):
        angle = -60 + (120 / (num - 1)) * i if num > 1 else 0
        angle_rad = math.radians(angle)
        length = canopy_radius * rng.uniform(0.3, 0.6)

        end_x = math.sin(angle_rad) * length
        end_y = -(canopy_h * 0.3 + math.cos(angle_rad) * length * 0.5)

        cp_x = end_x * 0.5 + rng.uniform(-3, 3)
        cp_y = end_y * 0.6

        branches.append(
            f"M 0,0 Q {cp_x:.1f},{cp_y:.1f} {end_x:.1f},{end_y:.1f}"
        )

    return branches


def _generate_canopy(
    canopy_h: float, crown_shape: CrownShape, rng: random.Random
) -> str:
    """Generate irregular closed canopy path."""
    # Base ellipse dimensions per crown shape
    if crown_shape == CrownShape.OAK:
        rx = canopy_h * 0.5
        ry = canopy_h * 0.45
    elif crown_shape == CrownShape.BIRCH:
        rx = canopy_h * 0.25
        ry = canopy_h * 0.55
    else:  # MAPLE
        rx = canopy_h * 0.65
        ry = canopy_h * 0.3

    # Center of canopy (above trunk top)
    cy = -(canopy_h * 0.55)
    n_points = rng.randint(16, 24)
    points: list[tuple[float, float]] = []

    for i in range(n_points):
        angle = (2 * math.pi * i) / n_points
        # Perturb radius for organic texture
        perturb = rng.uniform(-0.15, 0.15)
        px = (rx * (1 + perturb)) * math.cos(angle)
        py = cy + (ry * (1 + perturb)) * math.sin(angle)
        points.append((px, py))

    # Build path with quadratic Bezier through points
    if not points:
        return ""

    parts = [f"M {points[0][0]:.1f},{points[0][1]:.1f}"]
    for i in range(len(points)):
        curr = points[i]
        nxt = points[(i + 1) % len(points)]
        mid_x = (curr[0] + nxt[0]) / 2
        mid_y = (curr[1] + nxt[1]) / 2
        parts.append(f"Q {curr[0]:.1f},{curr[1]:.1f} {mid_x:.1f},{mid_y:.1f}")
    parts.append("Z")

    return " ".join(parts)
