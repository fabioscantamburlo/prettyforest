from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class CrownShape(Enum):
    OAK = "oak"
    BIRCH = "birch"
    MAPLE = "maple"


@dataclass
class ForestConfig:
    seed: int = 42
    season: Literal["summer", "autumn", "winter"] | None = None


@dataclass
class TreeVisuals:
    height: float
    trunk_width: float
    canopy_color: str
    crown_shape: CrownShape


@dataclass
class TreePaths:
    trunk: str
    branches: list[str] = field(default_factory=list)
    canopy: str = ""
