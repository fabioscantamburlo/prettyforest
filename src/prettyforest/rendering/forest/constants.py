MIN_HEIGHT = 80
MAX_HEIGHT = 250
MIN_TRUNK_WIDTH = 6
MAX_TRUNK_WIDTH = 20

TRUNK_HEIGHT_FRACTION = 0.35

SEASON_PALETTES: dict[str, dict[str, list[str] | str]] = {
    "summer": {
        "canopy": ["#2E8B57", "#3CB371", "#6B8E23", "#228B22", "#32CD32"],
        "ground": "#8cc97a",
    },
    "autumn": {
        "canopy": ["#D2691E", "#B22222", "#DAA520", "#CD853F", "#FF8C00"],
        "ground": "#8B6914",
    },
    "winter": {
        "canopy": [],
        "ground": "#B0C4DE",
        "branch_tones": ["#696969", "#808080", "#A9A9A9", "#8B7765"],
    },
}

METRIC_GRADIENT = ("#228B22", "#DAA520")
