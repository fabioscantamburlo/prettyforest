"""PrettyForest anywidget — works in Jupyter, Marimo, Colab, molab."""

from __future__ import annotations

from pathlib import Path

import anywidget
import traitlets

_STATIC = Path(__file__).parent / "static"


class PrettyForestWidget(anywidget.AnyWidget):
    """Interactive forest widget powered by anywidget.

    Renders the PrettyForest visualization directly in the notebook DOM
    (no iframe needed). Works in JupyterLab, Marimo, Colab, and molab.
    """

    _esm = _STATIC / "forest.js"
    _css = _STATIC / "forest.css"


    # The full HTML content (toolbar + SVG + scripts)
    html_content = traitlets.Unicode("").tag(sync=True)

    # Synced properties for live Python→JS control
    season = traitlets.Unicode("").tag(sync=True)
    sample_idx = traitlets.Int(-1).tag(sync=True)


def forest_widget(html: str | PrettyForestWidget) -> PrettyForestWidget:
    """Wrap PrettyForest HTML in an anywidget for notebook display.

    Usage:
        from prettyforest import prettygrow, forest_widget

        html = prettygrow(model, data=X)
        forest_widget(html)  # displays in Jupyter/Marimo/Colab
    """
    if isinstance(html, PrettyForestWidget):
        return html
    return PrettyForestWidget(html_content=html)
