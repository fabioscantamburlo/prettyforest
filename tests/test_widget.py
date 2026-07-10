"""Unit tests for PrettyForest anywidget integration and environment detection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prettyforest.api import _in_notebook, _handle_output
from prettyforest.widget import PrettyForestWidget, forest_widget


class TestPrettyForestWidget:
    def test_widget_static_assets_exist(self):
        """Verify that the static ESM and CSS assets pointed to by the widget exist."""
        esm_path = Path(getattr(PrettyForestWidget._esm, "_path", PrettyForestWidget._esm))
        css_path = Path(getattr(PrettyForestWidget._css, "_path", PrettyForestWidget._css))
        assert esm_path.exists()
        assert css_path.exists()
        assert esm_path.name == "forest.js"
        assert css_path.name == "forest.css"

    def test_widget_initialization(self):
        """Verify widget traits and default values."""
        sample_html = "<div class='pf-root'>Test Forest</div>"
        widget = PrettyForestWidget(html_content=sample_html)
        assert widget.html_content == sample_html
        assert widget.season == ""
        assert widget.sample_idx == -1

    def test_forest_widget_wrapper(self):
        """Verify forest_widget wraps HTML string and is idempotent for existing widget."""
        sample_html = "<div>Forest HTML</div>"
        w1 = forest_widget(sample_html)
        assert isinstance(w1, PrettyForestWidget)
        assert w1.html_content == sample_html

        w2 = forest_widget(w1)
        assert w2 is w1


class TestNotebookEnvironmentDetection:
    def test_not_in_notebook_by_default(self):
        """Standard pytest environment is not a notebook."""
        assert _in_notebook() is False

    def test_detect_marimo_notebook(self):
        """Detect Marimo notebook context when running_in_notebook is True."""
        mock_marimo = MagicMock()
        mock_marimo.running_in_notebook.return_value = True
        with patch.dict("sys.modules", {"marimo": mock_marimo}):
            assert _in_notebook() is True

    def test_detect_google_colab(self):
        """Detect Google Colab context."""
        with patch.dict("sys.modules", {"google.colab": MagicMock()}):
            assert _in_notebook() is True

    def test_handle_output_non_notebook(self):
        """When not in notebook, _handle_output returns raw html string."""
        sample_html = "<html>Forest</html>"
        result = _handle_output(sample_html, output_path=None)
        assert result == sample_html
