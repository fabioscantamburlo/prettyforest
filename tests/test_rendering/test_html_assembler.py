"""Unit tests for HTMLAssembler."""

import re

from prettyforest.rendering.html_assembler import HTMLAssembler


class TestSelfContainedHTML:
    def test_no_external_links(self):
        assembler = HTMLAssembler()
        html = assembler.assemble('<svg></svg>')

        # No external stylesheet links
        assert '<link' not in html or 'href="http' not in html
        # No external script sources
        external_scripts = re.findall(r'<script[^>]+src=', html)
        assert len(external_scripts) == 0

    def test_css_is_inlined(self):
        assembler = HTMLAssembler()
        html = assembler.assemble('<svg></svg>')

        assert '<style>' in html
        assert 'font-family' in html

    def test_js_is_inlined(self):
        assembler = HTMLAssembler()
        html = assembler.assemble('<svg></svg>')

        assert '<script>' in html
        assert 'addEventListener' in html

    def test_valid_html_structure(self):
        assembler = HTMLAssembler()
        html = assembler.assemble('<svg></svg>')

        assert '<!DOCTYPE html>' in html
        assert '<html' in html
        assert '</html>' in html
        assert '<head>' in html
        assert '</head>' in html
        assert '<body>' in html
        assert '</body>' in html

    def test_flow_mode_has_controls(self):
        assembler = HTMLAssembler()
        html = assembler.assemble('<svg></svg>', mode='flow')

        assert 'sample-index' in html
        assert 'highlight-btn' in html
        assert 'clear-btn' in html

    def test_blueprint_mode_no_input_controls(self):
        assembler = HTMLAssembler()
        html = assembler.assemble('<svg></svg>', mode='blueprint')

        # No input/button controls rendered in the HTML body
        assert '<input type="number" id="sample-index"' not in html
        assert '<button id="highlight-btn"' not in html
