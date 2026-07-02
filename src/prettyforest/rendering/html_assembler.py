"""Assemble self-contained HTML with inlined CSS and JavaScript."""

from __future__ import annotations

from prettyforest.models import EnsembleMeta


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #fafafa; }
.tree-container { overflow: auto; background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; }
svg rect[data-node-id] { cursor: pointer; transition: opacity 0.3s ease; }
svg line { transition: opacity 0.3s ease, stroke-width 0.3s ease; }
svg text { pointer-events: none; }
.controls { margin-bottom: 16px; display: flex; gap: 12px; align-items: center; }
.controls input { padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; width: 120px; }
.controls button { padding: 6px 16px; border: none; border-radius: 4px; background: #1565c0; color: white; cursor: pointer; }
.controls button:hover { background: #0d47a1; }
.error-msg { color: #c62828; font-size: 12px; margin-top: 4px; }
"""

_JS = r"""
(function() {
  var svg = document.querySelector('svg');
  if (!svg) return;

  var collapseState = {};
  var treeEl = document.getElementById('tree-data');
  var flowEl = document.getElementById('flow-data');
  var treeData = treeEl ? JSON.parse(treeEl.textContent) : {};
  var flowDataRaw = flowEl ? JSON.parse(flowEl.textContent) : {samples:{},total_rows:0};
  var flowSamples = flowDataRaw.samples || {};
  var nTotalRows = flowDataRaw.total_rows || 0;

  // --- Collapse/Expand ---
  svg.addEventListener('click', function(e) {
    var rect = e.target.closest('rect[data-node-id]');
    if (!rect) return;
    var nodeId = rect.dataset.nodeId;
    var node = findNode(treeData, nodeId);
    if (!node || node.type === 'leaf') return;
    collapseState[nodeId] = !collapseState[nodeId];
    toggleDescendants(nodeId, collapseState[nodeId]);
  });

  function findNode(tree, nodeId) {
    if (!tree || !tree.tree || !tree.tree.root) return null;
    var queue = [tree.tree.root];
    while (queue.length > 0) {
      var n = queue.shift();
      if (n.node_id === nodeId) return n;
      if (n.left) queue.push(n.left);
      if (n.right) queue.push(n.right);
    }
    return null;
  }

  function getDescendantIds(nodeId) {
    var node = findNode(treeData, nodeId);
    if (!node) return [];
    var ids = [];
    var stack = [];
    if (node.left) stack.push(node.left);
    if (node.right) stack.push(node.right);
    while (stack.length > 0) {
      var n = stack.pop();
      ids.push(n.node_id);
      if (n.left) stack.push(n.left);
      if (n.right) stack.push(n.right);
    }
    return ids;
  }

  function toggleDescendants(nodeId, hidden) {
    var ids = getDescendantIds(nodeId);
    var idSet = new Set(ids);
    svg.querySelectorAll('rect[data-node-id]').forEach(function(el) {
      if (idSet.has(el.dataset.nodeId)) el.style.display = hidden ? 'none' : '';
    });
    svg.querySelectorAll('line[data-edge]').forEach(function(el) {
      var parts = el.dataset.edge.split('-');
      if (idSet.has(parts[1]) || (idSet.has(parts[0]) && parts[0] !== nodeId)) {
        el.style.display = hidden ? 'none' : '';
      }
    });
    svg.querySelectorAll('text').forEach(function(el) {
      // Simple approach: hide all text near hidden nodes by checking proximity
    });
  }

  // --- Path Tracing ---
  function tracePath(sample) {
    if (!treeData || !treeData.tree || !treeData.tree.root) return [];
    var path = [];
    var node = treeData.tree.root;
    while (node && node.type !== 'leaf') {
      path.push(node.node_id);
      var value = sample[node.feature_name];
      if (value === undefined || value === null) break;
      var goLeft = false;
      switch(node.comparison_op) {
        case '<=': goLeft = value <= node.threshold; break;
        case '<':  goLeft = value < node.threshold; break;
        case '>=': goLeft = value >= node.threshold; break;
        case '>':  goLeft = value > node.threshold; break;
        case '==': goLeft = value == node.threshold; break;
        case '!=': goLeft = value != node.threshold; break;
        default:   goLeft = value <= node.threshold;
      }
      node = goLeft ? node.left : node.right;
    }
    if (node) path.push(node.node_id);
    return path;
  }

  function highlightPath(pathNodeIds) {
    var pathSet = new Set(pathNodeIds);
    var pathEdges = new Set();
    for (var i = 0; i < pathNodeIds.length - 1; i++) {
      pathEdges.add(pathNodeIds[i] + '-' + pathNodeIds[i+1]);
    }

    svg.querySelectorAll('rect[data-node-id]').forEach(function(el) {
      if (pathSet.has(el.dataset.nodeId)) {
        el.style.opacity = '1';
        el.style.stroke = '#1565c0';
        el.style.strokeWidth = '3';
      } else {
        el.style.opacity = '0.15';
      }
    });
    svg.querySelectorAll('line[data-edge]').forEach(function(el) {
      if (pathEdges.has(el.dataset.edge)) {
        el.style.opacity = '1';
        el.style.stroke = '#1565c0';
        el.style.strokeWidth = '4';
      } else {
        el.style.opacity = '0.1';
      }
    });
    svg.querySelectorAll('text').forEach(function(el) {
      el.style.opacity = '0.2';
    });
    // Re-show text inside path nodes
    pathSet.forEach(function(nodeId) {
      var rect = svg.querySelector('rect[data-node-id="' + nodeId + '"]');
      if (!rect) return;
      var rx = parseFloat(rect.getAttribute('x'));
      var ry = parseFloat(rect.getAttribute('y'));
      var rw = parseFloat(rect.getAttribute('width'));
      var rh = parseFloat(rect.getAttribute('height'));
      svg.querySelectorAll('text').forEach(function(t) {
        var tx = parseFloat(t.getAttribute('x'));
        var ty = parseFloat(t.getAttribute('y'));
        if (tx >= rx && tx <= rx + rw && ty >= ry - 5 && ty <= ry + rh + 10) {
          t.style.opacity = '1';
          t.style.fontWeight = 'bold';
        }
      });
    });
  }

  function clearHighlight() {
    svg.querySelectorAll('rect[data-node-id]').forEach(function(el) {
      el.style.opacity = '';
      el.style.stroke = '';
      el.style.strokeWidth = '';
    });
    svg.querySelectorAll('line[data-edge]').forEach(function(el) {
      el.style.opacity = '';
      el.style.stroke = '';
      el.style.strokeWidth = '';
    });
    svg.querySelectorAll('text').forEach(function(el) {
      el.style.opacity = '';
      el.style.fontWeight = '';
    });
  }

  // --- Controls ---
  var input = document.getElementById('sample-index');
  var btn = document.getElementById('highlight-btn');
  var clearBtn = document.getElementById('clear-btn');
  var errorEl = document.getElementById('highlight-error');

  if (btn && input) {
    btn.addEventListener('click', function() {
      var idx = parseInt(input.value, 10);
      if (isNaN(idx)) {
        if (errorEl) errorEl.textContent = 'Enter a valid integer index.';
        return;
      }
      if (idx < 0 || idx >= nTotalRows) {
        if (errorEl) errorEl.textContent = 'Index out of range. Valid: [0, ' + nTotalRows + ').';
        return;
      }
      var sample = flowSamples[String(idx)];
      if (!sample) {
        if (errorEl) errorEl.textContent = 'Row ' + idx + ' is not embedded. Use highlight_samples=[' + idx + '] when calling visualize().';
        return;
      }
      if (errorEl) errorEl.textContent = '';
      var path = tracePath(sample);
      if (path.length > 0) highlightPath(path);
    });
  }
  if (clearBtn) {
    clearBtn.addEventListener('click', function() {
      clearHighlight();
      if (errorEl) errorEl.textContent = '';
      if (input) input.value = '';
    });
  }
})();
"""


class HTMLAssembler:
    def assemble(
        self,
        svg_content: str,
        tree_json: str = "{}",
        flow_data_json: str = '{"samples": {}, "total_rows": 0}',
        mode: str = "blueprint",
        ensemble_meta: EnsembleMeta | None = None,
        title: str = "PrettyForest Visualization",
    ) -> str:
        controls = ""
        if mode == "flow":
            controls = (
                '<div class="controls">'
                '<input type="number" id="sample-index" placeholder="Row index"/>'
                '<button id="highlight-btn">Highlight Path</button>'
                '<button id="clear-btn">Clear</button>'
                '<div id="highlight-error" class="error-msg"></div>'
                "</div>"
            )

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            f"<title>{title}</title>\n"
            '<meta charset="utf-8"/>\n'
            f"<style>{_CSS}</style>\n"
            "</head>\n"
            "<body>\n"
            f"<h2>{title}</h2>\n"
            f"{controls}\n"
            '<div class="tree-container">\n'
            f"{svg_content}\n"
            "</div>\n"
            f'<script id="tree-data" type="application/json">{tree_json}</script>\n'
            f'<script id="flow-data" type="application/json">{flow_data_json}</script>\n'
            f"<script>{_JS}</script>\n"
            "</body>\n"
            "</html>"
        )
