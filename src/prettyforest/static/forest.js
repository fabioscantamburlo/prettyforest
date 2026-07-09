// PrettyForest anywidget ESM module (AFM spec: default export)

function render({ model, el }) {
  el.style.height = "700px";
  el.style.position = "relative";
  el.style.overflow = "hidden";

  // Parse the full HTML and extract parts
  var raw = model.get("html_content");

  // Extract <style> content
  var styleMatch = raw.match(/<style>([\s\S]*?)<\/style>/i);
  if (styleMatch) {
    var css = styleMatch[1];
    // Scope body selectors to .pf-root so they don't leak into the notebook
    css = css.replace(/\bbody\.dark\b/g, ".pf-root.dark");
    css = css.replace(/\bbody\b/g, ".pf-root");
    // In widget context, use 100% height instead of 100vh
    css = css.replace(/height:\s*100vh/g, "height: 100%");
    // Make the detail modal absolute within the widget rather than fixed on the viewport
    css = css.replace(
      /(\.detail-modal\s*\{[^}]*?)position:\s*fixed/,
      "$1position: absolute"
    );
    // Scope the universal reset to .pf-root children only
    css = css.replace(
      /\*\s*\{([^}]*)\}/,
      ".pf-root, .pf-root * {$1}"
    );
    // Ensure detail-body has a visible background (prevent black from notebook themes)
    css += "\n.pf-root .detail-body { background: #f9fdf9; }\n";
    css += ".pf-root.dark .detail-body { background: #0f1910; }\n";
    // Ensure node rects have explicit fill (fallback against notebook SVG resets)
    css += ".pf-root .detail-body svg rect.node-rect { fill: #fffde7 !important; stroke: #5D4037 !important; }\n";
    css += ".pf-root .detail-body svg rect.node-rect.leaf { fill: #e8f5e9 !important; stroke: #2e7d32 !important; }\n";
    css += ".pf-root .detail-body svg rect.node-rect.on-path { fill: #e3f2fd !important; stroke: #1565c0 !important; }\n";
    css += ".pf-root .detail-body svg rect.node-rect.leaf.on-path { fill: #bbdefb !important; stroke: #1565c0 !important; }\n";
    css += ".pf-root .detail-body svg text { fill: #333 !important; }\n";
    css += ".pf-root .detail-body svg line.edge-line { stroke: #8d6e63 !important; }\n";
    css += ".pf-root .detail-body svg line.edge-line.on-path { stroke: #1565c0 !important; }\n";
    var styleEl = document.createElement("style");
    styleEl.textContent = css;
    el.appendChild(styleEl);
  }

  // Extract body content
  var bodyMatch = raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  var bodyContent = bodyMatch ? bodyMatch[1] : raw;

  // Create a wrapper div that acts as the "body" for CSS purposes
  var root = document.createElement("div");
  root.className = "pf-root";
  root.style.position = "relative";
  root.style.height = "100%";
  root.style.overflow = "hidden";
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.innerHTML = bodyContent;
  el.appendChild(root);

  // Ensure the detail modal fully covers the widget when open
  var detailModal = root.querySelector("#detail-modal");
  if (detailModal) {
    detailModal.style.position = "absolute";
    detailModal.style.top = "0";
    detailModal.style.left = "0";
    detailModal.style.width = "100%";
    detailModal.style.height = "100%";
    detailModal.style.zIndex = "2000";
  }

  // Extract and execute the config script (var declarations)
  // and the main IIFE, passing `root` as ROOT
  var scripts = root.querySelectorAll("script");
  var configCode = "";
  var mainCode = "";

  scripts.forEach(function (script) {
    if (script.type === "application/json") return; // data scripts stay in DOM
    var code = script.textContent;
    if (code.match(/^\s*var METRIC_KEY/)) {
      // Config vars script
      configCode = code;
    } else if (code.match(/\(function\(ROOT\)/)) {
      // Main IIFE — extract the function body
      mainCode = code;
    }
    // Remove the script element (we'll execute manually)
    script.remove();
  });

  // Execute: config vars + main IIFE with root as ROOT
  if (configCode || mainCode) {
    // Replace (document) at the end of the IIFE with (root)
    // The IIFE ends with })(document);
    var fullCode = configCode + "\n" + mainCode;
    // We use Function constructor to execute with `root` in scope
    // The IIFE takes ROOT param, so we just replace the final (document) call
    fullCode = fullCode.replace(/\}\)\(document\)\s*;?\s*$/, "})(root);");
    try {
      new Function("root", fullCode)(root);
    } catch (e) {
      console.error("[PrettyForest widget] Script execution error:", e);
    }
  }

  // Listen for model changes (season switch from Python)
  model.on("change:season", function () {
    var select = root.querySelector("#season-toggle");
    if (select) {
      select.value = model.get("season") || "";
      select.dispatchEvent(new Event("change"));
    }
  });

  // Listen for sample_idx changes from Python
  model.on("change:sample_idx", function () {
    var idx = model.get("sample_idx");
    if (idx < 0) return;
    var input = root.querySelector("#predict-row");
    var btn = root.querySelector("#predict-go");
    if (input && btn) {
      input.value = idx;
      btn.click();
    }
  });
}

export default { render };
