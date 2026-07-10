// PrettyForest anywidget ESM module & interactive forest engine
// Works as both standalone script engine and AnyWidget ESM renderer without eval/new Function.

function initForest(ROOT, configOptions) {
  var cfg = configOptions || {};
  if (!cfg.METRIC_KEY) {
    var configEl = ROOT.querySelector('#forest-config');
    if (configEl) {
      try { cfg = JSON.parse(configEl.textContent); } catch(e) {}
    }
  }
  var METRIC_KEY = cfg.METRIC_KEY || (typeof window !== 'undefined' && window.METRIC_KEY) || 'variance';
  var METRIC_LABEL = cfg.METRIC_LABEL || (typeof window !== 'undefined' && window.METRIC_LABEL) || 'Pred Variance';
  var TOTAL = (cfg.TOTAL !== undefined) ? cfg.TOTAL : ((typeof window !== 'undefined' && window.TOTAL) ? window.TOTAL : 1);
  var PAGE_SIZE = (cfg.PAGE_SIZE !== undefined) ? cfg.PAGE_SIZE : ((typeof window !== 'undefined' && window.PAGE_SIZE) ? window.PAGE_SIZE : 200);
  var HAS_PREDICT = (cfg.HAS_PREDICT !== undefined) ? (cfg.HAS_PREDICT === true || cfg.HAS_PREDICT === 'true') : ((typeof window !== 'undefined' && window.HAS_PREDICT) ? window.HAS_PREDICT : false);
  var IS_BOOSTED = (cfg.IS_BOOSTED !== undefined) ? (cfg.IS_BOOSTED === true || cfg.IS_BOOSTED === 'true') : ((typeof window !== 'undefined' && window.IS_BOOSTED) ? window.IS_BOOSTED : false);
  var MODEL_NAME = cfg.MODEL_NAME || (typeof window !== 'undefined' && window.MODEL_NAME) || 'Unknown Model';

  var $ = function(id) { return ROOT.querySelector('#' + id); };
  var svg = $('forest-svg');
  var container = $('forest-container');
  var tooltip = $('tooltip');
  var sortBy = $('sort-by');
  var pagePrev = $('page-prev');
  var pageNext = $('page-next');
  var pageInfo = $('page-info');
  var resetAll = $('reset-all');
  var spotlightPanel = $('spotlight-panel');
  var spotlightClose = $('spotlight-close');
  var spotlightContent = $('spotlight-content');
  var zoomIn = $('zoom-in');
  var zoomOut = $('zoom-out');
  var zoomReset = $('zoom-reset');
  var zoomLabel = $('zoom-level');
  if (!svg || !container) return;

  // --- Collect all tree elements and their data ---
  var traceActive = false;
  var allTrees = Array.prototype.slice.call(svg.querySelectorAll('.visual-tree'));
  var treeData = allTrees.map(function(el) {
    return {
      el: el,
      idx: parseInt(el.getAttribute('data-tree-idx')) || 0,
      depth: parseInt(el.getAttribute('data-depth')) || 0,
      nodes: parseInt(el.getAttribute('data-nodes')) || 0,
      leaves: parseInt(el.getAttribute('data-leaves')) || 0,
      purity: parseFloat(el.getAttribute('data-purity')) || null,
      magnitude: parseFloat(el.getAttribute('data-magnitude')) || null,
      variance: parseFloat(el.getAttribute('data-variance')) || null,
      origTransform: el.getAttribute('transform') || ''
    };
  });

  function metric(d) {
    if (METRIC_KEY === 'purity') return d.purity;
    if (METRIC_KEY === 'magnitude') return d.magnitude;
    if (METRIC_KEY === 'variance') return d.variance;
    return null;
  }

  // --- Paging state ---
  var currentPage = 0;
  var sortedData = treeData.slice();
  var totalPages = Math.ceil(TOTAL / PAGE_SIZE);

  var positions = [];
  (function() {
    var visible = allTrees.filter(function(t) { return !t.classList.contains('hidden'); });
    for (var i = 0; i < visible.length; i++) {
      positions.push(visible[i].getAttribute('transform') || '');
    }
  })();

  function showPage() {
    var start = currentPage * PAGE_SIZE;
    var end = Math.min(start + PAGE_SIZE, sortedData.length);
    var count = end - start;

    treeData.forEach(function(d) {
      d.el.classList.add('hidden');
      d.el.classList.remove('grown','grow-trunk','grow-branches','grow-canopy','highlighted','spotlit');
      d.el.style.opacity = '';
    });

    for (var i = 0; i < count; i++) {
      var d = sortedData[start + i];
      var posIdx = i % positions.length;
      d.el.setAttribute('transform', positions[posIdx]);
      d.el.classList.remove('hidden');
      d.el.style.opacity = '1';
      d.el.classList.add('grown', 'grow-trunk', 'grow-branches', 'grow-canopy');
    }

    if (pageInfo) pageInfo.textContent = (start + 1) + '–' + end + ' of ' + sortedData.length;
    if (pagePrev) pagePrev.disabled = (currentPage === 0);
    if (pageNext) pageNext.disabled = (end >= sortedData.length);
  }

  showPage();

  if (pagePrev) pagePrev.addEventListener('click', function() {
    if (currentPage > 0) { currentPage--; showPage(); }
  });
  if (pageNext) pageNext.addEventListener('click', function() {
    if ((currentPage + 1) * PAGE_SIZE < sortedData.length) { currentPage++; showPage(); }
  });

  // --- Zoom & Pan ---
  var scale = 1, tx = 0, ty = 0, dragging = false, sx = 0, sy = 0;
  function applyZoom() {
    svg.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    svg.style.transformOrigin = 'center center';
    if (zoomLabel) zoomLabel.textContent = 'Zoom: ' + Math.round(scale * 100) + '%';
  }
  if (zoomIn) zoomIn.addEventListener('click', function() { scale = Math.min(scale * 1.25, 5); applyZoom(); });
  if (zoomOut) zoomOut.addEventListener('click', function() { scale = Math.max(scale / 1.25, 0.2); applyZoom(); });
  if (zoomReset) zoomReset.addEventListener('click', function() { scale = 1; tx = 0; ty = 0; applyZoom(); });

  var darkBtn = $('dark-toggle');
  var themeRoot = ROOT === document ? document.body : ROOT;
  if (darkBtn) {
    darkBtn.addEventListener('click', function() {
      themeRoot.classList.toggle('dark');
      darkBtn.textContent = themeRoot.classList.contains('dark') ? '☀️' : '🌙';
    });
  }

  var seasonSelect = $('season-toggle');
  if (seasonSelect) {
    var seasonPalettes = {
      spring: { canopy: ['#90EE90','#98FB98','#FFB7C5','#FF69B4','#DDA0DD','#87CEAB'], ground: '#a8d8a0', sky: '#f1fff1' },
      summer: { canopy: ['#2E8B57','#3CB371','#6B8E23','#228B22','#32CD32'], ground: '#8cc97a', sky: '#dceef5' },
      autumn: { canopy: ['#D2691E','#B22222','#DAA520','#CD853F','#FF8C00'], ground: '#8B6914', sky: '#fff3e0' },
      winter: { canopy: [], ground: '#B0C4DE', sky: '#e3f2fd', bare: true }
    };

    var groundStops = svg.querySelectorAll('#ground-gradient stop');
    var skyStops = svg.querySelectorAll('#sky-gradient stop');
    var patches = svg.querySelectorAll('ellipse[data-patch]');
    var origGround = []; groundStops.forEach(function(s) { origGround.push(s.getAttribute('stop-color')); });
    var origSky = []; skyStops.forEach(function(s) { origSky.push(s.getAttribute('stop-color')); });
    var origPatches = []; patches.forEach(function(e) { origPatches.push(e.getAttribute('fill')); });

    seasonSelect.addEventListener('change', function() {
      var season = this.value;
      var trees = svg.querySelectorAll('.visual-tree');

      if (!season) {
        trees.forEach(function(t) {
          var canopy = t.querySelector('.canopy');
          if (canopy && canopy.dataset.origFill) { canopy.setAttribute('fill', canopy.dataset.origFill); canopy.setAttribute('stroke', canopy.dataset.origStroke || canopy.dataset.origFill); }
          if (canopy) canopy.style.display = '';
        });
        groundStops.forEach(function(s, i) { if (origGround[i]) s.setAttribute('stop-color', origGround[i]); });
        skyStops.forEach(function(s, i) { if (origSky[i]) s.setAttribute('stop-color', origSky[i]); });
        patches.forEach(function(e, i) { if (origPatches[i]) e.setAttribute('fill', origPatches[i]); });
        return;
      }

      var pal = seasonPalettes[season];
      if (!pal) return;

      trees.forEach(function(t) {
        var canopy = t.querySelector('.canopy');
        if (!canopy) return;
        if (!canopy.dataset.origFill) { canopy.dataset.origFill = canopy.getAttribute('fill'); canopy.dataset.origStroke = canopy.getAttribute('stroke'); }
        if (pal.bare) { canopy.style.display = 'none'; }
        else { canopy.style.display = ''; var c = pal.canopy[Math.floor(Math.random()*pal.canopy.length)]; canopy.setAttribute('fill', c); canopy.setAttribute('stroke', c); }
      });

      if (groundStops.length >= 2) { groundStops[0].setAttribute('stop-color', pal.ground); groundStops[1].setAttribute('stop-color', pal.ground); }
      if (skyStops.length >= 1) { skyStops[0].setAttribute('stop-color', pal.sky); }
      patches.forEach(function(e) { e.setAttribute('fill', pal.ground); });
    });
  }

  container.addEventListener('wheel', function(e) {
    e.preventDefault();
    scale = Math.max(0.2, Math.min(5, scale * (e.deltaY > 0 ? 0.9 : 1.1)));
    applyZoom();
  }, {passive: false});
  container.addEventListener('mousedown', function(e) {
    if (e.target !== svg && !svg.contains(e.target)) return;
    if (e.button !== 0) return; dragging = true; sx = e.clientX - tx; sy = e.clientY - ty; svg.style.cursor = 'grabbing';
  });
  document.addEventListener('mousemove', function(e) {
    if (!dragging) return; tx = e.clientX - sx; ty = e.clientY - sy; svg.style.transition = 'none'; applyZoom();
  });
  document.addEventListener('mouseup', function() { dragging = false; svg.style.cursor = 'grab'; svg.style.transition = 'transform 0.15s ease'; });
  document.addEventListener('keydown', function(e) {
    switch(e.key) {
      case 'ArrowLeft': tx += 40; break; case 'ArrowRight': tx -= 40; break;
      case 'ArrowUp': ty += 40; break; case 'ArrowDown': ty -= 40; break;
      case '+': case '=': scale = Math.min(scale * 1.15, 5); break;
      case '-': scale = Math.max(scale / 1.15, 0.2); break;
      case 'Escape': closeSpotlight(); return;
      default: return;
    }
    e.preventDefault(); applyZoom();
  });

  // --- Tooltip ---
  function findTree(el) {
    while (el && el !== svg) {
      if (el.getAttribute && (el.getAttribute('class') || '').indexOf('visual-tree') !== -1) return el;
      el = el.parentNode;
    }
    return null;
  }
  if (tooltip) {
    svg.addEventListener('mouseover', function(e) {
      var tree = findTree(e.target);
      if (!tree) { tooltip.style.display = 'none'; return; }
      var d = treeData.find(function(t) { return t.el === tree; });
      if (!d) return;
      var h = '<strong>Tree #' + d.idx + '</strong><br>';
      h += 'Depth: ' + d.depth + '<br>Nodes: ' + d.nodes + '<br>Leaves: ' + d.leaves + '<br>';
      if (d.purity !== null) h += 'Purity: ' + (d.purity*100).toFixed(1) + '%<br>';
      if (d.magnitude !== null) h += 'Magnitude: ' + d.magnitude.toFixed(4) + '<br>';
      if (d.variance !== null) h += 'Variance: ' + d.variance.toFixed(2) + '<br>';
      tooltip.innerHTML = h; tooltip.style.display = 'block';
    });
    svg.addEventListener('mousemove', function(e) {
      if (tooltip.style.display === 'block') { tooltip.style.left=(e.clientX+14)+'px'; tooltip.style.top=(e.clientY+14)+'px'; }
    });
    svg.addEventListener('mouseout', function(e) { if (!findTree(e.target)) tooltip.style.display='none'; });
    svg.addEventListener('mouseleave', function() { tooltip.style.display='none'; });
  }

  // --- Sort ---
  if (sortBy) {
    sortBy.addEventListener('change', function() {
      var mode = this.value;
      if (mode === 'natural') {
        sortedData = treeData.slice();
      } else {
        sortedData = treeData.slice().sort(function(a, b) {
          if (mode === 'depth') return b.depth - a.depth;
          if (mode === 'nodes') return b.nodes - a.nodes;
          if (mode === 'leaves') return b.leaves - a.leaves;
          if (mode === 'metric') return (metric(b)||0) - (metric(a)||0);
          return 0;
        });
      }
      currentPage = 0;
      totalPages = Math.ceil(sortedData.length / PAGE_SIZE);
      showPage();
    });
  }

  // --- Reset ---
  if (resetAll) {
    resetAll.addEventListener('click', function() {
      closeSpotlight();
      if (sortBy) sortBy.value = 'natural';
      sortedData = treeData.slice();
      currentPage = 0;
      totalPages = Math.ceil(sortedData.length / PAGE_SIZE);
      treeData.forEach(function(d) {
        d.el.classList.remove('hidden', 'highlighted', 'spotlit');
        d.el.setAttribute('transform', d.origTransform);
      });
      showPage();
    });
  }

  // --- Click to spotlight ---
  var spotlitEl = null;
  var lastSpotlightClick = 0, lastSpotlightTree = null;
  svg.addEventListener('click', function(e) {
    var tree = findTree(e.target);
    if (!tree) { closeSpotlight(); return; }
    var now = Date.now();
    if (spotlitEl === tree) {
      if (lastSpotlightTree === tree && (now - lastSpotlightClick) < 380) {
        return;
      }
      closeSpotlight();
      return;
    }
    lastSpotlightClick = now;
    lastSpotlightTree = tree;

    if (spotlitEl) spotlitEl.classList.remove('spotlit');
    spotlitEl = tree;
    tree.classList.add('spotlit');

    var d = treeData.find(function(t) { return t.el === tree; });
    if (!d || !spotlightContent || !spotlightPanel) return;
    var h = '<strong>Tree #' + d.idx + '</strong>';
    h += '<div class="stat-row"><span class="stat-label">Depth</span><span class="stat-value">' + d.depth + '</span></div>';
    h += '<div class="stat-row"><span class="stat-label">Nodes</span><span class="stat-value">' + d.nodes + '</span></div>';
    h += '<div class="stat-row"><span class="stat-label">Leaves</span><span class="stat-value">' + d.leaves + '</span></div>';
    if (d.purity !== null) h += '<div class="stat-row"><span class="stat-label">Purity</span><span class="stat-value">' + (d.purity*100).toFixed(1) + '%</span></div>';
    if (d.magnitude !== null) h += '<div class="stat-row"><span class="stat-label">Magnitude</span><span class="stat-value">' + d.magnitude.toFixed(4) + '</span></div>';
    if (d.variance !== null) h += '<div class="stat-row"><span class="stat-label">Variance</span><span class="stat-value">' + d.variance.toFixed(2) + '</span></div>';
    var ranked = treeData.slice().filter(function(t) { return metric(t) !== null; });
    ranked.sort(function(a, b) { return (metric(b)||0) - (metric(a)||0); });
    var rank = ranked.findIndex(function(t) { return t.el === tree; }) + 1;
    if (rank > 0) h += '<div class="stat-row" style="margin-top:4px"><span class="stat-label">' + METRIC_LABEL + ' rank</span><span class="stat-value">#' + rank + '/' + ranked.length + '</span></div>';
    spotlightContent.innerHTML = h;
    spotlightPanel.classList.add('visible');
  });
  if (spotlightClose) spotlightClose.addEventListener('click', closeSpotlight);
  function closeSpotlight() {
    if (spotlitEl) spotlitEl.classList.remove('spotlit');
    spotlitEl = null;
    if (spotlightPanel) spotlightPanel.classList.remove('visible');
  }

  // --- Info button + Model description ---
  (function() {
    var infoBtn = $('info-btn');
    var svgInfoBtn = $('svg-info-btn');
    var infoPanel = $('model-info-panel');
    var infoClose = $('info-panel-close');
    var descEl = $('model-description');
    if (!infoPanel) return;

    var descriptions = {
      'Random Forest (Classification)': '<p><span class="key">How it works:</span> Trains multiple independent trees on random subsets of data (bagging). Each tree votes for a class. Final prediction = majority vote.</p><p><span class="key">Each tree:</span> Splits on original features with real class proportions in leaves. Fully interpretable individually.</p><p><span class="key">Reading tips:</span> Purity shows how cleanly each tree separates classes. High purity = the tree is confident in its leaves.</p>',
      'Random Forest (Regression)': '<p><span class="key">How it works:</span> Trains multiple independent trees on random subsets of data. Each tree predicts a value. Final prediction = average of all trees.</p><p><span class="key">Each tree:</span> Splits on original features with target means in leaves. Each leaf is a direct prediction.</p><p><span class="key">Reading tips:</span> Variance shows how spread out the leaf predictions are within each tree.</p>',
      'Gradient Boosting (Classification)': '<p><span class="key">How it works:</span> Trains trees sequentially. Each tree corrects the errors of the previous ensemble by fitting gradients (residuals).</p><p><span class="key">Each tree:</span> Splits on original features, but leaf values are small <em>gradient corrections</em>, not class predictions. A leaf value of +0.12 means "push the score slightly toward this class."</p><p><span class="key">Reading tips:</span> Final prediction = initial value + learning_rate × sum of all tree corrections. Individual tree leaf values are not standalone predictions.</p>',
      'Gradient Boosting (Regression)': '<p><span class="key">How it works:</span> Trains trees sequentially. Each tree predicts the residual (error) of the current ensemble.</p><p><span class="key">Each tree:</span> Splits on original features, leaf values are residual corrections. Final prediction = initial mean + lr × sum(leaf values).</p><p><span class="key">Reading tips:</span> Early trees make large corrections, later trees fine-tune. Magnitude decreases over iterations.</p>',
      'LightGBM (Classification)': '<p><span class="key">How it works:</span> Gradient boosting with histogram-based splits for speed. Trains trees on gradients sequentially.</p><p><span class="key">Each tree:</span> Splits on original features using histogram bins. Leaf values are log-odds corrections (not class probabilities).</p><p><span class="key">Reading tips:</span> Final prediction = sum of all leaf values, then softmax for probabilities. Individual leaves show gradient steps.</p>',
      'LightGBM (Regression)': '<p><span class="key">How it works:</span> Fast gradient boosting with histogram splits. Each tree predicts the residual error.</p><p><span class="key">Each tree:</span> Leaf values are additive corrections. Final prediction = sum of all tree leaf values.</p>',
      'CatBoost (Classification)': '<p><span class="key">How it works:</span> Ordered boosting with symmetric trees. Handles categorical features natively.</p><p><span class="key">Each tree:</span> Uses oblivious (symmetric) decision trees — same split at each depth level. Leaf values are gradient corrections.</p><p><span class="key">Reading tips:</span> Leaf values are small corrections. Final prediction comes from summing all trees. Exact reconstruction may differ slightly due to internal scaling.</p>',
      'CatBoost (Regression)': '<p><span class="key">How it works:</span> Ordered boosting with symmetric trees on residuals.</p><p><span class="key">Each tree:</span> Symmetric structure, leaf values are residual corrections summed for final prediction.</p>',
      'Decision Tree (Classification)': '<p><span class="key">How it works:</span> A single tree that recursively splits the data to separate classes.</p><p><span class="key">The tree:</span> Each split uses the feature that best separates classes (by Gini or entropy). Leaves show class proportions from training data.</p><p><span class="key">Reading tips:</span> The forest shows one tree. This IS the full model — no ensemble aggregation.</p>',
      'Decision Tree (Regression)': '<p><span class="key">How it works:</span> A single tree that recursively splits to minimize prediction error.</p><p><span class="key">The tree:</span> Leaves contain the mean target value of training samples that landed there. This is a direct prediction.</p>',
    };

    var desc = descriptions[MODEL_NAME] || '<p>Tree-based ensemble model.</p>';
    desc += '<hr style="margin:10px 0;border:none;border-top:1px solid rgba(0, 0, 0, 0.08)"><p style="font-size:11px;color:#666"><strong>Visual encoding:</strong> Tree height = depth, trunk width = node count, canopy color = green (pure/low variance) to amber (impure/high variance).</p>';
    if (descEl) descEl.innerHTML = desc;

    var toggleInfo = function(e) {
      e.stopPropagation();
      infoPanel.classList.toggle('visible');
    };

    if (infoBtn) infoBtn.addEventListener('click', toggleInfo);
    if (svgInfoBtn) svgInfoBtn.addEventListener('click', toggleInfo);
    if (infoClose) infoClose.addEventListener('click', function() { infoPanel.classList.remove('visible'); });
  })();

  // --- Double-click to open tree detail modal ---
  (function() {
    var modal = $('detail-modal');
    var modalBody = $('detail-body');
    var modalTitle = $('detail-title');
    var closeBtn = $('detail-close');
    var treesEl = $('trees-data');
    if (!modal || !treesEl) return;

    var allTreeStructures = JSON.parse(treesEl.textContent);
    var detailScale = 1, detailTx = 0, detailTy = 0;
    var detailSvg = null;

    function openDetail(treeIdx, depth, nodes) {
      var treeStruct = allTreeStructures[treeIdx];
      if (!treeStruct) return;

      if (modalTitle) modalTitle.textContent = 'Tree #' + treeIdx + ' — Depth: ' + depth + ', Nodes: ' + nodes;
      if (modalBody) modalBody.innerHTML = '';
      detailScale = 1; detailTx = 0; detailTy = 0;

      var noteEl = $('detail-note');
      if (noteEl) {
        if (IS_BOOSTED) {
          noteEl.textContent = '\u26a0\ufe0f This is a boosted tree — leaf values are gradient corrections (residuals), not final predictions. The sample path and splits are on original features.';
          noteEl.classList.add('visible');
        } else {
          noteEl.classList.remove('visible');
        }
      }

      var tracedSample = null;
      if (traceActive) {
        var predDataEl = $('predict-data');
        var rowInput = $('predict-row');
        if (predDataEl && rowInput) {
          try {
            var pd = JSON.parse(predDataEl.textContent);
            var idx = parseInt(rowInput.value);
            if (pd.samples && idx >= 0 && idx < pd.samples.length) {
              tracedSample = pd.samples[idx];
            }
          } catch(e) {}
        }
      }

      var detailSampleEl = $('detail-sample');
      if (detailSampleEl) {
        if (tracedSample) {
          var html = '<div class="sample-chips">';
          for (var key in tracedSample) {
            var val = tracedSample[key];
            var display = (typeof val === 'number') ? val.toFixed(3) : String(val);
            html += '<span class="chip"><span class="chip-name">' + key + ':</span><span class="chip-val">' + display + '</span></span>';
          }
          html += '</div>';
          detailSampleEl.innerHTML = html;
          detailSampleEl.classList.add('visible');
        } else {
          detailSampleEl.classList.remove('visible');
          detailSampleEl.innerHTML = '';
        }
      }

      detailSvg = renderTreeSVG(treeStruct, tracedSample);
      detailSvg.style.transition = 'none';
      detailSvg.style.transformOrigin = '0 0';
      if (modalBody) modalBody.appendChild(detailSvg);
      modal.classList.add('open');
      requestAnimationFrame(function() {
        if (!modalBody || !detailSvg) return;
        var bodyW = modalBody.clientWidth;
        var svgW = detailSvg.getBoundingClientRect().width / detailScale;
        detailTx = Math.max(0, (bodyW - svgW) / 2);
        detailTy = 0;
        detailSvg.style.transform = 'translate(' + detailTx + 'px,' + detailTy + 'px) scale(' + detailScale + ')';
        requestAnimationFrame(function() { if (detailSvg) detailSvg.style.transition = 'transform 0.15s ease'; });
      });
    }

    function triggerTreeDetail(tree) {
      if (!tree) return;
      var d = treeData.find(function(t) { return t.el === tree; });
      if (!d) return;
      openDetail(d.idx, d.depth, d.nodes);
    }

    svg.addEventListener('dblclick', function(e) {
      triggerTreeDetail(findTree(e.target));
    });

    var lastDetailClickTime = 0, lastDetailClickTree = null;
    svg.addEventListener('click', function(e) {
      var tree = findTree(e.target);
      if (!tree) return;
      var now = Date.now();
      if (lastDetailClickTree === tree && (now - lastDetailClickTime) < 380) {
        lastDetailClickTime = 0;
        lastDetailClickTree = null;
        triggerTreeDetail(tree);
      } else {
        lastDetailClickTime = now;
        lastDetailClickTree = tree;
      }
    });

    var spotlightEl = $('spotlight-content');
    if (spotlightEl) {
      spotlightEl.addEventListener('dblclick', function() {
        if (!spotlitEl) return;
        triggerTreeDetail(spotlitEl);
      });
    }


    if (closeBtn) closeBtn.addEventListener('click', closeDetail);
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && modal.classList.contains('open')) { closeDetail(); e.stopPropagation(); }
    });
    modal.addEventListener('click', function(e) { if (e.target === modal) closeDetail(); });

    function closeDetail() { modal.classList.remove('open'); detailSvg = null; }

    if (modalBody) {
      modalBody.addEventListener('wheel', function(e) {
        if (!detailSvg) return;
        e.preventDefault();
        detailScale = Math.max(0.3, Math.min(5, detailScale * (e.deltaY > 0 ? 0.9 : 1.1)));
        applyDetailTransform();
      }, {passive: false});

      var detailDrag = false, detailSx = 0, detailSy = 0;
      modalBody.addEventListener('mousedown', function(e) {
        if (e.button !== 0 || !detailSvg) return;
        detailDrag = true; detailSx = e.clientX - detailTx; detailSy = e.clientY - detailTy;
        modalBody.style.cursor = 'grabbing';
      });
      document.addEventListener('mousemove', function(e) {
        if (!detailDrag) return;
        detailTx = e.clientX - detailSx; detailTy = e.clientY - detailSy;
        if (detailSvg) detailSvg.style.transition = 'none';
        applyDetailTransform();
      });
      document.addEventListener('mouseup', function() {
        if (detailDrag) { detailDrag = false; modalBody.style.cursor = 'grab'; if (detailSvg) detailSvg.style.transition = 'transform 0.15s ease'; }
      });
    }

    function applyDetailTransform() {
      if (!detailSvg) return;
      detailSvg.style.transform = 'translate(' + detailTx + 'px,' + detailTy + 'px) scale(' + detailScale + ')';
      detailSvg.style.transformOrigin = '0 0';
    }

    var NODE_W = 160, NODE_H = 50, H_GAP = 16, V_GAP = 60;
    var INITIAL_DEPTH = 3;
    var currentTreeStruct = null, currentSample = null;
    var expandedNodes = {};

    function countNodes(node) {
      if (!node || node.t === 'l') return 1;
      return 1 + countNodes(node.l) + countNodes(node.r);
    }

    function computeLayout(node, depth, baseMax, path) {
      if (node.t === 'l') return { node: node, depth: depth, width: NODE_W, children: null, truncated: 0, path: path };
      var extra = expandedNodes[path] || 0;
      if (depth >= baseMax + extra) {
        return { node: node, depth: depth, width: NODE_W, children: null, truncated: countNodes(node) - 1, path: path };
      }
      var left = computeLayout(node.l, depth + 1, baseMax, path + '.l');
      var right = computeLayout(node.r, depth + 1, baseMax, path + '.r');
      var w = left.width + H_GAP + right.width;
      return { node: node, depth: depth, width: Math.max(w, NODE_W), children: [left, right], truncated: 0, path: path };
    }

    function assignPos(layout, cx, y, positions, nodeH) {
      positions.push({ layout: layout, x: cx, y: y });
      if (!layout.children) return;
      var totalW = layout.children[0].width + H_GAP + layout.children[1].width;
      assignPos(layout.children[0], cx - totalW/2 + layout.children[0].width/2, y + nodeH + V_GAP, positions, nodeH);
      assignPos(layout.children[1], cx + totalW/2 - layout.children[1].width/2, y + nodeH + V_GAP, positions, nodeH);
    }

    function traceP(node, sample) {
      if (!sample || node.t === 'l') return [];
      var fv = sample[node.f]; if (fv === undefined) return [];
      var gl = (node.op==='<='?fv<=node.th:node.op==='<'?fv<node.th:node.op==='>='?fv>=node.th:node.op==='>'?fv>node.th:fv<=node.th);
      return [gl?'l':'r'].concat(traceP(gl?node.l:node.r, sample));
    }

    function getPS(positions, ts, sample) {
      if (!sample) return new Set();
      var path = traceP(ts, sample), onP = new Set([0]), cur = positions[0].layout;
      for (var i = 0; i < path.length; i++) {
        if (!cur.children) break;
        var ch = cur.children[path[i]==='l'?0:1];
        for (var j = 0; j < positions.length; j++) { if (positions[j].layout === ch) { onP.add(j); break; } }
        cur = ch;
      }
      return onP;
    }

    function renderTreeSVG(ts, sample) { currentTreeStruct=ts; currentSample=sample; expandedNodes={}; return buildSvg(); }
    function rerender() { if(!currentTreeStruct) return; modalBody.innerHTML=''; detailSvg=buildSvg(); detailSvg.style.transition='none'; detailSvg.style.transformOrigin='0 0'; detailSvg.style.transform='translate('+detailTx+'px,'+detailTy+'px) scale('+detailScale+')'; modalBody.appendChild(detailSvg); requestAnimationFrame(function(){if(detailSvg)detailSvg.style.transition='transform 0.15s ease';}); }

    function buildSvg() {
      var sample=currentSample, ts=currentTreeStruct, NH=sample?70:NODE_H;
      var isDark = themeRoot.classList.contains('dark');
      var txtColor = isDark ? '#ecf3ed' : '#333';
      var splitFill = isDark ? '#2c2c1a' : '#fffde7';
      var splitStroke = isDark ? '#a1887f' : '#5D4037';
      var leafFill = isDark ? '#1a2e1a' : '#e8f5e9';
      var leafStroke = isDark ? '#66bb6a' : '#2e7d32';
      var pathFill = isDark ? '#1a2a3a' : '#e3f2fd';
      var pathLeafFill = isDark ? '#1a3a5a' : '#bbdefb';
      var pathStroke = isDark ? '#64b5f6' : '#1565c0';
      var edgeColor = isDark ? '#a1887f' : '#8d6e63';
      var edgeLabelColor = isDark ? '#bcaaa4' : '#5D4037';
      var sampleValColor = isDark ? '#64b5f6' : '#1565c0';
      var layout=computeLayout(ts,0,INITIAL_DEPTH,'R'), positions=[];
      assignPos(layout, layout.width/2, 20, positions, NH);
      var pathSet=getPS(positions,ts,sample), hasP=pathSet.size>0;
      var ns='http://www.w3.org/2000/svg', mxX=0,mxY=0;
      positions.forEach(function(p){mxX=Math.max(mxX,p.x+NODE_W/2);mxY=Math.max(mxY,p.y+NH);});
      var svgW=mxX+40,svgH=mxY+40;
      var el=document.createElementNS(ns,'svg');
      el.setAttribute('width',svgW);el.setAttribute('height',svgH);
      el.setAttribute('viewBox','0 0 '+svgW+' '+svgH);el.style.cursor='grab';

      for(var i=0;i<positions.length;i++){var p=positions[i];if(!p.layout.children)continue;var pOn=pathSet.has(i);
        p.layout.children.forEach(function(cl,ci){
          var cI=positions.findIndex(function(pp){return pp.layout===cl;});if(cI<0)return;
          var c=positions[cI],onE=pOn&&pathSet.has(cI),dim=hasP&&!onE;
          var ln=document.createElementNS(ns,'line');
          ln.setAttribute('x1',p.x);ln.setAttribute('y1',p.y+NH);ln.setAttribute('x2',c.x);ln.setAttribute('y2',c.y);
          ln.setAttribute('class','edge-line'+(onE?' on-path':'')+(dim?' dimmed':''));
          ln.style.stroke=onE?pathStroke:edgeColor;ln.style.strokeWidth=onE?'3.5':'2.5';ln.style.strokeLinecap='round';
          if(dim)ln.style.opacity=isDark?'0.3':'0.15';
          el.appendChild(ln);
          var lb=document.createElementNS(ns,'text');lb.setAttribute('x',(p.x+c.x)/2+(ci===0?-10:10));
          lb.setAttribute('y',(p.y+NH+c.y)/2);lb.setAttribute('class','edge-label'+(dim?' dimmed':''));
          lb.style.fill=onE?pathStroke:edgeLabelColor;lb.style.fontSize='11px';lb.style.textAnchor='middle';lb.style.fontWeight='600';
          if(dim)lb.style.opacity=isDark?'0.3':'0.15';
          lb.textContent=ci===0?'\u2713':'\u2717';el.appendChild(lb);
        });
      }

      for(var j=0;j<positions.length;j++){(function(j){
        var pos=positions[j],nd=pos.layout.node,trunc=pos.layout.truncated||0,nPath=pos.layout.path;
        var rx=pos.x-NODE_W/2,ry=pos.y,onP=pathSet.has(j),dim=hasP&&!onP;
        var rect=document.createElementNS(ns,'rect');
        rect.setAttribute('x',rx);rect.setAttribute('y',ry);rect.setAttribute('width',NODE_W);rect.setAttribute('height',NH);
        rect.setAttribute('class',(nd.t==='l'?'node-rect leaf':'node-rect')+(onP?' on-path':'')+(dim?' dimmed':''));
        if(onP){rect.style.fill=nd.t==='l'?pathLeafFill:pathFill;rect.style.stroke=pathStroke;rect.style.strokeWidth='3';}
        else if(nd.t==='l'){rect.style.fill=leafFill;rect.style.stroke=leafStroke;rect.style.strokeWidth='2';}
        else{rect.style.fill=splitFill;rect.style.stroke=splitStroke;rect.style.strokeWidth='2';}
        rect.setAttribute('rx','8');
        if(dim){rect.style.opacity=isDark?'0.35':'0.2';}
        el.appendChild(rect);
        var t1=document.createElementNS(ns,'text');t1.setAttribute('x',pos.x);t1.setAttribute('y',ry+(sample?20:18));t1.setAttribute('class','node-text'+(dim?' dimmed':''));
        t1.style.fill=txtColor;t1.style.fontSize='11px';t1.style.textAnchor='middle';t1.style.pointerEvents='none';
        if(dim)t1.style.opacity=isDark?'0.4':'0.2';
        var t2=document.createElementNS(ns,'text');t2.setAttribute('x',pos.x);t2.setAttribute('y',ry+(sample?36:33));t2.setAttribute('class','node-text'+(dim?' dimmed':''));
        t2.style.fill=txtColor;t2.style.fontSize='11px';t2.style.textAnchor='middle';t2.style.pointerEvents='none';
        if(dim)t2.style.opacity=isDark?'0.4':'0.2';

        if(trunc>0){
          t1.textContent=nd.f+' '+nd.op+' '+nd.th.toFixed(4);
          t2.textContent='\u25bc expand (+'+trunc+')';t2.style.fill=sampleValColor;t2.style.fontSize='9px';t2.style.cursor='pointer';
          rect.style.cursor='pointer';rect.style.strokeDasharray='4,2';
          var xp=function(e){e.stopPropagation();expandedNodes[nPath]=(expandedNodes[nPath]||0)+3;rerender();};
          rect.addEventListener('click',xp);t2.addEventListener('click',xp);
        }else if(nd.t==='s'){
          t1.textContent=nd.f+' '+nd.op+' '+nd.th.toFixed(4);t2.textContent='';
          if(sample&&onP&&sample[nd.f]!==undefined){var vt=document.createElementNS(ns,'text');vt.setAttribute('x',pos.x);vt.setAttribute('y',ry+55);vt.setAttribute('class','node-text sample-val');vt.style.fill=sampleValColor;vt.style.fontSize='10px';vt.style.fontStyle='italic';vt.style.fontWeight='500';vt.style.textAnchor='middle';vt.style.pointerEvents='none';vt.textContent=nd.f+' = '+sample[nd.f].toFixed(3);el.appendChild(vt);}
        }else{
          if(nd.c){var best='',bv=-1;for(var k in nd.c){if(nd.c[k]>bv){bv=nd.c[k];best=k;}}
            if(IS_BOOSTED){t1.textContent='🌿 Leaf correction';var vals=[];for(var k2 in nd.c){vals.push(k2+':'+nd.c[k2].toFixed(3));}t2.textContent=vals.join(' ');}
            else{t1.textContent='🌿 Class: '+best;t2.textContent=(bv*100).toFixed(0)+'%';}}
          else if(nd.v!==undefined){t1.textContent='🌿 '+nd.v.toFixed(4);t2.textContent='';}
        }
        el.appendChild(t1);if(t2.textContent)el.appendChild(t2);
      })(j);}

      return el;
    }
  })();

  // --- Prediction panel ---
  (function() {
    if (!HAS_PREDICT) return;
    var dataEl = $('predict-data');
    if (!dataEl) return;
    var predData = JSON.parse(dataEl.textContent);
    var goBtn = $('predict-go');
    var clearBtn = $('predict-clear');
    var rowInput = $('predict-row');
    var resultEl = $('predict-result');
    var closeBtn = $('predict-close');
    var panel = $('predict-panel');

    if (closeBtn && panel) {
      closeBtn.addEventListener('click', function() { panel.style.display = 'none'; });
    }

    function traceTree(node, sample) {
      if (node.t === 'l') {
        if (node.c) {
          var best = null, bestV = -1;
          for (var k in node.c) { if (node.c[k] > bestV) { bestV = node.c[k]; best = k; } }
          return { cls: best, dist: node.c };
        }
        return { val: node.v };
      }
      var fv = sample[node.f];
      if (fv === undefined) return { err: 'missing ' + node.f };
      var goLeft = false;
      switch(node.op) {
        case '<=': goLeft = fv <= node.th; break;
        case '<': goLeft = fv < node.th; break;
        case '>=': goLeft = fv >= node.th; break;
        case '>': goLeft = fv > node.th; break;
        case '==': goLeft = fv == node.th; break;
        case '!=': goLeft = fv != node.th; break;
        default: goLeft = fv <= node.th;
      }
      return traceTree(goLeft ? node.l : node.r, sample);
    }

    function clearBadges() {
      svg.querySelectorAll('.pred-label').forEach(function(el) { el.remove(); });
      if (resultEl) resultEl.textContent = '';
      var sd = $('sample-display');
      if (sd) sd.classList.remove('visible');
      traceActive = false;
    }

    function showSampleDisplay(sample, idx) {
      var sd = $('sample-display');
      if (!sd) return;
      var html = '<span class="sample-title">Sample #' + idx + '</span>';
      html += '<div class="sample-chips">';
      for (var key in sample) {
        var val = sample[key];
        var display = (typeof val === 'number') ? val.toFixed(3) : String(val);
        html += '<span class="chip"><span class="chip-name">' + key + ':</span><span class="chip-val">' + display + '</span></span>';
      }
      html += '</div>';
      sd.innerHTML = html;
      sd.classList.add('visible');
    }

    if (clearBtn) clearBtn.addEventListener('click', clearBadges);

    if (goBtn) goBtn.addEventListener('click', function() {
      clearBadges();
      var idx = parseInt(rowInput.value);
      if (isNaN(idx) || idx < 0 || idx >= predData.samples.length) {
        if (resultEl) resultEl.textContent = 'Row out of range (0–' + (predData.samples.length - 1) + ')';
        return;
      }
      var sample = predData.samples[idx];
      showSampleDisplay(sample, idx);
      traceActive = true;
      var predictions = [];

      for (var t = 0; t < predData.trees.length; t++) {
        predictions.push({ idx: t, result: traceTree(predData.trees[t], sample) });
      }

      var agg = predData.aggregation || 'avg';
      var visibleTrees = treeData.filter(function(d) { return !d.el.classList.contains('hidden'); });
      visibleTrees.forEach(function(d) {
        var pred = predictions[d.idx];
        if (!pred) return;
        var label = '';
        var color = '#1565c0';
        if (agg === 'sum') {
          if (pred.result.val !== undefined) {
            label = pred.result.val.toFixed(2);
            color = pred.result.val >= 0 ? '#2e7d32' : '#c62828';
          } else if (pred.result.cls !== undefined && pred.result.dist) {
            var vals = Object.values(pred.result.dist);
            label = vals[0].toFixed(2);
            color = vals[0] >= 0 ? '#2e7d32' : '#c62828';
          }
        } else {
          if (pred.result.cls !== undefined) {
            label = pred.result.cls;
            var colors = ['#1565c0','#c62828','#2e7d32','#f57c00','#6a1b9a','#00838f'];
            color = colors[parseInt(label) % colors.length];
          } else if (pred.result.val !== undefined) {
            label = pred.result.val.toFixed(2);
            color = '#333';
          }
        }
        if (!label) return;

        var badge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        badge.setAttribute('class', 'pred-label');
        var bgR = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bgR.setAttribute('x', '-18'); bgR.setAttribute('y', '-165');
        bgR.setAttribute('width', '36'); bgR.setAttribute('height', '16');
        bgR.setAttribute('rx', '3'); bgR.setAttribute('fill', color); bgR.setAttribute('opacity', '0.9');
        var txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        txt.setAttribute('x', '0'); txt.setAttribute('y', '-153');
        txt.setAttribute('text-anchor', 'middle'); txt.setAttribute('font-size', '10');
        txt.setAttribute('fill', 'white'); txt.setAttribute('font-weight', 'bold');
        txt.textContent = label;
        badge.appendChild(bgR); badge.appendChild(txt);
        d.el.appendChild(badge);
      });

      var modelPred = (predData.predictions && idx < predData.predictions.length) ? predData.predictions[idx] : null;

      if (resultEl) {
        if (modelPred !== null) {
          resultEl.innerHTML = 'Ensemble prediction: <strong>' + modelPred + '</strong>';
        } else if (predData.is_classifier) {
          if (agg === 'sum') {
            var classSums = {};
            predictions.forEach(function(p) {
              if (p.result.val !== undefined) { classSums['_'] = (classSums['_']||0) + p.result.val; }
              else if (p.result.dist) { for (var k in p.result.dist) { classSums[k] = (classSums[k]||0) + p.result.dist[k]; } }
            });
            var bestCls = '', bestSum = -Infinity;
            for (var k in classSums) { if (classSums[k] > bestSum) { bestSum = classSums[k]; bestCls = k; } }
            resultEl.innerHTML = 'Ensemble score: <strong>' + bestSum.toFixed(3) + '</strong> (' + predictions.length + ' trees)';
          } else {
            var votes = {};
            predictions.forEach(function(p) { if (p.result.cls) votes[p.result.cls] = (votes[p.result.cls]||0) + 1; });
            var best = '', bestCount = 0;
            for (var k in votes) { if (votes[k] > bestCount) { bestCount = votes[k]; best = k; } }
            resultEl.innerHTML = 'Ensemble prediction: <strong>' + best + '</strong> (' + bestCount + '/' + predictions.length + ' votes)';
          }
        } else {
          var sum = 0, cnt = 0;
          predictions.forEach(function(p) { if (p.result.val !== undefined) { sum += p.result.val; cnt++; } });
          if (agg === 'sum') {
            var final = sum;
            var detail = 'sum of ' + cnt + ' trees';
            if (predData.boosting) {
              var lr = predData.boosting.lr || 1;
              var init = predData.boosting.init || 0;
              final = init + lr * sum;
              detail = 'init(' + init.toFixed(2) + ') + ' + lr + ' × sum(' + sum.toFixed(2) + ')';
            }
            resultEl.innerHTML = 'Ensemble prediction: <strong>' + final.toFixed(4) + '</strong> (' + detail + ')';
          } else {
            var avg = cnt > 0 ? (sum / cnt).toFixed(4) : '?';
            resultEl.innerHTML = 'Ensemble prediction: <strong>' + avg + '</strong> (avg of ' + cnt + ' trees)';
          }
        }
        if (predData.targets && idx < predData.targets.length && predData.targets[idx] !== null) {
          var trueVal = predData.targets[idx];
          resultEl.innerHTML += ' | True: <strong style="color:#2e7d32">' + trueVal + '</strong>';
        }
      }
    });
  })();

  // --- Growth animation ---
  (function() {
    var visible = allTrees.filter(function(t) { return !t.classList.contains('hidden'); });
    if (visible.length > 200) {
      visible.forEach(function(t) { t.style.opacity='1'; t.classList.add('grown'); });
      return;
    }
    visible.sort(function(a, b) {
      var ay = parseFloat((a.getAttribute('transform')||'').replace(/.*translate\([^,]+,([^)]+)\).*/, '$1'))||0;
      var by = parseFloat((b.getAttribute('transform')||'').replace(/.*translate\([^,]+,([^)]+)\).*/, '$1'))||0;
      return ay - by;
    });
    var delay = Math.max(15, Math.min(50, 1200 / visible.length));
    visible.forEach(function(tree, i) {
      var d = i * delay;
      setTimeout(function() { tree.style.opacity='1'; tree.classList.add('grow-trunk'); }, d);
      setTimeout(function() { tree.classList.add('grow-branches'); }, d + 200);
      setTimeout(function() { tree.classList.add('grow-canopy'); }, d + 380);
      setTimeout(function() { tree.classList.add('grown'); }, d + 650);
    });
  })();
}

function render({ model, el }) {
  el.style.height = "100%";
  el.style.minHeight = "700px";
  el.style.position = "relative";
  el.style.overflow = "hidden";

  var raw = model.get("html_content");
  var bodyMatch = raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  var bodyContent = bodyMatch ? bodyMatch[1] : raw;

  var root = document.createElement("div");
  root.className = "pf-root";
  root.style.position = "relative";
  root.style.height = "100%";
  root.style.minHeight = "700px";
  root.style.overflow = "hidden";
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.innerHTML = bodyContent;
  el.appendChild(root);

  var detailModal = root.querySelector("#detail-modal");
  if (detailModal) {
    detailModal.style.position = "absolute";
    detailModal.style.top = "0";
    detailModal.style.left = "0";
    detailModal.style.width = "100%";
    detailModal.style.height = "100%";
    detailModal.style.zIndex = "2000";
  }

  var config = {};
  var configEl = root.querySelector("#forest-config");
  if (configEl) {
    try {
      config = JSON.parse(configEl.textContent);
    } catch (e) {
      console.error("[PrettyForest widget] Error parsing #forest-config JSON:", e);
    }
  }

  initForest(root, config);

  model.on("change:season", function () {
    var select = root.querySelector("#season-toggle");
    if (select) {
      select.value = model.get("season") || "";
      select.dispatchEvent(new Event("change"));
    }
  });

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
