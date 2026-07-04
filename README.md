# 🌲 PrettyForest

Interactive 2.5D forest visualization for tree-based ML ensembles. Explore your model's structure, trace predictions through individual trees, and understand how the ensemble makes decisions.

<p align="center">
  <img src="assets/demo.gif" alt="PrettyForest Demo" width="700"/>
</p>

## Features

- **2.5D isometric forest** — trees rendered with depth perspective, growth animation, seasonal themes
- **All major frameworks** — scikit-learn, LightGBM, CatBoost (RandomForest, GradientBoosting, single DecisionTree)
- **Prediction tracing** — select a data point, see per-tree corrections/votes, get the ensemble's actual prediction
- **True label comparison** — pass `target=y_test` to see if the model got it right
- **Interactive detail view** — double-click a tree to drill into its decision structure with per-node expansion
- **Boosted tree awareness** — correct display of gradient corrections vs class probabilities, with explanatory notes
- **Dark mode** — toggle with 🌙 button
- **Scales to thousands** — pagination, sorting, highlighting for large ensembles

## Installation

```bash
pip install prettyforest

# With optional frameworks
pip install prettyforest[all]  # includes lightgbm + catboost
```

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import polars as pl
from prettyforest import visualize

# Train
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Visualize
X_test_pl = pl.DataFrame({name: X_test[:, i] for i, name in enumerate(iris.feature_names)})
visualize(model, data=X_test_pl, target=y_test, output_path="forest.html")
```

## API

```python
from prettyforest import visualize

visualize(
    model,              # Trained tree model (sklearn, LightGBM, or CatBoost)
    *,
    data=None,          # Polars DataFrame or NumPy array for prediction tracing
    target=None,        # True labels/values — shown alongside predictions
    output_path=None,   # Write HTML to file; if None, returns string or displays in notebook
    feature_names=None, # Column names when data is a NumPy array
    seed=42,            # Random seed for layout
    season=None,        # Color theme: "summer", "autumn", "winter", or None (metric-based)
)
```

**Returns:** HTML string if no `output_path` and not in a notebook; `None` otherwise.

## Supported Models

| Model | Type | Trees | Leaf values |
|-------|------|-------|-------------|
| `DecisionTreeClassifier/Regressor` | Single tree | 1 | Direct predictions |
| `RandomForestClassifier/Regressor` | Bagging | N independent | Class proportions / target means |
| `GradientBoostingClassifier/Regressor` | Boosting | N × classes sequential | Gradient corrections |
| `LGBMClassifier/Regressor` | Boosting | N × classes sequential | Log-odds / residuals |
| `CatBoostClassifier/Regressor` | Boosting | N sequential | Ordered gradient corrections |

## Interaction Guide

| Action | What it does |
|--------|-------------|
| **Hover** a tree | Tooltip with depth, nodes, purity/magnitude |
| **Click** a tree | Spotlight panel with full stats + rank |
| **Double-click** a tree | Full decision structure with per-node expand |
| **Sort by** dropdown | Rearrange into grid by depth/nodes/leaves/metric |
| **◀ ▶** | Page through large ensembles (200/page) |
| **Top 3 / Bottom 3** | Show only best/worst trees by metric |
| **🌙** | Toggle dark mode |
| **?** | Model description + how the ensemble works |
| **Trace** | Show per-tree badges + ensemble prediction + true label |
| **Click a truncated node** | Expand that subtree 3 more levels |

## Seasons & Themes

Switch the visual theme live in the browser or set it via Python:

```python
visualize(model, season="spring")   # 🌸 Light greens + pink blossoms
visualize(model, season="summer")   # 🌿 Deep lush greens
visualize(model, season="autumn")   # 🍂 Warm oranges, reds, golds
visualize(model, season="winter")   # ❄️ Bare branches, blue-grey
visualize(model)                    # 🌳 Natural (metric-based coloring)
```

You can also switch seasons on the fly using the dropdown in the header — no need to re-run Python. The canopies, ground, sky, and grass patches all update instantly.

| Season | Canopy colors | Ground | Best for |
|--------|--------------|--------|----------|
| 🌳 Natural | Green→amber by metric | Soft green | Analysis (purity/variance encoded in color) |
| 🌸 Spring | Light green + pink/purple | Fresh green | Presentations |
| 🌿 Summer | Deep forest greens | Rich green | Dense forests |
| 🍂 Autumn | Orange, red, gold | Warm brown | Warm aesthetics |
| ❄️ Winter | Bare (no canopy) | Blue-grey | Seeing structure clearly |

## Prediction Display

When you trace a sample:

- **Per-tree badges** show on each tree's canopy:
  - *Random Forest*: the class vote (colored by class)
  - *Boosted models*: the raw correction value (green = positive, red = negative)
- **Ensemble prediction**: the actual `model.predict()` output — always correct
- **True label**: shown in green if `target` was provided

## Marimo Integration

```python
import marimo as mo
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import polars as pl
from prettyforest import visualize

iris = load_iris()
model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
model.fit(iris.data, iris.target)

X_pl = pl.DataFrame({name: iris.data[:, i] for i, name in enumerate(iris.feature_names)})
html = visualize(model, data=X_pl, target=iris.target)
mo.iframe(html, height="700px")
```

Use `mo.iframe()` (not `mo.Html()`) — PrettyForest needs JavaScript execution which marimo only allows inside iframes. Full interactivity works: zoom, pan, click, trace, expand.

## Jupyter Integration

```python
from prettyforest import visualize
# ... train model ...
visualize(model, data=X_test)  # auto-displays via IPython.display.HTML
```

## Understanding Boosted vs Bagged Trees

**Random Forest (bagged):** Each tree trains independently on a random data subset. Leaves contain real class proportions or target means. Ensemble averages/votes.

**Gradient Boosting / LightGBM / CatBoost (boosted):** Trees train sequentially — each corrects the previous ensemble's errors. Leaf values are small gradient adjustments, not standalone predictions. Ensemble sums corrections.

When you double-click a boosted tree, a warning appears:
> ⚠️ This is a boosted tree — leaf values are gradient corrections, not final predictions.

The splits and features are fully interpretable — they show which features matter and how the space is partitioned. The leaf values just represent "how much to adjust" rather than "what to predict."

## Development

```bash
git clone https://github.com/youruser/prettyforest.git
cd prettyforest
uv sync

# Run tests
uv run pytest

# Run experiments (all models)
uv run run_experiments.py --n-samples 1000 --n-trees 50 --max-depth 8

# Iris example (quick, all frameworks)
uv run examples/iris_forest.py

# MNIST example (larger, RF only)
uv run examples/mnist_forest.py
```

## Pre-commit

```bash
pre-commit install  # set up hooks
pre-commit run --all-files  # manual run
```

Hooks: trailing whitespace, end-of-file, YAML check, large file guard (500KB), ruff lint + format.

## License

MIT
