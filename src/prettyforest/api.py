"""Public API for PrettyForest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from prettyforest.analysis import numpy_to_polars
from prettyforest.extraction import TreeExtractorRegistry
from prettyforest.models import EnsembleType


def prettygrow(
    model: Any,
    *,
    data: pl.DataFrame | np.ndarray | None = None,
    target: pl.Series | np.ndarray | list | None = None,
    output_path: str | Path | None = None,
    feature_names: list[str] | None = None,
    seed: int = 42,
    season: Literal["spring", "summer", "autumn", "winter"] | None = None,
) -> str | None:
    """Visualize a trained tree-based model as an interactive forest.

    Args:
        model: Trained tree model (sklearn, LightGBM, or CatBoost)
        data: Dataset for prediction tracing (Polars DataFrame or NumPy array)
        target: True labels/values — shown alongside predictions for comparison
        output_path: File path to write HTML; if None, returns string or renders in Jupyter
        feature_names: Column names when data is a NumPy array
        seed: Random seed for layout (default 42)
        season: Color theme ("summer", "autumn", "winter", or None for metric-based)

    Returns:
        HTML string if no output_path and not in Jupyter; None otherwise
    """
    registry = TreeExtractorRegistry()
    result = registry.extract(model)

    if isinstance(result, list):
        trees = result
    else:
        trees = [result]

    ensemble_type = _detect_ensemble_type(model)
    model_name = _detect_model_name(model)

    from prettyforest.rendering.forest import ForestConfig, ForestRenderer

    config = ForestConfig(seed=seed, season=season)

    forest_data = None
    if data is not None:
        if isinstance(data, np.ndarray):
            names = feature_names or trees[0].feature_names
            forest_data = numpy_to_polars(data, names)
        else:
            forest_data = data

    # Convert target to a list for JSON embedding
    target_list = None
    if target is not None:
        if isinstance(target, np.ndarray):
            target_list = target.tolist()
        elif isinstance(target, pl.Series):
            target_list = target.to_list()
        else:
            target_list = [int(x) if hasattr(x, "item") else x for x in target]

    # Extract boosting metadata for correct prediction aggregation
    boosting_meta = _extract_boosting_meta(model)

    # Pre-compute the model's actual predictions for the embedded samples
    model_predictions = None
    if data is not None:
        try:
            if isinstance(data, np.ndarray):
                pred_input = data[:100]
            else:
                pred_input = data.head(100).to_numpy()
            model_predictions = model.predict(pred_input).tolist()
        except Exception:
            pass

    html = ForestRenderer().render(
        trees,
        config,
        ensemble_type=ensemble_type,
        data=forest_data,
        boosting_meta=boosting_meta,
        target=target_list,
        model_name=model_name,
        model_predictions=model_predictions,
    )
    return _handle_output(html, output_path)


def _detect_ensemble_type(model: Any) -> EnsembleType:
    module = type(model).__module__ or ""
    name = type(model).__name__

    if "RandomForest" in name:
        return EnsembleType.VOTE_BASED
    if "GradientBoosting" in name or "LGBM" in name or "CatBoost" in name:
        return EnsembleType.ADDITIVE
    if module.startswith("lightgbm") or module.startswith("catboost"):
        return EnsembleType.ADDITIVE
    return EnsembleType.SINGLE


def _extract_boosting_meta(model: Any) -> dict | None:
    """Extract learning rate and initial prediction for sklearn GBM."""
    name = type(model).__name__
    if "GradientBoosting" not in name:
        return None
    meta = {}
    if hasattr(model, "learning_rate"):
        meta["lr"] = float(model.learning_rate)
    if hasattr(model, "init_") and model.init_ != "zero":
        try:
            # For regressors: init_.constant_[0]
            if hasattr(model.init_, "constant_"):
                meta["init"] = float(model.init_.constant_.flat[0])
        except Exception:
            pass
    return meta if meta else None


def _handle_output(html: str, output_path: str | Path | None):
    if output_path is not None:
        path = Path(output_path)
        try:
            path.write_text(html, encoding="utf-8")
        except (OSError, PermissionError) as e:
            msg = f"Cannot write to '{path}': {e}"
            raise OSError(msg) from e
        return None

    if _in_notebook():
        try:
            from prettyforest.widget import PrettyForestWidget

            return PrettyForestWidget(html_content=html)
        except ImportError:
            try:
                from IPython.display import HTML, display

                display(HTML(html))
                return None
            except ImportError:
                pass

    # Not in a notebook — return raw HTML string
    return html


def _in_notebook() -> bool:
    """Detect if running inside an interactive notebook environment (Jupyter, Marimo, Colab)."""
    import sys

    # Marimo check
    if "marimo" in sys.modules:
        try:
            import marimo as mo

            if mo.running_in_notebook():
                return True
        except Exception:
            pass

    # Colab check
    if "google.colab" in sys.modules:
        return True

    # IPython / Jupyter check
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name in ("ZMQInteractiveShell", "Shell"):
            return True
        if hasattr(shell, "config") and "IPKernelApp" in shell.config:
            return True
    except (ImportError, AttributeError):
        pass

    return False



def _detect_model_name(model: Any) -> str:
    """Return a human-readable model name."""
    name = type(model).__name__
    mapping = {
        "DecisionTreeClassifier": "Decision Tree (Classification)",
        "DecisionTreeRegressor": "Decision Tree (Regression)",
        "RandomForestClassifier": "Random Forest (Classification)",
        "RandomForestRegressor": "Random Forest (Regression)",
        "GradientBoostingClassifier": "Gradient Boosting (Classification)",
        "GradientBoostingRegressor": "Gradient Boosting (Regression)",
        "LGBMClassifier": "LightGBM (Classification)",
        "LGBMRegressor": "LightGBM (Regression)",
        "CatBoostClassifier": "CatBoost (Classification)",
        "CatBoostRegressor": "CatBoost (Regression)",
    }
    return mapping.get(name, name)
