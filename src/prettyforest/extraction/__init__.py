from __future__ import annotations

from typing import Any

from prettyforest.extraction.catboost_extractor import CatBoostExtractor
from prettyforest.extraction.lightgbm_extractor import LightGBMExtractor
from prettyforest.extraction.protocol import TreeExtractor
from prettyforest.extraction.sklearn_extractor import SklearnExtractor
from prettyforest.models import UnifiedTree

SUPPORTED_TYPES = (
    "DecisionTreeClassifier, DecisionTreeRegressor, "
    "RandomForestClassifier, RandomForestRegressor, "
    "GradientBoostingClassifier, GradientBoostingRegressor, "
    "LGBMClassifier, LGBMRegressor, Booster, "
    "CatBoostClassifier, CatBoostRegressor"
)


class TreeExtractorRegistry:
    def __init__(self) -> None:
        self._extractors: dict[str, TreeExtractor] = {
            "sklearn": SklearnExtractor(),
            "lightgbm": LightGBMExtractor(),
            "catboost": CatBoostExtractor(),
        }

    def extract(
        self, model: Any, tree_index: int | None = None
    ) -> UnifiedTree | list[UnifiedTree]:
        """
        Dispatch to framework-specific extractor.

        Raises:
            TypeError: if model type is unsupported
            ValueError: if model is not fitted
            IndexError: if tree_index is out of range
        """
        framework = self._detect_framework(model)
        extractor = self._extractors[framework]

        if not extractor.is_fitted(model):
            msg = "Model must be fitted before visualization. Call model.fit(X, y) first."
            raise ValueError(msg)

        trees = extractor.extract(model)

        if tree_index is not None:
            n = len(trees)
            if tree_index < 0 or tree_index >= n:
                msg = f"Tree index {tree_index} out of range. Valid range: [0, {n})."
                raise IndexError(msg)
            return trees[tree_index]

        return trees[0] if len(trees) == 1 else trees

    def _detect_framework(self, model: Any) -> str:
        """Return 'sklearn', 'lightgbm', or 'catboost'."""
        module = type(model).__module__ or ""

        if module.startswith("sklearn"):
            return "sklearn"
        if module.startswith("lightgbm"):
            return "lightgbm"
        if module.startswith("catboost"):
            return "catboost"

        type_name = type(model).__qualname__
        msg = f"Unsupported model type '{type_name}'. Supported: {SUPPORTED_TYPES}"
        raise TypeError(msg)
