from __future__ import annotations

from typing import Any, Protocol

from prettyforest.models import UnifiedTree


class TreeExtractor(Protocol):
    def extract(self, model: Any) -> list[UnifiedTree]:
        """Extract all trees from the model."""
        ...

    def is_fitted(self, model: Any) -> bool:
        """Check if model has been trained."""
        ...

    def tree_count(self, model: Any) -> int:
        """Return number of trees in the model."""
        ...
