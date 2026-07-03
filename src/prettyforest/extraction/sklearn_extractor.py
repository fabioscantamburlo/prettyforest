from __future__ import annotations

from typing import Any


from prettyforest.models import ComparisonOp, UnifiedNode, UnifiedTree

TREE_LEAF = -1
TREE_UNDEFINED = -2


class SklearnExtractor:
    def extract(self, model: Any) -> list[UnifiedTree]:
        estimators = self._get_estimators(model)
        is_classifier = self._is_classifier(model)
        feature_names = self._get_feature_names(model, estimators[0])
        class_names = self._get_class_names(model) if is_classifier else None

        # For GradientBoosting classifiers, individual trees predict residuals (not classes)
        is_gbm = hasattr(model, "estimators_") and hasattr(model.estimators_, "ravel")
        if is_gbm and is_classifier:
            # Each sub-tree is a regressor (gradient for one class)
            return [
                self._extract_single(est, feature_names, False, None)
                for est in estimators
            ]

        return [
            self._extract_single(est, feature_names, is_classifier, class_names)
            for est in estimators
        ]

    def is_fitted(self, model: Any) -> bool:
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(model)
            return True
        except Exception:
            return hasattr(model, "tree_") or hasattr(model, "estimators_")

    def tree_count(self, model: Any) -> int:
        return len(self._get_estimators(model))

    def _get_estimators(self, model: Any) -> list[Any]:
        if hasattr(model, "estimators_"):
            # GradientBoosting stores estimators as 2D array (n_estimators, n_classes)
            estimators = model.estimators_
            if hasattr(estimators, "ravel"):
                return list(estimators.ravel())
            # RandomForest stores as a flat list
            return list(estimators)
        # Single tree
        return [model]

    def _is_classifier(self, model: Any) -> bool:
        from sklearn.base import is_classifier

        return is_classifier(model)

    def _get_feature_names(self, model: Any, fallback_estimator: Any) -> list[str]:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        tree = fallback_estimator.tree_
        n_features = tree.n_features
        return [f"feature_{i}" for i in range(n_features)]

    def _get_class_names(self, model: Any) -> list[str] | None:
        if hasattr(model, "classes_"):
            return [str(c) for c in model.classes_]
        return None

    def _extract_single(
        self,
        estimator: Any,
        feature_names: list[str],
        is_classifier: bool,
        class_names: list[str] | None,
    ) -> UnifiedTree:
        tree = estimator.tree_
        root = self._build_node(tree, 0, 0, feature_names, is_classifier, class_names)
        return UnifiedTree(
            root=root,
            node_count=tree.node_count,
            max_depth=int(tree.max_depth),
            feature_names=feature_names,
            is_classifier=is_classifier,
            class_names=class_names,
        )

    def _build_node(
        self,
        tree: Any,
        node_idx: int,
        depth: int,
        feature_names: list[str],
        is_classifier: bool,
        class_names: list[str] | None,
    ) -> UnifiedNode:
        left_child_idx = tree.children_left[node_idx]
        right_child_idx = tree.children_right[node_idx]
        feature_idx = tree.feature[node_idx]

        is_leaf = left_child_idx == TREE_LEAF or feature_idx == TREE_UNDEFINED

        if is_leaf:
            value = tree.value[node_idx]
            if is_classifier:
                counts = value.flatten().astype(float)
                total = counts.sum()
                proportions = (counts / total) if total > 0 else counts
                if class_names and len(class_names) == len(proportions):
                    distribution = {
                        class_names[i]: float(proportions[i])
                        for i in range(len(class_names))
                    }
                else:
                    distribution = {
                        str(i): float(proportions[i]) for i in range(len(proportions))
                    }
                return UnifiedNode(
                    node_id=str(node_idx),
                    depth=depth,
                    class_distribution=distribution,
                )
            else:
                prediction = float(value.flatten()[0])
                return UnifiedNode(
                    node_id=str(node_idx),
                    depth=depth,
                    prediction_value=prediction,
                )

        # Internal node
        threshold = float(tree.threshold[node_idx])
        fname = (
            feature_names[feature_idx]
            if feature_idx < len(feature_names)
            else f"feature_{feature_idx}"
        )

        left = self._build_node(
            tree, left_child_idx, depth + 1, feature_names, is_classifier, class_names
        )
        right = self._build_node(
            tree, right_child_idx, depth + 1, feature_names, is_classifier, class_names
        )

        return UnifiedNode(
            node_id=str(node_idx),
            depth=depth,
            feature_name=fname,
            threshold=threshold,
            comparison_op=ComparisonOp.LE,
            left_child=left,
            right_child=right,
        )
