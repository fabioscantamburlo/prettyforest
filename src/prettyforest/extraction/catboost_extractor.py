from __future__ import annotations

import json
import math
import tempfile
from typing import Any

from prettyforest.models import ComparisonOp, UnifiedNode, UnifiedTree


class CatBoostExtractor:
    def extract(self, model: Any) -> list[UnifiedTree]:
        model_json = self._export_json(model)
        is_classifier = self._is_classifier(model)
        feature_names = self._get_feature_names(model)
        class_names = self._get_class_names(model) if is_classifier else None
        n_classes = len(class_names) if class_names else 0

        trees: list[UnifiedTree] = []
        for tree_data in model_json["oblivious_trees"]:
            root, node_count, max_depth = self._build_tree(
                tree_data, feature_names, is_classifier, n_classes
            )
            trees.append(
                UnifiedTree(
                    root=root,
                    node_count=node_count,
                    max_depth=max_depth,
                    feature_names=feature_names,
                    is_classifier=is_classifier,
                    class_names=class_names,
                )
            )
        return trees

    def is_fitted(self, model: Any) -> bool:
        try:
            return model.tree_count_ > 0
        except Exception:
            return False

    def tree_count(self, model: Any) -> int:
        return model.tree_count_

    def _is_classifier(self, model: Any) -> bool:
        return type(model).__name__ == "CatBoostClassifier"

    def _get_feature_names(self, model: Any) -> list[str]:
        names = model.feature_names_
        if names:
            return list(names)
        n_features = len(model.feature_importances_)
        return [f"feature_{i}" for i in range(n_features)]

    def _get_class_names(self, model: Any) -> list[str] | None:
        try:
            return [str(c) for c in model.classes_]
        except Exception:
            return None

    def _export_json(self, model: Any) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="r") as f:
            model.save_model(f.name, format="json")
            return json.load(f)

    def _build_tree(
        self,
        tree_data: dict,
        feature_names: list[str],
        is_classifier: bool,
        n_classes: int,
    ) -> tuple[UnifiedNode, int, int]:
        # CatBoost splits are ordered bottom-to-top: splits[-1] is root, splits[0] is deepest
        splits = tree_data.get("splits", [])
        leaf_values = tree_data["leaf_values"]
        depth = len(splits)
        n_leaves = 2**depth

        # Determine values-per-leaf: for binary classification it's 1, for multi-class it's n_classes
        values_per_leaf = len(leaf_values) // n_leaves if n_leaves > 0 else 1

        if depth == 0:
            leaf = self._make_leaf(
                "0", 0, leaf_values, 0, is_classifier, n_classes, values_per_leaf
            )
            return leaf, 1, 0

        # Reverse splits so index 0 = root, index depth-1 = deepest
        splits_root_first = list(reversed(splits))
        counter = [0]

        def build(level: int, leaf_offset: int, num_leaves: int) -> UnifiedNode:
            node_id = str(counter[0])
            counter[0] += 1

            if level == depth:
                return self._make_leaf(
                    node_id, level, leaf_values, leaf_offset,
                    is_classifier, n_classes, values_per_leaf,
                )

            split = splits_root_first[level]
            feature_idx = split["float_feature_index"]
            threshold = split["border"]
            feat_name = (
                feature_names[feature_idx]
                if feature_idx < len(feature_names)
                else f"feature_{feature_idx}"
            )

            half = num_leaves // 2
            node = UnifiedNode(
                node_id=node_id,
                depth=level,
                feature_name=feat_name,
                threshold=threshold,
                comparison_op=ComparisonOp.LE,
            )
            node.left_child = build(level + 1, leaf_offset, half)
            node.right_child = build(level + 1, leaf_offset + half, half)
            return node

        root = build(0, 0, n_leaves)
        return root, counter[0], depth

    def _make_leaf(
        self,
        node_id: str,
        depth: int,
        leaf_values: list[float],
        leaf_index: int,
        is_classifier: bool,
        n_classes: int,
        values_per_leaf: int,
    ) -> UnifiedNode:
        if is_classifier and values_per_leaf > 1:
            # Multi-class: n_classes logits per leaf
            offset = leaf_index * values_per_leaf
            raw = leaf_values[offset : offset + values_per_leaf]
            max_val = max(raw)
            exp_vals = [math.exp(v - max_val) for v in raw]
            total = sum(exp_vals)
            proportions = [e / total for e in exp_vals]
            class_dist = {str(i): p for i, p in enumerate(proportions)}
            return UnifiedNode(
                node_id=node_id,
                depth=depth,
                class_distribution=class_dist,
            )
        elif is_classifier:
            # Binary classifier: single logit per leaf
            logit = leaf_values[leaf_index]
            prob = 1.0 / (1.0 + math.exp(-logit))
            class_dist = {"0": 1.0 - prob, "1": prob}
            return UnifiedNode(
                node_id=node_id,
                depth=depth,
                class_distribution=class_dist,
            )
        else:
            value = leaf_values[leaf_index]
            return UnifiedNode(
                node_id=node_id,
                depth=depth,
                prediction_value=value,
            )
