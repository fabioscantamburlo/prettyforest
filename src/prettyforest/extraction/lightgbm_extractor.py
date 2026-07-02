from __future__ import annotations

from typing import Any

from prettyforest.models import ComparisonOp, UnifiedNode, UnifiedTree


class LightGBMExtractor:
    def extract(self, model: Any) -> list[UnifiedTree]:
        booster = self._get_booster(model)
        dump = booster.dump_model()
        tree_infos = dump["tree_info"]
        feature_names = booster.feature_name()
        is_classifier = self._is_classifier(model)
        class_names = self._get_class_names(model) if is_classifier else None

        trees: list[UnifiedTree] = []
        for tree_info in tree_infos:
            root_dict = tree_info["tree_structure"]
            root, count, max_depth = self._build_node(root_dict, feature_names, depth=0)
            trees.append(
                UnifiedTree(
                    root=root,
                    node_count=count,
                    max_depth=max_depth,
                    feature_names=feature_names,
                    is_classifier=is_classifier,
                    class_names=class_names,
                )
            )
        return trees

    def is_fitted(self, model: Any) -> bool:
        import lightgbm as lgb

        if isinstance(model, lgb.Booster):
            return True
        return hasattr(model, "booster_")

    def tree_count(self, model: Any) -> int:
        booster = self._get_booster(model)
        dump = booster.dump_model()
        return len(dump["tree_info"])

    def _get_booster(self, model: Any) -> Any:
        import lightgbm as lgb

        if isinstance(model, lgb.Booster):
            return model
        return model.booster_

    def _is_classifier(self, model: Any) -> bool:
        import lightgbm as lgb

        if isinstance(model, lgb.LGBMClassifier):
            return True
        if isinstance(model, lgb.LGBMRegressor):
            return False
        # For raw Booster, check objective
        if isinstance(model, lgb.Booster):
            dump = model.dump_model()
            objective = dump.get("objective", "")
            return "binary" in objective or "multiclass" in objective
        return hasattr(model, "classes_")

    def _get_class_names(self, model: Any) -> list[str] | None:
        if hasattr(model, "classes_"):
            return [str(c) for c in model.classes_]
        return None

    def _build_node(
        self, node_dict: dict, feature_names: list[str], depth: int
    ) -> tuple[UnifiedNode, int, int]:
        if "leaf_value" in node_dict:
            node = UnifiedNode(
                node_id=str(node_dict.get("leaf_index", f"leaf_{depth}")),
                depth=depth,
                prediction_value=float(node_dict["leaf_value"]),
            )
            return node, 1, depth

        split_feature_idx = node_dict["split_feature"]
        feature_name = feature_names[split_feature_idx] if isinstance(split_feature_idx, int) else str(split_feature_idx)
        threshold = float(node_dict["threshold"])
        decision_type = node_dict.get("decision_type", "<=")
        comparison_op = self._parse_comparison_op(decision_type)

        left_child, left_count, left_max = self._build_node(
            node_dict["left_child"], feature_names, depth + 1
        )
        right_child, right_count, right_max = self._build_node(
            node_dict["right_child"], feature_names, depth + 1
        )

        node = UnifiedNode(
            node_id=str(node_dict.get("split_index", f"split_{depth}")),
            depth=depth,
            feature_name=feature_name,
            threshold=threshold,
            comparison_op=comparison_op,
            left_child=left_child,
            right_child=right_child,
        )
        count = 1 + left_count + right_count
        max_depth = max(left_max, right_max)
        return node, count, max_depth

    def _parse_comparison_op(self, decision_type: str) -> ComparisonOp:
        mapping = {
            "<=": ComparisonOp.LE,
            "<": ComparisonOp.LT,
            ">=": ComparisonOp.GE,
            ">": ComparisonOp.GT,
            "==": ComparisonOp.EQ,
            "!=": ComparisonOp.NE,
        }
        return mapping.get(decision_type, ComparisonOp.LE)
