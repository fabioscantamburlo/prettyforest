"""Serialize and deserialize UnifiedTree to/from JSON."""

from __future__ import annotations

import json

from .models import ComparisonOp, UnifiedNode, UnifiedTree

_VALID_OPS = {op.value for op in ComparisonOp}
_VALID_NODE_TYPES = {"internal", "leaf"}
_OP_FROM_STR = {op.value: op for op in ComparisonOp}


def serialize(tree: UnifiedTree) -> str:
    """Serialize a UnifiedTree to a JSON string, preserving float precision."""
    payload = {
        "version": "1.0",
        "tree": {
            "node_count": tree.node_count,
            "max_depth": tree.max_depth,
            "is_classifier": tree.is_classifier,
            "feature_names": tree.feature_names,
            "class_names": tree.class_names,
            "root": _serialize_node(tree.root),
        },
    }
    # Use allow_nan=False so invalid floats are caught.
    # Python's json.dumps with default float handling in CPython 3.1+
    # uses repr()-equivalent precision (17 significant digits).
    return json.dumps(payload)


def _serialize_node(node: UnifiedNode) -> dict:
    if node.is_leaf:
        d: dict = {
            "node_id": node.node_id,
            "depth": node.depth,
            "type": "leaf",
        }
        if node.prediction_value is not None:
            d["prediction_value"] = node.prediction_value
        if node.class_distribution is not None:
            d["class_distribution"] = node.class_distribution
        return d

    return {
        "node_id": node.node_id,
        "depth": node.depth,
        "type": "internal",
        "feature_name": node.feature_name,
        "threshold": node.threshold,
        "comparison_op": node.comparison_op.value if node.comparison_op else None,
        "left": _serialize_node(node.left_child),  # type: ignore[arg-type]
        "right": _serialize_node(node.right_child),  # type: ignore[arg-type]
    }


def deserialize(json_str: str) -> UnifiedTree:
    """Deserialize a JSON string into a UnifiedTree.

    Raises ValueError for malformed JSON, missing fields, or unknown operators/types.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize tree: invalid JSON at line {e.lineno} column {e.colno}: {e.msg}") from e

    if not isinstance(data, dict) or "tree" not in data:
        raise ValueError("Failed to deserialize tree: missing required field 'tree'")

    tree_data = data["tree"]
    for field in ("node_count", "max_depth", "is_classifier", "feature_names", "root"):
        if field not in tree_data:
            raise ValueError(f"Failed to deserialize tree: missing required field '{field}'")

    root = _deserialize_node(tree_data["root"])

    return UnifiedTree(
        root=root,
        node_count=tree_data["node_count"],
        max_depth=tree_data["max_depth"],
        feature_names=tree_data["feature_names"],
        is_classifier=tree_data["is_classifier"],
        class_names=tree_data.get("class_names"),
    )


def _deserialize_node(data: dict) -> UnifiedNode:
    for field in ("node_id", "type", "depth"):
        if field not in data:
            raise ValueError(f"Failed to deserialize tree: missing required field '{field}' in node")

    node_type = data["type"]
    if node_type not in _VALID_NODE_TYPES:
        raise ValueError(f"Failed to deserialize tree: unknown node type '{node_type}'")

    if node_type == "leaf":
        return UnifiedNode(
            node_id=data["node_id"],
            depth=data["depth"],
            prediction_value=data.get("prediction_value"),
            class_distribution=data.get("class_distribution"),
        )

    # Internal node
    for field in ("feature_name", "threshold", "comparison_op", "left", "right"):
        if field not in data:
            raise ValueError(f"Failed to deserialize tree: missing required field '{field}' in internal node '{data['node_id']}'")

    op_str = data["comparison_op"]
    if op_str not in _VALID_OPS:
        raise ValueError(f"Failed to deserialize tree: unknown comparison operator '{op_str}'")

    return UnifiedNode(
        node_id=data["node_id"],
        depth=data["depth"],
        feature_name=data["feature_name"],
        threshold=data["threshold"],
        comparison_op=_OP_FROM_STR[op_str],
        left_child=_deserialize_node(data["left"]),
        right_child=_deserialize_node(data["right"]),
    )
