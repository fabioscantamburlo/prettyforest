"""Reingold-Tilford tree layout algorithm."""

from __future__ import annotations

from prettyforest.models import NodePosition, UnifiedNode, UnifiedTree

NODE_WIDTH = 160.0
NODE_HEIGHT = 60.0
H_SEP = 20.0
V_SEP = 80.0


class LayoutEngine:
    def __init__(
        self,
        node_width: float = NODE_WIDTH,
        node_height: float = NODE_HEIGHT,
        h_sep: float = H_SEP,
        v_sep: float = V_SEP,
    ):
        self.node_width = node_width
        self.node_height = node_height
        self.h_sep = h_sep
        self.v_sep = v_sep

    def compute_layout(
        self,
        tree: UnifiedTree,
        collapse_state: dict[str, bool] | None = None,
    ) -> dict[str, NodePosition]:
        if collapse_state is None:
            collapse_state = {}

        positions: dict[str, NodePosition] = {}
        self._assign_x(tree.root, collapse_state, positions)
        return positions

    def _assign_x(
        self,
        root: UnifiedNode,
        collapse_state: dict[str, bool],
        positions: dict[str, NodePosition],
    ) -> None:
        # Phase 1: compute subtree widths
        widths: dict[str, float] = {}
        self._compute_widths(root, collapse_state, widths)

        # Phase 2: assign positions top-down
        self._position_node(root, 0.0, 0.0, collapse_state, widths, positions)

    def _compute_widths(
        self,
        node: UnifiedNode,
        collapse_state: dict[str, bool],
        widths: dict[str, float],
    ) -> float:
        is_collapsed = collapse_state.get(node.node_id, False)

        if node.is_leaf or is_collapsed:
            w = self.node_width
            widths[node.node_id] = w
            return w

        left_w = self._compute_widths(node.left_child, collapse_state, widths)
        right_w = self._compute_widths(node.right_child, collapse_state, widths)
        total = left_w + self.h_sep + right_w
        widths[node.node_id] = total
        return total

    def _position_node(
        self,
        node: UnifiedNode,
        x_center: float,
        y_top: float,
        collapse_state: dict[str, bool],
        widths: dict[str, float],
        positions: dict[str, NodePosition],
    ) -> None:
        positions[node.node_id] = NodePosition(
            x=x_center - self.node_width / 2,
            y=y_top,
            width=self.node_width,
            height=self.node_height,
        )

        is_collapsed = collapse_state.get(node.node_id, False)
        if node.is_leaf or is_collapsed:
            return

        total_w = widths[node.node_id]
        left_w = widths[node.left_child.node_id]
        right_w = widths[node.right_child.node_id]

        left_center = x_center - total_w / 2 + left_w / 2
        right_center = x_center + total_w / 2 - right_w / 2
        child_y = y_top + self.node_height + self.v_sep

        self._position_node(
            node.left_child, left_center, child_y, collapse_state, widths, positions
        )
        self._position_node(
            node.right_child, right_center, child_y, collapse_state, widths, positions
        )


def compute_initial_collapse_state(tree: UnifiedTree) -> dict[str, bool]:
    """Determine initial collapse state based on tree depth."""
    if tree.max_depth <= 5:
        return {}

    state: dict[str, bool] = {}
    for node in tree.iter_nodes():
        if not node.is_leaf and node.depth >= 3:
            state[node.node_id] = True
    return state


def count_descendants(node: UnifiedNode) -> int:
    """Count all descendant nodes (excluding the node itself)."""
    if node.is_leaf:
        return 0
    count = 0
    stack = []
    if node.left_child:
        stack.append(node.left_child)
    if node.right_child:
        stack.append(node.right_child)
    while stack:
        n = stack.pop()
        count += 1
        if n.left_child:
            stack.append(n.left_child)
        if n.right_child:
            stack.append(n.right_child)
    return count
