'''
Tree‑of‑Thought Framework Skeleton
Date: 2025‑05‑01
Description: A minimal, extensible template for implementing Tree‑of‑Thought style reasoning.
'''

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------
@dataclass(order=True)
class ThoughtNode:
    """A node in the thought tree.

    The dataclass is ordered by ``score`` so we can use it directly with
    priority‑queue utilities such as ``heapq``.
    """

    score: float
    state: Any = field(compare=False)
    parent: Optional['ThoughtNode'] = field(compare=False, default=None)
    action: Optional[Any] = field(compare=False, default=None)
    depth: int = field(compare=False, default=0)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def path(self) -> List['ThoughtNode']:
        """Return the sequence of nodes from the root to this node."""
        node, rev_path: Optional['ThoughtNode'], List['ThoughtNode'] = self, []
        while node:
            rev_path.append(node)
            node = node.parent
        return list(reversed(rev_path))

# -----------------------------------------------------------------------------
# Search controller
# -----------------------------------------------------------------------------
class TreeOfThought:
    """Beam‑search based Tree‑of‑Thought controller.

    Args:
        expand_fn:   Given a state, returns an iterable of (new_state, action)
        evaluate_fn: Returns a numeric score for a state
        max_depth:   Maximum search depth (tree height)
        beam_width:  Number of nodes to keep per level (beam width)
    """

    def __init__(
        self,
        expand_fn: Callable[[Any], Sequence[Tuple[Any, Any]]],
        evaluate_fn: Callable[[Any], float],
        max_depth: int = 5,
        beam_width: int = 3,
    ) -> None:
        self.expand_fn = expand_fn
        self.evaluate_fn = evaluate_fn
        self.max_depth = max_depth
        self.beam_width = beam_width

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def search(self, initial_state: Any) -> ThoughtNode:
        """Run beam search starting from *initial_state* and return the best node."""

        root = ThoughtNode(score=self.evaluate_fn(initial_state), state=initial_state)
        frontier: List[ThoughtNode] = [root]

        for depth in range(self.max_depth):
            logger.debug('Depth %d: exploring %d node(s)', depth, len(frontier))
            next_frontier: List[ThoughtNode] = []

            for node in frontier:
                for new_state, action in self.expand_fn(node.state):
                    score = self.evaluate_fn(new_state)
                    child = ThoughtNode(
                        score=score,
                        state=new_state,
                        parent=node,
                        action=action,
                        depth=node.depth + 1,
                    )
                    next_frontier.append(child)

            # Keep the top‑k nodes with highest score for the next layer
            frontier = heapq.nlargest(self.beam_width, next_frontier)
            if not frontier:
                break

        best = max(frontier, key=lambda n: n.score)
        logger.info('Best score: %.4f', best.score)
        return best

def _dummy_expand(state: Any) -> Sequence[Tuple[Any, Any]]:
    """Example expansion function that returns no children (stub)."""
    return []


def _dummy_evaluate(state: Any) -> float:
    """Example evaluation that assigns neutral score 0.0 (stub)."""
    return 0.0


if __name__ == '__main__':
    initial_state = 'START'
    tot = TreeOfThought(
        expand_fn=_dummy_expand,
        evaluate_fn=_dummy_evaluate,
        max_depth=3,
        beam_width=2,
    )
    best = tot.search(initial_state)
    print('Best path:', [node.state for node in best.path()])
