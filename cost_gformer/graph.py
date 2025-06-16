"""Utilities for expanded spatio-temporal graphs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .data import GraphSnapshot


class ExpandedGraph:
    """Construct an expanded spatio-temporal graph from snapshots.

    The expanded graph replicates each node for every time step in the
    provided window and links replicas using spatial edges from each
    snapshot and temporal edges between consecutive snapshots.
    """

    def __init__(self, snapshots: List[GraphSnapshot], num_nodes: int) -> None:
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        self.snapshots = snapshots
        self.num_nodes = num_nodes
        self.num_steps = len(snapshots)

        self.edges: List[Tuple[int, int]] = []
        self._build_edges()

    # ------------------------------------------------------------------
    def _build_edges(self) -> None:
        n = self.num_nodes
        T = self.num_steps
        for t, snap in enumerate(self.snapshots):
            base = t * n
            for (u, v) in snap.edges:
                self.edges.append((base + u, base + v))
        for t in range(T - 1):
            base = t * n
            next_base = (t + 1) * n
            for v in range(n):
                self.edges.append((base + v, next_base + v))

    # ------------------------------------------------------------------
    @property
    def num_expanded_nodes(self) -> int:
        return self.num_nodes * self.num_steps

    def edge_index(self) -> np.ndarray:
        arr = np.array(self.edges, dtype=np.int64).T
        return arr

__all__ = ["ExpandedGraph"]

