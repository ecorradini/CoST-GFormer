"""Utilities for expanded spatio-temporal graphs."""

from __future__ import annotations

from typing import Iterable, List, Tuple

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

        # Precompute spatio-temporal edges once during construction
        self.edges: np.ndarray | None = None
        self._build_edges()

    # ------------------------------------------------------------------
    def _build_edges(self) -> None:
        n = self.num_nodes
        T = self.num_steps

        spatial = [
            (t * n + u, t * n + v)
            for t, snap in enumerate(self.snapshots)
            for (u, v) in snap.edges
        ]

        temporal = [
            (t * n + v, (t + 1) * n + v)
            for t in range(T - 1)
            for v in range(n)
        ]

        self.edges = np.concatenate(
            [np.array(spatial, dtype=np.int64), np.array(temporal, dtype=np.int64)],
            axis=0,
        )

    # ------------------------------------------------------------------
    @property
    def num_expanded_nodes(self) -> int:
        return self.num_nodes * self.num_steps

    def edge_index(self) -> np.ndarray:
        if self.edges is None:
            return np.empty((2, 0), dtype=np.int64)
        return np.asarray(self.edges, dtype=np.int64).T


class DynamicGraphHandler:
    """Maintain fused adjacency matrices from static and dynamic components."""

    def __init__(
        self,
        num_nodes: int,
        static_edges: Iterable[Tuple[int, int]],
        alpha: float = 0.5,
        top_p: int = 5,
        decay: float = 0.1,
        remove_threshold: float = 0.01,
        add_edges: bool = True,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if top_p <= 0:
            raise ValueError("top_p must be positive")

        self.num_nodes = num_nodes
        self.alpha = float(alpha)
        self.top_p = int(top_p)

        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if not 0.0 <= remove_threshold <= 1.0:
            raise ValueError("remove_threshold must be in [0, 1]")

        self.decay = float(decay)
        self.remove_threshold = float(remove_threshold)
        self.add_edges = bool(add_edges)

        self.static_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        self.dynamic_mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v in static_edges:
            self.static_adj[u, v] = 1.0
            self.dynamic_mask[u, v] = 1.0

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_rows(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def _prune(self, adj: np.ndarray) -> np.ndarray:
        n, m = adj.shape
        k = min(self.top_p, m)

        # Get indices of the top-``k`` entries per row all at once.
        idx = np.argpartition(-adj, k - 1, axis=1)[:, :k]

        out = np.zeros_like(adj)
        rows = np.arange(n)[:, None]
        out[rows, idx] = adj[rows, idx]
        return out

    # ------------------------------------------------------------------
    def ingest(self, active_edges: Iterable[Tuple[int, int]]) -> None:
        """Ingest live service data and update the internal edge state."""

        self.dynamic_mask *= 1.0 - self.decay

        for u, v in active_edges:
            if self.add_edges and self.static_adj[u, v] == 0.0:
                self.static_adj[u, v] = 1.0
            self.dynamic_mask[u, v] = 1.0

        remove = self.dynamic_mask < self.remove_threshold
        self.dynamic_mask[remove] = 0.0
        self.static_adj[remove] = 0.0

    def update(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute fused adjacency from the latest embeddings."""

        scores = embeddings @ embeddings.T
        np.maximum(scores, 0.0, out=scores)
        dyn = self._softmax_rows(scores)

        fused_static = self.static_adj * self.dynamic_mask
        fused = self.alpha * fused_static + (1.0 - self.alpha) * dyn
        return self._prune(fused)

__all__ = ["ExpandedGraph", "DynamicGraphHandler"]

