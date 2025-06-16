"""Spatio-temporal embedding module for CoST-GFormer.

This implements a minimal version of the STM described in the project
specification.  It computes node embeddings by combining

- spectral coordinates extracted from the static graph Laplacian,
- periodic time encodings (hour of day and day of week), and
- aggregated dynamic edge features from the current snapshot.

A small two-layer MLP maps the concatenated features to the final
embedding space.  The implementation relies purely on ``numpy`` to keep
dependencies lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import Edge, GraphSnapshot


@dataclass
class MLP:
    """Tiny two-layer perceptron used by :class:`SpatioTemporalEmbedding`."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


class SpatioTemporalEmbedding:
    """Compute embeddings for nodes across a window of graph snapshots."""

    def __init__(
        self,
        num_nodes: int,
        static_edges: Iterable[Edge],
        dynamic_dim: int,
        embed_dim: int = 32,
        spectral_dim: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        self.num_nodes = num_nodes
        self.dynamic_dim = dynamic_dim

        # Build symmetric adjacency from the provided static edges.
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v in static_edges:
            adj[u, v] = 1.0
            adj[v, u] = 1.0

        # Normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}
        deg = adj.sum(axis=1)
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.diag(np.where(deg > 0, deg ** -0.5, 0.0))
        lap = np.eye(num_nodes) - d_inv_sqrt @ adj @ d_inv_sqrt

        # Eigen-decomposition and take the smallest non-zero eigenvectors.
        eigvals, eigvecs = np.linalg.eigh(lap)
        available = max(1, len(eigvals) - 1)
        sdim = min(spectral_dim, available)
        idx = np.argsort(eigvals)[1 : 1 + sdim]
        self.spectral = eigvecs[:, idx].astype(np.float32)
        spectral_dim = sdim

        in_dim = spectral_dim + 4 + dynamic_dim
        rng = np.random.default_rng(0)
        w1 = rng.standard_normal((in_dim, hidden_dim), dtype=np.float32) / np.sqrt(in_dim)
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = rng.standard_normal((hidden_dim, embed_dim), dtype=np.float32) / np.sqrt(hidden_dim)
        b2 = np.zeros(embed_dim, dtype=np.float32)
        self.mlp = MLP(w1, b1, w2, b2)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    @staticmethod
    def _time_encoding(t: int) -> np.ndarray:
        hour = t % 24
        day = t % 7
        return np.array(
            [
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day / 7),
                np.cos(2 * np.pi * day / 7),
            ],
            dtype=np.float32,
        )

    def _aggregate_dynamic(self, snapshot: GraphSnapshot) -> np.ndarray:
        agg = np.zeros((self.num_nodes, self.dynamic_dim), dtype=np.float32)
        count = np.zeros(self.num_nodes, dtype=np.float32)
        for (u, v) in snapshot.edges:
            feat = snapshot.dynamic_edge_feat[(u, v)]
            agg[u] += feat
            agg[v] += feat
            count[u] += 1
            count[v] += 1
        count[count == 0] = 1.0
        agg /= count[:, None]
        return agg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode_snapshot(self, snapshot: GraphSnapshot) -> np.ndarray:
        time_vec = self._time_encoding(snapshot.time)
        dyn = self._aggregate_dynamic(snapshot)
        out = np.zeros((self.num_nodes, self.mlp.b2.size), dtype=np.float32)
        for v in range(self.num_nodes):
            x = np.concatenate([self.spectral[v], time_vec, dyn[v]])
            out[v] = self.mlp(x)
        return out

    def encode_window(self, snaps: List[GraphSnapshot]) -> np.ndarray:
        return np.stack([self.encode_snapshot(s) for s in snaps])


# Backwards compatibility -------------------------------------------------
# Export a simple alias so earlier imports continue to work.
Embedding = SpatioTemporalEmbedding

__all__ = ["SpatioTemporalEmbedding", "Embedding"]

