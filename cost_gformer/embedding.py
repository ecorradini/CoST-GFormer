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
import torch

from .data import Edge, GraphSnapshot


@dataclass
class MLP:
    """Tiny two-layer perceptron used by :class:`SpatioTemporalEmbedding`."""

    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(x @ self.w1 + self.b1)
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
        device: str | torch.device = "cpu",
        use_sparse: bool = True,
        seed: int = 0,
    ) -> None:
        self.num_nodes = num_nodes
        self.dynamic_dim = dynamic_dim
        self.device = torch.device(device)
        self.use_sparse = use_sparse
        self.seed = seed

        if self.use_sparse and num_nodes > spectral_dim + 1:
            import scipy.sparse as sp
            import scipy.sparse.linalg as splinalg

            rows: List[int] = []
            cols: List[int] = []
            data: List[float] = []
            for u, v in static_edges:
                rows.extend([u, v])
                cols.extend([v, u])
                data.extend([1.0, 1.0])
            adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
            deg = np.asarray(adj.sum(axis=1)).reshape(-1)
            d_inv = np.where(deg > 0, deg ** -0.5, 0.0)
            d_inv_sqrt = sp.diags(d_inv)
            lap = sp.eye(num_nodes, dtype=np.float32) - d_inv_sqrt @ adj @ d_inv_sqrt

            k = min(spectral_dim + 1, num_nodes - 1)
            eigvals_np, eigvecs_np = splinalg.eigsh(lap, k=k, which="SM")
            order = np.argsort(eigvals_np)
            eigvals_np = eigvals_np[order]
            eigvecs_np = eigvecs_np[:, order]
            eigvals = torch.from_numpy(eigvals_np)
            eigvecs = torch.from_numpy(eigvecs_np)
        else:
            # Build symmetric adjacency from the provided static edges using dense tensors.
            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            for u, v in static_edges:
                adj[u, v] = 1.0
                adj[v, u] = 1.0

            # Normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}
            deg = adj.sum(dim=1)
            d_inv_sqrt = torch.diag(torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg)))
            lap = torch.eye(num_nodes) - d_inv_sqrt @ adj @ d_inv_sqrt

            # ``torch.linalg.eigh`` can fail on some platforms for large float32 matrices.
            # Using NumPy avoids those LAPACK issues.
            eigvals_np, eigvecs_np = np.linalg.eigh(lap.cpu().numpy())
            eigvals = torch.from_numpy(eigvals_np)
            eigvecs = torch.from_numpy(eigvecs_np)
        available = max(1, len(eigvals) - 1)
        sdim = min(spectral_dim, available)
        idx = torch.argsort(eigvals)[1 : 1 + sdim]
        self.spectral = eigvecs[:, idx].to(torch.float32)
        spectral_dim = sdim

        in_dim = spectral_dim + 4 + dynamic_dim
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        w1 = torch.randn((in_dim, hidden_dim), generator=g, dtype=torch.float32) / torch.sqrt(torch.tensor(in_dim, dtype=torch.float32))
        b1 = torch.zeros(hidden_dim, dtype=torch.float32)
        w2 = torch.randn((hidden_dim, embed_dim), generator=g, dtype=torch.float32) / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        b2 = torch.zeros(embed_dim, dtype=torch.float32)
        self.mlp = MLP(w1.to(self.device), b1.to(self.device), w2.to(self.device), b2.to(self.device))

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    @staticmethod
    def _time_encoding(t: int) -> torch.Tensor:
        hour = torch.tensor(float((t // 3600) % 24))
        day = torch.tensor(float((t // 86400) % 7))
        enc = torch.stack(
            [
                torch.sin(2 * torch.pi * hour / 24),
                torch.cos(2 * torch.pi * hour / 24),
                torch.sin(2 * torch.pi * day / 7),
                torch.cos(2 * torch.pi * day / 7),
            ]
        )
        return enc.to(torch.float32)

    def _aggregate_dynamic(self, snapshot: GraphSnapshot) -> torch.Tensor:
        # Convert edges and their dynamic features to dense tensors upfront.
        edges = torch.tensor(snapshot.edges, dtype=torch.long)
        feats = torch.stack(
            [torch.from_numpy(snapshot.dynamic_edge_feat[(int(u), int(v))]) for u, v in snapshot.edges],
            dim=0,
        )

        agg = torch.zeros((self.num_nodes, self.dynamic_dim), dtype=torch.float32)
        count = torch.zeros(self.num_nodes, dtype=torch.float32)

        # Sum features and counts using ``index_add_`` for efficiency.
        agg.index_add_(0, edges[:, 0], feats)
        agg.index_add_(0, edges[:, 1], feats)
        ones = torch.ones(edges.size(0), dtype=torch.float32)
        count.index_add_(0, edges[:, 0], ones)
        count.index_add_(0, edges[:, 1], ones)

        count[count == 0] = 1.0
        agg /= count[:, None]
        return agg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode_snapshot(self, snapshot: GraphSnapshot) -> np.ndarray:
        time_vec = self._time_encoding(snapshot.time).to(self.device)
        dyn = self._aggregate_dynamic(snapshot).to(self.device)
        spectral = self.spectral.to(self.device)

        # Repeat the time encoding for all nodes and compute all embeddings
        # in one batched matrix multiplication.
        time_mat = time_vec.repeat(self.num_nodes, 1)
        x = torch.cat([spectral, time_mat, dyn], dim=1)
        out = self.mlp(x)
        return out.cpu().numpy()

    def encode_window(self, snaps: List[GraphSnapshot]) -> np.ndarray:
        """Encode a sequence of snapshots in a single pass.

        Parameters
        ----------
        snaps:
            Ordered list of :class:`GraphSnapshot` objects.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(snaps), num_nodes, embed_dim)`` containing the
            embeddings for each snapshot.
        """

        n_steps = len(snaps)
        embed_dim = self.mlp.b2.numel()

        out = torch.empty(
            (n_steps, self.num_nodes, embed_dim),
            device=self.device,
            dtype=torch.float32,
        )

        spectral = self.spectral.to(self.device)
        for i, snap in enumerate(snaps):
            time_vec = self._time_encoding(snap.time).to(self.device)
            dyn = self._aggregate_dynamic(snap).to(self.device)

            time_mat = time_vec.repeat(self.num_nodes, 1)
            x = torch.cat([spectral, time_mat, dyn], dim=1)
            out[i] = self.mlp(x)

        return out.cpu().numpy()


# Backwards compatibility -------------------------------------------------
# Export a simple alias so earlier imports continue to work.
Embedding = SpatioTemporalEmbedding

__all__ = ["SpatioTemporalEmbedding", "Embedding"]

