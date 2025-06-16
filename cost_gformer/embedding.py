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
    ) -> None:
        self.num_nodes = num_nodes
        self.dynamic_dim = dynamic_dim
        self.device = torch.device(device)
        self.use_sparse = use_sparse

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
        g = torch.Generator().manual_seed(0)
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
        hour = torch.tensor(float(t % 24))
        day = torch.tensor(float(t % 7))
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
        agg = torch.zeros((self.num_nodes, self.dynamic_dim), dtype=torch.float32)
        count = torch.zeros(self.num_nodes, dtype=torch.float32)
        for (u, v) in snapshot.edges:
            feat = torch.from_numpy(snapshot.dynamic_edge_feat[(u, v)])
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
        time_vec = self._time_encoding(snapshot.time).to(self.device)
        dyn = self._aggregate_dynamic(snapshot).to(self.device)
        out = torch.zeros((self.num_nodes, self.mlp.b2.numel()), dtype=torch.float32, device=self.device)
        spectral = self.spectral.to(self.device)
        for v in range(self.num_nodes):
            x = torch.cat([spectral[v], time_vec, dyn[v]])
            out[v] = self.mlp(x)
        return out.cpu().numpy()

    def encode_window(self, snaps: List[GraphSnapshot]) -> np.ndarray:
        embeds = [torch.from_numpy(self.encode_snapshot(s)) for s in snaps]
        return torch.stack(embeds).cpu().numpy()


# Backwards compatibility -------------------------------------------------
# Export a simple alias so earlier imports continue to work.
Embedding = SpatioTemporalEmbedding

__all__ = ["SpatioTemporalEmbedding", "Embedding"]

