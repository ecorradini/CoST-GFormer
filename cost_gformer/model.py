"""Model definition for CoST-GFormer.

This module exposes the :class:`CoSTGFormer` class which combines embedding,
attention and memory components into a trainable model.  The Short Term
Memory (STM), Long Term Memory (LTM) and Unified Spatio‑Temporal Attention
(USTA) mechanisms are implemented in full, allowing the architecture to be used
for real experiments.
"""

import numpy as np
import torch

from .embedding import Embedding
from .attention import Attention, UnifiedSpatioTemporalAttention
from .memory import ShortTermMemory, LongTermMemory
from .heads import TravelTimeHead, CrowdingHead, mse_loss, cross_entropy_loss
from .data import GraphSnapshot


class CoSTGFormer:
    """Prototype implementation of the CoST-GFormer model."""

    def __init__(
        self,
        heads: int = 8,
        embedding: Embedding | None = None,
        num_nodes: int | None = None,
        device: str | torch.device = "cpu",
    ):
        self.embedding = embedding
        embed_dim = self.embedding.mlp.b2.numel() if embedding else 32
        self.attention = Attention(embed_dim=embed_dim, num_heads=heads)
        self.usta = UnifiedSpatioTemporalAttention(embed_dim=embed_dim, num_heads=heads)
        self.stm = ShortTermMemory(num_nodes=num_nodes, embed_dim=embed_dim)

        if num_nodes is None:
            self.ltm = None
        else:
            self.ltm = LongTermMemory(num_nodes=num_nodes, embed_dim=embed_dim)

        self.travel_head = TravelTimeHead(embed_dim, device=device)
        self.crowd_head = CrowdingHead(embed_dim, device=device)

    def forward(self, history: "list[GraphSnapshot]", horizon: int = 1):
        """Predict travel time and crowding for ``horizon`` future steps.

        Parameters
        ----------
        history:
            Sequence of :class:`GraphSnapshot` objects forming the input window.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of travel time and crowding predictions with shape
            ``(horizon, num_edges)``.
        """

        if self.embedding is None:
            raise ValueError("An Embedding module must be provided")

        # ------------------------------------------------------------------
        # 1) Compute spatio-temporal node embeddings for the input window
        # ------------------------------------------------------------------
        if horizon <= 0 or len(history) < horizon:
            raise ValueError("invalid horizon")

        embeds = self.embedding.encode_window(history)
        n_steps, num_nodes, dim = embeds.shape
        if self.ltm is None:
            self.ltm = LongTermMemory(num_nodes=num_nodes, embed_dim=dim)
        tt_preds: list[np.ndarray] = []
        cr_preds: list[np.ndarray] = []

        for t in range(n_steps):
            self.stm.write(embeds[t])
            self.ltm.write(embeds[t])
            attended = self.usta(embeds[: t + 1].reshape(-1, dim)).reshape(
                t + 1, num_nodes, dim
            )[-1]
            stm_ctx = self.stm.read_all()
            fused = 0.5 * attended + 0.5 * stm_ctx
            for v in range(num_nodes):
                fused[v] = self.ltm.fuse(v, fused[v])

            if t >= n_steps - horizon:
                edges = history[t].edges
                preds_tt = []
                preds_cr = []
                for u, v in edges:
                    pair = np.concatenate([fused[u], fused[v]], axis=-1)
                    preds_tt.append(self.travel_head.mlp(pair).squeeze(-1))
                    preds_cr.append(self.crowd_head.mlp(pair))
                tt_preds.append(np.stack(preds_tt))
                cr_preds.append(np.stack(preds_cr))

        travel = np.stack(tt_preds)
        crowd = np.stack(cr_preds)
        return travel, crowd

    # --------------------------------------------------------------
    def loss(
        self,
        history: "list[GraphSnapshot]",
        tt_target: "np.ndarray",
        cr_target: "np.ndarray",
        lambda_tt: float = 1.0,
        lambda_cr: float = 1.0,
        classification: bool = True,
    ) -> float:
        horizon = len(tt_target) if hasattr(tt_target, "__len__") else 1
        travel, crowd = self.forward(history, horizon=horizon)
        l_tt = mse_loss(travel, tt_target)
        if classification:
            l_cr = cross_entropy_loss(crowd, cr_target.astype(int))
        else:
            l_cr = mse_loss(crowd.squeeze(-1), cr_target)
        return lambda_tt * l_tt + lambda_cr * l_cr

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(heads={self.attention.num_heads})"
