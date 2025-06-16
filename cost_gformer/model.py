"""Model definition for CoST-GFormer.

This file contains the :class:`CoSTGFormer` class which ties together the
embedding, attention and memory modules. It is a skeleton only, describing how
Short Term Memory (STM), Long Term Memory (LTM) and Ultra Short Term Attention
(USTA) would fit in a full implementation.
"""

import numpy as np

from .embedding import Embedding
from .attention import Attention, UnifiedSpatioTemporalAttention
from .memory import ShortTermMemory, LongTermMemory
from .heads import TravelTimeHead, CrowdingHead, mse_loss, cross_entropy_loss
from .data import GraphSnapshot


class CoSTGFormer:
    """Simplified placeholder for the full model."""

    def __init__(self, heads: int = 8, embedding: Embedding | None = None, num_nodes: int | None = None):
        self.embedding = embedding
        embed_dim = self.embedding.mlp.b2.size if embedding else 32
        self.attention = Attention(heads=heads)
        self.usta = UnifiedSpatioTemporalAttention(embed_dim=embed_dim, num_heads=heads)
        self.stm = ShortTermMemory(num_nodes=num_nodes, embed_dim=embed_dim)

        if num_nodes is None:
            num_nodes = 0 if embedding is None else embedding.num_nodes
        self.ltm = LongTermMemory(num_nodes=num_nodes, embed_dim=embed_dim)

        self.travel_head = TravelTimeHead(embed_dim)
        self.crowd_head = CrowdingHead(embed_dim)

    def forward(self, history: "list[GraphSnapshot]"):
        """Predict travel time and crowding for the next step.

        Parameters
        ----------
        history:
            Sequence of :class:`GraphSnapshot` objects forming the input window.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Travel time and crowding predictions for the edges of the latest
            snapshot in ``history``.
        """

        if self.embedding is None:
            raise ValueError("An Embedding module must be provided")

        # ------------------------------------------------------------------
        # 1) Compute spatio-temporal node embeddings for the input window
        # ------------------------------------------------------------------
        embeds = self.embedding.encode_window(history)

        # Update short and long term memories with each step
        for step in embeds:
            self.stm.write(step)
            self.ltm.write(step)

        # ------------------------------------------------------------------
        # 2) Apply unified spatio-temporal attention over all embeddings
        # ------------------------------------------------------------------
        n_steps, num_nodes, dim = embeds.shape
        attended = self.usta(embeds.reshape(-1, dim)).reshape(n_steps, num_nodes, dim)

        # 3) Retrieve STM context and fuse with LTM for the latest step
        latest = attended[-1]
        stm_ctx = self.stm.read_all()
        fused = 0.5 * latest + 0.5 * stm_ctx
        for v in range(num_nodes):
            fused[v] = self.ltm.fuse(v, fused[v])

        # ------------------------------------------------------------------
        # 4) Compute multi-task predictions from shared edge representations
        # ------------------------------------------------------------------
        edges = history[-1].edges
        preds_tt = []
        preds_cr = []
        for u, v in edges:
            pair = np.concatenate([fused[u], fused[v]], axis=-1)
            preds_tt.append(self.travel_head.mlp(pair).squeeze(-1))
            preds_cr.append(self.crowd_head.mlp(pair))

        travel = np.stack(preds_tt)
        crowd = np.stack(preds_cr)
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
        travel, crowd = self.forward(history)
        l_tt = mse_loss(travel, tt_target)
        if classification:
            l_cr = cross_entropy_loss(crowd, cr_target.astype(int))
        else:
            l_cr = mse_loss(crowd.squeeze(-1), cr_target)
        return lambda_tt * l_tt + lambda_cr * l_cr

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(heads={self.attention.heads})"
