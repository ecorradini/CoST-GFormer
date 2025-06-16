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


class CoSTGFormer:
    """Simplified placeholder for the full model."""

    def __init__(self, heads: int = 8, embedding: Embedding | None = None, num_nodes: int | None = None):
        self.embedding = embedding
        embed_dim = self.embedding.mlp.b2.size if embedding else 32
        self.attention = Attention(heads=heads)
        self.usta = UnifiedSpatioTemporalAttention(embed_dim=embed_dim, num_heads=heads)
        self.stm = ShortTermMemory()

        if num_nodes is None:
            num_nodes = 0 if embedding is None else embedding.num_nodes
        self.ltm = LongTermMemory(num_nodes=num_nodes, embed_dim=embed_dim)

        self.travel_head = TravelTimeHead(embed_dim)
        self.crowd_head = CrowdingHead(embed_dim)

    def forward(self, embeddings: "np.ndarray", edges: "np.ndarray"):
        """Return predictions for the given edges."""

        preds_tt = []
        preds_cr = []
        for u, v in edges:
            u_emb = embeddings[u]
            v_emb = embeddings[v]
            preds_tt.append(self.travel_head(u_emb, v_emb))
            preds_cr.append(self.crowd_head(u_emb, v_emb))
        travel = np.stack(preds_tt)
        crowd = np.stack(preds_cr)
        return travel, crowd

    # --------------------------------------------------------------
    def loss(
        self,
        embeddings: "np.ndarray",
        edges: "np.ndarray",
        tt_target: "np.ndarray",
        cr_target: "np.ndarray",
        lambda_tt: float = 1.0,
        lambda_cr: float = 1.0,
        classification: bool = True,
    ) -> float:
        travel, crowd = self.forward(embeddings, edges)
        l_tt = mse_loss(travel, tt_target)
        if classification:
            l_cr = cross_entropy_loss(crowd, cr_target.astype(int))
        else:
            l_cr = mse_loss(crowd.squeeze(-1), cr_target)
        return lambda_tt * l_tt + lambda_cr * l_cr

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(heads={self.attention.heads})"
