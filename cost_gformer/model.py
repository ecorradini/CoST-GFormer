"""Model definition for CoST-GFormer.

This file contains the :class:`CoSTGFormer` class which ties together the
embedding, attention and memory modules. It is a skeleton only, describing how
Short Term Memory (STM), Long Term Memory (LTM) and Ultra Short Term Attention
(USTA) would fit in a full implementation.
"""

from .embedding import Embedding
from .attention import Attention, UnifiedSpatioTemporalAttention
from .memory import ShortTermMemory, LongTermMemory


class CoSTGFormer:
    """Simplified placeholder for the full model."""

    def __init__(self, heads: int = 8, embedding: Embedding | None = None):
        self.embedding = embedding
        self.attention = Attention(heads=heads)
        self.usta = UnifiedSpatioTemporalAttention(embed_dim=self.embedding.mlp.b2.size if embedding else 32,
                                                   num_heads=heads)
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()

    def forward(self, x):  # pragma: no cover - placeholder method
        """Fake forward pass that returns the input unchanged."""
        return x

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(heads={self.attention.heads})"
