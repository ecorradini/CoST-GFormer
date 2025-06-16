"""Model definition for CoST-GFormer.

This file contains the :class:`CoSTGFormer` class which ties together the
embedding, attention and memory modules. It is a skeleton only, describing how
Short Term Memory (STM), Long Term Memory (LTM) and Ultra Short Term Attention
(USTA) would fit in a full implementation.
"""

from .embedding import Embedding
from .attention import Attention
from .memory import UltraShortTermAttention, ShortTermMemory, LongTermMemory


class CoSTGFormer:
    """Simplified placeholder for the full model."""

    def __init__(self, d_model: int = 128, heads: int = 8):
        self.embedding = Embedding(dim=d_model)
        self.attention = Attention(heads=heads)
        self.usta = UltraShortTermAttention()
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()

    def forward(self, x):  # pragma: no cover - placeholder method
        """Fake forward pass that returns the input unchanged."""
        return x

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"{self.__class__.__name__}(d_model={self.embedding.dim}, heads={self.attention.heads})"
        )
