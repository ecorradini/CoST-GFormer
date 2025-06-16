"""Attention mechanisms for CoST-GFormer.

This module contains a placeholder :class:`Attention` that hints at multi-head
attention capable of operating over Short Term Memory (STM) as well as Long
Term Memory (LTM). Real implementations would include query/key/value
projections and memory-efficient computation.
"""


class Attention:
    """Placeholder attention block."""

    def __init__(self, heads: int = 8):
        self.heads = heads
        self.description = (
            "Attention layers would allow interaction between STM and LTM for "
            "context reasoning."
        )

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(heads={self.heads})"
