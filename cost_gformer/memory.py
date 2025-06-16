"""Memory buffers for CoST-GFormer.

The memory system is separated into three conceptual blocks:

- :class:`UltraShortTermAttention` (USTA): a very small buffer for the most
  recent tokens used directly by attention layers.
- :class:`ShortTermMemory` (STM): holds recent context for quick retrieval.
- :class:`LongTermMemory` (LTM): archives older information for long-range
  dependencies.

These classes are placeholders and only document the intended behavior.
"""


class UltraShortTermAttention:
    """Represents the USTA buffer."""

    def __init__(self, size: int = 4):
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(size={self.size})"


class ShortTermMemory:
    """Represents the STM buffer."""

    def __init__(self, size: int = 128):
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(size={self.size})"


class LongTermMemory:
    """Represents the LTM buffer."""

    def __init__(self, size: int = 1024):
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(size={self.size})"
