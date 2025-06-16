"""Data utilities for CoST-GFormer.

This module defines a simple :class:`DataModule` placeholder. It does not
provide real dataset handling but documents where Short Term Memory (STM) and
other components would interact with data loading pipelines.
"""


class DataModule:
    """Placeholder class representing the data pipeline."""

    def __init__(self):
        self.description = (
            "This module would normally prepare datasets and feed them into "
            "the model, utilizing STM and LTM buffers for streaming tasks."
        )

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}()"
