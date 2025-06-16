"""Data pipeline utilities for CoST-GFormer.

This module implements a minimal data pipeline capable of handling
sequences of dynamic graph snapshots.  It provides simple classes for
representing a single snapshot and for retrieving historical windows
used during training or evaluation.

The implementation is intentionally lightweight and does not depend on
external graph libraries.  The goal is to demonstrate how dynamic
spatio--temporal graphs could be loaded and fed to the model without
assuming a specific dataset format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


Edge = Tuple[int, int]


@dataclass
class GraphSnapshot:
    """Container for a single graph snapshot.

    Parameters
    ----------
    time : int
        Discrete time index of the snapshot.
    edges : List[Edge]
        List of directed edges ``(u, v)`` active at this time.
    static_edge_feat : Dict[Edge, np.ndarray]
        Mapping from edge to static feature vector.
    dynamic_edge_feat : Dict[Edge, np.ndarray]
        Mapping from edge to dynamic feature vector measured at ``time``.
    """

    time: int
    edges: List[Edge]
    static_edge_feat: Dict[Edge, np.ndarray]
    dynamic_edge_feat: Dict[Edge, np.ndarray]


class DynamicGraphDataset:
    """Sequence of :class:`GraphSnapshot` objects."""

    def __init__(self, snapshots: Iterable[GraphSnapshot]):
        self.snapshots: List[GraphSnapshot] = list(sorted(snapshots, key=lambda s: s.time))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> GraphSnapshot:  # pragma: no cover - trivial
        return self.snapshots[idx]


class DataModule:
    """Utility class to retrieve windows of snapshots.

    Parameters
    ----------
    dataset : DynamicGraphDataset
        Ordered sequence of graph snapshots.
    history : int
        Number of past snapshots that form the historical observation
        window ``T``.
    horizon : int
        Number of future steps to forecast ``H``.
    """

    def __init__(self, dataset: DynamicGraphDataset, history: int, horizon: int):
        if history <= 0 or horizon <= 0:
            raise ValueError("history and horizon must be positive")
        if len(dataset) < history + horizon:
            raise ValueError("dataset too small for the requested window size")

        self.dataset = dataset
        self.history = history
        self.horizon = horizon

    def __len__(self) -> int:
        """Number of available training examples."""
        return len(self.dataset) - self.history - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[List[GraphSnapshot], List[GraphSnapshot]]:
        """Return a pair ``(history, future)`` for the given index."""
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        start = idx
        mid = idx + self.history
        end = mid + self.horizon
        history = self.dataset.snapshots[start:mid]
        future = self.dataset.snapshots[mid:end]
        return history, future


# ---------------------------------------------------------------------------
# Helper utilities to generate synthetic datasets for examples and tests.
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    num_nodes: int,
    num_snapshots: int,
    static_dim: int = 3,
    dynamic_dim: int = 3,
    seed: int | None = None,
) -> DynamicGraphDataset:
    """Generate a small random :class:`DynamicGraphDataset`.

    This function creates a sequence of fully connected graphs with
    random static and dynamic features.  It is useful for unit tests or
    illustrative examples when no real dataset is available.
    """

    rng = np.random.default_rng(seed)
    snapshots: List[GraphSnapshot] = []
    edges = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v]

    static_feat = {e: rng.random(static_dim, dtype=np.float32) for e in edges}

    for t in range(num_snapshots):
        dyn_feat = {e: rng.random(dynamic_dim, dtype=np.float32) for e in edges}
        snap = GraphSnapshot(time=t, edges=edges, static_edge_feat=static_feat, dynamic_edge_feat=dyn_feat)
        snapshots.append(snap)

    return DynamicGraphDataset(snapshots)
