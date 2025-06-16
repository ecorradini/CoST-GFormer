"""Output heads for CoST-GFormer.

This module implements the minimal multi-task heads used by the model to
predict travel times and crowding levels on graph edges.  Both heads are
two-layer perceptrons written in ``numpy``.  They operate on concatenated
embeddings of the two end nodes of each edge.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MLP:
    """Simple two-layer multilayer perceptron."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


class TravelTimeHead:
    """Regress edge travel time via a small MLP."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64) -> None:
        in_dim = 2 * embed_dim
        rng = np.random.default_rng(0)
        w1 = rng.standard_normal((in_dim, hidden_dim), dtype=np.float32) / np.sqrt(in_dim)
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = rng.standard_normal((hidden_dim, 1), dtype=np.float32) / np.sqrt(hidden_dim)
        b2 = np.zeros(1, dtype=np.float32)
        self.mlp = MLP(w1, b1, w2, b2)

    def __call__(self, u_emb: np.ndarray, v_emb: np.ndarray) -> np.ndarray:
        x = np.concatenate([u_emb, v_emb], axis=-1)
        out = self.mlp(x)
        return out.squeeze(-1)


class CrowdingHead:
    """Predict crowding level either as classification or regression."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, num_classes: int = 3) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        in_dim = 2 * embed_dim
        rng = np.random.default_rng(1)
        w1 = rng.standard_normal((in_dim, hidden_dim), dtype=np.float32) / np.sqrt(in_dim)
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = rng.standard_normal((hidden_dim, num_classes), dtype=np.float32) / np.sqrt(hidden_dim)
        b2 = np.zeros(num_classes, dtype=np.float32)
        self.mlp = MLP(w1, b1, w2, b2)
        self.num_classes = num_classes

    def __call__(self, u_emb: np.ndarray, v_emb: np.ndarray) -> np.ndarray:
        x = np.concatenate([u_emb, v_emb], axis=-1)
        out = self.mlp(x)
        return out

    # --------------------------------------------------------------
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.mean((pred - target) ** 2))


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = CrowdingHead.softmax(logits)
    n = labels.shape[0]
    loss = -np.log(probs[np.arange(n), labels])
    return float(np.mean(loss))


__all__ = ["TravelTimeHead", "CrowdingHead", "mse_loss", "cross_entropy_loss"]
