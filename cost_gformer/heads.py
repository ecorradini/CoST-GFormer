"""Output heads for CoST-GFormer.

This module implements the minimal multi-task heads used by the model to
predict travel times and crowding levels on graph edges.  Both heads are
two-layer perceptrons written in ``numpy``.  They operate on concatenated
embeddings of the two end nodes of each edge.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Simple two-layer multilayer perceptron."""

    def __init__(self, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> None:
        super().__init__()
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(b1)
        self.w2 = nn.Parameter(w2)
        self.b2 = nn.Parameter(b2)

    def forward(self, x: "np.ndarray | torch.Tensor") -> "np.ndarray | torch.Tensor":
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).to(self.w1.device)
            hidden = torch.relu(x_t @ self.w1 + self.b1)
            out = hidden @ self.w2 + self.b2
            return out.detach().cpu().numpy()
        hidden = torch.relu(x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


class TravelTimeHead:
    """Regress edge travel time via a small MLP."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, device: str | torch.device = "cpu") -> None:
        in_dim = 2 * embed_dim
        rng = np.random.default_rng(0)
        w1 = torch.from_numpy(
            rng.standard_normal((in_dim, hidden_dim), dtype=np.float32)
        ) / torch.sqrt(torch.tensor(in_dim, dtype=torch.float32))
        b1 = torch.zeros(hidden_dim, dtype=torch.float32)
        w2 = torch.from_numpy(
            rng.standard_normal((hidden_dim, 1), dtype=np.float32)
        ) / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        b2 = torch.zeros(1, dtype=torch.float32)
        device = torch.device(device)
        self.mlp = MLP(w1.to(device), b1.to(device), w2.to(device), b2.to(device))

    def __call__(self, u_emb: "np.ndarray | torch.Tensor", v_emb: "np.ndarray | torch.Tensor") -> "np.ndarray | torch.Tensor":
        if isinstance(u_emb, np.ndarray):
            x = np.concatenate([u_emb, v_emb], axis=-1)
            out = self.mlp(x)
            return out.squeeze(-1)
        x = torch.cat([u_emb, v_emb], dim=-1)
        out = self.mlp(x)
        return out.squeeze(-1)


class CrowdingHead:
    """Predict crowding level either as classification or regression."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, num_classes: int = 3, device: str | torch.device = "cpu") -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        in_dim = 2 * embed_dim
        rng = np.random.default_rng(1)
        w1 = torch.from_numpy(
            rng.standard_normal((in_dim, hidden_dim), dtype=np.float32)
        ) / torch.sqrt(torch.tensor(in_dim, dtype=torch.float32))
        b1 = torch.zeros(hidden_dim, dtype=torch.float32)
        w2 = torch.from_numpy(
            rng.standard_normal((hidden_dim, num_classes), dtype=np.float32)
        ) / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        b2 = torch.zeros(num_classes, dtype=torch.float32)
        device = torch.device(device)
        self.mlp = MLP(w1.to(device), b1.to(device), w2.to(device), b2.to(device))
        self.num_classes = num_classes

    def __call__(self, u_emb: "np.ndarray | torch.Tensor", v_emb: "np.ndarray | torch.Tensor") -> "np.ndarray | torch.Tensor":
        if isinstance(u_emb, np.ndarray):
            x = np.concatenate([u_emb, v_emb], axis=-1)
            out = self.mlp(x)
            return out
        x = torch.cat([u_emb, v_emb], dim=-1)
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
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    probs = CrowdingHead.softmax(logits.reshape(-1, logits.shape[-1]))
    lbl = labels.reshape(-1)
    n = lbl.shape[0]
    loss = -np.log(probs[np.arange(n), lbl])
    return float(np.mean(loss))


__all__ = ["TravelTimeHead", "CrowdingHead", "mse_loss", "cross_entropy_loss"]
