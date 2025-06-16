"""Attention mechanisms for CoST-GFormer.

This module contains a placeholder :class:`Attention` that hints at multi-head
attention capable of operating over Short Term Memory (STM) as well as Long
Term Memory (LTM). Real implementations would include query/key/value
projections and memory-efficient computation.
"""


import numpy as np


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


class UnifiedSpatioTemporalAttention:
    """Simple implementation of USTA with mixture-of-experts and pruning."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_experts: int = 2,
        top_k: int = 5,
        rng: np.random.Generator | None = None,
    ) -> None:
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k

        rng = np.random.default_rng() if rng is None else rng

        head_dim = embed_dim // num_heads
        self.head_dim = head_dim

        # Query projection shared across experts
        self.W_q = (
            rng.standard_normal((embed_dim, num_heads, head_dim), dtype=np.float32)
            / np.sqrt(embed_dim)
        )

        # Expert-specific key/value projections
        self.W_k = (
            rng.standard_normal(
                (num_experts, embed_dim, num_heads, head_dim), dtype=np.float32
            )
            / np.sqrt(embed_dim)
        )
        self.W_v = (
            rng.standard_normal(
                (num_experts, embed_dim, num_heads, head_dim), dtype=np.float32
            )
            / np.sqrt(embed_dim)
        )

        self.W_g = rng.standard_normal((embed_dim, num_experts), dtype=np.float32)

        self.W_o = (
            rng.standard_normal((num_heads * head_dim, embed_dim), dtype=np.float32)
            / np.sqrt(num_heads * head_dim)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _phi(x: np.ndarray) -> np.ndarray:
        """Feature map for linear attention."""
        return np.maximum(0.0, x) + 1e-6

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def _attention_single(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Compute attention with pruning for one head."""

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        scores = phi_q @ phi_k.T

        # Prune to top-k neighbors for each query
        n = q.shape[0]
        out = np.zeros_like(q)
        for i in range(n):
            s = scores[i]
            idx = np.argpartition(-s, self.top_k)[: self.top_k]
            weights = self._softmax(s[idx])
            out[i] = weights @ v[idx]
        return out

    # ------------------------------------------------------------------
    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Apply USTA over a set of node-time embeddings."""

        n, _ = h.shape

        # Compute queries shared by experts
        q = np.einsum("nd,dhe->nhe", h, self.W_q)

        # Compute gating weights
        gate_logits = h @ self.W_g
        gates = self._softmax(gate_logits)

        head_outputs = []
        for head in range(self.num_heads):
            q_h = q[:, head, :]
            head_out = np.zeros((n, self.head_dim), dtype=np.float32)
            for e in range(self.num_experts):
                g = gates[:, e, None]
                k = h @ self.W_k[e, :, head, :]
                v = h @ self.W_v[e, :, head, :]
                attn = self._attention_single(q_h, k, v)
                head_out += g * attn
            head_outputs.append(head_out)

        concat = np.concatenate(head_outputs, axis=-1)
        out = concat @ self.W_o
        return out


__all__ = ["Attention", "UnifiedSpatioTemporalAttention"]
