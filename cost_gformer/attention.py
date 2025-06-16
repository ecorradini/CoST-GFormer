"""Attention mechanisms for CoST-GFormer.

This module contains a placeholder :class:`Attention` that hints at multi-head
attention capable of operating over Short Term Memory (STM) as well as Long
Term Memory (LTM). Real implementations would include query/key/value
projections and memory-efficient computation.
"""


import numpy as np  # for type hints
import torch


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
        device: str | torch.device = "cpu",
    ) -> None:
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k

        rng = np.random.default_rng() if rng is None else rng
        self.device = torch.device(device)

        head_dim = embed_dim // num_heads
        self.head_dim = head_dim

        # Query projection shared across experts
        self.W_q = (
            torch.from_numpy(rng.standard_normal((embed_dim, num_heads, head_dim), dtype=np.float32))
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)

        # Expert-specific key/value projections
        self.W_k = (
            torch.from_numpy(
                rng.standard_normal(
                    (num_experts, embed_dim, num_heads, head_dim), dtype=np.float32
                )
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)
        self.W_v = (
            torch.from_numpy(
                rng.standard_normal(
                    (num_experts, embed_dim, num_heads, head_dim), dtype=np.float32
                )
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)

        self.W_g = torch.from_numpy(
            rng.standard_normal((embed_dim, num_experts), dtype=np.float32)
        ).to(self.device)

        self.W_o = (
            torch.from_numpy(
                rng.standard_normal((num_heads * head_dim, embed_dim), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(num_heads * head_dim, dtype=torch.float32))
        ).to(self.device)

    # ------------------------------------------------------------------
    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        """Feature map for linear attention."""
        return torch.relu(x) + 1e-6

    @staticmethod
    def _softmax(x: torch.Tensor) -> torch.Tensor:
        x = x - x.max()
        e = torch.exp(x)
        return e / e.sum(dim=-1, keepdim=True)

    def _attention_single(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention with pruning for one head."""

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        scores = phi_q @ phi_k.T

        n = q.shape[0]
        out = torch.zeros_like(q)
        for i in range(n):
            s = scores[i]
            idx = torch.topk(s, self.top_k).indices
            weights = self._softmax(s[idx])
            out[i] = weights @ v[idx]
        return out

    # ------------------------------------------------------------------
    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Apply USTA over a set of node-time embeddings."""

        n, _ = h.shape
        h_t = torch.from_numpy(h).to(self.device)

        q = torch.einsum("nd,dhe->nhe", h_t, self.W_q)

        gate_logits = h_t @ self.W_g
        gates = self._softmax(gate_logits)

        head_outputs = []
        for head in range(self.num_heads):
            q_h = q[:, head, :]
            head_out = torch.zeros((n, self.head_dim), dtype=torch.float32, device=self.device)
            for e in range(self.num_experts):
                g = gates[:, e, None]
                k = h_t @ self.W_k[e, :, head, :]
                v = h_t @ self.W_v[e, :, head, :]
                attn = self._attention_single(q_h, k, v)
                head_out += g * attn
            head_outputs.append(head_out)

        concat = torch.cat(head_outputs, dim=-1)
        out = concat @ self.W_o
        return out.cpu().numpy()


__all__ = ["Attention", "UnifiedSpatioTemporalAttention"]
