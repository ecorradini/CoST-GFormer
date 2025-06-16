"""Attention mechanisms for CoST-GFormer.

The module provides fully functional attention layers.  The :class:`Attention`
class demonstrates how multi-head attention can operate over the Short Term
Memory (STM) and Long Term Memory (LTM), while
:class:`UnifiedSpatioTemporalAttention` implements a pruning mixture-of-experts
variant used by the model.
"""


import numpy as np  # for type hints
import torch
from torch.nn import Parameter


class Attention:
    """Multi-head attention supporting STM and LTM interactions."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        rng: np.random.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        rng = np.random.default_rng() if rng is None else rng
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.device = torch.device(device)

        self.W_q = Parameter(
            torch.from_numpy(
                rng.standard_normal((embed_dim, num_heads, self.head_dim), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)
        self.W_k = Parameter(
            torch.from_numpy(
                rng.standard_normal((embed_dim, num_heads, self.head_dim), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)
        self.W_v = Parameter(
            torch.from_numpy(
                rng.standard_normal((embed_dim, num_heads, self.head_dim), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)
        self.W_o = Parameter(
            torch.from_numpy(
                rng.standard_normal((self.num_heads * self.head_dim, embed_dim), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(self.num_heads * self.head_dim, dtype=torch.float32))
        ).to(self.device)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(heads={self.num_heads})"

    # ------------------------------------------------------------------
    def __call__(
        self,
        x: np.ndarray,
        stm: "np.ndarray | 'ShortTermMemory' | None" = None,
        ltm: "np.ndarray | 'LongTermMemory' | None" = None,
    ) -> np.ndarray:
        """Apply attention over current embeddings and optional memories."""

        from .memory import ShortTermMemory, LongTermMemory

        q = torch.from_numpy(x).to(self.device)

        memories = [q]
        if stm is not None:
            if isinstance(stm, ShortTermMemory):
                m = torch.from_numpy(stm.read_all()).to(self.device)
            else:
                m = torch.from_numpy(stm).to(self.device)
            memories.append(m)
        if ltm is not None:
            if isinstance(ltm, LongTermMemory):
                m = torch.from_numpy(ltm.read_all(x)).to(self.device)
            else:
                m = torch.from_numpy(ltm).to(self.device)
            memories.append(m)

        mem = torch.cat(memories, dim=0)

        q = torch.einsum("nd,dhe->nhe", q, self.W_q)
        k = torch.einsum("md,dhe->mhe", mem, self.W_k)
        v = torch.einsum("md,dhe->mhe", mem, self.W_v)

        scale = 1.0 / torch.sqrt(torch.tensor(float(self.head_dim)))
        scores = torch.einsum("nhe,mhe->nhm", q, k) * scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("nhm,mhe->nhe", weights, v)

        concat = out.reshape(out.shape[0], -1)
        result = concat @ self.W_o
        return result.cpu().numpy()

class UnifiedSpatioTemporalAttention:
    """USTA with learnable projections and gated mixture-of-experts."""

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

        # Learnable projections
        self.W_q = Parameter(
            torch.from_numpy(rng.standard_normal((embed_dim, num_heads, head_dim), dtype=np.float32))
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)

        self.W_k = Parameter(
            torch.from_numpy(
                rng.standard_normal((num_experts, embed_dim, num_heads, head_dim), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)
        self.W_v = Parameter(
            torch.from_numpy(
                rng.standard_normal((num_experts, embed_dim, num_heads, head_dim), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)

        self.W_g = Parameter(
            torch.from_numpy(
                rng.standard_normal((embed_dim, num_heads, num_experts), dtype=np.float32)
            )
            / torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        ).to(self.device)

        self.W_o = Parameter(
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
        x = x - x.max(dim=-1, keepdim=True).values
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

        gate_logits = torch.einsum("nd,dhe->nhe", h_t, self.W_g)
        gates = self._softmax(gate_logits)

        head_outputs = []
        for head in range(self.num_heads):
            q_h = q[:, head, :]
            head_out = torch.zeros((n, self.head_dim), dtype=torch.float32, device=self.device)
            for e in range(self.num_experts):
                g = gates[:, head, e, None]
                k = h_t @ self.W_k[e, :, head, :]
                v = h_t @ self.W_v[e, :, head, :]
                attn = self._attention_single(q_h, k, v)
                head_out += g * attn
            head_outputs.append(head_out)

        concat = torch.cat(head_outputs, dim=-1)
        out = concat @ self.W_o
        return out.cpu().numpy()


__all__ = ["Attention", "UnifiedSpatioTemporalAttention"]
