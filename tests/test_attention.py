import numpy as np
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_gformer.attention import Attention, UnifiedSpatioTemporalAttention
from cost_gformer.memory import ShortTermMemory, LongTermMemory


def _attention_single_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, top_k: int) -> torch.Tensor:
    phi_q = torch.relu(q) + 1e-6
    phi_k = torch.relu(k) + 1e-6
    scores = phi_q @ phi_k.T

    n = q.shape[0]
    out = torch.zeros_like(q)
    for i in range(n):
        s = scores[i]
        k_val = min(top_k, s.shape[0])
        idx = torch.topk(s, k_val).indices
        weights = torch.softmax(s[idx], dim=-1)
        out[i] = weights @ v[idx]
    return out


def _run_attention(device: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 8), dtype=np.float32)
    attn = Attention(embed_dim=8, num_heads=2, device=device)

    stm = ShortTermMemory(size=2, num_nodes=3, embed_dim=8, device=device)
    for _ in range(2):
        stm.write(rng.standard_normal((3, 8), dtype=np.float32))

    ltm = LongTermMemory(num_nodes=3, embed_dim=8, num_centroids=2, device=device)
    for _ in range(3):
        ltm.write(rng.standard_normal((3, 8), dtype=np.float32))
    ltm.build()

    out = attn(x, stm, ltm)
    assert out.shape == (3, 8)
    assert isinstance(out, np.ndarray)


def _run_usta(device: str) -> None:
    rng = np.random.default_rng(1)
    dim = 12
    h = rng.standard_normal((5, dim), dtype=np.float32)
    usta = UnifiedSpatioTemporalAttention(
        embed_dim=dim, num_heads=3, num_experts=2, top_k=2, device=device
    )
    out = usta(h)
    assert out.shape == (5, dim)
    assert isinstance(out, np.ndarray)


def _run_usta_equivalence(device: str) -> None:
    rng = np.random.default_rng(2)
    n = 4
    dim = 10
    h = rng.standard_normal((n, dim), dtype=np.float32)
    usta = UnifiedSpatioTemporalAttention(
        embed_dim=dim, num_heads=2, num_experts=2, top_k=3, device=device
    )

    out = usta(h)

    h_t = torch.from_numpy(h).to(device)
    q = torch.einsum("nd,dhe->nhe", h_t, usta.W_q)
    gate_logits = torch.einsum("nd,dhe->nhe", h_t, usta.W_g)
    gates = torch.softmax(gate_logits, dim=-1)

    head_outputs = []
    for head in range(usta.num_heads):
        q_h = q[:, head, :]
        head_out = torch.zeros((n, usta.head_dim), dtype=torch.float32, device=device)
        for e in range(usta.num_experts):
            g = gates[:, head, e, None]
            k = h_t @ usta.W_k[e, :, head, :]
            v = h_t @ usta.W_v[e, :, head, :]
            attn = _attention_single_reference(q_h, k, v, usta.top_k)
            head_out += g * attn
        head_outputs.append(head_out)
    concat = torch.cat(head_outputs, dim=-1)
    manual_out = concat @ usta.W_o
    manual_out = manual_out.detach().cpu().numpy()

    np.testing.assert_allclose(out, manual_out, rtol=1e-5, atol=1e-6)


def test_attention_cpu():
    _run_attention("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_attention_cuda():
    _run_attention("cuda")


def test_usta_cpu():
    _run_usta("cpu")


def test_usta_equivalence_cpu():
    _run_usta_equivalence("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_usta_cuda():
    _run_usta("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_usta_equivalence_cuda():
    _run_usta_equivalence("cuda")
