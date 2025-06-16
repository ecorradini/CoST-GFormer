import numpy as np
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_gformer.attention import Attention, UnifiedSpatioTemporalAttention
from cost_gformer.memory import ShortTermMemory, LongTermMemory


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


def test_attention_cpu():
    _run_attention("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_attention_cuda():
    _run_attention("cuda")


def test_usta_cpu():
    _run_usta("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_usta_cuda():
    _run_usta("cuda")
