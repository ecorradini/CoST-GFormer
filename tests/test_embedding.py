import numpy as np
import torch
import pytest

from cost_gformer.embedding import SpatioTemporalEmbedding
from cost_gformer.data import generate_synthetic_dataset


def test_sparse_embedding():
    dataset = generate_synthetic_dataset(num_nodes=6, num_snapshots=1, seed=0)
    edges = dataset[0].edges
    dyn_dim = len(next(iter(dataset[0].dynamic_edge_feat.values())))
    stm = SpatioTemporalEmbedding(
        num_nodes=6,
        static_edges=edges,
        dynamic_dim=dyn_dim,
        use_sparse=True,
    )
    emb = stm.encode_snapshot(dataset[0])
    assert emb.shape == (6, stm.mlp.b2.numel())
    assert stm.spectral.dtype == torch.float32
    assert emb.dtype == np.float32


def test_dense_embedding_and_window():
    dataset = generate_synthetic_dataset(num_nodes=4, num_snapshots=2, seed=1)
    edges = dataset[0].edges
    dyn_dim = len(next(iter(dataset[0].dynamic_edge_feat.values())))
    stm = SpatioTemporalEmbedding(
        num_nodes=4,
        static_edges=edges,
        dynamic_dim=dyn_dim,
        use_sparse=False,
    )
    emb = stm.encode_snapshot(dataset[0])
    assert emb.shape == (4, stm.mlp.b2.numel())
    window = stm.encode_window(dataset.snapshots)
    assert window.shape == (2, 4, stm.mlp.b2.numel())
    expected = np.stack([stm.encode_snapshot(s) for s in dataset.snapshots])
    np.testing.assert_allclose(window, expected, rtol=1e-6, atol=1e-7)
    assert window.dtype == np.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_encode_window_cuda():
    dataset = generate_synthetic_dataset(num_nodes=3, num_snapshots=2, seed=3)
    edges = dataset[0].edges
    dyn_dim = len(next(iter(dataset[0].dynamic_edge_feat.values())))
    stm = SpatioTemporalEmbedding(
        num_nodes=3,
        static_edges=edges,
        dynamic_dim=dyn_dim,
        use_sparse=False,
        device="cuda",
    )
    window = stm.encode_window(dataset.snapshots)
    expected = np.stack([stm.encode_snapshot(s) for s in dataset.snapshots])
    np.testing.assert_allclose(window, expected, rtol=1e-5, atol=1e-6)


def test_dynamic_aggregation_consistency():
    dataset = generate_synthetic_dataset(num_nodes=5, num_snapshots=1, seed=42)
    snap = dataset[0]
    dyn_dim = len(next(iter(snap.dynamic_edge_feat.values())))
    stm = SpatioTemporalEmbedding(
        num_nodes=5,
        static_edges=snap.edges,
        dynamic_dim=dyn_dim,
        use_sparse=False,
    )

    agg = stm._aggregate_dynamic(snap)

    ref = np.zeros((5, dyn_dim), dtype=np.float32)
    count = np.zeros(5, dtype=np.float32)
    for u, v in snap.edges:
        feat = snap.dynamic_edge_feat[(u, v)]
        ref[u] += feat
        ref[v] += feat
        count[u] += 1
        count[v] += 1
    count[count == 0] = 1.0
    ref /= count[:, None]

    assert np.allclose(agg.numpy(), ref)
