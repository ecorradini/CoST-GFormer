import numpy as np
import torch

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
