import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_gformer.graph import DynamicGraphHandler, ExpandedGraph
from cost_gformer.data import generate_synthetic_dataset


def test_live_edge_updates():
    handler = DynamicGraphHandler(
        num_nodes=3,
        static_edges=[(0, 1), (1, 2)],
        alpha=1.0,
        decay=0.5,
        remove_threshold=0.1,
    )

    assert handler.static_adj[1, 2] == 1.0
    assert handler.dynamic_mask[1, 2] == 1.0

    for _ in range(3):
        handler.ingest([(0, 1)])

    assert np.isclose(handler.dynamic_mask[1, 2], 0.125)
    assert handler.static_adj[1, 2] == 1.0

    handler.ingest([(0, 1)])
    assert handler.static_adj[1, 2] == 0.0
    assert handler.dynamic_mask[1, 2] == 0.0

    handler.ingest([(0, 1), (0, 2)])
    assert handler.static_adj[0, 2] == 1.0
    assert handler.dynamic_mask[0, 2] == 1.0

    emb = np.eye(3, dtype=np.float32)
    fused = handler.update(emb)
    assert fused.shape == (3, 3)
    assert fused[1, 2] == 0.0


def test_expanded_graph_edge_shape():
    dataset = generate_synthetic_dataset(num_nodes=3, num_snapshots=4, seed=0)
    history = dataset.snapshots[:3]
    graph = ExpandedGraph(history, num_nodes=3)
    edge_idx = graph.edge_index()

    n = 3
    T = 3
    expected = T * n * (n - 1) + (T - 1) * n
    assert edge_idx.shape == (2, expected)
