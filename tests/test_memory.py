import numpy as np
from cost_gformer.memory import ShortTermMemory, LongTermMemory


def test_stm_read_write():
    num_nodes = 3
    dim = 4
    stm = ShortTermMemory(size=3, num_nodes=num_nodes, embed_dim=dim)
    rng = np.random.default_rng(0)
    for _ in range(3):
        stm.write(rng.standard_normal((num_nodes, dim), dtype=np.float32))

    assert stm.buffer.shape == (3, num_nodes, dim)
    out = stm.read(0)
    assert out.shape == (dim,)
    all_out = stm.read_all()
    assert all_out.shape == (num_nodes, dim)


def test_ltm_build_and_retrieve():
    num_nodes = 2
    dim = 3
    rng = np.random.default_rng(0)
    history = rng.standard_normal((5, num_nodes, dim), dtype=np.float32)
    ltm = LongTermMemory(num_nodes=num_nodes, embed_dim=dim, num_centroids=2)
    for i in range(5):
        ltm.write(history[i])
    ltm.build()
    read = ltm.read(0, history[-1, 0])
    assert read.shape == (dim,)
    fused = ltm.fuse(0, history[-1, 0])
    assert fused.shape == (dim,)


def test_ltm_build_with_external_embeddings():
    num_nodes = 2
    dim = 3
    rng = np.random.default_rng(1)
    history = rng.standard_normal((4, num_nodes, dim), dtype=np.float32)
    ltm = LongTermMemory(num_nodes=num_nodes, embed_dim=dim, num_centroids=3)
    ltm.build(history, iters=2)
    out = ltm.read_all(history[-1])
    assert out.shape == (num_nodes, dim)
