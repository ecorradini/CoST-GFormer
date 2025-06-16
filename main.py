"""Example usage of the CoST-GFormer package."""

from cost_gformer import (
    CoSTGFormer,
    DataModule,
    SpatioTemporalEmbedding,
    ExpandedGraph,
    DynamicGraphHandler,
    UnifiedSpatioTemporalAttention,
    LongTermMemory,
    Trainer,
)
import numpy as np
from cost_gformer.data import generate_synthetic_dataset


def main() -> None:
    """Run a minimal example with synthetic data."""
    dataset = generate_synthetic_dataset(num_nodes=4, num_snapshots=10, seed=0)
    data = DataModule(dataset, history=3, horizon=2)

    # Retrieve the first training example just to demonstrate usage
    history, future = data[0]

    # Build spatio-temporal embeddings for the historical window
    static_edges = dataset[0].edges
    dyn_dim = len(next(iter(dataset[0].dynamic_edge_feat.values())))
    stm = SpatioTemporalEmbedding(num_nodes=4, static_edges=static_edges, dynamic_dim=dyn_dim)
    embeddings = stm.encode_window(history)

    # Construct expanded spatio-temporal graph using the same window
    graph = ExpandedGraph(history, num_nodes=4)

    # Compute fused dynamic adjacency for the latest snapshot
    dyn_graph = DynamicGraphHandler(num_nodes=4, static_edges=static_edges)
    latest_embed = embeddings[-1]
    fused_adj = dyn_graph.update(latest_embed)

    # Build lightweight temporal memory from the historical embeddings
    ltm = LongTermMemory(num_nodes=4, embed_dim=embeddings.shape[-1])
    ltm.build(embeddings)

    model = CoSTGFormer(embedding=stm, num_nodes=4)

    trainer = Trainer(model=model, data=data, epochs=2)
    trainer.fit()

    # Apply unified attention to the expanded embeddings as a demo
    usta = UnifiedSpatioTemporalAttention(embed_dim=embeddings.shape[-1])
    expanded_embeds = embeddings.reshape(-1, embeddings.shape[-1])
    attended = usta(expanded_embeds)

    # Predict travel time and crowding for the next snapshot
    next_edges = np.array(future[0].edges, dtype=np.int64)
    tt_pred, crowd_pred = model.forward(latest_embed, next_edges)

    print("Number of samples:", len(data))
    print("History length:", len(history))
    print("Future length:", len(future))
    print("Embeddings shape:", embeddings.shape)
    print("Model:", model)
    print("Expanded nodes:", graph.num_expanded_nodes)
    print("Edge index shape:", graph.edge_index().shape)
    print("Fused adjacency shape:", fused_adj.shape)
    sample_fused = ltm.fuse(0, latest_embed[0])
    print("LTM fused embedding shape:", sample_fused.shape)
    print("USTA output shape:", attended.shape)
    print("Travel time pred shape:", tt_pred.shape)
    print("Crowding pred shape:", crowd_pred.shape)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
