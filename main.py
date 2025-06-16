"""Example usage of the CoST-GFormer package."""

from cost_gformer import (
    CoSTGFormer,
    DataModule,
    SpatioTemporalEmbedding,
    ExpandedGraph,
    UnifiedSpatioTemporalAttention,
)
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

    model = CoSTGFormer(embedding=stm)

    # Apply unified attention to the expanded embeddings as a demo
    usta = UnifiedSpatioTemporalAttention(embed_dim=embeddings.shape[-1])
    expanded_embeds = embeddings.reshape(-1, embeddings.shape[-1])
    attended = usta(expanded_embeds)

    print("Number of samples:", len(data))
    print("History length:", len(history))
    print("Future length:", len(future))
    print("Embeddings shape:", embeddings.shape)
    print("Model:", model)
    print("Expanded nodes:", graph.num_expanded_nodes)
    print("Edge index shape:", graph.edge_index().shape)
    print("USTA output shape:", attended.shape)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
