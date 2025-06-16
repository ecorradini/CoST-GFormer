"""Example usage of the CoST-GFormer package."""

from cost_gformer import CoSTGFormer, DataModule
from cost_gformer.data import generate_synthetic_dataset


def main() -> None:
    """Run a minimal example with synthetic data."""
    dataset = generate_synthetic_dataset(num_nodes=4, num_snapshots=10, seed=0)
    data = DataModule(dataset, history=3, horizon=2)

    # Retrieve the first training example just to demonstrate usage
    history, future = data[0]

    model = CoSTGFormer()

    print("Number of samples:", len(data))
    print("History length:", len(history))
    print("Future length:", len(future))
    print("Model:", model)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
