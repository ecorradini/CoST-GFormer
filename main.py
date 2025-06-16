"""Example usage of the CoST-GFormer package."""

from cost_gformer import CoSTGFormer, DataModule


def main() -> None:
    """Run a trivial example demonstrating imports."""
    data = DataModule()
    model = CoSTGFormer()
    print("Data module:", data)
    print("Model:", model)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
