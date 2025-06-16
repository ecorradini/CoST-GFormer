import argparse
import torch
from cost_gformer.gtfs import load_gtfs
from cost_gformer.data import DataModule
from cost_gformer.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained CoST-GFormer model")
    p.add_argument("model", help="Path to saved model (created with torch.save)")
    p.add_argument("static", help="Path to GTFS static feed")
    p.add_argument("realtime", nargs="?", help="Optional path to GTFS realtime feed")
    p.add_argument("--history", type=int, default=3, help="Number of history steps")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--regression", action="store_true", help="Use regression for crowd level")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_gtfs(args.static, args.realtime)
    data = DataModule(dataset, history=args.history, horizon=args.horizon)
    model = torch.load(args.model, map_location=args.device)
    trainer = Trainer(
        model=model,
        data=data,
        epochs=1,
        batch_size=1,
        classification=not args.regression,
        device=args.device,
    )
    trainer.train_idx = []
    trainer.val_idx = list(range(len(data)))
    _, mae, rmse, acc = trainer.val_epoch()
    print(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nAccuracy: {acc:.4f}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
