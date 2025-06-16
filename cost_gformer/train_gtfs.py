"""Simple training script for GTFS datasets using the lightweight Trainer.

This entry point supports arbitrary forecasting horizons using the ``--horizon``
flag and relies on the updated :class:`Trainer` for multi-step targets.
"""

from __future__ import annotations

import argparse

import logging

from cost_gformer.gtfs import load_gtfs
from cost_gformer.data import DataModule
from cost_gformer.model import CoSTGFormer
from cost_gformer.trainer import Trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CoST-GFormer on GTFS data")
    p.add_argument("static", help="Path to GTFS static feed")
    p.add_argument("realtime", nargs="?", help="Optional path to GTFS realtime feed")
    p.add_argument("--history", type=int, default=3, help="Number of history steps")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument(
        "--lr-step-size",
        type=int,
        default=0,
        metavar="N",
        help="StepLR step size (0 to disable)",
    )
    p.add_argument(
        "--lr-gamma", type=float, default=0.95, help="StepLR decay factor"
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Computation device",
    )
    p.add_argument(
        "--regression",
        action="store_true",
        help="Use regression for crowd level instead of classification",
    )
    p.add_argument(
        "--log-file",
        metavar="PATH",
        help="Write verbose training log to the given file",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    dataset = load_gtfs(args.static, args.realtime)
    data = DataModule(dataset, history=args.history, horizon=args.horizon)
    model = CoSTGFormer()
    schedule = None
    if args.lr_step_size > 0:
        schedule = {"step_size": args.lr_step_size, "gamma": args.lr_gamma}
    trainer = Trainer(
        model=model,
        data=data,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        classification=not args.regression,
        lr_schedule=schedule,
        device=args.device,
    )
    trainer.fit()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
