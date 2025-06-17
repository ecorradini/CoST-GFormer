"""PyTorch trainer utilities for CoST-GFormer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict

import logging

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

import numpy as np
import torch
import torch.nn.functional as F

from .data import DataModule, GraphSnapshot, Edge
from .embedding import SpatioTemporalEmbedding
from .model import CoSTGFormer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _prepare_targets(
    snap: GraphSnapshot,
    num_classes: int,
    classification: bool,
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect targets for a ``GraphSnapshot`` as tensors on ``device``."""

    edge_list: List[Tuple[int, int]] = []
    tt_vals: List[float] = []
    cr_vals: List[int | float] = []

    for e in snap.edges:
        feat = snap.dynamic_edge_feat.get(e)
        if feat is None or len(feat) == 0:
            # Skip edges without dynamic data to avoid index errors during
            # training.  This mirrors the behaviour of ``encode_snapshot``
            # which uses zeros when no features are available.
            continue
        edge_list.append(e)

        # Always expect at least one dynamic feature for travel/ delay time
        if len(feat) < 1:
            raise ValueError("dynamic edge feature must contain at least one value")
        tt_vals.append(float(feat[0]))

        crowd_val = float(feat[1]) if len(feat) > 1 else 0.0
        if classification:
            label = int(crowd_val * num_classes)
            if label >= num_classes:
                label = num_classes - 1
            cr_vals.append(label)
        else:
            cr_vals.append(crowd_val)

    edges = torch.tensor(edge_list, dtype=torch.int64, device=device)
    tt_tgt = torch.tensor(tt_vals, dtype=torch.float32, device=device)
    if classification:
        cr_tgt = torch.tensor(cr_vals, dtype=torch.int64, device=device)
    else:
        cr_tgt = torch.tensor(cr_vals, dtype=torch.float32, device=device)

    return edges, tt_tgt, cr_tgt


# ---------------------------------------------------------------------------
@dataclass
class Trainer:
    """Simple trainer optimising only the output heads."""

    model: CoSTGFormer
    data: DataModule
    lr: float = 0.01
    epochs: int = 5
    batch_size: int = 1
    classification: bool = True
    lr_schedule: Dict[str, float] | None = None
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        n = len(self.data)
        split = int(n * 0.8)
        self.train_idx = list(range(0, split))
        self.val_idx = list(range(split, n))

        # Ensure an embedding module is present
        self.stm = self.model.embedding
        if self.stm is None:
            first = self.data.dataset[0]
            dyn_dim = len(next(iter(first.dynamic_edge_feat.values())))

            # Collect all nodes and edges across the dataset to properly size the
            # embedding module even when the first snapshot is sparse.
            all_edges: set[Edge] = set()
            all_nodes: set[int] = set()
            for snap in self.data.dataset.snapshots:
                all_edges.update(snap.edges)
                for u, v in snap.edges:
                    all_nodes.add(u)
                    all_nodes.add(v)

            if not all_nodes:
                raise ValueError("dataset contains no nodes")

            num_nodes = max(all_nodes) + 1

            self.stm = SpatioTemporalEmbedding(
                num_nodes=num_nodes,
                static_edges=all_edges,
                dynamic_dim=dyn_dim,
                device=self.device,
            )
            self.model.embedding = self.stm

        # Collect trainable parameters
        params = [
            self.model.travel_head.mlp.w1,
            self.model.travel_head.mlp.b1,
            self.model.travel_head.mlp.w2,
            self.model.travel_head.mlp.b2,
            self.model.crowd_head.mlp.w1,
            self.model.crowd_head.mlp.b1,
            self.model.crowd_head.mlp.w2,
            self.model.crowd_head.mlp.b2,
        ]
        for p in params:
            p.requires_grad_(True)
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        if self.lr_schedule is not None:
            step = int(self.lr_schedule.get("step_size", 1))
            gamma = float(self.lr_schedule.get("gamma", 0.95))
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step, gamma=gamma
            )
        else:
            self.scheduler = None

    # --------------------------------------------------------------
    def _forward_multi(
        self, history: List[GraphSnapshot], future: List[GraphSnapshot]
    ) -> Tuple[torch.Tensor, float, float, float, int]:
        if not future:
            return torch.tensor(0.0), 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            seq = history + future
            embeds = torch.from_numpy(self.stm.encode_window(seq)).to(self.device)

        start = len(history)
        total_loss = torch.tensor(0.0, device=self.device)
        mae_tt = 0.0
        rmse_tt = 0.0
        cr_acc = 0.0
        n_samples = 0

        for i, snap in enumerate(future):
            current = embeds[start + i]
            edges, tt_tgt, cr_tgt = _prepare_targets(
                snap,
                self.model.crowd_head.num_classes,
                self.classification,
                self.device,
            )
            if edges.numel() == 0:
                continue

            idx_u = edges[:, 0]
            idx_v = edges[:, 1]
            pairs = torch.cat([current[idx_u], current[idx_v]], dim=1)

            pred_tt = self.model.travel_head.mlp(pairs).squeeze(-1)
            l_tt = F.mse_loss(pred_tt, tt_tgt)
            step_mae = F.l1_loss(pred_tt, tt_tgt).item()
            step_rmse = torch.sqrt(F.mse_loss(pred_tt, tt_tgt)).item()

            logits = self.model.crowd_head.mlp(pairs)
            if self.classification:
                l_cr = F.cross_entropy(logits, cr_tgt.long())
                acc = (logits.argmax(dim=1) == cr_tgt).float().mean().item()
            else:
                pred = logits.squeeze(-1)
                l_cr = F.mse_loss(pred, cr_tgt)
                acc = 0.0

            total_loss = total_loss + l_tt + l_cr
            mae_tt += step_mae * edges.shape[0]
            rmse_tt += step_rmse * edges.shape[0]
            cr_acc += acc * edges.shape[0]
            n_samples += edges.shape[0]

        horizon = len(future)
        if n_samples == 0:
            return torch.tensor(0.0), 0.0, 0.0, 0.0, 0

        loss = total_loss / horizon
        return loss, mae_tt / n_samples, rmse_tt / n_samples, cr_acc / n_samples, n_samples

    # --------------------------------------------------------------
    def train_epoch(self) -> Tuple[float, float, float, float]:
        total_loss = 0.0
        total_samples = 0
        tt_mae = 0.0
        tt_rmse = 0.0
        cr_acc = 0.0

        for i in range(0, len(self.train_idx), self.batch_size):
            batch = self.train_idx[i : i + self.batch_size]
            self.optimizer.zero_grad()
            batch_loss = 0.0
            batch_samples = 0
            for idx in batch:
                hist, fut = self.data[idx]
                loss, mae, rmse, acc, n = self._forward_multi(hist, fut)
                batch_loss = batch_loss + loss
                batch_samples += n
                total_loss += float(loss) * n
                total_samples += n
                tt_mae += mae * n
                tt_rmse += rmse * n
                cr_acc += acc * n
            if batch_samples > 0:
                batch_loss.backward()
                self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if total_samples == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            total_loss / total_samples,
            tt_mae / total_samples,
            tt_rmse / total_samples,
            cr_acc / total_samples,
        )

    # --------------------------------------------------------------
    def val_epoch(self) -> Tuple[float, float, float, float]:
        if not self.val_idx:
            return 0.0, 0.0, 0.0, 0.0
        total_loss = 0.0
        total_samples = 0
        tt_mae = 0.0
        tt_rmse = 0.0
        cr_acc = 0.0
        with torch.no_grad():
            for idx in self.val_idx:
                hist, fut = self.data[idx]
                loss, mae, rmse, acc, n = self._forward_multi(hist, fut)
                total_loss += float(loss) * n
                total_samples += n
                tt_mae += mae * n
                tt_rmse += rmse * n
                cr_acc += acc * n
        if total_samples == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            total_loss / total_samples,
            tt_mae / total_samples,
            tt_rmse / total_samples,
            cr_acc / total_samples,
        )

    # --------------------------------------------------------------
    def fit(self) -> None:
        epoch_iter = range(1, self.epochs + 1)
        pbar = None
        if tqdm is not None:
            pbar = tqdm(epoch_iter, desc="Epochs")
            epoch_iter = pbar
        for epoch in epoch_iter:
            t_loss, t_mae, t_rmse, t_acc = self.train_epoch()
            v_loss, v_mae, v_rmse, v_acc = self.val_epoch()
            msg = (
                "Epoch %02d - train loss: %.4f mae: %.4f rmse: %.4f acc: %.4f - val loss: %.4f mae: %.4f rmse: %.4f acc: %.4f"
                % (
                    epoch,
                    t_loss,
                    t_mae,
                    t_rmse,
                    t_acc,
                    v_loss,
                    v_mae,
                    v_rmse,
                    v_acc,
                )
            )
            logger.info(msg)
            if pbar is not None:
                pbar.set_postfix(train_loss=t_loss, val_loss=v_loss)


__all__ = ["Trainer"]
