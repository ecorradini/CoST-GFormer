"""Simple training utilities for CoST-GFormer.

This module provides a lightweight trainer that optimises only the output
heads of the model using plain numpy. It is purely for demonstration and not
intended for large-scale use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .data import DataModule, GraphSnapshot
from .embedding import SpatioTemporalEmbedding
from .model import CoSTGFormer


# ---------------------------------------------------------------------------
# Helper functions for manual backpropagation of the tiny MLPs
# ---------------------------------------------------------------------------

def _mlp_forward(mlp, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(0.0, x @ mlp.w1 + mlp.b1)
    out = hidden @ mlp.w2 + mlp.b2
    return hidden, out


def _mlp_backward(
    mlp, x: np.ndarray, hidden: np.ndarray, grad_out: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grad_w2 = np.outer(hidden, grad_out)
    grad_b2 = grad_out
    grad_hidden = grad_out @ mlp.w2.T
    grad_hidden[hidden <= 0.0] = 0.0
    grad_w1 = np.outer(x, grad_hidden)
    grad_b1 = grad_hidden
    return grad_w1, grad_b1, grad_w2, grad_b2


def _update_mlp(mlp, grads, lr: float) -> None:
    gw1, gb1, gw2, gb2 = grads
    mlp.w1 -= lr * gw1
    mlp.b1 -= lr * gb1
    mlp.w2 -= lr * gw2
    mlp.b2 -= lr * gb2


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

def _prepare_targets(
    snap: GraphSnapshot, num_classes: int, classification: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges: List[Tuple[int, int]] = []
    tt: List[float] = []
    cr: List[int | float] = []
    for e in snap.edges:
        feat = snap.dynamic_edge_feat[e]
        edges.append(e)
        tt.append(float(feat[0]))
        if classification:
            label = int(feat[1] * num_classes)
            if label >= num_classes:
                label = num_classes - 1
            cr.append(label)
        else:
            cr.append(float(feat[1]))
    arr_edges = np.array(edges, dtype=np.int64)
    tt_tgt = np.array(tt, dtype=np.float32)
    cr_tgt = (
        np.array(cr, dtype=np.int64)
        if classification
        else np.array(cr, dtype=np.float32)
    )
    return arr_edges, tt_tgt, cr_tgt


# ---------------------------------------------------------------------------
@dataclass
class Trainer:
    """Very small trainer using SGD on the output heads."""

    model: CoSTGFormer
    data: DataModule
    lr: float = 0.01
    epochs: int = 5
    classification: bool = True

    def __post_init__(self) -> None:
        n = len(self.data)
        split = int(n * 0.8)
        self.train_idx = list(range(0, split))
        self.val_idx = list(range(split, n))
        self.stm = self.model.embedding
        if self.stm is None:
            first = self.data.dataset[0]
            dyn_dim = len(next(iter(first.dynamic_edge_feat.values())))
            num_nodes = len({u for u, _ in first.edges} | {v for _, v in first.edges})
            self.stm = SpatioTemporalEmbedding(
                num_nodes=num_nodes,
                static_edges=first.edges,
                dynamic_dim=dyn_dim,
            )
            self.model.embedding = self.stm

    # --------------------------------------------------------------
    def _run_batch(
        self,
        history: List[GraphSnapshot],
        target_snap: GraphSnapshot,
        update: bool,
    ) -> Tuple[float, float, float, float, int]:
        embeddings = self.stm.encode_window(history)
        current = embeddings[-1]
        edges, tt_tgt, cr_tgt = _prepare_targets(
            target_snap, self.model.crowd_head.num_classes, self.classification
        )
        loss = 0.0
        preds_tt: List[float] = []
        preds_cr: List[int] = []
        n_edges = len(edges)
        for (u, v), tt, cr in zip(edges, tt_tgt, cr_tgt):
            x = np.concatenate([current[u], current[v]])
            # Travel time regression
            h_tt, pred_tt = _mlp_forward(self.model.travel_head.mlp, x)
            pred_tt = float(pred_tt.squeeze())
            preds_tt.append(pred_tt)
            l_tt = (pred_tt - tt) ** 2
            grad_tt = 2.0 * (pred_tt - tt)
            if update:
                grads = _mlp_backward(
                    self.model.travel_head.mlp,
                    x,
                    h_tt,
                    np.array([grad_tt], dtype=np.float32),
                )
                _update_mlp(self.model.travel_head.mlp, grads, self.lr)
            # Crowding prediction
            h_cr, logits = _mlp_forward(self.model.crowd_head.mlp, x)
            if self.classification:
                logits = logits.squeeze()
                exp = np.exp(logits - logits.max())
                probs = exp / exp.sum()
                l_cr = -np.log(probs[int(cr)])
                grad_logits = probs
                grad_logits[int(cr)] -= 1.0
                preds_cr.append(int(np.argmax(logits)))
            else:
                pred = float(logits.squeeze())
                l_cr = (pred - cr) ** 2
                grad_logits = 2.0 * (pred - cr)
            if update:
                grads = _mlp_backward(
                    self.model.crowd_head.mlp, x, h_cr, grad_logits
                )
                _update_mlp(self.model.crowd_head.mlp, grads, self.lr)
            loss += l_tt + l_cr

        mae_tt = float(np.mean(np.abs(np.array(preds_tt) - tt_tgt))) if preds_tt else 0.0
        rmse_tt = float(np.sqrt(np.mean((np.array(preds_tt) - tt_tgt) ** 2))) if preds_tt else 0.0
        acc_cr = 0.0
        if self.classification and preds_cr:
            acc_cr = float(np.mean(np.array(preds_cr) == cr_tgt))
        return float(loss / n_edges), mae_tt, rmse_tt, acc_cr, n_edges

    # --------------------------------------------------------------
    def train_epoch(self) -> Tuple[float, float, float, float]:
        total = 0.0
        tt_mae = 0.0
        tt_rmse = 0.0
        cr_acc = 0.0
        n_samples = 0
        for idx in self.train_idx:
            hist, fut = self.data[idx]
            loss, mae, rmse, acc, n = self._run_batch(hist, fut[0], update=True)
            total += loss
            tt_mae += mae * n
            tt_rmse += rmse * n
            cr_acc += acc * n
            n_samples += n
        if n_samples == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            total / len(self.train_idx),
            tt_mae / n_samples,
            tt_rmse / n_samples,
            cr_acc / n_samples,
        )

    def val_epoch(self) -> Tuple[float, float, float, float]:
        if not self.val_idx:
            return 0.0, 0.0, 0.0, 0.0
        total = 0.0
        tt_mae = 0.0
        tt_rmse = 0.0
        cr_acc = 0.0
        n_samples = 0
        for idx in self.val_idx:
            hist, fut = self.data[idx]
            loss, mae, rmse, acc, n = self._run_batch(hist, fut[0], update=False)
            total += loss
            tt_mae += mae * n
            tt_rmse += rmse * n
            cr_acc += acc * n
            n_samples += n
        if n_samples == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            total / len(self.val_idx),
            tt_mae / n_samples,
            tt_rmse / n_samples,
            cr_acc / n_samples,
        )

    def fit(self) -> None:
        for epoch in range(1, self.epochs + 1):
            t_loss, t_mae, t_rmse, t_acc = self.train_epoch()
            v_loss, v_mae, v_rmse, v_acc = self.val_epoch()
            print(
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


__all__ = ["Trainer"]
