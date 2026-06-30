"""TriZOD continuous-disorder pretraining pipeline (P4-Ph1-S02).

Reference: P4_00 Amendment T-1; P4_03 §S02.

Stage 1 of the two-stage pretrain → finetune pipeline. The model trains a
regression head against per-residue continuous TriZOD G-scores (NMR-derived,
range [0, 1]; ~5.6% of positions are unmeasured and stored as NaN — these
must be masked from both loss and gradient).

Loss: Huber (smooth L1), more robust than MSE to outlier G-scores at the
distribution tails. NaN positions in the target tensor are masked entirely
(zero gradient).

Optimizer: AdamW, LR=1e-3 (default).
Stopping: early stop on TriZOD pretrain-val Huber loss with patience.

Output: BACKBONE state dict (CNN + Transformer + projection + final_norm).
The regression head is DISCARDED — the binary classification finetune in
Stage 2 attaches a fresh head and uses discriminative LR (see
src/training/trainer.py::build_finetune_optimizer).

Data loading: HDF5 produced by P4-Ph0-S03 (data/trizod/pretrain_train.h5,
pretrain_val.h5). Per-protein groups with `sequence` (bytes), `gscores`
(float32, NaN for unmeasured), `zscores` (float32, NaN for unmeasured).

For pretraining, ESM-2 embeddings of TriZOD sequences must be computed
on-the-fly via a feature provider (the existing TriZOD cluster reps are
not yet in features/esm2/). The feature_fn callable is dependency-injected
so this module remains decoupled from any specific ESM-2 cache path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# ─── Loss ────────────────────────────────────────────────────────────


def huber_loss_with_nan_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss with NaN-position masking.

    Args:
        pred:   [B, L] regression predictions (raw, unconstrained).
        target: [B, L] continuous targets in [0, 1]; NaN = unmeasured.
        delta:  Huber transition point (default 1.0; for targets in [0, 1] this
                effectively reduces to MSE, which is what we want — outliers
                in TriZOD are usually informative low-coverage edge cases, but
                Huber's robustness on raw predictions during early training
                still helps stabilize gradients).

    Returns:
        Scalar mean loss over MEASURED (non-NaN) positions only. Returns
        zero if no valid positions exist (degenerate batch).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} != target {target.shape}")
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0  # preserve grad graph, zero magnitude
    # Replace NaNs in target with zeros so the masked positions contribute zero
    # loss (and zero gradient through pred at those positions)
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    per_pos = F.smooth_l1_loss(pred, safe_target, beta=delta, reduction="none")
    per_pos = per_pos * mask.to(per_pos.dtype)
    return per_pos.sum() / mask.sum().clamp_min(1).to(per_pos.dtype)


# ─── Configuration ───────────────────────────────────────────────────


@dataclass
class PretrainConfig:
    """TriZOD pretraining configuration."""
    lr: float = 1e-3
    weight_decay: float = 1e-2
    huber_delta: float = 1.0
    max_epochs: int = 50
    patience: int = 5
    min_epochs: int = 3
    max_grad_norm: float = 1.0
    seed: int = 42
    residue_budget: int = 12000
    checkpoint_dir: str = "checkpoints/part4_pretrain"
    experiment_id: str = "trizod_pretrain"
    log_every_n_epochs: int = 1


@dataclass
class PretrainEpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    wall_time_s: float


# ─── Data Loading ────────────────────────────────────────────────────


def load_trizod_records(h5_path: Path) -> list[dict]:
    """Load per-protein records from a TriZOD pretrain HDF5 file.

    Returns a list of dicts with keys: bmrb_id, sequence (str), gscores
    (np.ndarray[float32], NaN for unmeasured)."""
    records = []
    with h5py.File(h5_path, "r") as f:
        for bmrb_id in f.keys():
            grp = f[bmrb_id]
            seq = grp["sequence"][()]
            if isinstance(seq, bytes):
                seq = seq.decode()
            gs = grp["gscores"][:]  # float32 with NaN
            records.append({"bmrb_id": bmrb_id, "sequence": str(seq).upper(),
                            "gscores": gs})
    return records


def build_pretrain_batches(
    records: list[dict],
    feature_fn: Callable[[str], np.ndarray],
    residue_budget: int,
    rng_seed: int,
) -> list[dict]:
    """Length-sorted dynamic batches respecting `residue_budget`.

    Each batch dict contains:
      'features': torch.Tensor [B, L_max, D]
      'targets':  torch.Tensor [B, L_max]   (NaN at unmeasured/pad)
      'lengths':  torch.Tensor [B]
      'ids':      list[str]
    """
    rng = np.random.default_rng(rng_seed)
    sorted_recs = sorted(records, key=lambda r: len(r["sequence"]))
    batches: list[dict] = []
    cur: list[dict] = []
    cur_max = 0
    for rec in sorted_recs:
        L = len(rec["sequence"])
        proposed_max = max(cur_max, L)
        if (len(cur) + 1) * proposed_max > residue_budget and cur:
            batches.append(_pack_pretrain_batch(cur, feature_fn))
            cur, cur_max = [], 0
        cur.append(rec)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(_pack_pretrain_batch(cur, feature_fn))
    rng.shuffle(batches)
    return batches


def _pack_pretrain_batch(recs: list[dict], feature_fn) -> dict:
    L_max = max(len(r["sequence"]) for r in recs)
    feats_list = [feature_fn(r["sequence"]) for r in recs]
    D = feats_list[0].shape[1]
    B = len(recs)
    feats = np.zeros((B, L_max, D), dtype=np.float32)
    targets = np.full((B, L_max), np.nan, dtype=np.float32)
    lengths = np.zeros(B, dtype=np.int64)
    for i, (rec, f) in enumerate(zip(recs, feats_list)):
        L = len(rec["sequence"])
        feats[i, :L] = f[:L]
        targets[i, :L] = rec["gscores"][:L]
        lengths[i] = L
    return {
        "features": torch.from_numpy(feats),
        "targets": torch.from_numpy(targets),
        "lengths": torch.from_numpy(lengths),
        "ids": [r["bmrb_id"] for r in recs],
    }


# ─── Pretrain Trainer ────────────────────────────────────────────────


class TriZODPretrainTrainer:
    """Stage 1 trainer: continuous regression on TriZOD G-scores.

    The model passed in must take `[B, L, input_dim]` and return `[B, L]`
    raw regression scores (e.g. `HybridRegressionHead`).
    """

    def __init__(
        self,
        model: nn.Module,
        config: PretrainConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self._set_seed(config.seed)
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.epoch_logs: list[PretrainEpochLog] = []
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_epoch(self, batches: list[dict]) -> float:
        self.model.train()
        total_loss = 0.0
        total_positions = 0
        for batch in batches:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(features)
            loss = huber_loss_with_nan_mask(pred, targets, delta=self.config.huber_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            n_valid = int(torch.isfinite(targets).sum().item())
            total_loss += loss.item() * n_valid
            total_positions += n_valid
        return total_loss / max(total_positions, 1)

    @torch.no_grad()
    def validate(self, batches: list[dict]) -> float:
        self.model.eval()
        total_loss = 0.0
        total_positions = 0
        for batch in batches:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            pred = self.model(features)
            loss = huber_loss_with_nan_mask(pred, targets, delta=self.config.huber_delta)
            n_valid = int(torch.isfinite(targets).sum().item())
            total_loss += loss.item() * n_valid
            total_positions += n_valid
        return total_loss / max(total_positions, 1)

    def fit(self, train_batches: list[dict], val_batches: list[dict]) -> dict:
        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_batches)
            val_loss = self.validate(val_batches)
            wall = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            self.epoch_logs.append(PretrainEpochLog(epoch, train_loss, val_loss, lr, wall))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                # Save BACKBONE-only checkpoint (regression head is discarded)
                ckpt_path = self.ckpt_dir / f"{self.config.experiment_id}_seed{self.config.seed}_backbone.pt"
                torch.save({
                    "backbone_state_dict": (
                        self.model.backbone_state_dict()
                        if hasattr(self.model, "backbone_state_dict")
                        else self.model.state_dict()
                    ),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": asdict(self.config),
                }, ckpt_path)
            else:
                self.patience_counter += 1

            if (epoch >= self.config.min_epochs and
                    self.patience_counter >= self.config.patience):
                return {
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": self.best_epoch,
                    "total_epochs": epoch,
                    "early_stopped": True,
                }
        return {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": self.config.max_epochs,
            "early_stopped": False,
        }

    def save_run_log(self, path: Path):
        log_data = {
            "experiment_id": self.config.experiment_id,
            "seed": self.config.seed,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.epoch_logs),
            "config": asdict(self.config),
            "epochs": [asdict(e) for e in self.epoch_logs],
        }
        with open(path, "w") as f:
            json.dump(log_data, f, indent=2)
