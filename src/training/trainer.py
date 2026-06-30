"""Training pipeline infrastructure.

Reference: Training Contract §9–§11, §14; P2_00 Amendments I-3, T-2, T-3.

Implements:
  - AdamW optimizer with configurable scheduler
    * mode="frozen" (Part 1): single param group, head LR only
    * mode="lora"   (Part 2): dual-LR param groups per Amendment T-2
        [{"params": head_params, "lr": head_lr},
         {"params": lora_params, "lr": lora_lr}]
  - Length-sorted dynamic batching with residue budget
  - Gradient clipping
  - Early stopping with patience and best-checkpoint restoration
  - Seeded reproducibility
  - Per-epoch logging
  - Checkpoint save/load
    * Frozen mode: save full model state_dict (Part 1 behaviour)
    * LoRA mode:   save only trainable_state_dict (LoRA adapters + head);
                   the 2.5GB frozen pLM base is NOT persisted.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.training.loss import masked_bce_loss


# ─── Configuration ──────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Full training configuration (all hyperparameters configurable)."""
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"  # "cosine", "plateau", "none"
    warmup_epochs: int = 2

    # Loss
    pos_weight: float | None = None

    # Batching
    residue_budget: int = 12000
    min_batch_size: int = 1

    # Training control
    max_epochs: int = 200
    patience: int = 15
    min_epochs: int = 10

    # Gradient
    max_grad_norm: float = 1.0

    # Reproducibility
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    experiment_id: str = "default"

    # Part 2 — LoRA training mode (Amendments I-3, T-2, T-3)
    mode: str = "frozen"  # "frozen" (Part 1) or "lora" (Part 2)
    # Dual-LR params — applied only in mode="lora" per Amendment T-2.
    # Head LR uses `lr` above. LoRA adapter LR is lower by convention.
    lora_lr: float = 5e-5           # 1e-5 to 1e-4 per Amendment T-2
    lora_weight_decay: float = 1e-2

    # Part 4 — Finetune-from-pretrain mode (Amendment T-1)
    # mode="finetune" loads a TriZOD-pretrained backbone and trains BCE
    # classification with discriminative LR (backbone at lr * pretrained_lr_factor,
    # newly-attached head at lr).
    pretrained_lr_factor: float = 0.1   # backbone LR multiplier in finetune mode
    freeze_early_cnn_blocks: int = 0    # number of CNN blocks frozen during warmup
    freeze_warmup_epochs: int = 3       # epochs to keep frozen blocks frozen

    def __post_init__(self):
        if self.mode not in ("frozen", "lora", "finetune"):
            raise ValueError(
                f"TrainingConfig.mode must be 'frozen', 'lora', or 'finetune', "
                f"got {self.mode!r}"
            )


# ─── Batching ───────────────────────────────────────────────────────

def create_length_sorted_batches(
    accessions: list[str],
    lengths: dict[str, int],
    residue_budget: int,
    min_batch_size: int = 1,
) -> list[list[str]]:
    """Create length-sorted batches with dynamic sizing.

    Groups proteins of similar length to minimize padding.
    Batch size determined by residue budget, not fixed protein count.

    Args:
        accessions: List of protein accessions.
        lengths: Dict mapping accession → sequence length.
        residue_budget: Max total residues per batch.
        min_batch_size: Minimum proteins per batch.

    Returns:
        List of batches, each a list of accessions.
    """
    # Sort by length
    sorted_accs = sorted(accessions, key=lambda a: lengths.get(a, 0))

    batches = []
    current_batch = []
    current_residues = 0
    current_max_len = 0

    for acc in sorted_accs:
        seq_len = lengths.get(acc, 0)
        new_max_len = max(current_max_len, seq_len)
        new_batch_size = len(current_batch) + 1
        # Total padded residues = batch_size × max_length
        new_total = new_batch_size * new_max_len

        if current_batch and new_total > residue_budget:
            batches.append(current_batch)
            current_batch = [acc]
            current_residues = seq_len
            current_max_len = seq_len
        else:
            current_batch.append(acc)
            current_residues = new_total
            current_max_len = new_max_len

    if current_batch:
        batches.append(current_batch)

    return batches


# ─── Checkpoint Management ──────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict,
    train_loss: float,
    config: dict | None = None,
    mode: str = "frozen",
):
    """Save training checkpoint.

    In mode="frozen" (Part 1 behaviour): persists full model.state_dict().
    In mode="lora":  persists model.trainable_state_dict() ONLY (adapter + head).
                     Saves ~10-50MB instead of ~2.5GB.
    """
    if mode == "lora":
        if not hasattr(model, "trainable_state_dict"):
            raise AttributeError(
                "mode='lora' requires model to expose trainable_state_dict() "
                "(see src.models.lora_adapter.LoRAExpertModel)"
            )
        model_state = model.trainable_state_dict()
    else:
        model_state = model.state_dict()

    torch.save({
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "train_loss": train_loss,
        "config": config,
        "mode": mode,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, device="cpu"):
    """Load training checkpoint.

    Mode is read from the checkpoint itself (falls back to "frozen" for
    Part 1 checkpoints that pre-date the mode field).

    Returns:
        Dict with epoch, val_metrics, train_loss, mode.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mode = ckpt.get("mode", "frozen")
    state = ckpt["model_state_dict"]

    if mode == "lora":
        if not hasattr(model, "load_trainable_state_dict"):
            raise AttributeError(
                "checkpoint mode='lora' requires model.load_trainable_state_dict"
            )
        model.load_trainable_state_dict(state, strict=True)
    else:
        model.load_state_dict(state)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "epoch": ckpt.get("epoch", 0),
        "val_metrics": ckpt.get("val_metrics", {}),
        "train_loss": ckpt.get("train_loss", float("inf")),
        "mode": mode,
    }


# ─── Finetune-from-pretrain helpers (Part 4 Amendment T-1) ─────────


def build_finetune_optimizer(
    model: nn.Module,
    base_lr: float = 1e-3,
    pretrained_lr_factor: float = 0.1,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> AdamW:
    """Build a discriminative-LR AdamW for the pretrain → finetune transition.

    Pretrained backbone params (everything matching `backbone_parameters()`)
    are placed at `base_lr * pretrained_lr_factor`; the freshly-attached
    classification head (matching `head_parameters()`) is at `base_lr`.

    The model must expose `backbone_parameters()` and `head_parameters()`
    iterators (CNNTransformerHybrid does). Each model parameter must end up in
    exactly one group — verified by checking ID-set equality with model.parameters().
    """
    if not (hasattr(model, "backbone_parameters") and hasattr(model, "head_parameters")):
        raise AttributeError(
            "build_finetune_optimizer requires model.backbone_parameters() and "
            "model.head_parameters() (e.g. CNNTransformerHybrid)"
        )
    backbone = list(model.backbone_parameters())
    head = list(model.head_parameters())
    if not backbone:
        raise ValueError("no backbone parameters discovered")
    if not head:
        raise ValueError("no head parameters discovered (replace_head_for_classification not called?)")
    return AdamW(
        [
            {"params": head,     "lr": base_lr,                          "weight_decay": weight_decay},
            {"params": backbone, "lr": base_lr * pretrained_lr_factor,   "weight_decay": weight_decay},
        ],
        betas=betas,
        eps=eps,
    )


def freeze_early_cnn_blocks(model: nn.Module, n_blocks: int) -> None:
    """Freeze the first `n_blocks` CNN blocks during finetune warmup.
    Looks for `cnn_blocks` (CNNTransformerHybrid) or `blocks`
    (DilatedResidualCNN). Leaves projection and downstream layers trainable.
    Idempotent — calling twice is safe."""
    block_list = getattr(model, "cnn_blocks", None) or getattr(model, "blocks", None)
    if block_list is None:
        raise AttributeError(
            "freeze_early_cnn_blocks requires model.cnn_blocks (CNNTransformerHybrid) "
            "or model.blocks (DilatedResidualCNN)"
        )
    n = min(n_blocks, len(block_list))
    for i in range(n):
        for p in block_list[i].parameters():
            p.requires_grad_(False)


def unfreeze_all(model: nn.Module) -> None:
    """Re-enable gradients on every parameter (post-warmup)."""
    for p in model.parameters():
        p.requires_grad_(True)


# ─── Training Loop ─────────────────────────────────────────────────

@dataclass
class EpochLog:
    """Per-epoch training log entry."""
    epoch: int
    train_loss: float
    val_loss: float | None = None
    val_p1: float | None = None
    val_p2: float | None = None
    supplementary_residue_pooled_ap: float | None = None  # diagnostic only, never for decisions
    lr: float = 0.0
    grad_norm_mean: float = 0.0
    grad_norm_max: float = 0.0
    wall_time_s: float = 0.0


class Trainer:
    """Training loop orchestrator.

    Manages the full training lifecycle: optimizer setup, epoch loop,
    validation, early stopping, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Seed
        self._set_seed(config.seed)

        # Optimizer — frozen mode uses a single param group (Part 1); LoRA
        # mode uses dual groups with lower LR on adapters per Amendment T-2.
        self.optimizer = self._build_optimizer(model, config)

        # Scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.max_epochs,
            )
        elif config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5,
            )
        else:
            self.scheduler = None

        # Early stopping state
        self.best_p1 = -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        # Logging
        self.epoch_logs: list[EpochLog] = []

        # Checkpoint dir
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self, model: nn.Module, config: TrainingConfig) -> AdamW:
        """Construct AdamW with the right param groups for this mode.

        frozen: one group, all trainable params at config.lr.
        lora:   two groups — head params at config.lr, LoRA adapter params at
                config.lora_lr per Amendment T-2.
        """
        if config.mode == "frozen":
            return AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )

        if config.mode == "finetune":
            # Discriminative LR for TriZOD pretrain → binary finetune
            # (Amendment T-1). Backbone at lr*pretrained_lr_factor, head at lr.
            return build_finetune_optimizer(
                model,
                base_lr=config.lr,
                pretrained_lr_factor=config.pretrained_lr_factor,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )

        # LoRA mode — deferred import to avoid circular dependency when the
        # model isn't a LoRA composite (e.g. unit tests that still use
        # mode="frozen").
        from src.models.lora_adapter import get_head_params, get_lora_params

        lora_params = get_lora_params(model)
        head_module = getattr(model, "head", None)
        if head_module is None:
            raise AttributeError(
                "mode='lora' requires model.head attribute (LoRAExpertModel)"
            )
        head_params = get_head_params(head_module)

        if not lora_params:
            raise ValueError("mode='lora' but no LoRA params discovered in model")
        if not head_params:
            raise ValueError("mode='lora' but no trainable head params discovered")

        return AdamW(
            [
                {
                    "params": head_params,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": lora_params,
                    "lr": config.lora_lr,
                    "weight_decay": config.lora_weight_decay,
                },
            ],
            betas=config.betas,
            eps=config.eps,
        )

    def train_epoch(
        self,
        batches: list[dict],
    ) -> float:
        """Run one training epoch.

        Args:
            batches: List of batch dicts with keys:
                'features': [B, L, D] tensor
                'labels': [B, L] tensor
                'lengths': [B] tensor (optional)

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_residues = 0

        for batch in batches:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch.get("lengths")

            self.optimizer.zero_grad()

            # Forward
            if hasattr(self.model, "forward") and lengths is not None:
                try:
                    logits = self.model(features, lengths=lengths)
                except TypeError:
                    logits = self.model(features)
            else:
                logits = self.model(features)

            # Loss
            loss = masked_bce_loss(logits, labels, pos_weight=self.config.pos_weight)

            # Backward
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm,
            )

            self.optimizer.step()

            # Accumulate
            from src.training.loss import build_loss_mask
            n_contributing = build_loss_mask(labels).sum().item()
            total_loss += loss.item() * n_contributing
            total_residues += n_contributing

        return total_loss / max(total_residues, 1)

    @torch.no_grad()
    def validate(
        self,
        batches: list[dict],
    ) -> dict:
        """Run validation and compute metrics.

        Returns:
            Dict with keys: loss, p1, p2 (p1/p2 may be None if
            evaluation orchestrator is not wired).
        """
        self.model.eval()
        total_loss = 0.0
        total_residues = 0
        all_probs = []
        all_labels = []

        for batch in batches:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch.get("lengths")

            if lengths is not None:
                try:
                    logits = self.model(features, lengths=lengths)
                except TypeError:
                    logits = self.model(features)
            else:
                logits = self.model(features)

            loss = masked_bce_loss(logits, labels, pos_weight=self.config.pos_weight)

            from src.training.loss import build_loss_mask
            n_contributing = build_loss_mask(labels).sum().item()
            total_loss += loss.item() * n_contributing
            total_residues += n_contributing

            # Collect predictions for P1/P2 computation
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        val_loss = total_loss / max(total_residues, 1)

        # Compute P1 (macro-averaged AUC-PR) and P2 (macro-averaged AUC-ROC)
        p1, p2 = None, None
        from src.evaluation.metrics import auc_pr, auc_roc
        from sklearn.metrics import average_precision_score

        per_protein_p1 = []
        per_protein_p2 = []
        # Supplementary residue-pooled AP (Benchmark Contract §9.5).
        # Diagnostic only — NEVER used for decisions, model selection,
        # ablation judgments, early stopping, or gate criteria.
        all_pooled_labels = []
        all_pooled_probs = []

        for probs_batch, labels_batch in zip(all_probs, all_labels):
            for i in range(probs_batch.shape[0]):
                p = probs_batch[i].float().numpy()
                l = labels_batch[i].numpy()
                # Filter to contributing residues
                mask = (l == 1) | (l == 0)
                if mask.sum() == 0:
                    continue
                p_masked = p[mask].astype(np.float64)
                l_masked = l[mask].astype(np.int32)
                # Accumulate for residue-pooled AP
                all_pooled_labels.append(l_masked)
                all_pooled_probs.append(p_masked)
                if len(np.unique(l_masked)) < 2:
                    continue  # single-class exclusion for macro metrics
                if np.any(np.isnan(p_masked)):
                    continue
                per_protein_p1.append(auc_pr(l_masked, p_masked))
                per_protein_p2.append(auc_roc(l_masked, p_masked))
        if per_protein_p1:
            p1 = float(np.mean(per_protein_p1))
            p2 = float(np.mean(per_protein_p2))

        # Compute supplementary residue-pooled AP (diagnostic only)
        supplementary_residue_pooled_ap = None
        if all_pooled_labels:
            pooled_l = np.concatenate(all_pooled_labels)
            pooled_p = np.concatenate(all_pooled_probs)
            if len(np.unique(pooled_l)) >= 2 and not np.any(np.isnan(pooled_p)):
                supplementary_residue_pooled_ap = float(np.clip(
                    average_precision_score(pooled_l, pooled_p), 0.0, 1.0,
                ))

        return {
            "loss": val_loss,
            "p1": p1,
            "p2": p2,
            "supplementary_residue_pooled_ap": supplementary_residue_pooled_ap,
        }

    def fit(
        self,
        train_batches: list[dict],
        val_batches: list[dict],
        shuffle_batches: bool = True,
    ) -> dict:
        """Run full training loop with early stopping.

        Args:
            train_batches: Training batch list.
            val_batches: Validation batch list.
            shuffle_batches: Whether to shuffle batch order each epoch.

        Returns:
            Dict with best_p1, best_epoch, total_epochs, early_stopped, and
            (in finetune mode) baseline_p1_epoch0 — val P1 measured BEFORE any
            finetune gradient update, used as the catastrophic-forgetting
            monitor per P4_03 §S02.
        """
        baseline_p1_epoch0 = None
        if self.config.mode == "finetune":
            # Catastrophic-forgetting monitor: val P1 at epoch 0 (before any
            # gradient step). If this is well below the Part 3 baseline (~0.91),
            # the pretrained features may not transfer — investigate before
            # spending GPU on a full finetune run.
            with torch.no_grad():
                baseline_p1_epoch0 = self.validate(val_batches).get("p1")

            # Apply warmup freeze on early CNN blocks (if configured)
            if self.config.freeze_early_cnn_blocks > 0:
                freeze_early_cnn_blocks(self.model, self.config.freeze_early_cnn_blocks)
                # Rebuild optimizer over currently trainable params
                self.optimizer = self._build_optimizer(self.model, self.config)

        for epoch in range(1, self.config.max_epochs + 1):
            # End-of-warmup unfreeze
            if (
                self.config.mode == "finetune"
                and self.config.freeze_early_cnn_blocks > 0
                and epoch == self.config.freeze_warmup_epochs + 1
            ):
                unfreeze_all(self.model)
                self.optimizer = self._build_optimizer(self.model, self.config)
            epoch_start = time.time()

            # Shuffle batch order (not within batches)
            if shuffle_batches:
                rng = np.random.RandomState(self.config.seed + epoch)
                indices = rng.permutation(len(train_batches))
                epoch_batches = [train_batches[i] for i in indices]
            else:
                epoch_batches = train_batches

            # Train
            train_loss = self.train_epoch(epoch_batches)

            # Validate
            val_result = self.validate(val_batches)

            # Scheduler step
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if val_result["p1"] is not None:
                        self.scheduler.step(val_result["p1"])
                else:
                    self.scheduler.step()

            # Log
            wall_time = time.time() - epoch_start
            log = EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_result["loss"],
                val_p1=val_result["p1"],
                val_p2=val_result["p2"],
                lr=lr,
                wall_time_s=wall_time,
            )
            self.epoch_logs.append(log)

            # Early stopping check
            if val_result["p1"] is not None and val_result["p1"] > self.best_p1:
                self.best_p1 = val_result["p1"]
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best checkpoint
                best_path = self.ckpt_dir / (
                    f"{self.config.experiment_id}_seed{self.config.seed}_best.pt"
                )
                save_checkpoint(
                    best_path, self.model, self.optimizer, epoch,
                    val_result, train_loss,
                    mode=self.config.mode,
                )
            else:
                self.patience_counter += 1

            # Check early stopping
            if (
                epoch >= self.config.min_epochs
                and self.patience_counter >= self.config.patience
            ):
                # Restore best checkpoint
                best_path = self.ckpt_dir / (
                    f"{self.config.experiment_id}_seed{self.config.seed}_best.pt"
                )
                if best_path.exists():
                    load_checkpoint(best_path, self.model, device=self.device)

                return {
                    "best_p1": self.best_p1,
                    "best_epoch": self.best_epoch,
                    "total_epochs": epoch,
                    "early_stopped": True,
                    "baseline_p1_epoch0": baseline_p1_epoch0,
                }

        # Max epochs reached — restore best
        best_path = self.ckpt_dir / (
            f"{self.config.experiment_id}_seed{self.config.seed}_best.pt"
        )
        if best_path.exists():
            load_checkpoint(best_path, self.model, device=self.device)

        return {
            "best_p1": self.best_p1,
            "best_epoch": self.best_epoch,
            "total_epochs": self.config.max_epochs,
            "early_stopped": False,
            "baseline_p1_epoch0": baseline_p1_epoch0,
        }

    def save_run_log(self, path: Path):
        """Save structured run log as JSON."""
        log_data = {
            "experiment_id": self.config.experiment_id,
            "seed": self.config.seed,
            "best_p1": self.best_p1,
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.epoch_logs),
            "config": asdict(self.config),
            "epochs": [asdict(e) for e in self.epoch_logs],
        }
        with open(path, "w") as f:
            json.dump(log_data, f, indent=2)
