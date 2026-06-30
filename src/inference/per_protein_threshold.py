"""Per-protein threshold optimization (Part 5 Phase 3 EXP-A03).

Reference: P5_05 §P5-Ph3-S05; CLAUDE.md rule 33 (train threshold MLP on VAL,
not train, predictions).

Mechanism:
  - For each val protein: sweep thresholds in [0.05, 0.50] step 0.005,
    compute F1, find optimal threshold per protein.
  - Train MLP: 5 per-protein features → 32 → 1 → sigmoid → scale [0.05, 0.50]
    to predict the optimal threshold.
  - Apply learned MLP at inference time on benchmark proteins.

Per-protein features (5):
  1. mean prediction
  2. std prediction
  3. max prediction
  4. fraction of predictions > 0.5
  5. protein length (log-normalized)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.50
SWEEP_STEP = 0.005


def per_protein_features(probs: np.ndarray, length: int | None = None) -> np.ndarray:
    """5 features for the threshold MLP.

    Args:
        probs: [L] per-residue disorder probability for one protein.
        length: protein length (defaults to len(probs)).

    Returns:
        [5] float feature vector.
    """
    if len(probs) == 0:
        return np.zeros(5, dtype=np.float32)
    L = length if length is not None else len(probs)
    return np.array([
        float(probs.mean()),
        float(probs.std() + 1e-8),
        float(probs.max()),
        float((probs > 0.5).mean()),
        float(np.log1p(L) / np.log(10000.0)),  # length scaled to ~[0, 1]
    ], dtype=np.float32)


def f1_at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float) -> float:
    """Binary F1 at threshold thr. Excludes labels < 0."""
    mask = labels >= 0
    if mask.sum() == 0:
        return 0.0
    p = (probs[mask] >= thr).astype(np.int64)
    y = labels[mask].astype(np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return 2 * prec * rec / (prec + rec + 1e-12)


def best_threshold_for_protein(
    probs: np.ndarray, labels: np.ndarray,
    thr_min: float = THRESHOLD_MIN, thr_max: float = THRESHOLD_MAX,
    step: float = SWEEP_STEP,
) -> tuple[float, float]:
    """Sweep thresholds, return (best_threshold, best_F1)."""
    valid = labels >= 0
    if valid.sum() == 0 or labels[valid].sum() == 0:
        return float((thr_min + thr_max) / 2), 0.0
    best_thr, best_f1 = thr_min, -1.0
    thr = thr_min
    while thr <= thr_max + 1e-9:
        f1 = f1_at_threshold(probs, labels, thr)
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr
        thr += step
    return best_thr, best_f1


def best_global_threshold(
    all_probs: np.ndarray, all_labels: np.ndarray,
    thr_min: float = THRESHOLD_MIN, thr_max: float = THRESHOLD_MAX,
    step: float = SWEEP_STEP,
) -> tuple[float, float]:
    """One global threshold across all residues (for baseline)."""
    return best_threshold_for_protein(all_probs, all_labels, thr_min, thr_max, step)


class ThresholdMLP(nn.Module):
    """5 features → 32 → 1, output scaled to [thr_min, thr_max]."""

    def __init__(
        self,
        n_features: int = 5,
        hidden_dim: int = 32,
        thr_min: float = THRESHOLD_MIN,
        thr_max: float = THRESHOLD_MAX,
    ):
        super().__init__()
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.net(x))                        # [N, 1]
        return self.thr_min + (self.thr_max - self.thr_min) * s.squeeze(-1)


def train_threshold_mlp(
    val_predictions: dict[str, np.ndarray],   # acc → [L] probs
    val_labels: dict[str, np.ndarray],         # acc → [L] labels
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> tuple[ThresholdMLP, dict]:
    """Train the threshold MLP. Returns (model, diagnostic_dict)."""
    torch.manual_seed(seed); np.random.seed(seed)

    accs = sorted(set(val_predictions.keys()) & set(val_labels.keys()))
    feats, targets, opt_f1s = [], [], []
    for acc in accs:
        p = val_predictions[acc]; lab = val_labels[acc]
        L = min(len(p), len(lab))
        thr, f1 = best_threshold_for_protein(p[:L], lab[:L])
        feats.append(per_protein_features(p[:L], L))
        targets.append(thr)
        opt_f1s.append(f1)
    if not feats:
        return ThresholdMLP(), {"error": "no proteins"}
    X = torch.tensor(np.array(feats), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.float32)

    # 80/20 split within val for early stopping
    perm = np.random.permutation(len(X))
    n_train = int(0.8 * len(X))
    tr_idx = perm[:n_train]; va_idx = perm[n_train:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    model = ThresholdMLP()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_va = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        pred = model(Xtr)
        loss = F.smooth_l1_loss(pred, ytr)
        optim.zero_grad(); loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            va_pred = model(Xva)
            va_loss = F.smooth_l1_loss(va_pred, yva).item()
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    diag = {
        "n_proteins": len(accs),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "mean_optimal_threshold": float(np.mean(targets)),
        "std_optimal_threshold": float(np.std(targets)),
        "mean_optimal_f1": float(np.mean(opt_f1s)),
        "best_val_loss": float(best_va),
    }
    return model, diag


def apply_per_protein_threshold(
    model: ThresholdMLP,
    probs_dict: dict[str, np.ndarray],
) -> dict[str, float]:
    """Predict per-protein threshold for each acc → returns {acc: threshold}."""
    feats = []
    accs = sorted(probs_dict.keys())
    for acc in accs:
        p = probs_dict[acc]
        feats.append(per_protein_features(p, len(p)))
    X = torch.tensor(np.array(feats), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        thr = model(X).cpu().numpy()
    return {a: float(t) for a, t in zip(accs, thr)}


def maxf1_with_thresholds(
    probs_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
    threshold_per_acc: dict[str, float] | float,
) -> dict:
    """Apply per-acc (or global) threshold, compute pooled F1.

    Returns {macro_f1, micro_f1, n_proteins, thresholds: array[N]}.
    """
    accs = sorted(set(probs_dict.keys()) & set(labels_dict.keys()))
    all_p, all_y, thrs = [], [], []
    for acc in accs:
        p = probs_dict[acc]; y = labels_dict[acc]
        L = min(len(p), len(y))
        thr = (threshold_per_acc[acc] if isinstance(threshold_per_acc, dict)
               else float(threshold_per_acc))
        valid = y[:L] >= 0
        if valid.sum() == 0:
            continue
        all_p.append((p[:L][valid] >= thr).astype(np.int64))
        all_y.append(y[:L][valid].astype(np.int64))
        thrs.append(thr)
    if not all_p:
        return {"micro_f1": 0.0, "n_proteins": 0, "thresholds": np.array([])}
    P = np.concatenate(all_p); Y = np.concatenate(all_y)
    tp = int(((P == 1) & (Y == 1)).sum())
    fp = int(((P == 1) & (Y == 0)).sum())
    fn = int(((P == 0) & (Y == 1)).sum())
    prec = tp / (tp + fp + 1e-12); rec = tp / (tp + fn + 1e-12)
    micro_f1 = 2 * prec * rec / (prec + rec + 1e-12) if tp > 0 else 0.0
    return {
        "micro_f1": float(micro_f1),
        "n_proteins": len(thrs),
        "thresholds": np.array(thrs, dtype=np.float32),
    }
