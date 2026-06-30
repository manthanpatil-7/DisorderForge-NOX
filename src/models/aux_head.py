"""Evidence-type auxiliary head (P3-Ph2-S03).

Reference: P3_00 Amendment M-3; P3_04 §S03.

Branches from an expert backbone's penultimate-layer features and predicts
one of 5 evidence-type classes (X-ray, NMR, CD, SAXS, Other) per residue.
Used as a multi-task auxiliary signal during training; its output is
diagnostic at inference and NOT used in final disorder predictions.

Loss: cross-entropy on residues whose protein has DisProt-annotated evidence
(label index in {0..4}). Residues with index < 0 (PDB-only / UNKNOWN proteins
or non-annotated regions) contribute zero gradient — same masking semantics
as the BCE/CRF disorder loss.

Combined training loss:
    total = disorder_loss + lambda_aux * evidence_loss

Default lambda_aux = 0.2 (P3_04 §S03).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EVIDENCE_CLASSES = ["X-ray", "NMR", "CD", "SAXS", "Other"]
NUM_EVIDENCE_CLASSES = len(EVIDENCE_CLASSES)


class EvidenceAuxHead(nn.Module):
    """Branches from backbone features → 5-class softmax per residue."""

    def __init__(self, hidden_dim: int, num_classes: int = NUM_EVIDENCE_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, L, hidden_dim] → logits [B, L, num_classes]."""
        return self.linear(self.drop(hidden))


def evidence_aux_loss(
    aux_logits: torch.Tensor,    # [B, L, C]
    evidence_labels: torch.Tensor,  # [B, L] long; -1 marks "no label, exclude"
    seq_mask: torch.Tensor | None = None,  # [B, L] bool
) -> torch.Tensor:
    """Cross-entropy over residues with valid evidence labels."""
    B, L, C = aux_logits.shape
    if seq_mask is None:
        seq_mask = torch.ones((B, L), dtype=torch.bool, device=aux_logits.device)
    valid = (evidence_labels >= 0) & seq_mask.bool()
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        # Return a finite zero that participates in autograd (no grad)
        return torch.tensor(0.0, device=aux_logits.device, requires_grad=True)
    flat_logits = aux_logits[valid]              # [N, C]
    flat_labels = evidence_labels[valid].long()  # [N]
    return F.cross_entropy(flat_logits, flat_labels)


def evidence_aux_accuracy(
    aux_logits: torch.Tensor,
    evidence_labels: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    """Top-1 accuracy on residues with valid evidence labels.

    Returns (accuracy, n_valid). accuracy=0.0 when n_valid=0.
    """
    if seq_mask is None:
        seq_mask = torch.ones(aux_logits.shape[:2], dtype=torch.bool, device=aux_logits.device)
    valid = (evidence_labels >= 0) & seq_mask.bool()
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return 0.0, 0
    preds = aux_logits[valid].argmax(dim=-1)
    correct = (preds == evidence_labels[valid].long()).sum().item()
    return correct / n_valid, n_valid


# ─── Wrapper that adds aux + main heads to any backbone with forward_features ──


class MultiTaskWrapper(nn.Module):
    """Wraps a backbone exposing `forward_features` to add disorder + aux heads.

    The disorder head is a fresh `Linear(hidden_dim, 1)` — we DON'T reuse the
    backbone's existing output_head because backbones may apply additional
    dropout or normalization between features and the original head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        aux_num_classes: int = NUM_EVIDENCE_CLASSES,
        aux_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.disorder_head = nn.Linear(hidden_dim, 1)
        self.aux_head = EvidenceAuxHead(hidden_dim, aux_num_classes, dropout=aux_dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, **backbone_kwargs):
        feats = self.backbone.forward_features(x, **backbone_kwargs)  # [B, L, hidden]
        disorder_logits = self.disorder_head(feats)                   # [B, L, 1]
        aux_logits = self.aux_head(feats)                             # [B, L, C]
        return disorder_logits, aux_logits, feats
