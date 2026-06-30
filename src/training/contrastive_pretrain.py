"""Contrastive disorder pretraining (Part 5 Phase 2 EXP-T01).

Reference: P5_04 §P5-Ph2-S05; CLAUDE.md rule 29 (contrastive operates on
representations, not labels).

Mechanism:
  - For each protein, sample (anchor, positive, hard_negative) triplets where:
      anchor: random residue
      positive: another residue with SAME disorder label, SAME protein
      hard negative: residue with OPPOSITE label, SAME protein
      easy negatives: residues from OTHER proteins in the batch (any label)
  - Project backbone features through Linear(hidden_dim -> 128) and L2-normalize.
  - InfoNCE loss with temperature tau=0.07.

Pretraining only operates on contributing residues (DISORDERED=1, ORDERED=0).
After pretraining: discard the projection head, finetune backbone for
classification with discriminative LR (pretrained at LR/10, new head at LR).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Label constants (must match src/data/labeling.py)
DISORDERED = 1
ORDERED = 0


class ProjectionHead(nn.Module):
    """Linear projection -> L2-normalized 128-dim vector."""

    def __init__(self, hidden_dim: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return F.normalize(z, p=2, dim=-1)


def sample_contrastive_triplets(
    labels: torch.Tensor,            # [B, L] integer labels
    n_anchors_per_protein: int = 32,
    rng: torch.Generator | None = None,
) -> dict:
    """For each protein in batch, sample anchor residues with positive +
    hard-negative partners from the same protein.

    Returns dict with:
        anchor_idx:   [N_total, 2] (batch_idx, residue_idx)
        pos_idx:      [N_total, 2]
        hardneg_idx:  [N_total, 2]
        valid:        [N_total] bool — False if protein lacks both labels
    """
    B, L = labels.shape
    device = labels.device

    anchors_b, anchors_r = [], []
    positives_b, positives_r = [], []
    hardneg_b, hardneg_r = [], []

    for b in range(B):
        lab = labels[b]
        dis_idx = (lab == DISORDERED).nonzero(as_tuple=True)[0]
        ord_idx = (lab == ORDERED).nonzero(as_tuple=True)[0]
        if len(dis_idx) < 2 or len(ord_idx) < 2:
            continue  # need >=2 of each so positive & hardneg exist

        for _ in range(n_anchors_per_protein):
            # 50/50 anchor-class split
            if torch.rand(1, generator=rng, device="cpu").item() < 0.5:
                anchor_pool = dis_idx
                pos_pool = dis_idx
                neg_pool = ord_idx
            else:
                anchor_pool = ord_idx
                pos_pool = ord_idx
                neg_pool = dis_idx

            # Sample anchor
            a_i = anchor_pool[torch.randint(len(anchor_pool), (1,), generator=rng).item()]
            # Sample positive (different residue, same class)
            p_choices = pos_pool[pos_pool != a_i]
            if len(p_choices) == 0:
                continue
            p_i = p_choices[torch.randint(len(p_choices), (1,), generator=rng).item()]
            n_i = neg_pool[torch.randint(len(neg_pool), (1,), generator=rng).item()]

            anchors_b.append(b); anchors_r.append(a_i.item())
            positives_b.append(b); positives_r.append(p_i.item())
            hardneg_b.append(b); hardneg_r.append(n_i.item())

    return {
        "anchor_b": torch.tensor(anchors_b, dtype=torch.long, device=device),
        "anchor_r": torch.tensor(anchors_r, dtype=torch.long, device=device),
        "pos_b": torch.tensor(positives_b, dtype=torch.long, device=device),
        "pos_r": torch.tensor(positives_r, dtype=torch.long, device=device),
        "hardneg_b": torch.tensor(hardneg_b, dtype=torch.long, device=device),
        "hardneg_r": torch.tensor(hardneg_r, dtype=torch.long, device=device),
    }


def info_nce_loss(
    anchor_proj: torch.Tensor,    # [N, D]  L2-normalized
    pos_proj: torch.Tensor,       # [N, D]
    hardneg_proj: torch.Tensor,   # [N, D]
    tau: float = 0.07,
    extra_neg_proj: torch.Tensor | None = None,  # [M, D] easy negatives (other proteins' residues)
) -> torch.Tensor:
    """InfoNCE: maximize anchor-positive similarity vs (hardneg + easy negs).

    For each anchor i, the candidate set is:
        positive_i  (1)
        hardneg_i   (1)
        all extra_neg residues (M shared across the batch)
        all OTHER anchors' positives + hardnegs (in-batch negatives)

    Loss = -log(exp(s_pos/tau) / sum(exp(s_*/tau)))
    """
    N = anchor_proj.size(0)
    device = anchor_proj.device

    # Build candidate matrix C  [N, K]
    # K = 1 (pos) + 1 (hardneg) + M (extra) + 2*(N-1) (in-batch others) — but
    # for simplicity, in-batch negatives use ALL other anchors' pos+hardneg,
    # then we mask self-positive at row i. So candidate pool = full
    # pos + hardneg pool (size 2N) plus optional extras.
    pool = [pos_proj, hardneg_proj]  # [N, D] each
    if extra_neg_proj is not None and extra_neg_proj.numel() > 0:
        pool.append(extra_neg_proj)
    cand = torch.cat(pool, dim=0)  # [2N + M, D]

    # Logits = anchor @ cand.T / tau
    logits = (anchor_proj @ cand.T) / tau  # [N, 2N+M]

    # Positive index for row i is i (pos_proj rows are first N rows)
    target = torch.arange(N, device=device)

    return F.cross_entropy(logits, target)


def contrastive_step(
    backbone: nn.Module,
    proj_head: ProjectionHead,
    features: torch.Tensor,    # [B, L, F] input embeddings (input view to backbone)
    labels: torch.Tensor,       # [B, L]
    tau: float = 0.07,
    n_anchors_per_protein: int = 32,
    n_easy_negs_per_protein: int = 8,
) -> tuple[torch.Tensor, dict]:
    """One contrastive forward+loss step.

    The backbone is expected to return a dict with key "penultimate" (or its
    forward_features) giving [B, L, hidden_dim] residue features. If the
    backbone returns logits only, we use the underlying penultimate via
    backbone.forward_features (Part 4 hybrid exposes this).
    """
    feats = backbone.forward_features(features)  # [B, L, hidden_dim]

    triplets = sample_contrastive_triplets(labels, n_anchors_per_protein=n_anchors_per_protein)
    if triplets["anchor_b"].numel() == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True), {"n_triplets": 0}

    a_proj = proj_head(feats[triplets["anchor_b"], triplets["anchor_r"]])
    p_proj = proj_head(feats[triplets["pos_b"], triplets["pos_r"]])
    n_proj = proj_head(feats[triplets["hardneg_b"], triplets["hardneg_r"]])

    # Easy negatives: random residues from random proteins (any label except masked)
    extra = None
    B, L = labels.shape
    if n_easy_negs_per_protein > 0 and B > 1:
        valid_mask = (labels == DISORDERED) | (labels == ORDERED)  # [B, L]
        valid_idx = valid_mask.nonzero(as_tuple=False)  # [V, 2]
        if valid_idx.size(0) > 0:
            n_take = min(n_easy_negs_per_protein * B, valid_idx.size(0))
            sel = valid_idx[torch.randperm(valid_idx.size(0))[:n_take]]
            extra = proj_head(feats[sel[:, 0], sel[:, 1]])

    loss = info_nce_loss(a_proj, p_proj, n_proj, tau=tau, extra_neg_proj=extra)
    diag = {
        "n_triplets": int(triplets["anchor_b"].numel()),
        "n_extra_negs": 0 if extra is None else int(extra.size(0)),
    }
    return loss, diag


def discriminative_param_groups(
    backbone: nn.Module,
    head: nn.Module,
    base_lr: float,
    backbone_lr_factor: float = 0.1,
    weight_decay: float = 1e-2,
) -> list[dict]:
    """Build optimizer param groups for finetuning a contrastively-pretrained
    backbone: backbone params at base_lr * backbone_lr_factor, head at base_lr.

    Same protocol as Part 4 TriZOD Stage 2 finetune.
    """
    return [
        {"params": list(backbone.parameters()), "lr": base_lr * backbone_lr_factor,
         "weight_decay": weight_decay, "name": "backbone_pretrained"},
        {"params": list(head.parameters()), "lr": base_lr,
         "weight_decay": weight_decay, "name": "head_new"},
    ]
