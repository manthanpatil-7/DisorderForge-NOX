"""CNN-Transformer Hybrid expert head (P4-Ph1-S01).

Reference: P4_00 Amendment M-1; P4_03 §S01.

Sequential architecture: ESM-2 (+ optional physicochemical features) →
input projection → 4 dilated residual CNN blocks (local motif extraction) →
2 rotary-PE Transformer encoder layers (long-range attention) → output linear.

Each residue's CNN feature vector is fed directly as a Transformer token —
no pooling, no reshaping. The CNN reduces working dim and captures local
disorder motifs; the Transformer attends across those motifs for context.

Reuses:
  - DilatedResBlock from src/models/family_a_cnn.py
  - RotaryTransformerLayer from src/models/transformer_head.py

Defaults (per P4_03 §S01):
  cnn_blocks       = 4
  cnn_dilations    = [1, 2, 4, 8]
  kernel_size      = 7
  working_dim      = 256          (also Transformer hidden_dim)
  transformer_layers = 2
  num_heads        = 4
  ff_dim           = 512
  dropout          = 0.2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.family_a_cnn import DilatedResBlock
from src.models.transformer_head import RotaryTransformerLayer


class CNNTransformerHybrid(nn.Module):
    """Sequential CNN → Transformer expert head.

    Args:
        input_dim: per-residue input feature dim (e.g. 1280 for ESM-2-650M
            alone; 1280 + ~40 for ESM-2 + physicochemical concat).
        working_dim: shared internal width — CNN working dim AND Transformer
            hidden_dim. Defaults to 256.
        cnn_blocks: number of dilated CNN blocks (default 4).
        cnn_dilation_schedule: dilation per block (default [1, 2, 4, 8]).
        kernel_size: CNN kernel size (default 7).
        transformer_layers: number of rotary-PE Transformer layers (default 2).
        num_heads: attention heads (default 4).
        ff_dim: Transformer feed-forward inner dim (default 512).
        dropout: dropout rate, applied uniformly in CNN blocks and Transformer
            layers (default 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        working_dim: int = 256,
        cnn_blocks: int = 4,
        cnn_dilation_schedule: list[int] | None = None,
        kernel_size: int = 7,
        transformer_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        if cnn_dilation_schedule is None:
            cnn_dilation_schedule = [1, 2, 4, 8]
        if len(cnn_dilation_schedule) != cnn_blocks:
            raise ValueError(
                f"cnn_dilation_schedule length {len(cnn_dilation_schedule)} "
                f"must equal cnn_blocks {cnn_blocks}"
            )

        self.input_dim = input_dim
        self.working_dim = working_dim

        # Input projection: input_dim → working_dim
        self.projection = nn.Linear(input_dim, working_dim)

        # CNN block: 4 dilated residual blocks, all at working_dim
        self.cnn_blocks = nn.ModuleList([
            DilatedResBlock(working_dim, kernel_size, d, dropout)
            for d in cnn_dilation_schedule
        ])

        # Transformer block: 2 layers, rotary PE applied inside attention
        self.transformer_layers = nn.ModuleList([
            RotaryTransformerLayer(working_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(working_dim)

        # Output head: working_dim → 1 logit per residue (pre-sigmoid)
        self.output_head = nn.Linear(working_dim, 1)

    def forward_features(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penultimate-layer features (post-Transformer, post-LayerNorm,
        pre-output-head) — same interface as Part 3 model heads for Phase 2
        auxiliary head branching.

        Args:
            x: [B, L, input_dim] per-residue features.
            key_padding_mask: optional [B, L] bool tensor (True = pad position).

        Returns:
            [B, L, working_dim] feature tensor.
        """
        h = self.projection(x)
        for block in self.cnn_blocks:
            h = block(h)
        for layer in self.transformer_layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        return self.final_norm(h)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, L, input_dim] → [B, L, 1] per-residue logits (pre-sigmoid)."""
        h = self.forward_features(x, key_padding_mask=key_padding_mask)
        return self.output_head(h)

    def backbone_state_dict(self) -> dict:
        """Backbone-only state dict: projection + cnn_blocks + transformer_layers
        + final_norm. EXCLUDES output_head and any regression head replacement.
        Used by TriZOD pretrain → finetune transition (the regression head is
        discarded; only the backbone weights transfer)."""
        prefixes = ("projection.", "cnn_blocks.", "transformer_layers.", "final_norm.")
        return {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
            if k.startswith(prefixes)
        }

    def load_backbone_state_dict(self, sd: dict, strict: bool = True) -> None:
        """Load a backbone state dict produced by `backbone_state_dict()`.
        Output head weights are left untouched."""
        own = self.state_dict()
        missing = []
        for k in own:
            if any(k.startswith(p) for p in
                   ("projection.", "cnn_blocks.", "transformer_layers.", "final_norm.")):
                if k in sd:
                    own[k] = sd[k]
                else:
                    missing.append(k)
        if strict and missing:
            raise KeyError(f"backbone state_dict missing {len(missing)} keys, "
                           f"first: {missing[:3]}")
        self.load_state_dict(own)

    def backbone_parameters(self):
        """Iterator over backbone parameters (projection + CNN + Transformer
        + final_norm). For discriminative-LR optimizer construction in
        finetune mode."""
        for name, p in self.named_parameters():
            if name.startswith(("projection.", "cnn_blocks.",
                                "transformer_layers.", "final_norm.")):
                yield p

    def head_parameters(self):
        """Iterator over output_head parameters (the only 'new' params after
        finetune head swap)."""
        for name, p in self.named_parameters():
            if name.startswith("output_head."):
                yield p

    def replace_head_for_classification(self) -> None:
        """Replace the output head with a freshly-initialized Linear→1 logit
        head. Called after loading TriZOD pretrained backbone weights to
        transition from regression pretraining to binary classification
        finetuning (the pretrain regression head is discarded)."""
        self.output_head = nn.Linear(self.working_dim, 1)


class HybridRegressionHead(nn.Module):
    """Hybrid backbone with a regression output head for TriZOD continuous
    G-score pretraining. Output is a raw scalar per residue (NO sigmoid).

    Wraps `CNNTransformerHybrid` and replaces the binary classification head
    with a regression projection. After pretraining, backbone weights are
    extracted via `hybrid.backbone_state_dict()` and loaded into a fresh
    `CNNTransformerHybrid` for binary finetuning.
    """

    def __init__(
        self,
        input_dim: int,
        working_dim: int = 256,
        cnn_blocks: int = 4,
        cnn_dilation_schedule: list[int] | None = None,
        kernel_size: int = 7,
        transformer_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Reuse the hybrid backbone — but we'll bypass its output_head.
        self.hybrid = CNNTransformerHybrid(
            input_dim=input_dim,
            working_dim=working_dim,
            cnn_blocks=cnn_blocks,
            cnn_dilation_schedule=cnn_dilation_schedule,
            kernel_size=kernel_size,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        # Pretraining regression head — discarded after pretrain.
        self.regression_head = nn.Linear(working_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, L, input_dim] → [B, L] raw continuous G-score predictions
        (NO sigmoid — outputs are unconstrained reals; TriZOD G-scores live
        in [0, 1] but the pretrain Huber loss does not require clamping)."""
        h = self.hybrid.forward_features(x, key_padding_mask=key_padding_mask)
        return self.regression_head(h).squeeze(-1)

    def backbone_state_dict(self) -> dict:
        return self.hybrid.backbone_state_dict()


def count_parameters(model: nn.Module) -> dict:
    """Diagnostic helper — total / trainable / per-block parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_block = {}
    for name, _ in model.named_children():
        per_block[name] = sum(p.numel() for p in getattr(model, name).parameters())
    return {"total": total, "trainable": trainable, "per_block": per_block}
