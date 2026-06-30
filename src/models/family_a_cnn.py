"""Family A — Dilated Residual CNN expert head.

Reference: Model Family Contract §9.2

Architecture:
  Input projection → [Dilated Conv Block × N] → Output head
  Each block: Conv1d(dilated) → LayerNorm → GELU → Dropout → residual add

Hyperparameter ranges (tuned in Phase 4):
  working_dim: 256–512
  num_blocks: 4–8
  kernel_size: 5, 7, or 9
  dilation_schedule: exponentially increasing [1, 2, 4, 8, 16, ...]
  dropout: 0.1–0.3
"""

import torch
import torch.nn as nn


class DilatedResBlock(nn.Module):
    """Single dilated convolution block with residual connection."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] → [B, L, D]"""
        # Conv1d expects [B, D, L]
        residual = x
        h = x.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)  # back to [B, L, D]
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h + residual


class DilatedResidualCNN(nn.Module):
    """Dilated Residual CNN expert head for per-residue disorder prediction.

    Args:
        input_dim: Dimension of input features (determined by input view).
        working_dim: Internal working dimensionirtual (256–512).
        num_blocks: Number of dilated conv blocks (4–8).
        kernel_size: Convolution kernel size (5, 7, or 9).
        dilation_schedule: List of dilation factors per block.
            If None, uses exponential schedule [1, 2, 4, 8, ...].
        dropout: Dropout rate (0.1–0.3).
    """

    def __init__(
        self,
        input_dim: int,
        working_dim: int = 384,
        num_blocks: int = 6,
        kernel_size: int = 7,
        dilation_schedule: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.working_dim = working_dim

        # Input projection
        self.projection = nn.Linear(input_dim, working_dim)

        # Dilation schedule
        if dilation_schedule is None:
            dilation_schedule = [2**i for i in range(num_blocks)]
        assert len(dilation_schedule) == num_blocks

        # Dilated residual blocks
        self.blocks = nn.ModuleList([
            DilatedResBlock(working_dim, kernel_size, d, dropout)
            for d in dilation_schedule
        ])

        # Output head: working_dim → 1 logit per residue
        self.output_head = nn.Linear(working_dim, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer features [B, L, working_dim] (pre-output-head)."""
        h = self.projection(x)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, L, input_dim] — per-residue input features.

        Returns:
            [B, L, 1] — per-residue logits (pre-sigmoid).
        """
        return self.output_head(self.forward_features(x))  # [B, L, 1]

    # ── Part 4 finetune-from-pretrain plumbing (Amendment T-1) ──

    def backbone_state_dict(self) -> dict:
        """Backbone-only state dict: projection + blocks. EXCLUDES output_head
        so the binary classification head can be replaced after a TriZOD
        regression pretrain."""
        return {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
            if k.startswith(("projection.", "blocks."))
        }

    def load_backbone_state_dict(self, sd: dict, strict: bool = True) -> None:
        own = self.state_dict()
        missing = []
        for k in own:
            if k.startswith(("projection.", "blocks.")):
                if k in sd:
                    own[k] = sd[k]
                else:
                    missing.append(k)
        if strict and missing:
            raise KeyError(f"backbone state_dict missing {len(missing)} keys")
        self.load_state_dict(own)

    def backbone_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith(("projection.", "blocks.")):
                yield p

    def head_parameters(self):
        for name, p in self.named_parameters():
            if name.startswith("output_head."):
                yield p

    def replace_head_for_classification(self) -> None:
        self.output_head = nn.Linear(self.working_dim, 1)


class DilatedResidualCNNRegressionHead(nn.Module):
    """Wraps `DilatedResidualCNN` with a regression head for TriZOD continuous
    G-score pretraining (raw output, no sigmoid). After pretraining, the
    backbone state dict is loaded into a fresh `DilatedResidualCNN` for binary
    classification finetuning."""

    def __init__(
        self,
        input_dim: int,
        working_dim: int = 384,
        num_blocks: int = 6,
        kernel_size: int = 7,
        dilation_schedule: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = DilatedResidualCNN(
            input_dim=input_dim,
            working_dim=working_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dilation_schedule=dilation_schedule,
            dropout=dropout,
        )
        self.regression_head = nn.Linear(working_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn.forward_features(x)
        return self.regression_head(h).squeeze(-1)

    def backbone_state_dict(self) -> dict:
        return self.cnn.backbone_state_dict()
