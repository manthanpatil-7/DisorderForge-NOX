"""Family C — Residue-Local MLP expert head.

Reference: Model Family Contract §11.2

Architecture:
  Input projection → Hidden layers (per-residue, no inter-residue context) → Output head

Each residue is processed independently. This family serves as a pLM sufficiency
probe and ensemble decorrelation candidate.

Hyperparameter ranges (tuned in Phase 4):
  working_dim: 128–256
  num_layers: 1–2 (hidden)
  dropout: 0.1–0.5
"""

import torch
import torch.nn as nn


class ResidueLocalMLP(nn.Module):
    """Residue-local MLP expert head for per-residue disorder prediction.

    No inter-residue context — each residue processed independently.
    Serves as pLM sufficiency probe and ensemble diversity candidate.

    Args:
        input_dim: Dimension of input features (determined by input view).
        working_dim: Hidden layer dimension (128–256).
        num_layers: Number of hidden layers (1–2).
        dropout: Dropout rate (0.1–0.5).
    """

    def __init__(
        self,
        input_dim: int,
        working_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.working_dim = working_dim

        layers: list[nn.Module] = [nn.Linear(input_dim, working_dim)]

        for _ in range(num_layers):
            layers.extend([
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(working_dim, working_dim),
            ])

        self.hidden = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(working_dim)
        self.output_head = nn.Linear(working_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — each residue processed independently.

        Args:
            x: [B, L, input_dim] — per-residue input features.

        Returns:
            [B, L, 1] — per-residue logits (pre-sigmoid).
        """
        h = self.hidden(x)  # [B, L, working_dim]
        h = self.norm(h)
        return self.output_head(h)  # [B, L, 1]
