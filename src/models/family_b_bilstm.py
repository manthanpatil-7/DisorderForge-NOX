"""Family B — Bidirectional LSTM expert head.

Reference: Model Family Contract §10.2

Architecture:
  Input projection → BiLSTM(num_layers) → Post-LSTM projection → Output head

Hyperparameter ranges (tuned in Phase 4):
  working_dim: 256–512
  hidden_dim: 256–384
  num_layers: 1–3
  dropout: 0.1–0.3
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMHead(nn.Module):
    """Bidirectional LSTM expert head for per-residue disorder prediction.

    Args:
        input_dim: Dimension of input features (determined by input view).
        hidden_dim: LSTM hidden dimension per direction (256–384).
        num_layers: Number of BiLSTM layers (1–3).
        working_dim: Post-LSTM projection dimension (256–512).
        dropout: Dropout rate between LSTM layers (0.1–0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        working_dim: int = 384,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.working_dim = working_dim

        # Input projection
        self.projection = nn.Linear(input_dim, working_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=working_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Post-LSTM projection: 2 * hidden_dim → working_dim
        self.post_lstm = nn.Linear(2 * hidden_dim, working_dim)
        self.norm = nn.LayerNorm(working_dim)
        self.drop = nn.Dropout(dropout)

        # Output head: working_dim → 1 logit per residue
        self.output_head = nn.Linear(working_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional packed-sequence support.

        Args:
            x: [B, L, input_dim] — per-residue input features.
                For variable-length batches, padded to max length.
            lengths: [B] — actual sequence lengths for packed-sequence
                processing. If None, assumes all sequences have length L.

        Returns:
            [B, L, 1] — per-residue logits (pre-sigmoid).
        """
        h = self.projection(x)  # [B, L, working_dim]

        if lengths is not None:
            # Pack for efficient variable-length processing
            packed = pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            h, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            h, _ = self.lstm(h)

        # h: [B, L, 2 * hidden_dim]
        h = self.post_lstm(h)  # [B, L, working_dim]
        h = self.norm(h)
        h = self.drop(h)
        return self.output_head(h)  # [B, L, 1]


class BiLSTMHeadLateFusion(nn.Module):
    """B-esm late-fusion variant per P3-00 Amendment M-4.

    Differs from BiLSTMHead in feature routing:
      Path A (contextual): ESM-2 (esm2_dim) → BiLSTM → hidden representation
      Path B (positional/biophysical): lightweight features (lw_dim) bypass the BiLSTM
      Fusion: concat(BiLSTM_hidden, lightweight) → Linear → ReLU → Linear → 1 logit

    The lightweight path preserves position- and chemistry-level signal that
    early fusion can dilute when ESM-2's high-dimensional context dominates
    the BiLSTM's input projection.

    Forward expects features in the order assembled by colab_train: ESM-2
    embedding first, then lightweight features (per src.models.routing V-ESM
    layout). The split index is the configured esm2_dim.
    """

    def __init__(
        self,
        esm2_dim: int = 1280,
        lw_dim: int = 41,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.esm2_dim = esm2_dim
        self.lw_dim = lw_dim
        self.input_dim = esm2_dim + lw_dim  # exposed for routing/sanity checks

        self.lstm = nn.LSTM(
            input_size=esm2_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fuse = nn.Linear(2 * hidden_dim + lw_dim, fusion_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(fusion_dim)
        self.drop = nn.Dropout(dropout)
        self.output_head = nn.Linear(fusion_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, esm2_dim + lw_dim]
        esm2 = x[..., : self.esm2_dim]
        lw = x[..., self.esm2_dim : self.esm2_dim + self.lw_dim]
        h_lstm, _ = self.lstm(esm2)  # [B, L, 2*hidden_dim]
        h = torch.cat([h_lstm, lw], dim=-1)  # late fusion
        h = self.fuse(h)
        h = self.act(h)
        h = self.norm(h)
        h = self.drop(h)
        return self.output_head(h)  # [B, L, 1]
