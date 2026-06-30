"""1D Vision Transformer (patch-based) for per-residue disorder prediction.

Reference: P4_00 Amendment M-5 (conditional ViT-1D architecture).

Architecture:
  Input [B, L, 1320] (ESM-2 1280 + LIGHTWEIGHT 41)
    → Pad L up to next multiple of patch_size (default 8); record original L.
    → 1D patch embedding: reshape into [B, L_pad/8, 1320*8]; linear → 512-dim
      patch tokens. Add learned positional embeddings.
    → Transformer: 4 layers, 8 heads, hidden 512, ff 1024, dropout 0.2.
    → Per-residue recovery: each patch token → Linear(512 → 512*8) → reshape
      to [B, L_pad, 512]; trim to original L.
    → Residual concat with original ESM-2 embeddings ([B, L, 1280]) →
      Linear(512 + 1280 = 1792 → 256) → ReLU → output_head Linear(256 → 1).

`forward_features()` returns the 256-dim pre-final features [B, L, 256].
Same interface as `DilatedResidualCNN.forward_features` and
`CNNTransformerHybrid.forward_features` — lever components plug in unchanged.

Padding correctness: an input of length L=50 is padded to 56 (next multiple
of 8), processed in 7 patches, then output is trimmed back to 50 residues.
For L=2000 (already a multiple of 8), no padding/trimming applied. Verified
in RT-30.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer_head import RotaryTransformerLayer


class ViTHead(nn.Module):
    """1D patch-based ViT for per-residue disorder prediction.

    Args:
        input_dim: per-residue input feature dim (e.g. 1320 = ESM-2 + LIGHTWEIGHT).
        esm2_dim: dim of the ESM-2 slice in `input_dim` for the residual path
            (the residual concat uses ONLY the ESM-2 columns, not LIGHTWEIGHT;
            default 1280; first `esm2_dim` columns of input).
        patch_size: number of residues per patch (default 8).
        patch_token_dim: patch token dim after linear projection (default 512).
        num_layers: Transformer encoder layers (default 4).
        num_heads: attention heads (default 8).
        ff_dim: Transformer feed-forward inner dim (default 1024).
        max_seq_len: maximum sequence length supported (default 4096; the
            learned positional embedding has this many slots).
        working_dim: post-residual MLP hidden dim (default 256).
        dropout: dropout rate (default 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        esm2_dim: int = 1280,
        patch_size: int = 8,
        patch_token_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        max_seq_len: int | None = None,   # kept for API compat; no longer used
        working_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.esm2_dim = esm2_dim
        self.patch_size = patch_size
        self.patch_token_dim = patch_token_dim
        self.working_dim = working_dim

        # Patch embedding: each patch is patch_size residues × input_dim.
        self.patch_embed = nn.Linear(input_dim * patch_size, patch_token_dim)
        # NO learned pos_embed — RotaryTransformerLayer applies rotary PE
        # inside attention, which is sequence-length-agnostic.
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.layers = nn.ModuleList([
            RotaryTransformerLayer(patch_token_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.transformer_norm = nn.LayerNorm(patch_token_dim)

        # Per-residue recovery: each patch token → patch_size residue features.
        self.patch_recover = nn.Linear(patch_token_dim, patch_token_dim * patch_size)

        # Residual concat + post-MLP. Residual uses ONLY the ESM-2 columns
        # of the original input; LIGHTWEIGHT columns are not concatenated to
        # keep the residual path semantically clean (pLM-only signal).
        self.post_mlp = nn.Sequential(
            nn.Linear(patch_token_dim + esm2_dim, working_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(working_dim)
        self.output_head = nn.Linear(working_dim, 1)

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad the L dim of x to the next multiple of patch_size.
        Returns (padded_x, original_L, padded_L)."""
        B, L, D = x.shape
        rem = L % self.patch_size
        if rem == 0:
            return x, L, L
        pad = self.patch_size - rem
        # Replicate the last position for padding (sequence-aware extension).
        pad_block = x[:, -1:, :].expand(B, pad, D)
        return torch.cat([x, pad_block], dim=1), L, L + pad

    def forward_features(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, L, input_dim] → [B, L, working_dim] (pre-output-head)."""
        B, L_orig, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {D}")

        # Pad L → multiple of patch_size
        x_pad, _, L_pad = self._pad_to_patch(x)
        n_patches = L_pad // self.patch_size

        # Reshape into patches: [B, n_patches, patch_size * input_dim]
        patches = x_pad.reshape(B, n_patches, self.patch_size * self.input_dim)

        # Patch embedding (rotary PE applied inside the transformer, not here)
        tokens = self.patch_embed(patches)                              # [B, n_patches, P]
        tokens = self.embed_dropout(tokens)

        # Transformer encoder
        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=None)
        tokens = self.transformer_norm(tokens)

        # Per-residue recovery: each token → patch_size residue features
        recovered = self.patch_recover(tokens)                           # [B, n_patches, P*patch_size]
        recovered = recovered.reshape(B, L_pad, self.patch_token_dim)
        recovered = recovered[:, :L_orig, :]                             # trim to original L

        # Residual concat with the original ESM-2 slice of the input
        esm2_slice = x[:, :, :self.esm2_dim]
        fused = torch.cat([recovered, esm2_slice], dim=-1)               # [B, L, P + esm2_dim]
        fused = self.post_mlp(fused)
        return self.final_norm(fused)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, L, input_dim] → [B, L, 1] per-residue logits."""
        h = self.forward_features(x, key_padding_mask=key_padding_mask)
        return self.output_head(h)
