"""Family T — Transformer expert head (P3-Ph2-S01).

Reference: P3_00 Amendment M-1; P3_04 §S01.

Lightweight Transformer encoder for per-residue disorder prediction:
  Input projection → N TransformerEncoderLayers (rotary position encoding) → Linear → 1 logit

Defaults:
  num_layers   = 2
  num_heads    = 4
  hidden_dim   = 256
  ff_dim       = 512
  dropout      = 0.2

Variable-length proteins are handled by passing a key-padding mask through
the encoder; padded positions are excluded from attention. Rotary position
encoding is applied inside the attention block (see RotaryAttention) so no
additive position embeddings are added at the input.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Rotary Position Encoding ────────────────────────────────────────


def _build_rotary_cache(seq_len: int, head_dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos, sin) caches of shape [seq_len, head_dim]."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [L, head_dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype)
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair-wise rotation: (x_2i, x_2i+1) → (-x_2i+1, x_2i)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x [..., L, head_dim]."""
    return x * cos + _rotate_half(x) * sin


# ─── Multi-head Self-Attention with Rotary ───────────────────────────


class RotaryMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, hidden]; key_padding_mask: [B, L] (True = pad)
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = _build_rotary_cache(L, self.head_dim, x.device, x.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        # Standard scaled dot-product attention with optional key_padding_mask
        attn_mask = None
        if key_padding_mask is not None:
            # broadcast to [B, 1, 1, L]
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~attn_mask if attn_mask is not None else None,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, L, self.hidden_dim)
        return self.out(out)


# ─── Transformer Encoder Layer ───────────────────────────────────────


class RotaryTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = RotaryMultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.drop(h)
        return x


# ─── Public Transformer Head ─────────────────────────────────────────


class TransformerHead(nn.Module):
    """Family T expert head.

    Args:
        input_dim:  Per-residue input feature dim (e.g. 1321 for V-ESM).
        hidden_dim: Internal working dim (default 256).
        num_layers: Number of transformer layers (default 2).
        num_heads:  Attention heads (default 4).
        ff_dim:     Feed-forward inner dim (default 512).
        dropout:    Dropout for attention + FF (default 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            RotaryTransformerLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward_features(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return penultimate-layer features [B, L, hidden_dim]."""
        h = self.projection(x)
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        return self.final_norm(h)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.forward_features(x, key_padding_mask=key_padding_mask)
        return self.output_head(h)  # [B, L, 1]
