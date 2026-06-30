"""CNN–Transformer hybrid head — the per-residue scoring network used by every
member of the DisorderForge-NOX ensemble.

Self-contained: depends only on torch. This is a vendored, inference-only copy of
the head architecture (input projection → 4 dilated residual CNN blocks →
2 rotary-PE Transformer layers → linear logit). It must match the architecture
the shipped checkpoints were trained with; the box parity test is the guard.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── dilated residual CNN block ─────────────────────────
class DilatedResBlock(nn.Module):
    """Single dilated convolution block with residual connection."""

    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              dilation=dilation, padding=padding)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] → [B, L, D]"""
        residual = x
        h = x.transpose(1, 2)        # Conv1d expects [B, D, L]
        h = self.conv(h)
        h = h.transpose(1, 2)        # back to [B, L, D]
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h + residual


# ───────────────────────── rotary position encoding ─────────────────────────
def _build_rotary_cache(seq_len, head_dim, device, dtype):
    """Compute (cos, sin) caches of shape [seq_len, head_dim]."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device,
                                             dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
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


# ───────────────────────── rotary multi-head attention ─────────────────────────
class RotaryMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # x: [B, L, hidden]; key_padding_mask: [B, L] (True = pad)
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = _build_rotary_cache(L, self.head_dim, x.device, x.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~attn_mask if attn_mask is not None else None,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, L, self.hidden_dim)
        return self.out(out)


# ───────────────────────── rotary transformer layer ─────────────────────────
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

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.drop(h)
        return x


# ───────────────────────── the hybrid head ─────────────────────────
class CNNTransformerHybrid(nn.Module):
    """Sequential CNN → Transformer per-residue head (inference-only)."""

    def __init__(self, input_dim, working_dim=256, cnn_blocks=4,
                 cnn_dilation_schedule=None, kernel_size=7, transformer_layers=2,
                 num_heads=4, ff_dim=512, dropout=0.2):
        super().__init__()
        if cnn_dilation_schedule is None:
            cnn_dilation_schedule = [1, 2, 4, 8]
        if len(cnn_dilation_schedule) != cnn_blocks:
            raise ValueError(
                f"cnn_dilation_schedule length {len(cnn_dilation_schedule)} "
                f"must equal cnn_blocks {cnn_blocks}")
        self.input_dim = input_dim
        self.working_dim = working_dim
        self.projection = nn.Linear(input_dim, working_dim)
        self.cnn_blocks = nn.ModuleList([
            DilatedResBlock(working_dim, kernel_size, d, dropout)
            for d in cnn_dilation_schedule])
        self.transformer_layers = nn.ModuleList([
            RotaryTransformerLayer(working_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)])
        self.final_norm = nn.LayerNorm(working_dim)
        self.output_head = nn.Linear(working_dim, 1)

    def forward_features(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """[B, L, input_dim] → [B, L, working_dim] penultimate features."""
        h = self.projection(x)
        for block in self.cnn_blocks:
            h = block(h)
        for layer in self.transformer_layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        return self.final_norm(h)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """[B, L, input_dim] → [B, L, 1] per-residue logits (pre-sigmoid)."""
        h = self.forward_features(x, key_padding_mask=key_padding_mask)
        return self.output_head(h)
