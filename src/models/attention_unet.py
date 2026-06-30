"""Attention U-Net (1D) for per-residue disorder prediction.

Reference: P4_00 Amendment M-4 (conditional U-Net architecture).

Architecture:
  Input [B, L, 1320] (ESM-2 1280 + LIGHTWEIGHT 41)
    → Encoder path (3 Conv1d blocks; strides 1, 2, 2 → resolutions L, L/2, L/4)
    → Bottleneck (2-layer rotary-PE Transformer over L/4 tokens)
    → Decoder path (2 TransposeConv blocks, each adds attention-gated skip
      from corresponding encoder block)
    → forward_features() output [B, L, working_dim] before the final linear.
    → Linear(working_dim → 1) — pre-sigmoid logit.

Attention gates (Oktay et al. 2018, "Attention U-Net"): for skip features `x`
and gating signal `g` (the decoder upsampled features), the gate is

    α = σ(ψ(ReLU(W_g(g) + W_x(x))))

and the gated skip is `α ⊙ x`. α is per-position scalar in [0, 1] — verified
in RT-29.

Length handling: stride-2 convolutions halve sequence length each block.
Boundary padding ensures no information is dropped at sequence end. Decoder
TransposeConv2 produces L/2; TransposeConv1 produces L. Output length is
clamped to original L (handles odd-L cases via deterministic length book-keeping).

Exposes the same `forward_features()` interface as `DilatedResidualCNN` and
`CNNTransformerHybrid`, so existing lever components (evidence aux head,
function multi-task head, CRF, late fusion, protein context, etc.) plug in
without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer_head import RotaryTransformerLayer


class ConvBlock1D(nn.Module):
    """Conv1d → LayerNorm (over channels) → GELU → Dropout, optionally strided."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 5,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] → [B, D, L] → conv → [B, D', L'] → [B, L', D']
        h = x.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h = self.norm(h)
        h = self.act(h)
        return self.drop(h)


class TransposeConvBlock1D(nn.Module):
    """ConvTranspose1d (stride 2 upsample) → LayerNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # kernel_size=4, stride=2, padding=1 produces exact 2× upsample for
        # inputs whose spatial dim was floor(L/2) (matches the stride-2 forward).
        self.deconv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=kernel_size,
                                         stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.deconv(h)
        h = h.transpose(1, 2)
        # Adjust output length to match target_len (handles odd input lengths).
        if target_len is not None and h.shape[1] != target_len:
            if h.shape[1] > target_len:
                h = h[:, :target_len, :]
            else:
                # Pad by replicating the last position
                pad = h[:, -1:, :].expand(-1, target_len - h.shape[1], -1)
                h = torch.cat([h, pad], dim=1)
        h = self.norm(h)
        h = self.act(h)
        return self.drop(h)


class AttentionGate1D(nn.Module):
    """Attention gate from Oktay et al. 2018, adapted for 1D sequences.

    α = σ(ψ(ReLU(W_g(g) + W_x(x))))
    out = α ⊙ x

    g (gating signal): decoder features at the target resolution.
    x (skip features): encoder features at the same resolution.
    Both must have the same shape [B, L, D]. α has shape [B, L, 1] and is
    in (0, 1) by construction.
    """

    def __init__(self, gate_dim: int, skip_dim: int, inter_dim: int):
        super().__init__()
        self.W_g = nn.Linear(gate_dim, inter_dim, bias=False)
        self.W_x = nn.Linear(skip_dim, inter_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(inter_dim))
        self.psi = nn.Linear(inter_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor):
        # Returns (gated_skip, alpha) — alpha is exposed for unit tests.
        h = self.W_g(g) + self.W_x(x) + self.bias
        h = self.relu(h)
        alpha = self.sigmoid(self.psi(h))   # [B, L, 1] in (0, 1)
        return alpha * x, alpha


class AttentionUNet(nn.Module):
    """1D Attention U-Net per-residue disorder predictor.

    Args:
        input_dim: per-residue input feature dim (e.g. 1320 = ESM-2 1280 + LIGHTWEIGHT 41).
        working_dim: shared internal width (encoder/decoder/bottleneck). Default 256.
        bottleneck_layers: rotary-PE Transformer layers in the bottleneck (default 2).
        bottleneck_heads: attention heads (default 4).
        bottleneck_ff_dim: feed-forward inner dim (default 512).
        kernel_size: encoder/decoder Conv kernel (default 5).
        dropout: dropout rate (default 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        working_dim: int = 256,
        bottleneck_layers: int = 2,
        bottleneck_heads: int = 4,
        bottleneck_ff_dim: int = 512,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.working_dim = working_dim

        # ── Encoder path ────────────────────────────────────────────
        # Block 1: full resolution L; project input_dim → working_dim.
        self.enc1 = ConvBlock1D(input_dim, working_dim,
                                kernel_size=kernel_size, stride=1, dropout=dropout)
        # Block 2: stride 2 → L/2.
        self.enc2 = ConvBlock1D(working_dim, working_dim,
                                kernel_size=kernel_size, stride=2, dropout=dropout)
        # Block 3: stride 2 → L/4.
        self.enc3 = ConvBlock1D(working_dim, working_dim,
                                kernel_size=kernel_size, stride=2, dropout=dropout)

        # ── Bottleneck (rotary-PE Transformer) ──────────────────────
        self.bottleneck_layers = nn.ModuleList([
            RotaryTransformerLayer(working_dim, bottleneck_heads,
                                   bottleneck_ff_dim, dropout)
            for _ in range(bottleneck_layers)
        ])
        self.bottleneck_norm = nn.LayerNorm(working_dim)

        # ── Decoder path ────────────────────────────────────────────
        # Up3: L/4 → L/2; gated skip from enc2 (L/2 features).
        self.up3 = TransposeConvBlock1D(working_dim, working_dim,
                                        kernel_size=4, dropout=dropout)
        self.gate2 = AttentionGate1D(gate_dim=working_dim, skip_dim=working_dim,
                                     inter_dim=working_dim // 2)
        self.merge2 = ConvBlock1D(working_dim * 2, working_dim,
                                  kernel_size=kernel_size, stride=1, dropout=dropout)

        # Up2: L/2 → L; gated skip from enc1 (L features).
        self.up2 = TransposeConvBlock1D(working_dim, working_dim,
                                        kernel_size=4, dropout=dropout)
        self.gate1 = AttentionGate1D(gate_dim=working_dim, skip_dim=working_dim,
                                     inter_dim=working_dim // 2)
        self.merge1 = ConvBlock1D(working_dim * 2, working_dim,
                                  kernel_size=kernel_size, stride=1, dropout=dropout)

        self.final_norm = nn.LayerNorm(working_dim)
        self.output_head = nn.Linear(working_dim, 1)

    def forward_features(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, L, input_dim] → [B, L, working_dim] (pre-output-head).

        Same interface as `DilatedResidualCNN.forward_features` and
        `CNNTransformerHybrid.forward_features` so all lever components
        (aux head, CRF, function MT, late fusion, ctx) plug in unchanged.
        """
        # Encoder
        e1 = self.enc1(x)              # [B, L, D]
        e2 = self.enc2(e1)             # [B, L/2, D]
        e3 = self.enc3(e2)             # [B, L/4, D]

        # Bottleneck (Transformer over L/4 tokens — no key_padding_mask
        # propagation here because the stride-2 down-samples have already
        # mixed padded/non-padded positions; we pass None to the rotary attn
        # which is robust to slightly mixed inputs at the sequence boundary).
        h = e3
        for layer in self.bottleneck_layers:
            h = layer(h, key_padding_mask=None)
        h = self.bottleneck_norm(h)

        # Decoder + attention-gated skips
        d2 = self.up3(h, target_len=e2.shape[1])           # back to L/2
        gated_e2, _ = self.gate2(g=d2, x=e2)
        d2 = self.merge2(torch.cat([d2, gated_e2], dim=-1))

        d1 = self.up2(d2, target_len=e1.shape[1])          # back to L
        gated_e1, _ = self.gate1(g=d1, x=e1)
        d1 = self.merge1(torch.cat([d1, gated_e1], dim=-1))

        return self.final_norm(d1)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, L, input_dim] → [B, L, 1] per-residue logits."""
        h = self.forward_features(x, key_padding_mask=key_padding_mask)
        return self.output_head(h)

    # ── Inspection helper for unit tests / RT-29 ───────────────────

    def attention_gates_alpha(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha2, alpha1) — the attention-gate maps from each gate.
        Each is shape [B, L', 1] in (0, 1). Used by RT-29 to verify gates
        produce valid probabilities."""
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        h = e3
        for layer in self.bottleneck_layers:
            h = layer(h, key_padding_mask=None)
        h = self.bottleneck_norm(h)
        d2 = self.up3(h, target_len=e2.shape[1])
        _, alpha2 = self.gate2(g=d2, x=e2)
        merged_d2 = self.merge2(torch.cat([d2, alpha2 * e2], dim=-1))
        d1 = self.up2(merged_d2, target_len=e1.shape[1])
        _, alpha1 = self.gate1(g=d1, x=e1)
        return alpha2, alpha1
