"""Torch RM gated-residual head — byte-faithful copy of
p11_common.GatedResidualStudent + gated_final. Architecture must match the
RM_seed*.pt checkpoints. Kept separate from _core.py so the pure numerics import
without torch."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GatedResidualStudent(nn.Module):
    """[base+evo+msat] -> (raw_residual, raw_gate).
    final = base + sigmoid(gate)*scale*tanh(res)."""

    def __init__(self, d_in, d=256, dconv=128, dropout=0.1, init_gate=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d, dconv, k, padding=k // 2, groups=1),
                          nn.SiLU()) for k in (9, 25, 65)])
        self.merge = nn.Linear(3 * dconv, dconv)
        self.mlp1 = nn.Linear(dconv, dconv)
        self.gate_mlp = nn.Linear(dconv, 2 * dconv)
        self.mlp2 = nn.Linear(dconv, 64)
        self.res_out = nn.Linear(64, 1)
        self.gate_out = nn.Linear(64, 1)
        nn.init.zeros_(self.res_out.weight)
        nn.init.zeros_(self.res_out.bias)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias,
                          math.log(init_gate / (1 - init_gate)))

    def forward(self, x):                          # x [L, d_in]
        h = self.act(self.proj(self.norm(x)))
        hc = h.transpose(0, 1).unsqueeze(0)
        cat = torch.cat([b(hc) for b in self.branches], dim=1)[0].transpose(0, 1)
        h = self.act(self.merge(cat))
        h = self.drop(h)
        h = self.act(self.mlp1(h))
        a, b = self.gate_mlp(h).chunk(2, dim=-1)
        h = self.act(self.mlp2(a * torch.sigmoid(b)))
        return self.res_out(h).squeeze(-1), self.gate_out(h).squeeze(-1)


def gated_final(student, feat_t, base_logit_t, scale):
    """final logit = base + sigmoid(gate)*scale*tanh(res)."""
    raw_res, raw_gate = student(feat_t)
    gate = torch.sigmoid(raw_gate)
    res = gate * scale * torch.tanh(raw_res)
    return base_logit_t + res, res, gate
