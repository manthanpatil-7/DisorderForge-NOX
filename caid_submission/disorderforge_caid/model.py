"""DisorderForge-NOX predictor: frozen 5-member champion + 3 gated-residual
students, run on CPU from precomputed embeddings. Byte-faithful to the validated
production form (CAID3-NOX rpAP 0.6944). Self-contained: no external imports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import _core as C
from .cnn_transformer_hybrid import CNNTransformerHybrid
from .features import all_lightweight
from .rm_head import GatedResidualStudent, gated_final

# Member roster (order matters: assemble_base assumes 3 SaProt then 2 ESM-2).
SAPROT_SEEDS = (42, 123, 456)
ESM2_SEEDS = (42, 123)
RM_SEEDS = (42, 123, 456)
DIM_EMB = 1280
INPUT_DIM = DIM_EMB + C.LW_DIM   # 1321


def _new_head(device):
    return CNNTransformerHybrid(
        input_dim=INPUT_DIM, working_dim=256, cnn_blocks=4,
        cnn_dilation_schedule=[1, 2, 4, 8], kernel_size=7,
        transformer_layers=2, num_heads=4, ff_dim=512, dropout=0.2).to(device)


def _load_head(ckpt_path, device):
    head = _new_head(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(ck["model_state_dict"], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def _load_student(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    st = GatedResidualStudent(
        ck["d_in"], init_gate=ck["cfg"]["init_gate"]).to(device)
    st.load_state_dict(ck["student"], strict=True)
    st.eval()
    for p in st.parameters():
        p.requires_grad_(False)
    scale = float(ck["residual_scale"])
    emask = np.asarray(ck["evo_mask"], np.float32)
    use_msat = int(ck.get("use_msat", 0))
    if not use_msat:
        raise ValueError(f"{ckpt_path}: use_msat=0 — not an RM (MSA-T) student")
    if ck["d_in"] != C.BASE_DIM + C.EVO_DIM + C.MSAT_DIM:
        raise ValueError(f"{ckpt_path}: unexpected d_in={ck['d_in']}")
    return st, scale, emask


class DisorderForgeRM:
    """Loads all checkpoints once; predict() scores one protein."""

    def __init__(self, checkpoints_dir, rm_ckpt_dir, device="cpu"):
        self.device = device
        cd, rd = Path(checkpoints_dir), Path(rm_ckpt_dir)
        self.saprot_heads = [_load_head(cd / f"EXP-S02-v3_seed{s}_best.pt", device)
                             for s in SAPROT_SEEDS]
        self.esm2_heads = [_load_head(cd / f"EXP-E13H_seed{s}_best.pt", device)
                           for s in ESM2_SEEDS]
        self.students = [_load_student(rd / f"RM_seed{s}.pt", device)
                         for s in RM_SEEDS]

    @torch.no_grad()
    def _member_logit(self, head, emb, lw):
        """Halo-windowed head pass over a precomputed embedding -> logit[L].
        Same centre-valid halo stitching as the validated full-residue pipeline,
        with the live encoder replaced by the precomputed embedding."""
        L = emb.shape[0]
        logit = np.zeros(L, np.float32)
        filled = np.zeros(L, bool)
        for (es, ee, vs, ve) in C.halo_windows(L, 0):
            sub = np.concatenate([emb[es:ee], lw[es:ee]], axis=1)   # [n, 1321]
            n = sub.shape[0]
            x = torch.from_numpy(sub).float().unsqueeze(0).to(self.device)
            sm = torch.ones(1, n, dtype=torch.bool, device=self.device)
            out = head(x, key_padding_mask=~sm)[0].squeeze(-1)      # [n]
            out = out.float().cpu().numpy()
            logit[vs:ve] = out[(vs - es):(ve - es)]
            filled[vs:ve] = True
        assert filled.all(), f"head coverage gap (L={L})"
        return logit

    @torch.no_grad()
    def predict(self, sequence, saprot_emb, esm2_emb, msat_emb):
        """Return per-residue disorder probability [L]. Embeddings are [L, dim]
        aligned to len(sequence)."""
        L = len(sequence)
        lw = np.asarray(all_lightweight(sequence), np.float32)      # [L, 41]
        assert lw.shape == (L, C.LW_DIM), f"lightweight dim {lw.shape}"
        ml = np.zeros((L, C.N_MEM), np.float32)
        for i, head in enumerate(self.saprot_heads):
            ml[:, i] = self._member_logit(head, saprot_emb, lw)
        for j, head in enumerate(self.esm2_heads):
            ml[:, C.N_SAPROT + j] = self._member_logit(head, esm2_emb, lw)
        elog = C.ensemble_logit_of(ml)                              # [L]
        base = C.assemble_base(ml, lw)                              # [L, 63]
        evo = np.zeros((L, C.EVO_DIM), np.float32)                  # zeroed (evo none)
        et = torch.from_numpy(elog).float().to(self.device)
        psum = np.zeros(L, np.float32)
        for st, scale, emask in self.students:
            feat = np.concatenate([base, evo * emask, msat_emb], axis=1).astype(np.float32)
            ft = torch.from_numpy(feat).float().to(self.device)
            fin, _, _ = gated_final(st, ft, et, scale)
            psum += torch.sigmoid(fin).float().cpu().numpy()
        p = psum / len(self.students)
        assert len(p) == L and np.all(np.isfinite(p)), "predict len/finite fail"
        return p
