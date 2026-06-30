"""Pure numerics for the DisorderForge-NOX CAID predictor: halo windowing, base
feature assembly, and the champion ensemble logit. Self-contained and byte-
faithful to the validated production pipeline; the box parity test is the guard.

Pure numpy: no torch, no filesystem, no network (so the windowing/feature logic
is testable without a torch install). The torch RM head lives in rm_head.py.
"""
from __future__ import annotations

import numpy as np

# ───────────────────────── windowing ─────────────────────────
WIN_MAX = 1022          # max biological residues per encoder/head window
HALO = 128              # context residues each side whose predictions are discarded
STRIDE = 766            # = VALID block width (valid blocks tile contiguously)
WORKING_DIM = 256       # CNNTransformerHybrid penultimate width

# ───────────────────────── feature dims ────────────────────
N_MEM = 5               # 3 SaProt-seed heads + 2 ESM-2-seed heads
N_SAPROT = 3
LW_DIM = 41             # all_lightweight Tier-1 dim
LOCAL_WINS = (9, 21, 51, 101)   # local champion-score summary windows
# ens_logit(1)+member_logits(5)+var(1)+sap(1)+esm(1)+disagree(1)+lw(41)+local(12)=63
BASE_DIM = 1 + N_MEM + 1 + 1 + 1 + 1 + LW_DIM + len(LOCAL_WINS) * 3
EVO_DIM = 29            # conservation block (zeroed for RM: evo-level none)
MSAT_DIM = 768          # MSA-Transformer per-residue embedding


def halo_windows(L, offset=0):
    """Partition [0,L) into valid blocks (<=STRIDE) -> per-window
    (enc_s, enc_e, val_s, val_e). Each residue is OWNED by exactly one window
    (val block); enc block adds HALO context each side, clipped at termini.
    Exact full coverage, no double counting."""
    if L <= WIN_MAX:
        return [(0, L, 0, L)]
    edges = [0]
    first = STRIDE - offset if offset else STRIDE
    p = min(first, L)
    if p > 0:
        edges.append(p)
    while p < L:
        p = min(p + STRIDE, L)
        edges.append(p)
    wins = []
    for i in range(len(edges) - 1):
        vs, ve = edges[i], edges[i + 1]
        if ve <= vs:
            continue
        es = max(0, vs - HALO)
        ee = min(L, ve + HALO)
        assert ee - es <= WIN_MAX, f"window {ee - es} > {WIN_MAX} (L={L})"
        wins.append((es, ee, vs, ve))
    cov = sorted((vs, ve) for _, _, vs, ve in wins)
    assert cov[0][0] == 0 and cov[-1][1] == L, f"coverage gap (L={L})"
    for a, b in zip(cov, cov[1:]):
        assert a[1] == b[0], f"valid blocks not contiguous (L={L})"
    return wins


# ───────────────────────── base-feature assembly ──────────
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ensemble_logit_of(member_logits):
    """Champion ensemble = MEAN of member PROBABILITIES (Part-5 Level-1 fusion);
    returned as a logit so a zero residual sits exactly on the champion."""
    mean_prob = (1.0 / (1.0 + np.exp(-member_logits))).mean(1)
    return _logit(mean_prob).astype(np.float32)


def local_summaries(score):
    """mean/max/std of `score` over centred windows -> [L, len(LOCAL_WINS)*3]."""
    L = len(score)
    cols = []
    for w in LOCAL_WINS:
        pad = w // 2
        s = np.pad(score, (pad, pad), mode="edge")
        idx = np.arange(L)[:, None] + np.arange(w)[None, :]
        win = s[idx]
        cols += [win.mean(1), win.max(1), win.std(1)]
    return np.stack(cols, axis=1).astype(np.float32)


def assemble_base(member_logits, lw):
    """member_logits [L,5], lw [L,41] -> [L, BASE_DIM(63)] float32.
   """
    ens_logit = ensemble_logit_of(member_logits)[:, None]
    var = member_logits.var(1, keepdims=True)
    sap = member_logits[:, :N_SAPROT].mean(1, keepdims=True)
    esm = member_logits[:, N_SAPROT:].mean(1, keepdims=True)
    disagree = np.abs(sap - esm)
    ens_prob = 1.0 / (1.0 + np.exp(-ens_logit[:, 0]))
    loc = local_summaries(ens_prob)
    return np.concatenate([ens_logit, member_logits, var, sap, esm, disagree,
                           lw[:len(member_logits)], loc], axis=1).astype(np.float32)

# The torch RM gated-residual head lives in rm_head.py (kept out of this pure
# module so the windowing / feature numerics import without torch).
