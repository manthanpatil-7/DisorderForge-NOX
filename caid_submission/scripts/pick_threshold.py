#!/usr/bin/env python3
"""GATE 8 — pick the binary-call threshold on an INDEPENDENT set (CAID2-NOX),
never on the CAID3 test set. Runs the packaged predictor over CAID2-NOX, pools
per-residue scores vs labels, finds the MaxF1 threshold, and writes it into
config.yaml. Records the selection set + metric.

CAID ranks on the continuous score (AP/AUC, threshold-free); this only sets the
binary column's operating point.

    python caid_submission/scripts/pick_threshold.py            # writes config.yaml
    python caid_submission/scripts/pick_threshold.py --track caid2_nox --no-write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
REPO_ROOT = PKG_ROOT.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import p7_ft_common as F            # noqa: E402
import p8_fullres_common as P8     # noqa: E402
import p11_common as P             # noqa: E402
from disorderforge_caid import DisorderForgeRM   # noqa: E402
from disorderforge_caid import _core as C        # noqa: E402


@torch.no_grad()
def encoder_fullres(enc, tok, view, seq, di, device):
    L = len(seq)
    emb = np.zeros((L, 1280), np.float32)
    for (es, ee, vs, ve) in C.halo_windows(L, 0):
        e = F._encode(enc, tok, seq[es:ee], (di[es:ee] if di else None),
                      view, device, grad=False, struct_dp=None, rng=None)
        emb[vs:ve] = e[(vs - es):(ve - es)].float().cpu().numpy()
    return emb


def maxf1_threshold(scores, labels, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    y = labels.astype(int)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="caid2_nox",
                    help="INDEPENDENT selection set (NOT caid3_*)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    assert "caid3" not in args.track, "never tune the threshold on CAID3"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DisorderForgeRM(REPO_ROOT / "checkpoints",
                            REPO_ROOT / "results/part11/ckpt", device=device)
    mods = {}
    for view, seed in P.MEMBERS:
        if view not in mods:
            enc, tok, _ = P8.load_teacher(view, seed, device)
            mods[view] = (enc, tok)

    d2u = F.dp_to_uniprot()
    scores, labels = [], []
    rows = list(F.load_bench(F.BENCHES[args.track], d2u))
    for i, (dp, uni, seq, lab) in enumerate(rows):
        di = F.load_3di(uni)
        sap = encoder_fullres(mods["saprot"][0], mods["saprot"][1], "saprot", seq, di, device)
        esm = encoder_fullres(mods["esm2"][0], mods["esm2"][1], "esm2", seq, di, device)
        msat, _ = P.msat_for(uni, len(seq))
        p = model.predict(seq, sap, esm, msat)
        n = min(len(p), len(lab))
        scores.append(p[:n]); labels.append(lab[:n])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    s = np.concatenate(scores); y = np.concatenate(labels)
    m = (y == 0) | (y == 1)
    s, y = s[m], y[m]
    t, f1 = maxf1_threshold(s, y)
    print(f"\nselection set : {args.track}  ({len(rows)} proteins, {len(y)} labelled residues, "
          f"{int(y.sum())} disordered)")
    print(f"MaxF1 threshold = {t:.3f}   (selection MaxF1 = {f1:.4f})")

    cfg_path = PKG_ROOT / "config.yaml"
    if not args.no_write:
        txt = cfg_path.read_text()
        txt = re.sub(r"(?m)^threshold:.*$", f"threshold: {t:.3f}", txt)
        cfg_path.write_text(txt)
        print(f"wrote threshold {t:.3f} -> {cfg_path}")
        print(f"RECORD in CAID_SUBMISSION_READINESS.md GATE 8: "
              f"selection set = {args.track}, metric = MaxF1, threshold = {t:.3f}")
    else:
        print("(--no-write: config.yaml not modified)")


if __name__ == "__main__":
    main()
