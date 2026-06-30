#!/usr/bin/env python3
"""Generate the three per-residue embeddings (SaProt, ESM-2, MSA-Transformer) for
a CAID benchmark track and save them as <id>.h5 files — the exact format CAID
will mount. Use this to produce real embedding inputs for predict.py / Docker /
robustness / determinism testing.

Embeddings follow EMBEDDINGS.md: full-length, ownership-stitched halo windowing,
BOS/EOS stripped, keyed by FASTA id (= DisProt id used in the benchmark FASTA).

    python caid_submission/scripts/dump_embeddings.py --track caid3_nox --out /content/emb
    # then:
    python caid_submission/predict.py --fasta data/benchmarks/caid3/disorder_nox.fasta \
        --saprot-emb /content/emb/saprot --esm2-emb /content/emb/esm2 \
        --msat-emb /content/emb/msat --out /content/caid_out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
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


def save_h5(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("embedding", data=arr.astype(np.float32),
                         compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="caid3_nox")
    ap.add_argument("--out", required=True, help="output root (saprot/ esm2/ msat/)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)

    mods = {}
    for view, seed in P.MEMBERS:
        if view not in mods:
            enc, tok, _ = P8.load_teacher(view, seed, device)
            mods[view] = (enc, tok)

    d2u = F.dp_to_uniprot()
    rows = list(F.load_bench(F.BENCHES[args.track], d2u))
    if args.limit:
        rows = rows[:args.limit]
    for i, (dp, uni, seq, lab) in enumerate(rows):
        di = F.load_3di(uni)
        sap = encoder_fullres(mods["saprot"][0], mods["saprot"][1], "saprot", seq, di, device)
        esm = encoder_fullres(mods["esm2"][0], mods["esm2"][1], "esm2", seq, di, device)
        msat, _ = P.msat_for(uni, len(seq))
        save_h5(out / "saprot" / f"{dp}.h5", sap)
        save_h5(out / "esm2" / f"{dp}.h5", esm)
        save_h5(out / "msat" / f"{dp}.h5", msat)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    print(f"\nsaved {len(rows)} x 3 embeddings -> {out}/{{saprot,esm2,msat}}/<id>.h5")
    print("files keyed by DisProt id (matches the benchmark FASTA headers).")


if __name__ == "__main__":
    main()
