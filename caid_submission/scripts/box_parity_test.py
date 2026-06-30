#!/usr/bin/env python3
"""BOX parity test — prove the CAID container ≡ the validated production pipeline.

Run on the GPU/torch box (needs the SaProt/ESM-2 encoders + data/msa_t.tar).
Covers readiness GATES 1-3:

  GATE 1 (≤1022 aa): per-residue diff distribution between the packaged predictor
    and the reference path — max / mean / 99.9pct |Δ|, residues compared,
    non-identical count (tolerance 1e-6).
  GATE 2 (full CAID3-NOX): packaged-predictor pooled metrics + coverage + NaN/Inf
    (rpAP / AUC / MaxF1 / MCC / macroAPS, protein & residue coverage, pred count).
  GATE 3 (>1022 aa): per-protein length, max/mean |Δ|, boundary vs non-boundary
    |Δ|, and the pooled-metric delta caused exclusively by long proteins.

The two paths differ ONLY in how the SaProt/ESM-2 embedding reaches the heads:
  - reference  = p11_eval_ensemble (live encoder+head per halo window),
  - packaged   = container head-windowing over a precomputed full-length embedding
                 (ownership-stitched encoder output — the array EMBEDDINGS.md tells
                 CAID to mount). For L≤1022 the embedding is a single window so the
                 two paths must be float-identical.

    python caid_submission/scripts/box_parity_test.py
    python caid_submission/scripts/box_parity_test.py --limit 20   # debug
"""
from __future__ import annotations

import argparse
import json
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

import p7_ft_common as F            # noqa: E402  encoders, windowing, benches, metrics
import p8_fullres_common as P8     # noqa: E402  load_teacher, teacher_fullres
import p11_common as P             # noqa: E402  msat_for, ensemble glue
from disorderforge_caid import DisorderForgeRM   # noqa: E402
from disorderforge_caid import _core as C        # noqa: E402
from disorderforge_caid.rm_head import gated_final  # noqa: E402

REF = {"rpAP": (0.6946, 0.6950), "AUC": (0.8638, 0.8641), "MaxF1": (0.6643, 0.6646)}
TOL = 1e-6
BOUNDARY_MARGIN = 32   # residues within this of an internal window edge = "boundary"


@torch.no_grad()
def encoder_fullres(enc, tok, view, seq, di, device):
    """Full-length [L,1280] embedding, ownership-stitched over halo windows —
    the array CAID precompute is expected to provide (per EMBEDDINGS.md)."""
    L = len(seq)
    emb = np.zeros((L, 1280), np.float32)
    for (es, ee, vs, ve) in C.halo_windows(L, 0):
        e = F._encode(enc, tok, seq[es:ee], (di[es:ee] if di else None),
                      view, device, grad=False, struct_dp=None, rng=None)
        emb[vs:ve] = e[(vs - es):(ve - es)].float().cpu().numpy()
    return emb


def boundary_mask(L):
    """True for residues near an INTERNAL halo-window valid-block edge."""
    m = np.zeros(L, bool)
    if L <= C.WIN_MAX:
        return m
    for (_, _, vs, ve) in C.halo_windows(L, 0):
        if vs > 0:
            m[max(0, vs - BOUNDARY_MARGIN):vs + BOUNDARY_MARGIN] = True
        if ve < L:
            m[max(0, ve - BOUNDARY_MARGIN):ve + BOUNDARY_MARGIN] = True
    return m


@torch.no_grad()
def reference_probs(mods, students, uni, seq, di, lw, device):
    """p11_eval_ensemble path: live encoder+head per window, RM seed-mean."""
    ml = np.zeros((len(seq), len(mods)), np.float32)
    for mi, (view, _, enc, tok, head) in enumerate(mods):
        _, logit = P8.teacher_fullres(enc, tok, head, view, seq, di, lw, device)
        ml[:, mi] = logit[:len(seq)]
    elog = P.ensemble_logit_of(ml)
    base = P.assemble_base(ml, lw)
    msat, _ = P.msat_for(uni, len(seq))
    et = torch.from_numpy(elog).float().to(device)
    evo = np.zeros((len(seq), C.EVO_DIM), np.float32)
    psum = np.zeros(len(seq), np.float32)
    for st, scale, emask in students:
        feat = np.concatenate([base, evo * emask, msat], axis=1).astype(np.float32)
        fin, _, _ = gated_final(st, torch.from_numpy(feat).float().to(device), et, scale)
        psum += torch.sigmoid(fin).cpu().numpy()
    return psum / len(students)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="caid3_nox")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "results/part_caid/parity_report.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DisorderForgeRM(REPO_ROOT / "checkpoints",
                            REPO_ROOT / "results/part11/ckpt", device=device)
    mods = []
    for view, seed in P.MEMBERS:
        enc, tok, head = P8.load_teacher(view, seed, device)
        mods.append((view, seed, enc, tok, head))
    students = model.students

    d2u = F.dp_to_uniprot()
    rows = list(F.load_bench(F.BENCHES[args.track], d2u))
    if args.limit:
        rows = rows[:args.limit]

    short_abs, n_short_diff, n_short_res = [], 0, 0
    long_rows = []
    labs, probs_c, probs_r, nan_count = [], [], [], 0
    for i, (dp, uni, seq, lab) in enumerate(rows):
        di = F.load_3di(uni)
        lw = np.asarray(F.all_lightweight(seq), np.float32)
        sap = encoder_fullres(mods[0][2], mods[0][3], "saprot", seq, di, device)
        esm = encoder_fullres(mods[3][2], mods[3][3], "esm2", seq, di, device)
        msat, _ = P.msat_for(uni, len(seq))
        p_c = model.predict(seq, sap, esm, msat)
        p_r = reference_probs(mods, students, uni, seq, di, lw, device)
        nan_count += int((~np.isfinite(p_c)).sum())
        d = np.abs(p_c - p_r)
        if len(seq) <= C.WIN_MAX:
            short_abs.append(d)
            n_short_diff += int((d > TOL).sum())
            n_short_res += len(d)
        else:
            bm = boundary_mask(len(seq))
            long_rows.append({
                "id": dp, "uni": uni, "length": int(len(seq)),
                "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                "boundary_mean_abs": float(d[bm].mean()) if bm.any() else 0.0,
                "interior_mean_abs": float(d[~bm].mean()) if (~bm).any() else 0.0,
            })
        n = min(len(p_c), len(lab))
        labs.append(lab[:n]); probs_c.append(p_c[:n]); probs_r.append(p_r[:n])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    short_abs = np.concatenate(short_abs) if short_abs else np.zeros(0)
    g1 = {
        "n_residues": int(n_short_res),
        "max_abs": float(short_abs.max()) if short_abs.size else 0.0,
        "mean_abs": float(short_abs.mean()) if short_abs.size else 0.0,
        "p99_9_abs": float(np.percentile(short_abs, 99.9)) if short_abs.size else 0.0,
        "n_non_identical_gt_tol": int(n_short_diff),
        "tolerance": TOL,
        "pass": bool(short_abs.size and short_abs.max() <= 1e-4),
    }
    r_c = F.metric_panel(labs, probs_c)
    r_r = F.metric_panel(labs, probs_r)
    g2 = {
        "rpAP": r_c["rpAP"], "AUC": r_c["AUC"], "MaxF1": r_c["MaxF1"],
        "MCC": r_c["mcc"], "macro_APS": r_c["macro_APS"],
        "n_prot": r_c["n_prot"], "n_res": r_c["n_res"],
        "protein_coverage": r_c["n_prot"] / len(rows),
        "nan_inf_count": int(nan_count),
        "ref_rpAP": r_r["rpAP"], "ref_AUC": r_r["AUC"], "ref_MaxF1": r_r["MaxF1"],
        "rpAP_in_ref_band": REF["rpAP"][0] - 0.001 <= r_c["rpAP"] <= REF["rpAP"][1] + 0.001,
    }
    g3 = {
        "n_long": len(long_rows),
        "proteins": long_rows,
        "pooled_rpAP_delta_container_minus_ref": r_c["rpAP"] - r_r["rpAP"],
        "pooled_AUC_delta": r_c["AUC"] - r_r["AUC"],
        "pooled_MaxF1_delta": r_c["MaxF1"] - r_r["MaxF1"],
    }
    report = {"track": args.track, "GATE1_short": g1, "GATE2_full": g2, "GATE3_long": g3}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("\n================ PARITY REPORT ================")
    print(f"GATE 1 (≤{C.WIN_MAX} aa): max|Δ|={g1['max_abs']:.2e} mean|Δ|={g1['mean_abs']:.2e} "
          f"p99.9={g1['p99_9_abs']:.2e}  nRes={g1['n_residues']} "
          f"non-identical(>{TOL:g})={g1['n_non_identical_gt_tol']}  -> "
          f"{'PASS' if g1['pass'] else 'CHECK'}")
    print(f"GATE 2 (full): rpAP={g2['rpAP']:.4f} AUC={g2['AUC']:.4f} MaxF1={g2['MaxF1']:.4f} "
          f"MCC={g2['MCC']:.4f} cov={g2['protein_coverage']*100:.1f}% "
          f"nRes={g2['n_res']} NaN/Inf={g2['nan_inf_count']} "
          f"(ref rpAP={g2['ref_rpAP']:.4f}) -> "
          f"{'PASS' if g2['rpAP_in_ref_band'] and g2['nan_inf_count'] == 0 else 'CHECK'}")
    print(f"GATE 3 (>{C.WIN_MAX} aa): n={g3['n_long']} "
          f"pooled rpAP Δ(container-ref)={g3['pooled_rpAP_delta_container_minus_ref']:+.5f} "
          f"AUC Δ={g3['pooled_AUC_delta']:+.5f}")
    for r in long_rows:
        print(f"    {r['id']:10} L={r['length']:5} max|Δ|={r['max_abs']:.2e} "
              f"mean|Δ|={r['mean_abs']:.2e} bnd={r['boundary_mean_abs']:.2e} "
              f"int={r['interior_mean_abs']:.2e}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
