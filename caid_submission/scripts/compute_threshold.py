#!/usr/bin/env python3
"""Pick the binary-call threshold on a HELD-OUT set (never the CAID test).

The disorder SCORE column drives the threshold-free metrics CAID ranks on
(AP / AUC); the binary column is a fixed operating point. DisorderForge scores
are well-ranked but low-magnitude (SaProt is under-calibrated), so the default
0.10 reflects the validation MaxF1 operating point — re-derive it here if you
regenerate predictions.

Input: a .npz with pooled per-residue 'scores' (float [N]) and 'labels'
(int {0,1} [N]) over your validation proteins. Produce it on the box, e.g. by
running predict.py on the validation FASTA + val embeddings and pooling the
.caid scores against the known labels.

    python compute_threshold.py --npz val_pool.npz [--write-config ../config.yaml]
"""
from __future__ import annotations

import argparse

import numpy as np


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
    ap.add_argument("--npz", required=True, help="npz with 'scores' and 'labels'")
    ap.add_argument("--write-config", default=None, help="config.yaml to update")
    args = ap.parse_args()
    d = np.load(args.npz)
    scores, labels = np.asarray(d["scores"], float), np.asarray(d["labels"], int)
    m = (labels == 0) | (labels == 1)
    scores, labels = scores[m], labels[m]
    t, f1 = maxf1_threshold(scores, labels)
    print(f"validation residues: {len(labels)}  positives: {int(labels.sum())}")
    print(f"MaxF1 threshold = {t:.3f}  (val MaxF1 = {f1:.4f})")
    if args.write_config:
        import re
        txt = open(args.write_config).read()
        new = re.sub(r"(?m)^threshold:.*$", f"threshold: {t:.3f}", txt)
        open(args.write_config, "w").write(new)
        print(f"updated threshold -> {args.write_config}")


if __name__ == "__main__":
    main()
