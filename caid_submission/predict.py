#!/usr/bin/env python3
"""DisorderForge-NOX — CAID 4 command-line predictor.

Reads a FASTA file and, for each sequence, loads the three precomputed
per-residue embeddings (SaProt, ESM-2, MSA-Transformer) mounted by CAID, scores
disorder on CPU, and writes one <id>.caid file per protein plus a timings.csv.

Runs on CPU only, offline, no GPU. See README.md for the full contract and
EMBEDDINGS.md for the exact embedding-generation recipe.

Example:
    python predict.py \
        --fasta input.fasta \
        --saprot-emb /mnt/emb/saprot --esm2-emb /mnt/emb/esm2 --msat-emb /mnt/emb/msat \
        --out ./output --threads 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from disorderforge_caid.embeddings import load_embedding          # noqa: E402
from disorderforge_caid.io_caid import Timings, read_fasta, write_caid  # noqa: E402

DIM_SAPROT, DIM_ESM2, DIM_MSAT = 1280, 1280, 768


def load_config(path):
    cfg = {}
    if path and os.path.exists(path):
        import yaml
        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
    return cfg


def resolve(cli_val, cfg, key, default=None):
    if cli_val is not None:
        return cli_val
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def main():
    ap = argparse.ArgumentParser(description="DisorderForge-NOX CAID predictor")
    ap.add_argument("--fasta", required=True, help="input FASTA file")
    ap.add_argument("--out", default=None, help="output root (disorder/ + timings.csv)")
    ap.add_argument("--saprot-emb", default=None, help="dir of SaProt embeddings")
    ap.add_argument("--esm2-emb", default=None, help="dir of ESM-2 embeddings")
    ap.add_argument("--msat-emb", default=None, help="dir of MSA-Transformer embeddings")
    ap.add_argument("--checkpoints-dir", default=None, help="dir with head ckpts")
    ap.add_argument("--rm-ckpt-dir", default=None, help="dir with RM_seed*.pt")
    ap.add_argument("--threshold", type=float, default=None, help="binary-call threshold")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all)")
    ap.add_argument("--flavor", default="disorder", help="output subdir name")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo_root = HERE.parent
    saprot_dir = resolve(args.saprot_emb, cfg, "saprot_emb")
    esm2_dir = resolve(args.esm2_emb, cfg, "esm2_emb")
    msat_dir = resolve(args.msat_emb, cfg, "msat_emb")
    ckpt_dir = resolve(args.checkpoints_dir, cfg, "checkpoints_dir",
                       str(repo_root / "checkpoints"))
    rm_dir = resolve(args.rm_ckpt_dir, cfg, "rm_ckpt_dir",
                     str(repo_root / "results/part11/ckpt"))
    out_root = resolve(args.out, cfg, "out", "./output")
    threshold = float(resolve(args.threshold, cfg, "threshold", 0.10))
    threads = resolve(args.threads, cfg, "threads")

    for name, d in (("--saprot-emb", saprot_dir), ("--esm2-emb", esm2_dir),
                    ("--msat-emb", msat_dir)):
        if not d:
            ap.error(f"{name} not provided (CLI or config)")

    import torch  # imported after arg parsing so --help is fast
    if threads:
        torch.set_num_threads(int(threads))
    torch.set_grad_enabled(False)
    from disorderforge_caid.model import DisorderForgeRM

    t0 = time.perf_counter()
    print(f"[DisorderForge-NOX] loading checkpoints ({ckpt_dir}, {rm_dir}) ...",
          file=sys.stderr)
    model = DisorderForgeRM(ckpt_dir, rm_dir, device="cpu")
    print(f"[DisorderForge-NOX] loaded in {time.perf_counter() - t0:.1f}s; "
          f"threads={torch.get_num_threads()} threshold={threshold}", file=sys.stderr)

    disorder_dir = os.path.join(out_root, args.flavor)
    timings = Timings(os.path.join(out_root, "timings.csv"))
    started = time.strftime("%a %b %d %H:%M:%S %Z %Y")

    n_ok = n_err = 0
    seen = set()
    for pid, seq in read_fasta(args.fasta):
        L = len(seq)
        if pid in seen:
            print(f"[WARN] duplicate protein id '{pid}' — later record overwrites "
                  f"earlier output", file=sys.stderr)
        seen.add(pid)
        if L == 0:
            print(f"[WARN] {pid}: empty sequence, skipped", file=sys.stderr)
            n_err += 1
            continue
        try:
            sap = load_embedding(saprot_dir, pid, L, DIM_SAPROT)
            esm = load_embedding(esm2_dir, pid, L, DIM_ESM2)
            msat = load_embedding(msat_dir, pid, L, DIM_MSAT)
            tic = time.perf_counter()
            scores = model.predict(seq, sap, esm, msat)
            ms = (time.perf_counter() - tic) * 1000.0
            binary = (scores >= threshold).astype(int)
            write_caid(disorder_dir, pid, seq, scores, binary)
            timings.add(pid, ms)
            n_ok += 1
        except Exception as e:           # log clearly, keep going, fail at the end
            print(f"[ERROR] {pid} (L={L}) FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            n_err += 1

    timings.write(started_str=started)
    print(f"[DisorderForge-NOX] done: {n_ok} ok, {n_err} failed -> {disorder_dir}",
          file=sys.stderr)
    # GATE 6: a missing/malformed REQUIRED embedding is a hard failure. Successful
    # proteins are still written, but a non-zero exit signals the problem.
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
