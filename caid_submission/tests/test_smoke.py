#!/usr/bin/env python3
"""Smoke tests for the DisorderForge-NOX CAID predictor.

Runs without torch for the IO / windowing / ambiguous-char checks; the
end-to-end model test self-skips if torch (or the checkpoints) are unavailable.
Run:  python caid_submission/tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
REPO_ROOT = PKG_ROOT.parent
for p in (str(PKG_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from disorderforge_caid import _core as C            # noqa: E402
from disorderforge_caid.io_caid import (             # noqa: E402
    read_fasta, write_caid)


def test_halo_windows_cover_exactly():
    for L in (1, 50, 1022, 1023, 2000, 5000, 34350):
        wins = C.halo_windows(L, 0)
        covered = np.zeros(L, bool)
        for (es, ee, vs, ve) in wins:
            assert 0 <= es <= vs < ve <= ee <= L
            assert ee - es <= C.WIN_MAX
            assert not covered[vs:ve].any(), f"overlap at L={L}"
            covered[vs:ve] = True
        assert covered.all(), f"gap at L={L}"
    print("ok  halo_windows exact coverage (incl. L=34350)")


def test_fasta_ambiguous_chars():
    seq = "MKTAYIAKQRBZJUOX" * 3   # includes all ambiguous codes
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "in.fasta")
        with open(fp, "w") as fh:
            fh.write(">P_TEST desc here\n" + seq[:24] + "\n" + seq[24:] + "\n")
        got = list(read_fasta(fp))
    assert got == [("P_TEST", seq)], got
    print("ok  FASTA parse tolerates wrapping + ambiguous chars")


def test_lightweight_no_crash_on_ambiguous():
    from src.features.sequence_features import all_lightweight
    seq = "BZJUOXMKTAYIAKQR"
    lw = np.asarray(all_lightweight(seq), np.float32)
    assert lw.shape == (len(seq), 41), lw.shape
    assert np.all(np.isfinite(lw))
    # ambiguous residues zero-fill the one-hot/physico rows (no KeyError/IndexError)
    print("ok  all_lightweight handles B/Z/J/U/O/X (zero-fill, finite)")


def test_assemble_base_dim():
    L = 40
    ml = np.random.RandomState(0).randn(L, C.N_MEM).astype(np.float32)
    lw = np.zeros((L, C.LW_DIM), np.float32)
    base = C.assemble_base(ml, lw)
    assert base.shape == (L, C.BASE_DIM), base.shape
    feat_dim = C.BASE_DIM + C.EVO_DIM + C.MSAT_DIM
    assert feat_dim == 860, feat_dim
    print(f"ok  assemble_base -> [L,{C.BASE_DIM}]; RM d_in = {feat_dim}")


def test_caid_output_format():
    with tempfile.TemporaryDirectory() as d:
        seq = "MKTAYX"
        scores = np.array([0.9, 0.1, 0.5, 0.05, 0.8, 0.2], np.float32)
        binary = (scores >= 0.10).astype(int)
        path = write_caid(d, "P0", seq, scores, binary)
        lines = open(path).read().splitlines()
    assert lines[0] == ">P0"
    cols = lines[1].split("\t")
    assert cols[0] == "1" and cols[1] == "M" and len(cols) == 4
    assert cols[3] == "1"                       # score 0.9 >= 0.10 -> 1
    assert lines[4].split("\t")[3] == "0"       # residue 4, score 0.05 < 0.10 -> 0
    print("ok  .caid format: >id then idx\\tAA\\tscore\\tbinary")


def test_embedding_loader_fails_loud():
    """GATE 6: missing / wrong-length / wrong-dim / NaN embeddings must raise,
    never silently zero-fill."""
    from disorderforge_caid.embeddings import load_embedding
    with tempfile.TemporaryDirectory() as d:
        # missing file
        try:
            load_embedding(d, "NOPE", 10, 1280); raise SystemExit("no raise")
        except FileNotFoundError:
            pass
        # wrong length
        np.save(os.path.join(d, "P1.npy"), np.zeros((8, 1280), np.float32))
        try:
            load_embedding(d, "P1", 10, 1280); raise SystemExit("no raise (len)")
        except ValueError:
            pass
        # wrong dim
        np.save(os.path.join(d, "P2.npy"), np.zeros((10, 768), np.float32))
        try:
            load_embedding(d, "P2", 10, 1280); raise SystemExit("no raise (dim)")
        except ValueError:
            pass
        # NaN
        bad = np.zeros((10, 1280), np.float32); bad[0, 0] = np.nan
        np.save(os.path.join(d, "P3.npy"), bad)
        try:
            load_embedding(d, "P3", 10, 1280); raise SystemExit("no raise (nan)")
        except ValueError:
            pass
        # exact match OK
        np.save(os.path.join(d, "P4.npy"), np.ones((10, 1280), np.float32))
        e = load_embedding(d, "P4", 10, 1280)
        assert e.shape == (10, 1280) and e.dtype == np.float32
    print("ok  embedding loader fails loud on missing/len/dim/NaN; exact OK")


def test_end_to_end_if_available():
    try:
        import torch  # noqa: F401
    except Exception:
        print("skip end-to-end (torch not installed)")
        return
    ckpt = REPO_ROOT / "checkpoints" / "EXP-S02-v3_seed42_best.pt"
    if not ckpt.exists():
        print("skip end-to-end (checkpoints not present)")
        return
    from disorderforge_caid.model import DisorderForgeRM
    model = DisorderForgeRM(REPO_ROOT / "checkpoints",
                            REPO_ROOT / "results/part11/ckpt", device="cpu")
    L = 60
    rng = np.random.RandomState(1)
    seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), L))
    sap = rng.randn(L, 1280).astype(np.float32)
    esm = rng.randn(L, 1280).astype(np.float32)
    msat = rng.randn(L, 768).astype(np.float32)
    p = model.predict(seq, sap, esm, msat)
    assert p.shape == (L,) and np.all((p >= 0) & (p <= 1))
    print(f"ok  end-to-end predict -> [L] probs in [0,1] (mean={p.mean():.3f})")


if __name__ == "__main__":
    test_halo_windows_cover_exactly()
    test_fasta_ambiguous_chars()
    test_lightweight_no_crash_on_ambiguous()
    test_assemble_base_dim()
    test_caid_output_format()
    test_embedding_loader_fails_loud()
    test_end_to_end_if_available()
    print("\nALL SMOKE TESTS PASSED")
