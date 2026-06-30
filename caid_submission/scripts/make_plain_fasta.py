#!/usr/bin/env python3
"""Write a PLAIN FASTA (>id / sequence) from a CAID benchmark track.

The CAID *benchmark* files are a 3-line scoring format (header / sequence /
label-string). The predictor's real input is a standard FASTA. Use this to make
a plain FASTA for predict.py / Docker / determinism testing.

    python caid_submission/scripts/make_plain_fasta.py --track caid3_nox --out /content/caid3_nox.fasta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import p7_ft_common as F   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="caid3_nox")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d2u = F.dp_to_uniprot()
    rows = F.load_bench(F.BENCHES[args.track], d2u)
    with open(args.out, "w") as fh:
        for dp, uni, seq, lab in rows:
            fh.write(f">{dp}\n{seq}\n")
    print(f"wrote {len(rows)} plain-FASTA sequences -> {args.out}")


if __name__ == "__main__":
    main()
