# DisorderForge

Residue-level intrinsic protein disorder prediction.

## CAID 4 submission → [`caid_submission/`](caid_submission/)

The CAID 4 predictor (**DisorderForge-NOX**, disorder-NOX flavor) and everything
needed to install, run, and reproduce it live in **[`caid_submission/`](caid_submission/)**:

- [`caid_submission/README.md`](caid_submission/README.md) — install + run + CAID compliance
- [`caid_submission/EMBEDDINGS.md`](caid_submission/EMBEDDINGS.md) — exact embedding-generation spec (for the precompute step)
- [`caid_submission/predict.py`](caid_submission/predict.py) — CPU-only command-line entry point
- [`caid_submission/Dockerfile`](caid_submission/Dockerfile) · [`conda-env.yml`](caid_submission/conda-env.yml) · [`requirements.txt`](caid_submission/requirements.txt)
- [`caid_submission/CAID_SUBMISSION_READINESS.md`](caid_submission/CAID_SUBMISSION_READINESS.md) — validation-gate status

**Method in one line:** a frozen ensemble of five CNN–Transformer heads (3 on
SaProt, 2 on ESM-2 embeddings) fused by probability averaging, refined by a gated
MSA-Transformer residual. CPU-only at prediction time; the three per-residue
embeddings (SaProt, ESM-2, MSA-Transformer) are precomputed and passed as
`.h5`/`.npy` files. CAID3-NOX: rpAP ≈ 0.695 / AUC ≈ 0.864 / MaxF1 ≈ 0.664 at 100% coverage.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r caid_submission/requirements.txt
python caid_submission/tests/test_smoke.py        # sanity check (no GPU needed)
python caid_submission/predict.py --help
```

See [`caid_submission/README.md`](caid_submission/README.md) for full usage.

> Note: the research history (Parts 1–12) and large data/embeddings are not part
> of the submission and are excluded from version control; only source, the
> submission package, and the trained head weights are tracked.
