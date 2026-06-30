# DisorderForge-NOX — CAID 4 submission

Per-residue intrinsic-disorder predictor. **CPU-only, offline, no GPU.** The
heavy protein-language-model work is done *upstream* by CAID precompute: this
predictor consumes three precomputed per-residue embeddings and runs a small
ensemble of CNN–Transformer heads plus a gated MSA-Transformer residual.

- **Flavor:** `disorder` — **this is the disorder-NOX submission ONLY.** It does
  not produce Binding, Binding-IDR, Linker, or PDB outputs and makes no such claim.
- **Held-out performance (CAID3-NOX, 100 % residue coverage):** rpAP 0.695 / AUC 0.864 / MaxF1 0.664
- **Footprint:** ~107 MB of weights, < 2 GB RAM at runtime, well under the 48 GB / 24-thread / 6-h limits.

> **Validation status: implementation-complete, validation-pending.** The decisive
> parity / Docker / robustness gates run on the production box — see
> [BOX_CHECKLIST.md](BOX_CHECKLIST.md) and [CAID_SUBMISSION_READINESS.md](CAID_SUBMISSION_READINESS.md).
> Do not submit until every gate records a PASS.

---

## What the method needs (precomputed by CAID)

For every input sequence the predictor reads **three** per-residue embedding
files (one folder per embedding type, `<protein_id>.h5` or `<protein_id>.npy`,
shape `[L, dim]`). See [EMBEDDINGS.md](EMBEDDINGS.md) for the **exact** model
versions, layers and generation commands.

| Embedding | Model | dim | Needs |
|---|---|---|---|
| SaProt | `westlake-repl/SaProt_650M_AF2` | 1280 | AlphaFold structure → Foldseek 3Di |
| ESM-2 | `facebook/esm2_t33_650M_UR50D` | 1280 | sequence only |
| MSA-Transformer | `esm_msa1b_t12_100M` | 768 | an MSA (a3m) |

No language-model weights, MSAs, structures, or databases are bundled in the
container (per CAID best practice) — only our trained heads.

## Install

```bash
# option A: pip (CPU torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# option B: conda
conda env create -f conda-env.yml && conda activate disorderforge-rm
```

## Run (command line)

```bash
python caid_submission/predict.py \
    --fasta input.fasta \
    --saprot-emb /mnt/emb/saprot \
    --esm2-emb   /mnt/emb/esm2 \
    --msat-emb   /mnt/emb/msat \
    --out ./output \
    --threads 8
```

Outputs:
- `output/disorder/<id>.caid` — one file per protein, `index⇥AA⇥score⇥binary`
- `output/timings.csv` — per-sequence runtime (`sequence,milliseconds`)

Paths can also be set in [config.yaml](config.yaml); CLI flags override it. No
hard-coded absolute paths — checkpoint paths resolve against the repo root.

## Run (Docker)

```bash
# build from the REPOSITORY ROOT (context needs the checkpoints)
docker build -f caid_submission/Dockerfile -t disorderforge-rm .

docker run --rm \
  -v "$PWD/input.fasta:/data/input.fasta:ro" \
  -v /path/to/emb/saprot:/emb/saprot:ro \
  -v /path/to/emb/esm2:/emb/esm2:ro \
  -v /path/to/emb/msat:/emb/msat:ro \
  -v "$PWD/output:/out" \
  disorderforge-rm \
    --fasta /data/input.fasta \
    --saprot-emb /emb/saprot --esm2-emb /emb/esm2 --msat-emb /emb/msat \
    --out /out --threads 8
```

## CAID requirement compliance

| Requirement | Status |
|---|---|
| Runs on CPU, no GPU | ✅ pure-CPU torch; container has no CUDA |
| Command-line, non-interactive | ✅ `predict.py` |
| Works offline | ✅ no network at runtime (`HF_*_OFFLINE=1`); embeddings mounted |
| < 48 GB RAM | ✅ ~1–2 GB (small heads, one protein at a time) |
| ≤ 24 threads | ✅ `--threads` / `OMP_NUM_THREADS` |
| < 6 h per sequence | ✅ ms–seconds per protein (only the small heads run here) |
| Ambiguous chars B/Z/J/U/O/X | ✅ never crash (lightweight features zero-fill; pLM handled upstream) |
| No public DBs / pLM weights bundled | ✅ embeddings precomputed by CAID; only our heads shipped |
| No hard-coded paths | ✅ config + CLI args |
| Dependency descriptor | ✅ `requirements.txt`, `conda-env.yml`, `Dockerfile` |

## Layout

```
caid_submission/
├── predict.py                       # CLI entrypoint
├── config.yaml                      # default paths / threshold / threads
├── disorderforge_caid/
│   ├── model.py                     # 5 heads + 3 RM students; full forward
│   ├── cnn_transformer_hybrid.py    # vendored head network (torch)
│   ├── features.py                  # vendored 41-d lightweight features
│   ├── rm_head.py                   # vendored RM gated-residual head (torch)
│   ├── _core.py                     # vendored pure numerics (windowing, base feats)
│   ├── embeddings.py                # .h5/.npy loader (id-keyed, length-checked)
│   └── io_caid.py                   # FASTA in, .caid + timings.csv out
├── scripts/compute_threshold.py     # (optional) re-derive binary threshold on val
├── tests/test_smoke.py              # IO / windowing / ambiguous-char + end-to-end
├── Dockerfile · conda-env.yml · requirements.txt
├── README.md · EMBEDDINGS.md
```

The package is **fully self-contained**: the head network, the 41-d lightweight
features, and all numerics are vendored into `disorderforge_caid/` (torch +
numpy only). No external/research code is imported — nothing beyond this folder
and the trained weights is needed to run the predictor.

## Notes

- The disorder **score** column drives the threshold-free metrics CAID ranks on
  (AP / AUC). Scores are well-ranked but low-magnitude (SaProt is
  under-calibrated); the binary column uses a validation operating point
  (default 0.10, see `scripts/compute_threshold.py`).
- For sequences ≤ 1022 aa the precomputed-embedding path is bit-identical to the
  validated pipeline; for longer sequences the heads are applied with the same
  1022/128/766 halo windowing (see EMBEDDINGS.md).
