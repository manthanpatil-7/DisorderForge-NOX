# BOX_CHECKLIST — DisorderForge-NOX CAID validation

> **Note (record):** these gates were executed and PASSED on the box on
> 2026-07-01 — see `CAID_SUBMISSION_READINESS.md` for measured results. The
> GPU parity/reference harness (`caid_box_parity.py`) and the embedding
> extractors run against the full research environment and are **not part of
> this submission repo**; they live in the authors' private workspace. The
> predictor itself (`predict.py`) is fully self-contained and CPU-only. This
> checklist is retained as documentation of the validation procedure.

Exact, copy-paste commands to validate the predictor on the **production box**.
The predictor is CPU-only; the parity *reference* needs GPU + the research env.

```bash
export DF="$PWD"          # repo root (contains caid_submission/, checkpoints/)
cd "$DF"
```

---

## 1. Environment & hardware inspection

```bash
uname -a
python3 --version
nproc                                   # CPU threads available
free -g 2>/dev/null || vm_stat          # RAM
nvidia-smi || echo "no GPU (fine for the predictor; needed only for parity ref)"
df -h "$DF"
```

## 2. Install in a FRESH environment

```bash
python3 -m venv /tmp/df_caid_env
source /tmp/df_caid_env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -r caid_submission/requirements.txt
python -c "import torch, numpy, h5py, yaml; print('deps OK', torch.__version__)"
python caid_submission/tests/test_smoke.py        # must end: ALL SMOKE TESTS PASSED
```

## 3. Production-checkpoint discovery

```bash
ls -la checkpoints/EXP-S02-v3_seed42_best.pt checkpoints/EXP-S02-v3_seed123_best.pt \
       checkpoints/EXP-S02-v3_seed456_best.pt checkpoints/EXP-E13H_seed42_best.pt \
       checkpoints/EXP-E13H_seed123_best.pt
ls -la results/part11/ckpt/RM_seed42.pt results/part11/ckpt/RM_seed123.pt \
       results/part11/ckpt/RM_seed456.pt
python - <<'PY'
import torch
for f in ["checkpoints/EXP-S02-v3_seed42_best.pt","results/part11/ckpt/RM_seed42.pt"]:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    print(f, "keys:", list(ck.keys())[:6])
PY
```

## 4–6. Parity (GATES 1, 2, 3 in one run — needs GPU for the reference path)

```bash
python caid_submission/scripts/box_parity_test.py --track caid3_nox
# -> prints GATE 1 (≤1022 diff distribution), GATE 2 (full metrics+coverage),
#    GATE 3 (per long-protein + pooled delta); writes results/part_caid/parity_report.json
cat results/part_caid/parity_report.json
```

Read off and record:
- **GATE 1:** `max_abs`, `mean_abs`, `p99_9_abs`, `n_residues`, `n_non_identical_gt_tol` (target max ≈ 0, ≤ 1e-6 absent a documented numerical reason).
- **GATE 2:** `rpAP`/`AUC`/`MaxF1`/`MCC`, `protein_coverage`, `n_res`, `nan_inf_count` (RM band rpAP 0.6946–0.6950, AUC 0.8638–0.8641, MaxF1 0.6643–0.6646; coverage 100%; NaN/Inf 0).
- **GATE 3:** per-protein `max_abs`/`boundary_mean_abs`/`interior_mean_abs` and `pooled_rpAP_delta_container_minus_ref` (must be immaterial; if material, fix the embedding/windowing contract).

## GATE 4 — embedding-contract parity (the embeddings CAID will actually mount)

The parity run above generates embeddings via `encoder_fullres` = the exact
halo windowing of EMBEDDINGS.md. To prove the *contract* (not just the code),
also generate embeddings with the repo's extraction scripts and confirm the
predictor reproduces GATE 2, and run a sensitivity check against an alternative
chunking:

```bash
# (a) generate per EMBEDDINGS.md (exact model/layer/3Di/MSA), e.g.:
#   scripts/colab_p5_ph0_s03_saprot_extract.py   (SaProt 1280, AA+3Di, BOS/EOS stripped)
#   <ESM-2 t33 650M last layer, BOS/EOS stripped>
#   scripts/colab_p6_ph0_s02_msat_extract.py     (MSA-T 768, query row, 1023-col tiling)
# write them to  /tmp/emb/{saprot,esm2,msat}/<id>.h5   (dataset key 'embedding')

# (b) run the packaged predictor on those mounted embeddings:
python caid_submission/predict.py \
    --fasta data/benchmarks/caid3/disorder_nox.fasta \
    --saprot-emb /tmp/emb/saprot --esm2-emb /tmp/emb/esm2 --msat-emb /tmp/emb/msat \
    --out /tmp/caid_out --threads "$(nproc)"
# (c) score /tmp/caid_out/disorder/*.caid vs labels and confirm GATE-2 metrics.
```
Verify against EMBEDDINGS.md: model versions, selected layers, BOS/EOS removal,
dim, dtype, residue alignment, ambiguous-AA handling, MSA depth/subsampling,
SaProt 3Di construction, long-sequence windowing, overlap/halo rules.

## 7. Docker build (from a clean checkout, repo root context)

```bash
docker build -f caid_submission/Dockerfile -t disorderforge-rm .
docker image inspect disorderforge-rm --format 'image size: {{.Size}} bytes'
```

## 8. Docker OFFLINE inference

```bash
docker run --rm --network none \
  -v "$DF/caid_submission/examples/example.fasta:/data/input.fasta:ro" \
  -v /tmp/emb/saprot:/emb/saprot:ro -v /tmp/emb/esm2:/emb/esm2:ro \
  -v /tmp/emb/msat:/emb/msat:ro -v /tmp/dout:/out \
  disorderforge-rm \
    --fasta /data/input.fasta \
    --saprot-emb /emb/saprot --esm2-emb /emb/esm2 --msat-emb /emb/msat \
    --out /out --threads 8
ls -R /tmp/dout            # expect disorder/<id>.caid + timings.csv
echo "exit: $?"
```

## 9. CPU-only execution + 10. RAM/thread/runtime monitoring

```bash
docker run --rm --network none --cpus 24 --memory 48g \
  -v "$DF/data/benchmarks/caid3/disorder_nox.fasta:/data/input.fasta:ro" \
  -v /tmp/emb/saprot:/emb/saprot:ro -v /tmp/emb/esm2:/emb/esm2:ro \
  -v /tmp/emb/msat:/emb/msat:ro -v /tmp/dout2:/out \
  disorderforge-rm --fasta /data/input.fasta \
    --saprot-emb /emb/saprot --esm2-emb /emb/esm2 --msat-emb /emb/msat \
    --out /out --threads 24 &
DPID=$!
# monitor peak RAM / threads / wall clock while it runs:
while kill -0 $DPID 2>/dev/null; do
  docker stats --no-stream --format '{{.MemUsage}} {{.PIDs}}' 2>/dev/null
  sleep 5
done
cat /tmp/dout2/timings.csv | head     # per-protein milliseconds
# confirm: peak RAM < 48 GB, threads ≤ 24, no GPU used, longest protein < 6 h
```

## 11. Output-format validation

```bash
python - <<'PY'
import glob, sys
bad = 0
for f in glob.glob("/tmp/dout2/disorder/*.caid"):
    lines = open(f).read().splitlines()
    assert lines[0].startswith(">"), f
    for ln in lines[1:]:
        c = ln.split("\t")
        assert len(c) == 4, (f, ln)
        int(c[0]); float(c[2]); assert c[3] in ("0", "1"), (f, ln)
        assert 0.0 <= float(c[2]) <= 1.0
print("all .caid files well-formed; one score per residue")
# timings.csv schema
h = open("/tmp/dout2/timings.csv").read().splitlines()
assert any(x.startswith("sequence,milliseconds") for x in h[:2]), h[:2]
print("timings.csv schema OK")
PY
```

## 12. Determinism (two independent processes)

```bash
python caid_submission/predict.py --fasta caid_submission/examples/example.fasta \
  --saprot-emb /tmp/emb/saprot --esm2-emb /tmp/emb/esm2 --msat-emb /tmp/emb/msat \
  --out /tmp/det1 --threads 4
python caid_submission/predict.py --fasta caid_submission/examples/example.fasta \
  --saprot-emb /tmp/emb/saprot --esm2-emb /tmp/emb/esm2 --msat-emb /tmp/emb/msat \
  --out /tmp/det2 --threads 4
diff -r /tmp/det1/disorder /tmp/det2/disorder && echo "DETERMINISTIC: identical .caid"
```

## 13. Failure-mode testing (GATE 6)

```bash
# empty fasta
printf "" > /tmp/empty.fasta
python caid_submission/predict.py --fasta /tmp/empty.fasta \
  --saprot-emb /tmp/emb/saprot --esm2-emb /tmp/emb/esm2 --msat-emb /tmp/emb/msat \
  --out /tmp/fm; echo "empty exit: $?"            # expect 0, no crash

# missing embedding -> must be a CLEAR non-zero failure, NOT silent zero-fill
printf ">MISSING_X\nMKTAYIAKQR\n" > /tmp/miss.fasta
python caid_submission/predict.py --fasta /tmp/miss.fasta \
  --saprot-emb /tmp/emb/saprot --esm2-emb /tmp/emb/esm2 --msat-emb /tmp/emb/msat \
  --out /tmp/fm; echo "missing-embedding exit: $?"   # expect non-zero, [ERROR] logged

# also exercise: multiline FASTA, duplicate ids, 1-residue seq, >1022 seq,
# B/Z/J/U/O/X, lowercase residues, wrong-length / wrong-dim / NaN embedding,
# malformed .h5, unwritable --out. Each must fail loud (non-zero) or handle
# gracefully per the documented contract — never silently zero-fill a pLM modality.
python caid_submission/tests/test_smoke.py   # covers loader fail-loud + ambiguous chars
```

## 14. Final repository audit (no large/secret leak)

```bash
git ls-files -o --exclude-standard -z | xargs -0 stat -f '%z %N' 2>/dev/null \
  | awk '$1>26214400{printf "%.1f MB  %s\n",$1/1048576,$2}' | sort -rn   # >25 MB
git ls-files -o --exclude-standard | grep -E '\.(pem|key|env)$|id_rsa|credentials' \
  && echo "SECRET LEAK" || echo "no secrets in committable set"
git ls-files -o --exclude-standard | wc -l            # committable file count
git check-ignore -v data/msa_t.tar features/ results/part9   # confirm heavy trees ignored
```

---

When every gate has a recorded PASS, update `CAID_SUBMISSION_READINESS.md` to
**READY** and only then `git add`/`commit`/`push`.
