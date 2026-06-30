# DisorderForge-NOX — CPU-only CAID 4 predictor.
# Build from the REPOSITORY ROOT (so the checkpoints are in build context):
#   docker build -f caid_submission/Dockerfile -t disorderforge-nox .
# Run (mount the CAID-provided embeddings + an output dir):
#   docker run --rm \
#     -v /path/to/input.fasta:/data/input.fasta:ro \
#     -v /path/to/emb/saprot:/emb/saprot:ro \
#     -v /path/to/emb/esm2:/emb/esm2:ro \
#     -v /path/to/emb/msat:/emb/msat:ro \
#     -v /path/to/output:/out \
#     disorderforge-rm \
#       --fasta /data/input.fasta \
#       --saprot-emb /emb/saprot --esm2-emb /emb/esm2 --msat-emb /emb/msat \
#       --out /out --threads 8
FROM python:3.11-slim

# CPU-only torch + light deps, baked at build time (no network at run time).
RUN pip install --no-cache-dir \
        torch>=2.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy h5py PyYAML

WORKDIR /app

# Self-contained model code (vendored network + features; no external src/).
COPY caid_submission/ /app/caid_submission/

# Trained weights (small: 5 heads ~13 MB + 3 RM students ~14 MB).
COPY checkpoints/EXP-S02-v3_seed42_best.pt   /app/checkpoints/
COPY checkpoints/EXP-S02-v3_seed123_best.pt  /app/checkpoints/
COPY checkpoints/EXP-S02-v3_seed456_best.pt  /app/checkpoints/
COPY checkpoints/EXP-E13H_seed42_best.pt     /app/checkpoints/
COPY checkpoints/EXP-E13H_seed123_best.pt    /app/checkpoints/
COPY results/part11/ckpt/RM_seed42.pt        /app/results/part11/ckpt/
COPY results/part11/ckpt/RM_seed123.pt       /app/results/part11/ckpt/
COPY results/part11/ckpt/RM_seed456.pt       /app/results/part11/ckpt/

# Offline + deterministic-ish defaults.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/caid_submission/predict.py"]
CMD ["--help"]
