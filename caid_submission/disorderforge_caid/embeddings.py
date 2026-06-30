"""Load per-residue embeddings provided by CAID precompute.

CAID convention: one file per protein in a mounted directory, with the file name
(without extension) matching the FASTA protein id, e.g. ">P04637" -> "P04637.h5"
or "P04637.npy". Embeddings are per-residue arrays [L, dim].

This loader is format-tolerant (.h5 or .npy). A REQUIRED pLM embedding that is
missing, malformed, the wrong width, the wrong length, or non-finite raises a
clear error — it is never silently zero-filled (a missing language-model
modality would change the prediction). The per-residue length must equal the
sequence length L exactly (embeddings are stripped of BOS/EOS upstream; see
EMBEDDINGS.md). Returns array[L, dim] float32.
"""
from __future__ import annotations

import os

import numpy as np

# h5 dataset keys we accept, in priority order (our extractors write "embedding").
_H5_KEYS = ("embedding", "emb", "representations", "features", "data")


def _load_raw(path):
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float32)
    if path.endswith(".h5") or path.endswith(".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            key = next((k for k in _H5_KEYS if k in f), None)
            if key is None:
                # fall back to the first 2-D dataset
                key = next((k for k in f.keys()
                            if hasattr(f[k], "ndim") and f[k].ndim == 2), None)
            if key is None:
                raise KeyError(f"{path}: no usable 2-D embedding dataset "
                               f"(keys={list(f.keys())})")
            return np.asarray(f[key], dtype=np.float32)
    raise ValueError(f"unsupported embedding format: {path}")


def find_embedding_file(emb_dir, protein_id):
    """Return the path to protein_id's embedding (any supported ext) or None."""
    for ext in (".h5", ".hdf5", ".npy"):
        p = os.path.join(emb_dir, protein_id + ext)
        if os.path.exists(p):
            return p
    return None


def load_embedding(emb_dir, protein_id, L, dim):
    """Return emb[L, dim] float32 for a REQUIRED embedding, or raise.

    Raises FileNotFoundError (missing), ValueError (wrong width, wrong length,
    non-finite, unreadable). Never silently zero-fills — a missing/short pLM
    modality must surface as a hard failure, not a degraded prediction."""
    path = find_embedding_file(emb_dir, protein_id)
    if path is None:
        raise FileNotFoundError(
            f"no embedding for '{protein_id}' in {emb_dir} (expected "
            f"{protein_id}.h5 or {protein_id}.npy)")
    e = _load_raw(path)
    if e.ndim == 1:                      # [L*dim] flattened -> reshape if divisible
        if e.size % dim == 0:
            e = e.reshape(-1, dim)
        else:
            raise ValueError(f"{path}: 1-D array of size {e.size} "
                             f"not divisible by dim={dim}")
    if e.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D [L,{dim}] array, got {e.shape}")
    if e.shape[1] != dim:
        if e.shape[0] == dim:            # tolerate a transposed [dim, L] array
            e = e.T
        else:
            raise ValueError(
                f"{path}: embedding width {e.shape[1]} != expected {dim}")
    if e.shape[0] != L:
        raise ValueError(
            f"{path}: embedding length {e.shape[0]} != sequence length {L} "
            f"(expect one per-residue row, BOS/EOS stripped — see EMBEDDINGS.md)")
    if not np.all(np.isfinite(e)):
        raise ValueError(f"{path}: embedding contains NaN/Inf")
    return np.ascontiguousarray(e, dtype=np.float32)
