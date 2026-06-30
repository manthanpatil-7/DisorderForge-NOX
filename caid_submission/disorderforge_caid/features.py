"""Lightweight sequence features (Tier 1 — Default Baseline).

Reference: Input Stack Contract §10.1–§10.3

Feature families:
  - One-hot amino acid identity: 20 dims (§10.1)
  - Physicochemical properties: 15 dims (§10.2)
  - Positional / terminal encodings: 6 dims (§10.3)
  Total: 41 dims per residue

All features are computed on-the-fly from the amino acid sequence.
No pre-computation or caching required.
"""

import math

import numpy as np

# ─── Constants ──────────────────────────────────────────────────────

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

ONEHOT_DIM = 20
PHYSICO_DIM = 15
POSITIONAL_DIM = 6
LIGHTWEIGHT_DIM = ONEHOT_DIM + PHYSICO_DIM + POSITIONAL_DIM  # 41

# ─── Physicochemical scales (per standard AA) ───────────────────────
# Each dict maps single-letter AA code → raw value.
# Z-score normalized across the 20 standard AAs at module load.

# Kyte-Doolittle hydrophobicity (§10.2)
_HYDROPHOBICITY_RAW = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Charge at pH 7.0: +1 (K, R), -1 (D, E), 0 (others); H treated as 0
_CHARGE_RAW = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0,
    "G": 0, "H": 0, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

# Molecular weight (daltons, amino acid residue masses)
_MOLWEIGHT_RAW = {
    "A": 89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
    "G": 75.0, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
    "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2, "R": 174.2,
    "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}

# Vihinen flexibility index
_FLEXIBILITY_RAW = {
    "A": 0.984, "C": 0.906, "D": 1.068, "E": 1.094, "F": 0.915,
    "G": 1.031, "H": 0.950, "I": 0.927, "K": 1.102, "L": 0.935,
    "M": 0.952, "N": 1.048, "P": 1.049, "Q": 1.037, "R": 1.008,
    "S": 1.046, "T": 0.997, "V": 0.931, "W": 0.904, "Y": 0.929,
}

# Grantham polarity
_POLARITY_RAW = {
    "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2,
    "G": 9.0, "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9,
    "M": 5.7, "N": 11.6, "P": 8.0, "Q": 10.5, "R": 10.5,
    "S": 9.2, "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2,
}

# Aromaticity: binary (F, W, Y, H = 1; others = 0)
_AROMATICITY_RAW = {
    "A": 0, "C": 0, "D": 0, "E": 0, "F": 1,
    "G": 0, "H": 1, "I": 0, "K": 0, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": 0,
    "S": 0, "T": 0, "V": 0, "W": 1, "Y": 1,
}

# Zamyatnin side-chain volume (Å³)
_VOLUME_RAW = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

# Disorder propensity (amino acid composition bias in known IDRs)
# Based on TOP-IDP scale (Campen et al., 2008)
_DISORDER_PROPENSITY_RAW = {
    "A": 0.06, "C": -0.02, "D": 0.19, "E": 0.74, "F": -0.41,
    "G": 0.16, "H": -0.01, "I": -0.49, "K": 0.59, "L": -0.34,
    "M": -0.19, "N": 0.13, "P": 0.54, "Q": 0.56, "R": 0.18,
    "S": 0.34, "T": 0.04, "V": -0.38, "W": -0.50, "Y": -0.34,
}

# Beta-sheet propensity (Chou-Fasman)
_BETA_PROPENSITY_RAW = {
    "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38,
    "G": 0.75, "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30,
    "M": 1.05, "N": 0.89, "P": 0.55, "Q": 1.10, "R": 0.93,
    "S": 0.75, "T": 1.19, "V": 1.70, "W": 1.37, "Y": 1.47,
}

# Alpha-helix propensity (Chou-Fasman)
_HELIX_PROPENSITY_RAW = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}

# Turn propensity (Chou-Fasman)
_TURN_PROPENSITY_RAW = {
    "A": 0.66, "C": 1.19, "D": 1.46, "E": 0.74, "F": 0.60,
    "G": 1.56, "H": 0.95, "I": 0.47, "K": 1.01, "L": 0.59,
    "M": 0.60, "N": 1.56, "P": 1.52, "Q": 0.98, "R": 0.95,
    "S": 1.43, "T": 0.96, "V": 0.50, "W": 0.96, "Y": 1.14,
}

# Coil propensity (derived as complement)
_COIL_PROPENSITY_RAW = {
    "A": 0.82, "C": 0.78, "D": 1.20, "E": 0.92, "F": 0.67,
    "G": 1.27, "H": 1.05, "I": 0.60, "K": 1.07, "L": 0.64,
    "M": 0.70, "N": 1.21, "P": 1.34, "Q": 0.80, "R": 1.03,
    "S": 1.13, "T": 1.04, "V": 0.66, "W": 0.73, "Y": 0.85,
}

# Solvent accessibility (relative, Janin scale)
_ACCESSIBILITY_RAW = {
    "A": 0.49, "C": 0.26, "D": 0.81, "E": 0.84, "F": 0.42,
    "G": 0.48, "H": 0.66, "I": 0.34, "K": 0.97, "L": 0.40,
    "M": 0.48, "N": 0.78, "P": 0.75, "Q": 0.84, "R": 0.95,
    "S": 0.65, "T": 0.70, "V": 0.36, "W": 0.51, "Y": 0.76,
}


def _zscore_scale(raw_dict: dict[str, float]) -> dict[str, float]:
    """Z-score normalize a scale across the 20 standard AAs."""
    vals = [raw_dict[aa] for aa in STANDARD_AA]
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    if std < 1e-12:
        return {aa: 0.0 for aa in STANDARD_AA}
    return {aa: (raw_dict[aa] - mean) / std for aa in STANDARD_AA}


# Pre-compute z-scored scales at module load
_PHYSICO_SCALES_RAW = [
    _HYDROPHOBICITY_RAW,
    _CHARGE_RAW,
    _MOLWEIGHT_RAW,
    _FLEXIBILITY_RAW,
    _POLARITY_RAW,
    _AROMATICITY_RAW,
    _VOLUME_RAW,
    _DISORDER_PROPENSITY_RAW,
    _BETA_PROPENSITY_RAW,
    _HELIX_PROPENSITY_RAW,
    _TURN_PROPENSITY_RAW,
    _COIL_PROPENSITY_RAW,
    _ACCESSIBILITY_RAW,
]

_PHYSICO_SCALES = [_zscore_scale(s) for s in _PHYSICO_SCALES_RAW]

# Additional binary features (not z-scored):
# - is_proline (binary) — unique structural role in disorder
# - is_glycine (binary) — unique flexibility role
# These bring PHYSICO_DIM to 15
assert len(_PHYSICO_SCALES) + 2 == PHYSICO_DIM


# ─── Build lookup tables as numpy arrays for vectorized access ──────

def _build_onehot_table() -> np.ndarray:
    """Build 256 × 20 lookup table (ASCII ordinal → one-hot row)."""
    table = np.zeros((256, ONEHOT_DIM), dtype=np.float32)
    for aa, idx in AA_TO_IDX.items():
        table[ord(aa)] = 0.0
        table[ord(aa)][idx] = 1.0
    return table


def _build_physico_table() -> np.ndarray:
    """Build 256 × PHYSICO_DIM lookup table."""
    table = np.zeros((256, PHYSICO_DIM), dtype=np.float32)
    for aa in STANDARD_AA:
        row = [scale[aa] for scale in _PHYSICO_SCALES]
        row.append(1.0 if aa == "P" else 0.0)  # is_proline
        row.append(1.0 if aa == "G" else 0.0)  # is_glycine
        table[ord(aa)] = row
    return table


_ONEHOT_TABLE = _build_onehot_table()
_PHYSICO_TABLE = _build_physico_table()


# ─── Public API ─────────────────────────────────────────────────────

def one_hot(sequence: str) -> np.ndarray:
    """One-hot encode amino acid sequence.

    Args:
        sequence: Amino acid sequence string.

    Returns:
        Array of shape [L, 20], float32.
        Non-standard residues (X, U, etc.) → all-zeros row.
    """
    indices = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _ONEHOT_TABLE[indices].copy()


def physicochemical(sequence: str) -> np.ndarray:
    """Compute physicochemical property features.

    Args:
        sequence: Amino acid sequence string.

    Returns:
        Array of shape [L, 15], float32.
        Z-score normalized across standard 20 AAs.
        Non-standard residues → all-zeros (post-normalization mean).
    """
    indices = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _PHYSICO_TABLE[indices].copy()


def positional(sequence: str) -> np.ndarray:
    """Compute positional / terminal encoding features.

    Per Input Stack §10.3:
      - Relative position: position / length → [0, 1]
      - N-terminal proximity: max(0, 1 - position / 30)
      - C-terminal proximity: max(0, 1 - (length - position) / 30)
      - Log protein length: log10(length), broadcast
      - Sine of relative position
      - Cosine of relative position

    Args:
        sequence: Amino acid sequence string.

    Returns:
        Array of shape [L, 6], float32.
    """
    length = len(sequence)
    positions = np.arange(length, dtype=np.float32)

    rel_pos = positions / max(length - 1, 1)  # [0, 1]
    n_term = np.maximum(0.0, 1.0 - positions / 30.0)
    c_term = np.maximum(0.0, 1.0 - (length - 1 - positions) / 30.0)
    log_len = np.full(length, math.log10(max(length, 1)), dtype=np.float32)
    sin_pos = np.sin(rel_pos * math.pi).astype(np.float32)
    cos_pos = np.cos(rel_pos * math.pi).astype(np.float32)

    return np.stack([rel_pos, n_term, c_term, log_len, sin_pos, cos_pos], axis=1)


def local_composition(sequence: str, window: int = 51) -> np.ndarray:
    """Compute local amino acid composition in a sliding window.

    Reference: Input Stack Contract §10.4 (Tier 3)

    For each residue, computes the frequency of each of the 20 standard
    amino acids within a centered window. Padded at termini.

    Args:
        sequence: Amino acid sequence string.
        window: Window size (default 51, odd).

    Returns:
        Array of shape [L, 20], float32.
    """
    L = len(sequence)
    half = window // 2
    oh = one_hot(sequence)  # [L, 20]
    result = np.zeros((L, 20), dtype=np.float32)

    # Cumulative sum for efficient window computation
    cumsum = np.zeros((L + 1, 20), dtype=np.float64)
    cumsum[1:] = np.cumsum(oh, axis=0)

    for i in range(L):
        start = max(0, i - half)
        end = min(L, i + half + 1)
        window_size = end - start
        result[i] = (cumsum[end] - cumsum[start]) / window_size

    return result


LOCAL_COMPOSITION_DIM = 20


def all_lightweight(
    sequence: str,
    include_physico: bool = True,
    include_positional: bool = True,
    include_local_comp: bool = False,
) -> np.ndarray:
    """Compute lightweight features with optional components.

    Default (Tier 1): one-hot + physicochemical + positional = 41 dims.
    With local_comp (Tier 3): adds 20 dims = 61 dims.
    Without physico: 26 dims. Without positional: 35 dims.

    Returns:
        Array of shape [L, D], float32.
    """
    parts = [one_hot(sequence)]
    if include_physico:
        parts.append(physicochemical(sequence))
    if include_positional:
        parts.append(positional(sequence))
    if include_local_comp:
        parts.append(local_composition(sequence))
    return np.concatenate(parts, axis=1)
