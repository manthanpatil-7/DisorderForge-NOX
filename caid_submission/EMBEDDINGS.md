# Embedding generation spec (for CAID precompute)

DisorderForge-NOX consumes **three** precomputed per-residue embeddings per
sequence. This document gives the exact model, layer and preprocessing for each,
so the precomputed arrays match what the trained heads expect.

**File convention (all three):** one file per protein in a dedicated folder,
named `<protein_id>.h5` (dataset key `embedding`, shape `[L, dim]`, float16/32)
or `<protein_id>.npy` (array `[L, dim]`). `<protein_id>` is the first
whitespace token of the FASTA header. `L` = sequence length; row `i` = residue
`i` (no BOS/EOS rows). The loader tolerates `.npy`/`.h5`, a transposed
`[dim, L]`, and length mismatch (truncate/zero-pad).

Each section below gives the exact public model, layer, and preprocessing needed
to reproduce the embeddings the trained heads expect.

---

## 1. SaProt — `westlake-repl/SaProt_650M_AF2`, dim 1280

Structure-aware pLM. Needs a per-residue **3Di** structural token string, from
an AlphaFold/ColabFold structure via Foldseek.

1. **Structure:** AlphaFold2 / ColabFold model for the sequence (mmCIF/PDB).
   *(CAID note: do not bundle ColabFold — run it once and mount the structure.)*
2. **3Di tokens:** `foldseek structureto3didescriptor` on the structure → a 3Di
   string of length `L` (lowercase letters; `#` / `X` for missing/low-confidence).
3. **SaProt input:** interleave amino acid (UPPER) + 3Di (lower) per residue:
   residue `i` → `AA[i].upper() + 3Di[i].lower()` (e.g. `M` + `d` → `Md`),
   concatenated into one string of `L` combined tokens.
4. **Embed:** tokenize with the SaProt tokenizer, run the encoder
   (`SaProtForMaskedLM(...).esm` / `AutoModel`), take `last_hidden_state`, and
   **drop the BOS and EOS rows** → `[L, 1280]`.

## 2. ESM-2 — `facebook/esm2_t33_650M_UR50D`, dim 1280

Sequence-only. Tokenize the raw sequence, run the encoder, take the **final
layer** `last_hidden_state`, **drop BOS/EOS** → `[L, 1280]`.

## 3. MSA-Transformer — `esm_msa1b_t12_100M`, dim 768

Needs an MSA (a3m) for the query.

1. **MSA:** build an a3m (e.g. ColabFold/MMseqs2 against UniRef30; the training
   set used `uniref30_2302`). *(CAID note: provide the search command; mount the
   precomputed a3m.)*
2. **Subsample:** query as row 0, up to **128** sequences total.
3. **Embed:** run MSA-Transformer; for sequences longer than the column limit,
   tile in chunks of **≤ 1023 columns** (the `+1` is the BOS token). Take the
   **query row (row 0)** of the final-layer representation → `[L, 768]`.

---

## Long sequences (L > 1022) — windowing

The trained heads were applied with **centre-valid halo windowing**
(`WIN_MAX = 1022`, `HALO = 128`, `STRIDE = 766`): each residue is owned by
exactly one 766-wide valid block, embedded inside a ≤1022-wide window with 128
residues of context each side (clipped at termini).

- **L ≤ 1022:** a single full-length embedding is bit-identical to the validated
  pipeline. (89 % of CAID3 NOX proteins.)
- **L > 1022:** ESM-2/SaProt cannot ingest > ~1024 tokens at once, so the
  embedding **must** be produced by windowing. To match the heads exactly, embed
  each 1022-wide window (128-residue halo each side, 766-residue stride) and
  write, for every residue, the value from the window that **owns** it (its
  766-wide valid block). MSA-Transformer column-tiling follows the same idea.

If CAID's standard precompute emits a single full-length embedding via a
different windowing scheme, results for the small > 1022 aa subset may differ by
a negligible margin at window boundaries; the parity test in the submission
README quantifies it.

---

## Quick checklist

- [ ] SaProt `[L,1280]`, AA+3Di interleaved, BOS/EOS stripped
- [ ] ESM-2 `[L,1280]`, final layer, BOS/EOS stripped
- [ ] MSA-Transformer `[L,768]`, query row, ≤128 seqs, 1023-col tiling
- [ ] files named `<protein_id>.h5`/`.npy`, one per sequence, in three folders
- [ ] long proteins embedded with 1022/128/766 halo windowing
