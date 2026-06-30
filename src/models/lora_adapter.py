"""LoRA adapter integration for ESM-2-650M.

Reference: P2_00 Amendment I-1; P2_02 §P2-Ph1-S07.

Implements Low-Rank Adaptation (Hu et al. 2021) for ESM-2's attention layers.
Only the Q and V projections in the last N transformer layers are adapted; all
other model weights are frozen. The expert head (A/B/C from Part 1) remains
unchanged and is trained alongside the LoRA adapter via a dual-LR optimizer
(see Amendment T-2).

LoRA forward (per adapted Linear):
    y = W x + (B A x) * (alpha / rank)
where W is the frozen pretrained weight, A has shape (rank, in) with Kaiming
init, B has shape (out, rank) with zero init. At initialization the adapter
output is zero, so the model starts identical to frozen ESM-2.

Gradient checkpointing is applied to each ESM-2 transformer layer via
torch.utils.checkpoint.checkpoint — required to fit LoRA training in GPU
memory (Amendment T-3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


# ─── LoRA module ────────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """Wrap an nn.Linear with a LoRA low-rank adapter.

    The original Linear is kept and frozen; two new parameters (lora_A, lora_B)
    carry the adaptation. Dropout is applied on the LoRA branch only.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = original.in_features
        self.out_features = original.out_features

        # Keep and freeze the original Linear.
        self.original = original
        for p in self.original.parameters():
            p.requires_grad_(False)

        # LoRA parameters.
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        # Standard LoRA init: A ~ Kaiming, B = 0 (so adapter delta = 0 at start).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original(x)
        # Cast LoRA parameters to match input dtype so this adapter works with
        # half-precision pLMs (e.g. fair-esm loaded with .half() on CUDA) without
        # requiring torch.cuda.amp.autocast. The cast is differentiable — grads
        # flow back to lora_A / lora_B tensors in their native dtype.
        A = self.lora_A.to(x.dtype)
        B = self.lora_B.to(x.dtype)
        adapted = self.dropout(x) @ A.T @ B.T
        return orig + adapted * self.scaling

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}")


# ─── Apply LoRA to ESM-2 ────────────────────────────────────────────────────


def apply_lora_to_esm2(
    model: nn.Module,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.05,
    num_layers_to_adapt: int = 6,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    enable_gradient_checkpointing: bool = True,
) -> nn.Module:
    """Apply LoRA to the last N layers of an ESM-2 model, in-place.

    Args:
        model: ESM-2 model from esm.pretrained.esm2_t33_650M_UR50D().
        rank: LoRA rank (Amendment I-1: 8–16).
        alpha: LoRA scaling; defaults to 2*rank per standard convention.
        dropout: Dropout on LoRA branch (Amendment I-1: 0.05–0.1).
        num_layers_to_adapt: Number of final transformer layers to wrap (4–8).
        target_modules: Attribute names on each attention module to adapt.
                        Amendment I-1 specifies ("q_proj", "v_proj").
        enable_gradient_checkpointing: Enable activation checkpointing on each
                                       transformer layer's forward (Amendment T-3).

    Returns:
        The same model, with LoRA adapters inserted. Base weights are frozen.
    """
    if alpha is None:
        alpha = 2.0 * rank
    if not (4 <= num_layers_to_adapt <= 8):
        raise ValueError(
            f"num_layers_to_adapt={num_layers_to_adapt} outside Amendment I-1 range [4, 8]"
        )
    if not (8 <= rank <= 16):
        raise ValueError(
            f"rank={rank} outside Amendment I-1 range [8, 16]"
        )

    # Freeze all parameters first. LoRA params added below will be trainable.
    for p in model.parameters():
        p.requires_grad_(False)

    layers = _get_transformer_layers(model)
    total_layers = len(layers)
    if num_layers_to_adapt > total_layers:
        raise ValueError(
            f"num_layers_to_adapt={num_layers_to_adapt} exceeds model depth {total_layers}"
        )
    start_idx = total_layers - num_layers_to_adapt

    adapted = 0
    for layer_idx in range(start_idx, total_layers):
        layer = layers[layer_idx]
        attn = _get_self_attention(layer)
        for name in target_modules:
            if not hasattr(attn, name):
                raise AttributeError(
                    f"attention module has no attribute '{name}' at layer {layer_idx}"
                )
            original = getattr(attn, name)
            if not isinstance(original, nn.Linear):
                raise TypeError(
                    f"{name} at layer {layer_idx} is {type(original).__name__}, expected nn.Linear"
                )
            setattr(attn, name, LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout))
            adapted += 1

    if enable_gradient_checkpointing:
        _wrap_layers_with_checkpointing(layers, start_idx, total_layers)

    model._lora_config = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "num_layers_to_adapt": num_layers_to_adapt,
        "target_modules": list(target_modules),
        "adapted_modules_count": adapted,
        "total_transformer_layers": total_layers,
        "adapted_layer_indices": list(range(start_idx, total_layers)),
        "gradient_checkpointing": enable_gradient_checkpointing,
    }
    return model


def _get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer layers list on an ESM-2 or ESM-2-like model."""
    # fair-esm ESM-2 exposes .layers directly.
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers
    # Fallback: search for a ModuleList of TransformerLayer-like modules.
    for _name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            return child
    raise AttributeError("model has no discoverable transformer layer list")


def _get_self_attention(layer: nn.Module) -> nn.Module:
    """Return the self-attention submodule of a transformer layer."""
    for attr in ("self_attn", "attention", "self_attention"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    raise AttributeError(f"layer {type(layer).__name__} has no known self-attention attribute")


def _wrap_layers_with_checkpointing(layers: nn.ModuleList, start: int, end: int) -> None:
    """Wrap each layer's forward with torch.utils.checkpoint.checkpoint.

    Applied only to adapted layers (where activations would otherwise balloon).
    Non-reentrant mode is used to cooperate with frozen-param branches.
    """
    for idx in range(start, end):
        layer = layers[idx]
        if getattr(layer, "_lora_checkpointed", False):
            continue  # idempotent
        original_forward = layer.forward

        def checkpointed_forward(*args, _of=original_forward, **kwargs):
            if torch.is_grad_enabled():
                return cp.checkpoint(_of, *args, use_reentrant=False, **kwargs)
            return _of(*args, **kwargs)

        layer.forward = checkpointed_forward  # type: ignore[method-assign]
        layer._lora_checkpointed = True


# ─── Param-group helpers ─────────────────────────────────────────────────────


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return all LoRA adapter parameters in a LoRA-wrapped model."""
    params: list[nn.Parameter] = []
    for n, p in model.named_parameters():
        if ("lora_A" in n or "lora_B" in n) and p.requires_grad:
            params.append(p)
    return params


def get_head_params(head: nn.Module) -> list[nn.Parameter]:
    """Return all trainable parameters of an expert head module."""
    return [p for p in head.parameters() if p.requires_grad]


def count_lora_parameters(
    rank: int,
    embed_dim: int,
    num_layers_to_adapt: int,
    num_target_modules: int = 2,
) -> int:
    """Expected LoRA parameter count.

    Each adapted Linear has rank*(in_features + out_features) params. For square
    Q/V projections (in=out=embed_dim), that's 2*rank*embed_dim per module.
    """
    per_module = 2 * rank * embed_dim
    return per_module * num_target_modules * num_layers_to_adapt


def lora_summary(model: nn.Module) -> dict:
    """Report trainable/frozen param counts for a LoRA-adapted model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for p in get_lora_params(model))
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "lora_parameters": lora,
        "frozen_parameters": total - trainable,
        "trainable_fraction": round(trainable / total, 6) if total else 0.0,
        "lora_config": getattr(model, "_lora_config", None),
    }


# ─── Composite pLM + head model for LoRA training ───────────────────────────


class LoRAExpertModel(nn.Module):
    """pLM + LoRA adapter + expert head, composed as a single trainable module.

    Forward contract:
      input:  batch_tokens   int tensor [B, T]   (ESM-2 token ids, with BOS/EOS)
              lengths        int tensor [B]      (per-protein residue counts, excluding BOS/EOS)
      output: logits         float tensor [B, L_max, 1]  (L_max = max residue count in batch)

    The pLM forward is included in the autograd graph so LoRA gradients flow.
    Intermediate activations are checkpointed per-layer inside the pLM (see
    _wrap_layers_with_checkpointing).

    Auto-dispatch: if the max residue count in the batch exceeds window_size
    (default 1022, ESM-2's context window), this module transparently switches
    to a per-protein sliding-window forward that:
      1. Splits each long protein into overlapping windows using Part 1's
         compute_window_schedule (window=1022, overlap=256, C-terminal-anchored).
      2. Runs pLM+LoRA on each window INDEPENDENTLY (no cross-window attention —
         same approximation as Part 1 frozen extraction at P3-S03).
      3. Mean-aggregates embeddings in overlap regions via a differentiable
         scatter_add (gradient flows back to each window's LoRA path).
      4. Wraps each window's pLM forward in torch.utils.checkpoint.checkpoint
         to bound backward memory to one window's activations regardless of
         protein length.
    Requires `alphabet` to tokenize windows and `sequences` arg passed to forward
    for batches that contain proteins > window_size.

    Checkpoint save/load (see trainer) persists only {LoRA adapter + head}
    state_dicts — the 2.5GB frozen base is not stored.
    """

    DEFAULT_WINDOW_SIZE = 1022
    DEFAULT_OVERLAP = 256

    def __init__(
        self,
        plm: nn.Module,
        head: nn.Module,
        extraction_layer: int,
        alphabet=None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.plm = plm
        self.head = head
        self.extraction_layer = extraction_layer
        self.alphabet = alphabet  # for sliding-window tokenization (only needed if L>window)
        self.window_size = window_size
        self.overlap = overlap
        self.embed_dim = embed_dim

    def forward(
        self,
        batch_tokens: torch.Tensor,
        lengths: torch.Tensor | None = None,
        extra_features: torch.Tensor | None = None,
        sequences: list[str] | None = None,
    ) -> torch.Tensor:
        """Run pLM + head on tokenized batch.

        Args:
            batch_tokens: [B, T] int tokens with T = max_length + 2 (BOS/EOS).
            lengths: Optional [B] per-protein residue counts for head-side masking.
            extra_features: Optional [B, L, D_extra] of pre-computed per-residue
                features (e.g. Part 1's lightweight 41-dim features) concatenated
                to the pLM's ESM-2 embeddings along the feature axis BEFORE the
                head forward. L must match the stripped hidden length (T-2).
            sequences: Required iff any protein in the batch has residue count
                > self.window_size. List of raw sequence strings (length B) used
                to re-tokenize each window during sliding-window forward.

        Returns:
            Head output (typically [B, L, 1] logits).
        """
        max_len = self._batch_max_len(batch_tokens, lengths)
        if max_len <= self.window_size:
            return self._batched_forward(batch_tokens, lengths, extra_features)
        if sequences is None:
            raise ValueError(
                f"batch contains protein of length {max_len} > window_size={self.window_size}; "
                f"sliding-window forward requires `sequences` (raw strings) to tokenize windows"
            )
        return self._windowed_forward(lengths, extra_features, sequences)

    def _batch_max_len(self, batch_tokens: torch.Tensor, lengths: torch.Tensor | None) -> int:
        if lengths is not None:
            return int(lengths.max().item())
        return batch_tokens.size(1) - 2  # strip BOS/EOS

    def _batched_forward(
        self,
        batch_tokens: torch.Tensor,
        lengths: torch.Tensor | None,
        extra_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Standard single-window batched forward (unchanged from pre-windowing)."""
        results = self.plm(batch_tokens, repr_layers=[self.extraction_layer])
        hidden = results["representations"][self.extraction_layer]
        hidden = hidden[:, 1:-1, :]

        if extra_features is not None:
            if extra_features.shape[1] != hidden.shape[1]:
                raise ValueError(
                    f"extra_features length {extra_features.shape[1]} != "
                    f"pLM hidden length {hidden.shape[1]}"
                )
            hidden = torch.cat([hidden, extra_features.to(hidden.dtype)], dim=-1)

        if lengths is not None:
            try:
                return self.head(hidden, lengths=lengths)
            except TypeError:
                pass
        return self.head(hidden)

    def _windowed_forward(
        self,
        lengths: torch.Tensor | None,
        extra_features: torch.Tensor | None,
        sequences: list[str],
    ) -> torch.Tensor:
        """Per-protein sliding-window forward for batches containing L > window_size.

        Each protein is processed independently so variable-length per-window
        outputs can be stitched back into a full-length embedding before the
        head forward.
        """
        if self.alphabet is None:
            raise AttributeError(
                "LoRAExpertModel.alphabet must be set to use sliding-window forward"
            )
        device = next(self.plm.parameters()).device

        per_protein_logits: list[torch.Tensor] = []
        per_protein_lens: list[int] = []
        for i, seq in enumerate(sequences):
            L = int(lengths[i].item()) if lengths is not None else len(seq)
            per_protein_lens.append(L)
            embedding = self._compute_embedding_windowed(seq, L, device)  # [L, D]
            if extra_features is not None:
                ext = extra_features[i, :L].to(embedding.dtype)
                embedding = torch.cat([embedding, ext], dim=-1)  # [L, D+extra]
            embedding_b = embedding.unsqueeze(0)  # [1, L, D_total]
            len_tensor = torch.tensor([L])
            try:
                logit = self.head(embedding_b, lengths=len_tensor)
            except TypeError:
                logit = self.head(embedding_b)
            per_protein_logits.append(logit.squeeze(0))  # [L, 1]

        max_len = max(per_protein_lens)
        padded = []
        for logit, L in zip(per_protein_logits, per_protein_lens):
            if L < max_len:
                pad_rows = max_len - L
                logit = torch.nn.functional.pad(logit, (0, 0, 0, pad_rows))
            padded.append(logit)
        return torch.stack(padded, dim=0)  # [B, max_len, 1]

    def _compute_embedding_windowed(
        self,
        sequence: str,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """ESM-2 per-residue embedding for one protein, with sliding-window over windows > self.window_size."""
        # Deferred import to avoid circularity at module import time.
        from src.features.esm2_extract import compute_window_schedule

        if length <= self.window_size:
            batch_converter = self.alphabet.get_batch_converter()
            _, _, tokens = batch_converter([("p", sequence)])
            tokens = tokens.to(device)
            return self._forward_tokens(tokens)

        windows = compute_window_schedule(length, self.window_size, self.overlap)
        batch_converter = self.alphabet.get_batch_converter()

        embedding_sum = torch.zeros(
            length, self.embed_dim, device=device, dtype=torch.float32,
        )
        count = torch.zeros(length, device=device, dtype=torch.float32)

        for (w_start, w_end) in windows:
            window_seq = sequence[w_start:w_end]
            _, _, tokens = batch_converter([("p", window_seq)])
            tokens = tokens.to(device)
            # Per-window checkpoint: bounds backward memory to one window's activations
            # regardless of how many windows the protein spans (critical for titin-scale).
            window_emb = torch.utils.checkpoint.checkpoint(
                self._forward_tokens, tokens, use_reentrant=False,
            ).to(torch.float32)
            w_len = w_end - w_start
            idx = torch.arange(
                w_start, w_end, device=device, dtype=torch.long,
            ).unsqueeze(-1).expand(-1, self.embed_dim)
            # Out-of-place scatter_add: differentiable, keeps gradient graph tidy.
            embedding_sum = embedding_sum.scatter_add(0, idx, window_emb)
            count[w_start:w_end] += 1

        # Mean aggregation over overlapping windows. All positions covered by
        # at least one window (compute_window_schedule guarantees this via
        # C-terminal anchor + stride).
        return embedding_sum / count.unsqueeze(-1)

    def _forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """pLM forward on single-window tokens, return [L_win, D] per-residue embedding."""
        results = self.plm(tokens, repr_layers=[self.extraction_layer])
        hidden = results["representations"][self.extraction_layer]
        return hidden[0, 1:-1, :]  # strip batch dim + BOS/EOS

    def trainable_state_dict(self) -> dict:
        """Return state_dict containing only trainable (LoRA + head) params.

        Used by the trainer when checkpointing to avoid persisting the
        ~2.5GB frozen pLM base.
        """
        sd: dict = {}
        # LoRA params (named with 'lora_A' / 'lora_B')
        for n, p in self.plm.named_parameters():
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad:
                sd[f"plm.{n}"] = p.detach().clone()
        # All head params (heads are fully trainable)
        for n, p in self.head.named_parameters():
            sd[f"head.{n}"] = p.detach().clone()
        return sd

    def load_trainable_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Restore LoRA + head weights from a trainable state_dict.

        Ignores any keys that don't match (with strict=False) — useful if the
        checkpoint predates a new non-trainable head buffer.
        """
        own_params = {n: p for n, p in self.named_parameters()}
        missing = []
        for key, tensor in state_dict.items():
            if key in own_params:
                with torch.no_grad():
                    own_params[key].copy_(tensor)
            elif strict:
                missing.append(key)
        if missing and strict:
            raise KeyError(f"unexpected keys in trainable state_dict: {missing[:5]}")
