"""Test-Time Augmentation via Monte Carlo dropout (P4 Phase 3 S02).

Reference: P4_00 Amendment T-6.

Standard `model.eval()` disables dropout layers, which is undesirable for
TTA — we WANT dropout active so each forward pass produces a different
prediction sampled from the model's posterior over weights. Per the CLAUDE.md
gotcha §: "TTA requires dropout at inference time. This is NON-STANDARD —
verify that model.eval() does NOT disable dropout for TTA runs. Use a custom
inference mode."

`enable_dropout_only` puts the model in eval mode (LayerNorm/BatchNorm in
inference) AND then re-enables training mode on every nn.Dropout module
(including nn.Dropout1d/2d/3d, nn.AlphaDropout, etc.).

`tta_forward` runs N stochastic forward passes and returns the mean and
std prediction tensors per residue.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn

DROPOUT_TYPES = (
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
    nn.AlphaDropout, nn.FeatureAlphaDropout,
)


def enable_dropout_only(model: nn.Module) -> int:
    """Set model to eval(), then flip every Dropout submodule back to train().
    Returns the count of Dropout modules flipped."""
    model.eval()
    n = 0
    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES):
            module.train()
            n += 1
    return n


@contextmanager
def tta_mode(*models: nn.Module):
    """Context manager — enable dropout-only on `models` for the duration of
    the block; restore eval() afterwards."""
    counts = [enable_dropout_only(m) for m in models]
    try:
        yield counts
    finally:
        for m in models:
            m.eval()


@torch.no_grad()
def tta_forward(
    forward_fn: Callable[[], torch.Tensor],
    n_passes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `forward_fn()` n_passes times under dropout-active inference.

    `forward_fn` must already capture the model + inputs (closure or partial).
    The caller is responsible for putting the relevant models into TTA mode
    via `enable_dropout_only()` or `tta_mode()` BEFORE calling this.

    Returns:
        (mean_probs, std_probs): each shape matches forward_fn() output.
    """
    if n_passes < 2:
        raise ValueError(f"n_passes must be >= 2 to compute std, got {n_passes}")
    samples = []
    for _ in range(n_passes):
        out = forward_fn()
        if isinstance(out, torch.Tensor):
            samples.append(out.detach())
        else:
            raise TypeError(f"forward_fn must return a torch.Tensor, got {type(out)}")
    stacked = torch.stack(samples, dim=0)  # [N, ...]
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    return mean, std
