"""Iterative refinement at inference (Part 5 Phase 3 EXP-A02).

Reference: P5_05 §P5-Ph3-S04; CLAUDE.md rule 34 (max 3 passes).

Mechanism (only applicable when the model has a self-refinement channel):
  Pass 1: feed Part 4 predictions in the SR channel.
  Pass 2: feed pass-1 predictions back into the SR channel.
  Pass 3: feed pass-2 predictions back into the SR channel.

Per-pass rpAP must be non-decreasing; if pass N < pass N-1, return pass N-1.

Phase 2 best (= Phase 1 best, EXP-S02 SaProt alone) does NOT include a
self-refinement channel — there is no SR slot to iterate on. This module
exposes `applicable_to_view(view_id)` and a `IterativeRefinementSkipped`
helper for the Phase 3 trainer to record the skip cleanly.
"""

from __future__ import annotations

import numpy as np
from src.models.routing import VIEWS


def applicable_to_view(view_id: str) -> bool:
    """True iff the view contains a self-refinement channel."""
    if view_id not in VIEWS:
        return False
    return VIEWS[view_id].requires_self_refinement


class IterativeRefinementSkipped(Exception):
    """Raised when iterative refinement is requested but the view has no SR slot."""


def iterate(
    model,
    forward_fn,
    features: dict,        # {acc: [L, D] np.float32}
    channel_slices: dict,  # {acc: {channel_name: (s, e)}}
    max_passes: int = 3,
    early_stop_metric=None,  # callable(predictions_dict) -> float
) -> dict:
    """Run multi-pass inference, replacing SR channel with previous-pass output.

    Returns:
        dict with keys:
            'pass_predictions': list of {acc: [L]} per-pass predictions
            'best_pass_idx': index of best pass per `early_stop_metric`
            'kl_to_part4': KL divergence between Part 4 (pass-1 SR input) and
                           model's pass-N predictions, per pass.
    """
    # Verify SR channel exists in at least one protein's slice spec
    has_sr = any("self_refinement" in cs for cs in channel_slices.values())
    if not has_sr:
        raise IterativeRefinementSkipped(
            "No self_refinement channel in feature slices. Iterative refinement "
            "is not applicable to this model/view; skip per planning doc §S04."
        )

    pass_preds = []
    metrics = []
    kls = []

    current_features = {a: f.copy() for a, f in features.items()}
    part4_sr_snapshot = {}  # store original Part 4 SR to compute KL

    for p in range(max_passes):
        preds = {}
        for acc, feats in current_features.items():
            cs = channel_slices.get(acc, {})
            x_pred = forward_fn(model, feats)  # [L] sigmoid prob
            preds[acc] = x_pred

            if "self_refinement" in cs:
                s, e = cs["self_refinement"]
                if p == 0:
                    part4_sr_snapshot[acc] = current_features[acc][:, s:e].copy()
                # Replace SR channel with this pass's prediction (broadcast to [:, 1])
                current_features[acc][:, s:e] = x_pred.reshape(-1, 1)

        pass_preds.append(preds)

        # Diagnostic: KL divergence vs original Part 4 predictions
        if part4_sr_snapshot:
            kl = 0.0; n = 0
            for acc, p4 in part4_sr_snapshot.items():
                pn = preds[acc]
                p4_flat = p4.reshape(-1).clip(1e-6, 1 - 1e-6)
                pn_flat = pn[:len(p4_flat)].clip(1e-6, 1 - 1e-6)
                L = min(len(p4_flat), len(pn_flat))
                kl += float(np.mean(
                    p4_flat[:L] * np.log(p4_flat[:L] / pn_flat[:L])
                    + (1 - p4_flat[:L]) * np.log((1 - p4_flat[:L]) / (1 - pn_flat[:L]))
                ))
                n += 1
            kls.append(kl / max(n, 1))

        if early_stop_metric is not None:
            m = early_stop_metric(preds)
            metrics.append(m)
            if p > 0 and m < metrics[p - 1]:
                # Pass degraded — stop, return pass p-1
                return {
                    "pass_predictions": pass_preds,
                    "best_pass_idx": p - 1,
                    "kl_to_part4": kls,
                    "metrics_per_pass": metrics,
                    "stopped_early": True,
                }

    best_idx = (max(range(len(metrics)), key=lambda i: metrics[i])
                if metrics else len(pass_preds) - 1)
    return {
        "pass_predictions": pass_preds,
        "best_pass_idx": best_idx,
        "kl_to_part4": kls,
        "metrics_per_pass": metrics,
        "stopped_early": False,
    }
